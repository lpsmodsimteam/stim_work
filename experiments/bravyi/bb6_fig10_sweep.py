#!/usr/bin/env python3
"""Reproduce the Figure-10 BB(6) Relay-BP line of arXiv:2511.15177 with all 3 techniques.

Figure 10 reports the logical error rate vs physical error rate of the distance-6
bivariate-bicycle code ``BB(6) = [[72,12,6]]`` under circuit-level depolarizing noise,
decoded with the Relay-BP decoder, using the three "fail-fast" techniques:

  * Technique I  — failure-spectrum ansatz fit + extrapolation   (importance_sampling.py)
  * Technique II — exact min-weight onset properties (D, w0, f0)  (min_weight.py)
  * Technique III — multi-seeded Metropolis splitting cross-check (splitting.py)

This driver glues those library modules onto the BB(6) memory circuit and is built to
run unattended for hours: the weight-stratified importance-sampling sweep checkpoints
after **every weight**, so a crash resumes with at most one weight of lost work
(mirrors gross_code_sweep.py).

Pipeline (run_all):
  0. (optional) onset scan to size the contiguous WEIGHTS block
  1. Technique II  — circuit fault distance D, optimal onset w0=D/2, onset fraction f0
  2. IS sweep      — sample the failure spectrum f(w), checkpointed, then reweight to LER(p)
  3. Technique I   — fit the ansatz (pinned by w0/f0 from step 1) and extrapolate LER(p)
  4. Technique III — splitting estimate over an overlapping p-ladder (cross-check)
  5. plot LER(p): raw IS points, Technique-I ansatz curve, Technique-III splitting points

Usage (from anywhere in the repo, in the qec env which has relay_bp + ldpc):
    python notebooks/bb_code/bb6_fig10_sweep.py --smoke            # tiny end-to-end (<~1 min)
    python notebooks/bb_code/bb6_fig10_sweep.py --plot             # full production run (hours)
    python notebooks/bb_code/bb6_fig10_sweep.py --onset-scan --plot

The --smoke flag swaps in a tiny Config but exercises *every* code path the production
run hits, so `python ... --smoke` (or tests/test_bb6_fig10.py) catches errors before the
multi-hour job is launched.

Outputs (under --outdir, default notebooks/bb_code/bb6_fig10_out/):
    bb6.spectrum.json   incremental per-weight IS checkpoint (resume source of truth)
    distance.json       Technique-II distance / onset / onset-fraction
    ansatz_fit.json     Technique-I ansatz params + extrapolated LER
    splitting.json      Technique-III ladder estimate
    bb6_fig10.npz       combined arrays (p grid + every curve)
    bb6_fig10.png       the reproduced figure (only with --plot)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

# Force UTF-8 stdout/stderr: this job prints μ/→/← in progress lines, and the default
# Windows console encoding (cp1252) raises UnicodeEncodeError on them — which would crash
# the multi-hour run. Harmless no-op where stdout can't be reconfigured (e.g. pytest capture).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --- locate repo src/ regardless of CWD (mirrors gross_code_sweep.py) ------------
def _add_src_to_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "src" / "bb_code_sim.py").exists():
            sys.path.insert(0, str(cand / "src"))
            return cand
    raise RuntimeError(f"could not locate repo src/ starting from {here}")


REPO_ROOT = _add_src_to_path()

from scipy.special import gammaln  # noqa: E402

from bb_code_sim import (BBCodeSimulator, BBCodeParams, BB_72_12_6, BB_144_12_12,  # noqa: E402
                         RelayBPDecoder)

# Code registry for the --code seam (default BB(6); others are future bicycle codes).
CODES = {"bb72": (BB_72_12_6, "BB(6)=[[72,12,6]]"), "bb144": (BB_144_12_12, "BB(12)=[[144,12,12]]")}
from surface_code_sim import ErrorModel  # noqa: E402
from importance_sampling import (  # noqa: E402
    FailureSpectrum,
    ImportanceSamplingResult,
    _parse_dem,
    _expand,
    _sample_failures_at_weight,
    fit_failure_spectrum,
    logical_error_rate_from_ansatz,
    predict_failure_fraction,
    shots_to_hit_failures,
)
from repo_paths import RUNS  # noqa: E402


# =================================================================================
# Configuration — production vs smoke. Both drive the SAME code paths in run_all;
# only the numbers differ, so --smoke is a faithful dry run of the multi-hour job.
# =================================================================================
@dataclass
class Config:
    label: str

    # code (the --code seam; default BB(6)). build_circuit + the toric symmetry read this, so
    # other bivariate-bicycle codes are a one-line change (set code= and rounds=code.distance).
    code: BBCodeParams = field(default_factory=lambda: BB_72_12_6)
    code_label: str = "BB(6)=[[72,12,6]]"

    # circuit / noise
    p_ref: float = 0.003                 # circuit is built at this physical error rate
    # Measurement-error multiplier (the --p-meas-factor seam): p_meas = p_meas_factor * p_phys.
    # 1.0 = symmetric (paper default). The fail-fast reweighting is rate-agnostic, so a fixed ratio
    # swept by overall strength p stays a valid single-parameter family — no downstream change.
    p_meas_factor: float = 1.0
    # Optional single-channel noise isolation (None/'full' = symmetric). Filters instructions on the
    # built circuit to one physical channel at the SAME base rate: 'cz' (two-qubit DEPOLARIZE2),
    # 'meas'/'prep' (X_ERROR before M / after R), 'idle' (DEPOLARIZE1 not right after H). Applied in
    # build_circuit, so every technique (I/II/III) sees the filtered circuit.
    noise_model: Optional[str] = None
    rounds: int = BB_72_12_6.distance    # syndrome rounds (= d for the memory experiment)

    # Relay-BP decoder settings (paper §2.4 for BB(6): γ0=0.125, leg1=80 it,
    # legs=60 it up to 600 legs, γ~Unif[-0.24,0.66], S=6).
    relay_gamma0: float = 0.125
    relay_pre_iter: int = 80
    relay_num_sets: int = 600
    relay_set_max_iter: int = 60
    relay_gamma_lo: float = -0.24
    relay_gamma_hi: float = 0.66
    relay_stop_nconv: int = 6

    # importance-sampling sweep
    weights: List[int] = field(default_factory=lambda: list(range(2, 60)))
    # True when the weight plan was set explicitly (e.g. --weights), in which case the onset-scan
    # auto-bracket must NOT override it (the user is deliberately skipping/choosing weights).
    weights_explicit: bool = False
    # 500 (was 3000): the IS sweep dominates the runtime, and the Technique-I ansatz pools
    # ~45 weights so it tolerates the larger per-weight binomial noise; only the raw IS points
    # get wider error bars. Override with --shots for tighter raw points.
    shots_per_weight: int = 500
    # Optional per-weight shot overrides {w: T}; weights not listed fall back to shots_per_weight.
    # Pour shots into the rare near-onset bins (tiny f(w) — a flat budget yields 0 failures there and
    # only a 3/T upper limit) and spend fewer on the high-weight bins. The per-weight SE and the
    # bootstrap band are per-weight throughout, so heterogeneous T flows correctly into the error bars.
    shots_by_weight: Optional[Dict[int, int]] = None
    # Adaptive 'hit N failures per weight': sweep weights high->low and size each weight's shots from
    # the extrapolated f(w) to expect ~adaptive_failures failures (T_w = N/f_pred, clamped). The top
    # weights (f≈1) cost ~N shots; cost grows toward the onset. Overrides shots_per_weight/by_weight.
    adaptive: bool = False
    adaptive_failures: int = 50
    adaptive_shots_min: int = 4
    adaptive_shots_max: int = 20000
    adaptive_predict_window: int = 3
    # Sample only every Nth weight (1 = all) for speed; the ansatz interpolates the skipped weights.
    weight_stride: int = 1
    seed: int = 42

    # reweight / extrapolation target grid
    p_lo: float = 1e-4
    p_hi: float = 1e-2
    n_p: int = 30

    # Technique I — ansatz
    ansatz_model: str = "f3"
    pin_onset: bool = True               # pin ansatz w0/f0 from Technique II

    # Technique II — min-weight search
    mw_osd_order: int = 10
    mw_max_iter: int = 200
    mw_max_trials: int = 2000
    mw_workers: int = 1                  # >1 → run the independent BP-OSD decodes across processes
    mw_systematic: bool = True           # enumerate all 2^K-1 syndrome classes before random trials
    mw_use_symmetry: bool = True         # exploit Z_6×Z_6 toric shift automorphisms (36× expansion)
    mw_single_sector: bool = True        # paper's Z-type decoding: single CSS detector sector
    mw_sector_type: int = 0              # detector coord-type for the sector (0 = Z-checks)
    # Exact Technique-II onset pin (from the bit-exact reference enumeration, bb6_exact_enum_mitm.py:
    # D=6, |L(D)|=1524, |F|=3.825e8, f0=2.324e-5). When set, run_technique_ii returns these
    # directly instead of re-running the multi-minute BP-OSD search (which plateaus ~0.5% low).
    mw_f0_override: Optional[float] = 2.3239e-5
    mw_w0_override: Optional[int] = 3
    # II.3 feasibility cap: the exact Proposition-1 onset fraction (even D) enumerates every
    # weight-D/2 restriction of every min-weight logical — ~C(D, D/2)·|L(D)| subsets. For large
    # even D with a big L(D) (the cz channel: D=22, |L(D)|=17856 → ~1.3e10 restrictions) this OOMs
    # and never finishes. Above this cap, skip the exact enumeration and leave f0 unpinned (same as
    # the odd-D branch); for bb144 an unpinned onset is the correct choice anyway (see run_technique_i).
    mw_prop1_max_restrictions: int = 100_000_000

    # Decoder-convergence diagnostic (--decoder-conv): Relay-BP LER vs # legs on the full DEM.
    dconv_p: float = 0.005               # near-threshold rate with plenty of failures
    dconv_shots: int = 2000
    dconv_num_sets: Tuple[int, ...] = (1, 2, 5, 10, 30, 100, 300, 600)

    # Technique III — splitting cross-check
    split_p_high: float = 0.006
    # p_low=0.003: at lower p, q~6.7e-5 and single-flip chains need O(1/q)~15k steps to
    # make one move — chains below p≈0.003 are frozen even with warm-starting.
    split_p_low: float = 0.003
    # More levels (20 vs 6) → smaller per-step ratios → easier to estimate each step.
    split_n_levels: int = 20
    # chain_steps=50000: enough for ~3-4 accepted moves at p=0.003 (q≈2e-4, ~1/q=5k steps
    # per move). Combined with warm-starting this gives genuine mixing near p≈0.003.
    split_n_seeds: int = 32
    split_chain_steps: int = 50000
    split_burn_in: int = 5000
    split_anchor_shots: int = 3000
    split_min_weight_max_trials: int = 400   # BP-OSD trials for splitting's min-weight seeds
    # Replica-exchange (parallel-tempering) params: swaps between adjacent rungs mix the weight
    # space for codes with many inequivalent logicals (BB(12)), where the swap-less sequential
    # splitting freezes. These drive replica_exchange_estimate (see split_fulldem.py / split_bb144.py).
    split_re_walkers: int = 6
    split_re_local_steps: int = 6
    split_re_sweeps: int = 200
    split_re_burn_in: int = 60

    # onset scan (only used with --onset-scan)
    onset_shots: int = 200
    onset_coarse_step: int = 1           # BB(6) onset is small (~3); a fine walk is cheap

    @property
    def p_targets(self) -> np.ndarray:
        return np.logspace(np.log10(self.p_lo), np.log10(self.p_hi), self.n_p)

    @classmethod
    def production(cls) -> "Config":
        return cls(label="production")

    @classmethod
    def smoke(cls) -> "Config":
        """Tiny config: every technique runs end-to-end in well under a minute.

        Uses the *fast* Relay settings (not the paper's 600-leg accuracy settings) and a
        handful of shots/weights/chain-steps. Numbers are chosen only so each code path
        executes and returns finite, sane shapes — NOT for accuracy.
        """
        return cls(
            label="smoke",
            shots_per_weight=20,
            weights=list(range(2, 11)),
            n_p=6,
            # fast Relay settings (RelayBPDecoder's speed-tuned defaults)
            relay_gamma0=0.1, relay_pre_iter=20, relay_num_sets=20,
            relay_set_max_iter=20, relay_gamma_lo=-0.24, relay_gamma_hi=0.66,
            relay_stop_nconv=5,
            mw_max_trials=50, mw_systematic=False,
            split_p_high=0.01, split_p_low=0.005, split_n_levels=2, split_n_seeds=2,
            split_chain_steps=30, split_burn_in=10, split_anchor_shots=80,
            split_min_weight_max_trials=20,
            split_re_walkers=2, split_re_local_steps=2, split_re_sweeps=4, split_re_burn_in=2,
            onset_shots=40,
            dconv_shots=40, dconv_num_sets=(1, 5, 20),
        )


def make_decoder(cfg: Config) -> RelayBPDecoder:
    """Relay-BP decoder configured from cfg (paper §2.4 settings in production)."""
    return RelayBPDecoder(
        gamma0=cfg.relay_gamma0,
        pre_iter=cfg.relay_pre_iter,
        num_sets=cfg.relay_num_sets,
        set_max_iter=cfg.relay_set_max_iter,
        gamma_dist_interval=(cfg.relay_gamma_lo, cfg.relay_gamma_hi),
        stop_nconv=cfg.relay_stop_nconv,
    )


_NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}
_NOISE_PRED = {
    "cz":   lambda i, p, n: i.name == "DEPOLARIZE2",                                       # two-qubit gate
    "meas": lambda i, p, n: i.name == "X_ERROR" and n is not None and n.name == "M",        # before a measurement
    "prep": lambda i, p, n: i.name == "X_ERROR" and p is not None and p.name == "R",        # after a reset
    "idle": lambda i, p, n: i.name == "DEPOLARIZE1" and not (p is not None and p.name == "H"),  # idle data (not post-H)
}
def _filter_noise(circ, model):
    """Isolate one physical noise channel by filtering instructions on the built circuit (same base rate)."""
    keep = _NOISE_PRED.get(model)
    if keep is None:                       # None / 'full' / unknown -> full symmetric (no filtering)
        return circ
    insts = list(circ.flattened()); out = stim.Circuit()
    for i, inst in enumerate(insts):
        if inst.name in _NOISE:
            prev = insts[i - 1] if i > 0 else None
            nxt = insts[i + 1] if i + 1 < len(insts) else None
            if not keep(inst, prev, nxt):
                continue
        out.append(inst)
    return out


def build_circuit(cfg: Config):
    """The cfg.code (default BB(6)=[[72,12,6]]) memory circuit at p_ref with cfg.rounds rounds,
    optionally filtered to a single noise channel (cfg.noise_model). Every technique reads this.
    Measurement error is p_meas_factor * p_phys (1.0 = symmetric)."""
    em = ErrorModel(p_phys=cfg.p_ref, p_meas=cfg.p_ref * cfg.p_meas_factor)
    return _filter_noise(BBCodeSimulator(cfg.code).build_circuit(em, rounds=cfg.rounds), cfg.noise_model)


# ================================ checkpoint I/O ==================================
def _atomic_write_json(path: pathlib.Path, obj: dict, *, retries: int = 8) -> None:
    """Write JSON to a temp file then rename, so a crash mid-write can't corrupt it.

    On Windows ``os.replace`` can transiently raise PermissionError (WinError 5) when an
    AV scanner / search indexer momentarily holds the destination open — common under
    Desktop/OneDrive paths. Since this checkpoint is written after *every* weight of a
    multi-hour run, we retry the rename with backoff rather than letting a transient lock
    kill the whole job.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def _load_json(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


# ================================ reweighting =====================================
def reweight_spectrum(spectrum: FailureSpectrum, p_values: np.ndarray) -> ImportanceSamplingResult:
    """LER(p) from an accumulated failure spectrum (mirrors importance_sample's reweighting)."""
    N_exp = spectrum.n_expanded
    p_arr = np.asarray(p_values, dtype=float)
    q_targets = np.clip(spectrum.q_base * (p_arr / spectrum.p_ref), 1e-300, 1.0 - 1e-15)
    log_q = np.log(q_targets)
    log_1mq = np.log(1.0 - q_targets)

    P_logical = np.zeros_like(p_arr)
    var = np.zeros_like(p_arr)
    for w, T, F in zip(spectrum.weights, spectrum.trials, spectrum.failures):
        f = F / T if T > 0 else 0.0
        f_se = np.sqrt(f * (1.0 - f) / T) if T > 0 else 0.0
        log_binom = gammaln(N_exp + 1) - gammaln(w + 1) - gammaln(N_exp - w + 1)
        weight = np.exp(log_binom + w * log_q + (N_exp - w) * log_1mq)
        P_logical += f * weight
        var += (f_se * weight) ** 2

    return ImportanceSamplingResult(
        p_values=p_arr, P_logical=P_logical, P_logical_se=np.sqrt(var), spectrum=spectrum,
    )


def _shots_for(cfg: "Config", w: int) -> int:
    """Per-weight shot budget: shots_by_weight[w] if given, else the scalar shots_per_weight."""
    if cfg.shots_by_weight:
        t = cfg.shots_by_weight.get(int(w))
        if t is not None:
            return int(t)
    return int(cfg.shots_per_weight)


def _parse_shots_by_weight(spec: str) -> Dict[int, int]:
    """Parse a CLI spec like '3:8000,4:8000,5-7:4000' into {3:8000, 4:8000, 5:4000, 6:4000, 7:4000}."""
    out: Dict[int, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        wsel, _, tstr = part.partition(":")
        if not tstr:
            raise ValueError(f"bad --shots-by-weight token {part!r}; expected 'w:T' or 'lo-hi:T'")
        T = int(tstr)
        if "-" in wsel:
            lo, hi = (int(x) for x in wsel.split("-", 1))
            for w in range(lo, hi + 1):
                out[w] = T
        else:
            out[int(wsel)] = T
    return out


def _parse_weight_spec(spec: str) -> List[int]:
    """Parse an explicit IS weight plan like '2-10,15,20-50' into a sorted, de-duplicated weight list.
    Everything not listed is skipped (never sampled)."""
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _spectrum_from_checkpoint(ckpt: dict) -> FailureSpectrum:
    done: Dict[int, int] = {int(w): int(F) for w, F in ckpt["failures_by_weight"].items()}
    # Per-weight trials (current format); fall back to the scalar shots_per_weight for old checkpoints.
    tbw: Dict[int, int] = {int(w): int(t) for w, t in ckpt.get("trials_by_weight", {}).items()}
    shots = int(ckpt.get("shots_per_weight", 0))
    weights = [w for w in ckpt["weights_plan"] if w in done]
    return FailureSpectrum(
        weights=weights,
        trials=[tbw.get(w, shots) for w in weights],
        failures=[done[w] for w in weights],
        n_expanded=int(ckpt["n_expanded"]),
        q_base=float(ckpt["q_base"]),
        p_ref=float(ckpt["p_ref"]),
    )


# ============================ Section 0: onset scan ===============================
ONSET_SEED_STREAM = 1000  # distinct RNG stream so scan draws never collide with the sweep


def onset_scan(cfg: Config, outdir: pathlib.Path) -> dict:
    """Locate the failure onset and size a contiguous WEIGHTS block (mirrors gross sweep).

    Walks weight upward in cfg.onset_coarse_step steps until the first weight with an
    observed failure, then suggests WEIGHTS = [onset-1 .. w_hi] where w_hi brackets the
    dominant binomial mass μ(p)=N·q over the target grid (+4σ). Checkpointed + deterministic.
    """
    print("Section 0 — onset scan", flush=True)
    circuit = build_circuit(cfg)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, None)
    n_expanded = int(col_to_mech.shape[0])
    decoder = make_decoder(cfg)
    decoder.setup(circuit)

    d = BB_72_12_6.distance
    mu = n_expanded * q_base * (cfg.p_targets / cfg.p_ref)
    mu_max = float(mu.max())
    w_hi = int(np.ceil(mu_max + 4.0 * np.sqrt(mu_max)))
    print(f"  N_expanded={n_expanded}, q_base={q_base:.5f}, μ(p) over grid: "
          f"{mu.min():.1f}..{mu_max:.1f}, w_hi={w_hi}", flush=True)

    ckpt_path = outdir / "onset.spectrum.json"
    ck = _load_json(ckpt_path)
    if ck is not None and (int(ck.get("shots", -1)) != cfg.onset_shots or int(ck.get("seed", -1)) != cfg.seed):
        ck = None  # different shots/seed → rescan fresh
    cache: Dict[int, int] = {int(w): int(F) for w, F in ck["failures_by_weight"].items()} if ck else {}

    def scan(w: int) -> int:
        w = int(w)
        if w in cache:
            return cache[w]
        rng = np.random.default_rng([cfg.seed, ONSET_SEED_STREAM, w])
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, cfg.onset_shots, decoder, rng)
        cache[w] = F
        _atomic_write_json(ckpt_path, {
            "shots": cfg.onset_shots, "seed": cfg.seed,
            "failures_by_weight": {str(k): int(v) for k, v in sorted(cache.items())},
        })
        print(f"    w={w:>4}: F/T = {F:>3}/{cfg.onset_shots} = {F/cfg.onset_shots:.3f}", flush=True)
        return F

    onset: Optional[int] = None
    w = max(d // 2, 2)                    # onset >= ceil(d/2); start a touch below
    while w <= w_hi:
        if scan(w) > 0:
            onset = w
            break
        w += cfg.onset_coarse_step
    if onset is None:
        print(f"  no failures up to w_hi={w_hi}; falling back to cfg.weights")
        result = {"onset": None, "w_hi": w_hi, "n_expanded": n_expanded, "q_base": q_base,
                  "suggested_weights": [cfg.weights[0], cfg.weights[-1]]}
        _atomic_write_json(outdir / "onset.json", result)
        return result

    last_zero = max(onset - 1, 1)
    suggested = list(range(last_zero, w_hi + 1))
    print(f"  onset w*={onset}, WEIGHTS = {last_zero}..{w_hi} ({len(suggested)} weights)")
    result = {"onset": int(onset), "last_zero": int(last_zero), "w_hi": int(w_hi),
              "n_expanded": n_expanded, "q_base": float(q_base),
              "suggested_weights": [int(last_zero), int(w_hi)]}
    _atomic_write_json(outdir / "onset.json", result)
    return result


# ============================ Technique II: min-weight ============================
def run_technique_ii(cfg: Config, outdir: pathlib.Path) -> dict:
    """Circuit fault distance D, optimal onset w0=D/2, and exact onset fraction f0=f*(D/2).

    Three BP-OSD (ldpc) phases, each timed + progress-printed since the L(D) search can run
    many minutes silently: (1) distance, (2) L(D) search, (3) exact onset fraction. For BB(6),
    D=6 is even, so the Proposition-1 exact onset fraction applies; w0/f0 pin the Technique-I
    ansatz. Resumable: a matching distance.json (same rounds) is reused so a restart skips this.
    """
    from min_weight import (
        dem_check_action_matrices, single_sector_dem, compute_distance,
        find_min_weight_logicals, min_weight_fail_count, expanded_logical_count,
        build_circuit_translation_perms,
    )

    # Paper Table-2 targets (BB(6)-circuit, Z-type / single-sector representation).
    PAPER_NCOMP, PAPER_NEXP = 2233, 46224
    PAPER_LD_EXP, PAPER_FAILS = 6.01e12, 3.83e8

    # Exact onset pin from the bit-exact reference enumeration (bb6_exact_enum_mitm.py):
    # skip the multi-minute BP-OSD search and use the validated D/w0/f0 directly.
    if cfg.mw_f0_override is not None:
        w0 = int(cfg.mw_w0_override)
        out = {"rounds": cfg.rounds, "single_sector": bool(cfg.mw_single_sector),
               "sector_type": int(cfg.mw_sector_type) if cfg.mw_single_sector else None,
               "distance": 2 * w0, "onset": w0, "onset_fraction": float(cfg.mw_f0_override),
               "exact_pin": True}
        print(f"\nTechnique II — exact onset pin (reference enumeration): "
              f"D={2*w0}, w0={w0}, f0={cfg.mw_f0_override:.3e} "
              f"(paper 2.33e-5; |L(D)|=1524, |F|=3.825e8 validated separately)", flush=True)
        _atomic_write_json(outdir / "distance.json", out)
        return out

    # Resume: Technique II is not cheap and depends only on the circuit (rounds), not on p or
    # shots, so a prior distance.json with the same rounds + sector mode can be reused on restart.
    cached = _load_json(outdir / "distance.json")
    if (cached is not None and int(cached.get("rounds", -1)) == cfg.rounds
            and bool(cached.get("single_sector", False)) == cfg.mw_single_sector):
        _cf0 = cached.get("onset_fraction")
        _cf0s = f"{_cf0:.3e}" if isinstance(_cf0, (int, float)) else "None (unpinned)"
        print(f"\nTechnique II — reusing cached distance.json "
              f"(D={cached['distance']}, onset w0={cached['onset']}, f0={_cf0s})", flush=True)
        return cached

    sector = cfg.mw_sector_type if cfg.mw_single_sector else None
    mode = f"single-sector (type {sector})" if cfg.mw_single_sector else "full both-sector"
    print(f"\nTechnique II — min-weight onset, {mode} DEM "
          f"(BP-OSD, ~1.2s/decode; prints progress) ...", flush=True)
    circuit = build_circuit(cfg)

    # Build H/A/mult in the chosen representation. single_sector_dem also returns the
    # renumbered detector coordinates the toric symmetry builder needs.
    det_coords = None
    if cfg.mw_single_sector:
        H, A, mult, priors, det_coords = single_sector_dem(circuit, detector_type=cfg.mw_sector_type)
    else:
        H, A, mult, priors = dem_check_action_matrices(circuit)
    n_comp = H.shape[1]
    n_exp = int(mult.sum())
    print(f"  Ñ (compressed)={n_comp} (paper {PAPER_NCOMP}), "
          f"N (expanded)={n_exp} (paper {PAPER_NEXP})", flush=True)
    t0 = time.perf_counter()

    print(f"  [II.1] fault distance D over {A.shape[0]} logicals ...", flush=True)
    dr = compute_distance(circuit, osd_order=cfg.mw_osd_order, max_iter=cfg.mw_max_iter,
                          priors=priors, progress=True, workers=cfg.mw_workers, sector=sector)
    D = dr.distance
    print(f"  [II.1] D={D}, onset w0={D // 2}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    t1 = time.perf_counter()
    K_obs = circuit.num_observables
    n_sys = (1 << K_obs) - 1 if (cfg.mw_systematic and K_obs <= 20) else 0

    sym_perms = None
    if cfg.mw_use_symmetry:
        print(f"  [II.2] building Z_6×Z_6 toric translation permutations ...", flush=True)
        try:
            sym_perms = build_circuit_translation_perms(circuit, H, l=cfg.code.l, m=cfg.code.m,
                                                        det_coords=det_coords, verbose=True)
            print(f"  [II.2] {len(sym_perms)} toric perms ready "
                  f"(each logical found → up to {len(sym_perms)} for free)", flush=True)
        except (KeyError, ValueError) as exc:
            print(f"  [II.2] WARNING: toric sym build failed ({exc}); falling back to no symmetry",
                  flush=True)
            sym_perms = None

    print(f"  [II.2] L(D) search: {n_sys} systematic + up to {cfg.mw_max_trials} random trials ...",
          flush=True)
    logicals, search_trace = find_min_weight_logicals(
        circuit, D, max_trials=cfg.mw_max_trials, osd_order=cfg.mw_osd_order,
        max_iter=cfg.mw_max_iter, priors=priors, seed=cfg.seed,
        progress_every=max(max(n_sys, cfg.mw_max_trials) // 40, 1), workers=cfg.mw_workers,
        systematic=cfg.mw_systematic, symmetry_perms=sym_perms, sector=sector,
        return_trace=True,
    )
    ld_comp = len(logicals)
    # Search-saturation trace: |L(D)| found vs cumulative trials. The plateau establishes the
    # search found all min-weight logicals — the completeness check for the search-derived
    # full-DEM Table 2 (no exact MITM enumeration there). Validated against the exact MITM count
    # where available (single-sector).
    _atomic_write_json(outdir / "search_convergence.json", {
        "trace": [[int(t), int(c)] for (t, c) in search_trace],
        "n_systematic": int(n_sys), "max_trials": int(cfg.mw_max_trials),
        "single_sector": bool(cfg.mw_single_sector), "distance": int(D), "final": int(ld_comp),
    })
    # Persist the L(D) supports so the exact onset fraction (II.3) can be computed offline — even
    # when the inline Proposition-1 enumeration is skipped below for being too large (e.g. cz:
    # D=22, |L(D)|=17856 → ~1.3e10 restrictions). Tiny: |L(D)|×D ints. All supports have weight D,
    # so a rectangular int array round-trips cleanly.
    try:
        _ld_arr = (np.array([sorted(s) for s in logicals], dtype=np.int32)
                   if logicals else np.empty((0, 0), dtype=np.int32))
        np.savez(outdir / "logicals_LD.npz", supports=_ld_arr, distance=int(D),
                 single_sector=bool(cfg.mw_single_sector),
                 sector=int(cfg.mw_sector_type) if cfg.mw_single_sector else -1)
    except Exception as _exc:                       # best-effort; never let persistence kill the run
        print(f"  [II.2] WARNING: could not persist L(D) supports ({_exc})", flush=True)
    ld_exp = expanded_logical_count(logicals, mult)
    print(f"  [II.2] |L(D)| compressed={ld_comp}, expanded={ld_exp:.4g} "
          f"(paper {PAPER_LD_EXP:.3g})   ({time.perf_counter() - t1:.0f}s)", flush=True)

    half = (D + 1) // 2          # onset weight w0 = ceil(D/2) (correct for even AND odd D)
    # Feasibility of the exact Proposition-1 enumeration: ~C(D, D/2)·|L(D)| weight-D/2 restrictions.
    from math import comb
    restr_est = comb(D, half) * ld_comp if (logicals and D % 2 == 0) else 0
    if logicals and D % 2 == 0 and restr_est <= cfg.mw_prop1_max_restrictions:
        # Even D, tractable: exact onset failure fraction f*(D/2) via Proposition 1.
        print(f"  [II.3] exact onset fraction f*(D/2) via Proposition 1 "
              f"(~{restr_est:.2g} restrictions) ...", flush=True)
        fails, n_exp = min_weight_fail_count(H, A, logicals, mult)
        log_choose = gammaln(n_exp + 1) - gammaln(half + 1) - gammaln(n_exp - half + 1)
        f_star = float(fails / np.exp(log_choose))
    else:
        # Odd D, no weight-D logicals found, OR an even-D enumeration too large to run (e.g. cz:
        # D=22, |L(D)|=17856 → ~1.3e10 restrictions → OOM). In all three cases the exact onset
        # fraction is unavailable/inapplicable, so report |L(D)| as a lower bound and leave f0
        # unpinned — Technique I then FITS w0/f0 from the IS spectrum instead of pinning a wrong
        # value, and a multi-hour run never dies here.
        n_exp = int(mult.sum())
        fails, f_star = 0, None
        if not logicals:
            why = "no weight-D logicals found"
        elif D % 2 != 0:
            why = f"D={D} is odd (Prop. 1 is even-D only)"
        else:
            why = (f"D={D} even but ~{restr_est:.2g} restrictions exceed cap "
                   f"{cfg.mw_prop1_max_restrictions:.2g} (exact onset infeasible)")
        print(f"  [II.3] {why} → onset fraction unpinned (f0=None)", flush=True)

    out = {
        "rounds": cfg.rounds,
        "single_sector": bool(cfg.mw_single_sector),
        "sector_type": int(cfg.mw_sector_type) if cfg.mw_single_sector else None,
        "method": "search",                          # symmetry-augmented BP-OSD search
        "distance": int(D),                          # upper bound on the circuit fault distance
        "distance_is_bound": True,                   # BP-OSD search → D is an upper bound (paper convention)
        "onset": int(half),                          # w0 = ceil(D/2)
        "n_compressed": int(n_comp),
        "n_expanded": int(n_exp),
        "n_min_logicals": int(ld_comp),              # |L(D)| lower bound (search-found, paper convention ≥)
        "n_min_logicals_expanded": int(ld_exp),
        "fail_count": int(fails),
        "onset_fraction": f_star,                    # f0 = f*(D/2) for even D; None otherwise
    }
    f0s = f"{f_star:.3e}" if f_star is not None else "None"
    print(f"  D≤{D}, onset w0={half}, |L(D)|≥{ld_comp} (exp {ld_exp:.3g}), |F|={fails}, "
          f"f0={f0s}   (total {time.perf_counter() - t0:.0f}s)")
    _atomic_write_json(outdir / "distance.json", out)
    return out


# ============================ IS sweep (checkpointed) ============================
def run_is_sweep(cfg: Config, outdir: pathlib.Path, weights_plan: List[int]) -> ImportanceSamplingResult:
    """Sample the failure spectrum f(w) over weights_plan, checkpointing after every weight."""
    ckpt_path = outdir / "bb6.spectrum.json"
    ckpt = _load_json(ckpt_path)
    if ckpt is not None:
        mismatch = (
            ckpt.get("weights_plan") != weights_plan
            or int(ckpt.get("shots_per_weight", -1)) != cfg.shots_per_weight
            or int(ckpt.get("seed", -1)) != cfg.seed
        )
        if mismatch:
            raise SystemExit(
                f"existing checkpoint {ckpt_path} was made with a different plan/shots/seed. "
                f"Delete it to start fresh, or restore the original parameters to resume."
            )
        failures_by_weight: Dict[int, int] = {int(w): int(F) for w, F in ckpt["failures_by_weight"].items()}
        trials_by_weight: Dict[int, int] = {int(w): int(t) for w, t in ckpt.get("trials_by_weight", {}).items()}
        if not trials_by_weight:   # old scalar-only checkpoint: every done weight used shots_per_weight
            _sc = int(ckpt.get("shots_per_weight", cfg.shots_per_weight))
            trials_by_weight = {w: _sc for w in failures_by_weight}
        print(f"[is] resuming: {len(failures_by_weight)}/{len(weights_plan)} weights done")
    else:
        failures_by_weight = {}
        trials_by_weight = {}

    remaining = [w for w in weights_plan if w not in failures_by_weight]

    # Heavy setup (DEM parse + decoder) — only if there's sampling left to do.
    if remaining:
        print("[is] building circuit + decoder ...", flush=True)
        circuit = build_circuit(cfg)
        if cfg.mw_single_sector:
            # Stim circuit is the source; project to the single Z-sector (the paper's
            # representation) and decode those matrices with Relay-BP. det_mat/obs_mat are
            # mechanism-indexed (rows = mechanisms), matching _sample_failures_at_weight.
            from min_weight import single_sector_dem
            H, A, mult, probs, _ = single_sector_dem(circuit, detector_type=cfg.mw_sector_type)
            det_mat = H.T.astype(bool); obs_mat = A.T.astype(bool)
            # q_base = the expansion base rate (~p/15, the full-DEM minimum) that single_sector_dem
            # used to build `mult`; reweighting binomials use N_expanded slots each at rate q_base.
            q_base = float(np.asarray(probs_full := _parse_dem(circuit)[0]).min())
            col_to_mech = np.repeat(np.arange(H.shape[1], dtype=np.int32), mult)
            n_expanded = int(mult.sum())
            decoder = make_decoder(cfg)
            decoder.setup_from_matrices(H, probs, A)
            print(f"[is] single-sector (type {cfg.mw_sector_type}): {H.shape[0]} detectors, "
                  f"{H.shape[1]} mechanisms, N_expanded={n_expanded}, q_base={q_base:.5f}", flush=True)
        else:
            probs, det_mat, obs_mat = _parse_dem(circuit)
            col_to_mech, q_base, _ = _expand(probs, None)
            n_expanded = int(col_to_mech.shape[0])
            decoder = make_decoder(cfg)
            decoder.setup(circuit)
            print(f"[is] full DEM: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
                  f"N_expanded={n_expanded}, q_base={q_base:.5f}", flush=True)
    else:
        # nothing to sample — reconstruct metadata from the checkpoint
        assert ckpt is not None
        n_expanded, q_base = int(ckpt["n_expanded"]), float(ckpt["q_base"])

    def save() -> None:
        _atomic_write_json(ckpt_path, {
            "n_expanded": int(n_expanded),
            "q_base": float(q_base),
            "p_ref": cfg.p_ref,
            "shots_per_weight": cfg.shots_per_weight,
            "trials_by_weight": {str(w): int(t) for w, t in sorted(trials_by_weight.items())},
            "seed": cfg.seed,
            "weights_plan": weights_plan,
            "failures_by_weight": {str(w): int(F) for w, F in sorted(failures_by_weight.items())},
        })

    # Adaptive mode sweeps high->low so each weight's shots can be predicted from the (higher) weights
    # already measured; the static path keeps the plan's order.
    order = sorted(remaining, reverse=True) if cfg.adaptive else remaining
    t0 = time.perf_counter()
    for i, w in enumerate(order):
        rng = np.random.default_rng([cfg.seed, w])   # per-weight RNG → resume-order-independent
        if cfg.adaptive:
            measured_f = {ww: failures_by_weight[ww] / trials_by_weight[ww]
                          for ww in failures_by_weight if trials_by_weight.get(ww)}
            f_pred = predict_failure_fraction(measured_f, w, cfg.adaptive_predict_window)
            T_w = shots_to_hit_failures(f_pred, cfg.adaptive_failures,
                                        cfg.adaptive_shots_min, cfg.adaptive_shots_max)
        else:
            T_w = _shots_for(cfg, w)
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, T_w, decoder, rng)
        failures_by_weight[w] = F
        trials_by_weight[w] = T_w
        save()  # checkpoint BEFORE moving on — a crash here loses nothing already saved
        done = len(failures_by_weight)
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (len(order) - i - 1) / rate if rate > 0 else float("nan")
        print(f"[is] w={w:>4}: F/T={F:>4}/{T_w} = {F/T_w:.3f}"
              + (f"  (f_pred={f_pred:.2g})" if cfg.adaptive else "")
              + f"   ({done}/{len(weights_plan)}, {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    if remaining:
        save()
    final = _load_json(ckpt_path)
    assert final is not None
    spectrum = _spectrum_from_checkpoint(final)
    return reweight_spectrum(spectrum, cfg.p_targets)


# ============================ Technique I: ansatz ===============================
def run_technique_i(cfg: Config, is_result: ImportanceSamplingResult,
                    tech2: Optional[dict], outdir: pathlib.Path) -> dict:
    """Fit the failure-spectrum ansatz and extrapolate LER(p) (Eq. 10 of the paper).

    If cfg.pin_onset and Technique II ran, w0/f0 are pinned from the exact onset properties;
    otherwise they are fit. K (for the saturation a=1-2^-K) is read from the circuit.
    """
    K = build_circuit(cfg).detector_error_model(decompose_errors=False).num_observables
    w0 = f0 = None
    # Pin the onset (w0,f0) ONLY when Technique II determined the fraction exactly (even D, MITM/
    # search). For bb144 (odd D, f0 unknown) PINNING w0=ceil(D/2)=6 BREAKS the ansatz: the true
    # onset is ~40 weights below where failures become observable (w≈46), a gap the f3/f5 form can't
    # bridge (it collapses to LER≈1). So leave w0 free there — the fit then tracks the observable
    # spectrum; the low-p extrapolation is unreliable and the splitting cross-check covers that regime.
    if cfg.pin_onset and tech2 is not None and tech2.get("onset_fraction") is not None:
        w0, f0 = float(tech2["onset"]), float(tech2["onset_fraction"])
    print(f"\nTechnique I — ansatz '{cfg.ansatz_model}', K={K}, "
          f"pinned w0={w0}, f0={f0}", flush=True)

    # A failed ansatz fit (e.g. an all-zero spectrum because the weights missed the failure
    # onset) must NOT discard the expensive IS + Technique-III results — warn and return a
    # P_ext-less result so the rest of the run still completes and is saved.
    try:
        fit = fit_failure_spectrum(is_result.spectrum, K=K, model=cfg.ansatz_model, w0=w0, f0=f0)
        P_ext = logical_error_rate_from_ansatz(fit, list(cfg.p_targets))
    except ValueError as e:
        print(f"  [WARN] ansatz fit failed — {e}\n"
              f"  [WARN] skipping Technique-I extrapolation; IS/splitting results are unaffected.")
        out = {"model": cfg.ansatz_model, "K": int(K), "error": str(e), "P_ext": None,
               "pinned": bool(cfg.pin_onset and tech2 is not None)}
        _atomic_write_json(outdir / "ansatz_fit.json", out)
        return out

    out = {
        "model": cfg.ansatz_model, "K": int(K),
        "params": {k: float(v) for k, v in fit.params.items()},
        "n_points": int(fit.n_points), "cost": float(fit.cost),
        "pinned": bool(cfg.pin_onset and tech2 is not None),
        "P_ext": P_ext.tolist(),
    }
    print("  params: " + ", ".join(f"{k}={v:.3g}" for k, v in fit.params.items())
          + f"   (n_points={fit.n_points}, cost={fit.cost:.3g})")
    _atomic_write_json(outdir / "ansatz_fit.json", out)
    return out


# ============================ Technique III: splitting ==========================
def run_technique_iii(cfg: Config, tech2: Optional[dict], outdir: pathlib.Path) -> dict:
    """Replica-exchange (parallel-tempering) splitting cross-check over the p-ladder.

    Uses replica_exchange_estimate — walkers per rung with SWAPS between adjacent rungs — not the
    swap-less splitting_estimate, whose independent chains freeze for codes with many inequivalent
    logicals (BB(12)). Prints [tempering] swap-accept progress; returns the flat ladder schema (plus
    swap diagnostics) consumed by run_all/make_plot.
    """
    from splitting import replica_exchange_estimate

    print("\nTechnique III — replica-exchange splitting cross-check ...", flush=True)
    circuit = build_circuit(cfg)
    distance = int(tech2["distance"]) if tech2 is not None else cfg.code.distance
    temper, diag = replica_exchange_estimate(
        circuit, make_decoder(cfg),
        p_ref=cfg.p_ref, p_high=cfg.split_p_high, p_low=cfg.split_p_low,
        n_levels=cfg.split_n_levels, n_walkers=cfg.split_re_walkers,
        local_steps=cfg.split_re_local_steps, n_sweeps=cfg.split_re_sweeps,
        burn_in=cfg.split_re_burn_in, anchor_shots=cfg.split_anchor_shots,
        distance=distance, seed=cfg.seed,
        single_sector=cfg.mw_single_sector, sector=cfg.mw_sector_type,
    )
    out = {
        "p_ladder": np.asarray(temper.p_ladder).tolist(),
        "P_logical": np.asarray(temper.P_logical).tolist(),
        "P_logical_se": np.asarray(temper.P_logical_se).tolist(),
        "swap_accept": list(diag["swap_accept"]),
        "mean_weight": list(diag["mean_weight"]),
    }
    print(f"  ladder {temper.p_ladder[0]:.2e}..{temper.p_ladder[-1]:.2e}, "
          f"P {np.asarray(temper.P_logical)[0]:.2e}..{np.asarray(temper.P_logical)[-1]:.2e}; "
          f"swap-accept {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f}")
    _atomic_write_json(outdir / "splitting.json", out)
    return out


# ================================== plotting =====================================
def make_plot(cfg: Config, is_result: ImportanceSamplingResult, tech1: dict,
              tech3: Optional[dict], outdir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    p = is_result.p_values
    lo = np.maximum(is_result.P_logical - is_result.P_logical_se, 1e-300)
    hi = is_result.P_logical + is_result.P_logical_se
    ax.fill_between(p, lo, hi, color="steelblue", alpha=0.2)
    ax.plot(p, is_result.P_logical, "o", color="steelblue", ms=4, label="IS reweighted (raw)")
    if tech1.get("P_ext"):
        ax.plot(cfg.p_targets, tech1["P_ext"], "-", color="crimson", lw=2,
                label=f"Technique I ansatz ({tech1['model']})")
    if tech3 is not None:
        ax.plot(tech3["p_ladder"], tech3["P_logical"], "s", color="seagreen", ms=5,
                label="Technique III splitting")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Physical error rate $p$"); ax.set_ylabel("Logical error rate")
    ax.set_title(f"{cfg.code_label} — Fig. 10 reproduction (Relay-BP, {cfg.label})")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "bb6_fig10.png", dpi=150)
    print(f"  wrote {outdir/'bb6_fig10.png'}")


def run_decoder_convergence(cfg: Config, outdir: pathlib.Path, *, p: Optional[float] = None,
                            shots: Optional[int] = None, num_sets_grid=None, seed: int = 0) -> dict:
    """Relay-BP decoder convergence on the FULL DEM: logical error rate (and disagreement with the
    most-legs decoder) vs the number of relay legs (num_sets), at a fixed near-threshold p. As
    num_sets grows the LER plateaus and the disagreement -> 0, showing the decoder has enough legs
    to be reliable. Writes decoder_convergence.json. Defaults from cfg.dconv_* (smoke overrides)."""
    p = float(p if p is not None else cfg.dconv_p)
    shots = int(shots if shots is not None else cfg.dconv_shots)
    num_sets_grid = tuple(num_sets_grid if num_sets_grid is not None else cfg.dconv_num_sets)
    circuit = BBCodeSimulator(cfg.code).build_circuit(ErrorModel.symmetric(p), rounds=cfg.rounds)
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    print(f"\nDecoder convergence (full DEM): p={p:.3g}, {shots} shots, "
          f"num_sets grid {list(num_sets_grid)} ...", flush=True)
    preds_by_ns, rows = {}, []
    for ns in num_sets_grid:
        dec = RelayBPDecoder(gamma0=cfg.relay_gamma0, pre_iter=cfg.relay_pre_iter, num_sets=int(ns),
                             set_max_iter=cfg.relay_set_max_iter,
                             gamma_dist_interval=(cfg.relay_gamma_lo, cfg.relay_gamma_hi),
                             stop_nconv=cfg.relay_stop_nconv)
        dec.setup(circuit)                                  # full both-sector DEM
        t0 = time.perf_counter()
        preds = np.asarray(dec.decode_batch(det))
        ler = float(np.any(preds != obs, axis=1).mean())
        preds_by_ns[int(ns)] = preds
        rows.append({"num_sets": int(ns), "ler": ler,
                     "ler_se": float(np.sqrt(max(ler * (1 - ler), 1e-12) / shots))})
        print(f"  num_sets={ns:>4}: LER={ler:.3e}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    best = preds_by_ns[int(num_sets_grid[-1])]              # most-legs decoder = reference
    for r in rows:
        r["disagree_with_best"] = float(np.any(preds_by_ns[r["num_sets"]] != best, axis=1).mean())
    out = {"p": p, "shots": int(shots), "rounds": cfg.rounds, "code": cfg.code_label,
           "single_sector": False, "rows": rows}
    _atomic_write_json(outdir / "decoder_convergence.json", out)
    print(f"wrote {outdir / 'decoder_convergence.json'}", flush=True)
    return out


# ================================== driver =======================================
def run_all(cfg: Config, outdir: pathlib.Path, *, do_onset: bool = False,
            do_split: bool = True, do_plot: bool = False, do_decoder_conv: bool = False) -> dict:
    """Full pipeline. Returns a dict of every result (also written to disk)."""
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.label}] repo={REPO_ROOT}\n[{cfg.label}] outdir={outdir}")
    print(f"[{cfg.label}] p_ref={cfg.p_ref}, rounds={cfg.rounds}, "
          f"shots/weight={cfg.shots_per_weight}, seed={cfg.seed}")
    _atomic_write_json(outdir / "config.json", dataclasses.asdict(cfg))

    weights_plan = list(cfg.weights)
    if do_onset and not cfg.weights_explicit:
        onset = onset_scan(cfg, outdir)
        if onset.get("last_zero") is not None:
            weights_plan = list(range(onset["last_zero"], onset["w_hi"] + 1))
            print(f"  → onset-derived WEIGHTS {weights_plan[0]}..{weights_plan[-1]}")
    elif do_onset and cfg.weights_explicit:
        print(f"  → explicit --weights plan ({len(weights_plan)} weights); skipping onset auto-bracket")

    tech2 = run_technique_ii(cfg, outdir)                       # Technique II (pins ansatz)
    is_result = run_is_sweep(cfg, outdir, weights_plan)         # IS sweep → raw LER(p)
    tech1 = run_technique_i(cfg, is_result, tech2, outdir)      # Technique I  (ansatz)
    tech3 = run_technique_iii(cfg, tech2, outdir) if do_split else None  # Technique III

    np.savez(
        outdir / "bb6_fig10.npz",
        p_values=is_result.p_values,
        is_P_logical=is_result.P_logical,
        is_P_logical_se=is_result.P_logical_se,
        ansatz_p=cfg.p_targets,
        ansatz_P=np.asarray(tech1["P_ext"]) if tech1.get("P_ext") else np.array([]),
        spectrum_weights=np.array(is_result.spectrum.weights),
        spectrum_failures=np.array(is_result.spectrum.failures),
        spectrum_trials=np.array(is_result.spectrum.trials),
        n_expanded=is_result.spectrum.n_expanded,
        q_base=is_result.spectrum.q_base,
        distance=tech2["distance"], onset=tech2["onset"], onset_fraction=tech2["onset_fraction"],
        split_p=np.asarray(tech3["p_ladder"]) if tech3 else np.array([]),
        split_P=np.asarray(tech3["P_logical"]) if tech3 else np.array([]),
    )
    print(f"\nwrote {outdir/'bb6_fig10.npz'}")

    dconv = run_decoder_convergence(cfg, outdir) if do_decoder_conv else None

    if do_plot:
        print("\nplotting ...")
        make_plot(cfg, is_result, tech1, tech3, outdir)

    print(f"\n[{cfg.label}] done.")
    return {"tech1": tech1, "tech2": tech2, "tech3": tech3, "is": is_result, "dconv": dconv}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=RUNS / "bravyi" / "bb6_fig10_out")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end dry run (fast Relay settings, ~20 shots) — exercises "
                         "every code path the production run hits, in well under a minute")
    ap.add_argument("--rounds", type=int, default=None, help="override syndrome rounds (default d=6)")
    ap.add_argument("--p-ref", type=float, default=None, help="override circuit build rate")
    ap.add_argument("--shots", type=int, default=None, help="override shots per weight (the base budget)")
    ap.add_argument("--shots-by-weight", type=str, default=None,
                    help="per-weight shot overrides on top of --shots, e.g. '3:8000,4:8000,5-7:4000'. "
                         "Weights not listed use --shots. Pours shots into the rare near-onset bins so "
                         "their f(w) lifts out of the 0-failure (3/T upper-limit) regime; error bars "
                         "are computed per-weight, so this tightens exactly those bins.")
    ap.add_argument("--weights-hi", type=int, default=None,
                    help="cap the IS sweep weights to range(2, hi+1) (full-DEM decodes are slow at "
                         "mid weights; the ansatz saturates by w~40)")
    ap.add_argument("--weights", type=str, default=None,
                    help="explicit IS weight plan, e.g. '2-10,15,20-50' — anything not listed is "
                         "SKIPPED (never sampled). Overrides --weights-hi. Safe for the Technique-I "
                         "ansatz (it fits a smooth f(w) and reweights over ALL w), but skipping "
                         "weights INSIDE the dominant binomial mass biases the RAW IS points, so keep "
                         "a representative bracket there.")
    ap.add_argument("--weight-stride", type=int, default=None,
                    help="sample only every Nth weight (1=all) for speed; the ansatz interpolates the "
                         "skipped weights. Applied on top of --weights / --weights-hi.")
    ap.add_argument("--adaptive", action="store_true",
                    help="adaptive per-weight shots: sweep weights high->low and size each weight to "
                         "expect ~--adaptive-failures failures, with f(w) predicted from the weights "
                         "already done (T=N/f_pred, clamped). Overrides --shots / --shots-by-weight.")
    ap.add_argument("--adaptive-failures", type=int, default=None,
                    help="target failures per weight in --adaptive mode (default 50; higher = tighter "
                         "f(w) and more shots)")
    ap.add_argument("--adaptive-shots-min", type=int, default=None,
                    help="floor shots/weight in --adaptive mode (default 4)")
    ap.add_argument("--adaptive-shots-max", type=int, default=None,
                    help="cap shots/weight in --adaptive mode (default 20000; bounds the rare onset bins)")
    ap.add_argument("--seed", type=int, default=None, help="override base RNG seed")
    ap.add_argument("--mw-max-trials", type=int, default=None,
                    help="cap Technique-II L(D) search trials (lower = faster, less complete L(D); "
                         "production default 2000)")
    ap.add_argument("--mw-workers", type=int, default=None,
                    help="parallelize Technique-II BP-OSD decodes across N processes "
                         "(default 1 = serial; try 24 on this machine)")
    ap.add_argument("--no-symmetry", action="store_true",
                    help="disable Z_6×Z_6 toric symmetry expansion for Technique-II L(D) search")
    ap.add_argument("--full-dem", action="store_true",
                    help="use the full both-sector DEM for Technique II instead of the paper's "
                         "single Z-check sector (default is single-sector, matching Table 2)")
    ap.add_argument("--onset-scan", action="store_true", help="run onset scan to size WEIGHTS")
    ap.add_argument("--no-split", action="store_true", help="skip Technique III (splitting)")
    ap.add_argument("--split-only", action="store_true",
                    help="skip IS + ansatz; reuse existing spectrum checkpoint and only re-run splitting")
    ap.add_argument("--split-p-low", type=float, default=None,
                    help="override splitting p_low (default 0.003)")
    ap.add_argument("--split-n-levels", type=int, default=None,
                    help="override number of splitting ladder levels (default 20)")
    ap.add_argument("--split-chain-steps", type=int, default=None,
                    help="override chain steps per level (default 50000)")
    ap.add_argument("--split-n-seeds", type=int, default=None,
                    help="override number of seed chains per level (default 32)")
    ap.add_argument("--code", choices=sorted(CODES), default=None,
                    help="bivariate-bicycle code (default bb72=BB(6); bb144=BB(12)). Sets rounds=d "
                         "unless --rounds is given.")
    ap.add_argument("--noise-model", choices=["full", "cz", "meas", "prep", "idle"], default=None,
                    help="isolate ONE physical noise channel (default full symmetric): cz=two-qubit gate "
                         "(DEPOLARIZE2), meas=measurement, prep=reset/state-prep, idle=idle-data. Filters "
                         "the circuit at the same base rate; feeds every technique.")
    ap.add_argument("--p-meas-factor", type=float, default=None,
                    help="measurement error = factor * gate error (1.0 = symmetric, default). "
                         "Asymmetric (e.g. 5) is rate-agnostic for the reweighting; clears the "
                         "symmetric onset pin so Technique II recomputes.")
    ap.add_argument("--decoder-conv", action="store_true",
                    help="also measure Relay-BP decoder convergence (LER vs num_sets legs) on the "
                         "full DEM -> decoder_convergence.json")
    ap.add_argument("--plot", action="store_true", help="save the reproduced figure PNG")
    ap.add_argument("--max-cores", type=int, default=None,
                    help="cap parallelism to N cores so the machine stays responsive (sets "
                         "RAYON_NUM_THREADS for relay + BLAS thread envs, and caps --mw-workers). "
                         f"This box has {os.cpu_count()} logical cores; ~90%% is "
                         f"{int((os.cpu_count() or 1) * 0.9)}.")
    args = ap.parse_args()

    # Cap parallelism BEFORE relay/BLAS initialise their thread pools. relay_bp (rayon) reads
    # RAYON_NUM_THREADS on first parallel decode; spawned Technique-II workers inherit these
    # env vars and the capped worker count. Leaves headroom so the desktop stays usable.
    if args.max_cores is not None:
        n = max(1, int(args.max_cores))
        os.environ["RAYON_NUM_THREADS"] = str(n)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = str(n)
        print(f"[cfg] capping parallelism to {n}/{os.cpu_count()} cores "
              f"(RAYON_NUM_THREADS={n}, mw_workers≤{n})")

    cfg = Config.smoke() if args.smoke else Config.production()
    if args.noise_model is not None:
        cfg.noise_model = None if args.noise_model == "full" else args.noise_model
    if args.code is not None:
        cfg.code, cfg.code_label = CODES[args.code]
        if args.rounds is None: cfg.rounds = cfg.code.distance   # default rounds = code distance
        if args.code != "bb72":
            # The exact onset pin (mw_f0_override/mw_w0_override) is the BB(6) reference enumeration
            # (D=6, f0=2.3239e-5) — invalid for any other code. Clear it so Technique II actually
            # runs the symmetry-augmented BP-OSD search for this code instead of shortcutting to D=6.
            cfg.mw_f0_override = None
            cfg.mw_w0_override = None
    if args.rounds is not None: cfg.rounds = args.rounds
    if args.p_ref is not None:  cfg.p_ref = args.p_ref
    if args.shots is not None:  cfg.shots_per_weight = args.shots
    if args.shots_by_weight is not None: cfg.shots_by_weight = _parse_shots_by_weight(args.shots_by_weight)
    if args.weights is not None:
        cfg.weights = _parse_weight_spec(args.weights); cfg.weights_explicit = True
    elif args.weights_hi is not None:
        cfg.weights = list(range(2, args.weights_hi + 1))
    if args.weight_stride is not None and args.weight_stride > 1:
        cfg.weight_stride = args.weight_stride
        cfg.weights = sorted(cfg.weights)[::args.weight_stride]   # decimate the plan
        cfg.weights_explicit = True                              # don't let the onset auto-bracket undo it
    if args.adaptive: cfg.adaptive = True
    if args.adaptive_failures is not None: cfg.adaptive_failures = args.adaptive_failures
    if args.adaptive_shots_min is not None: cfg.adaptive_shots_min = args.adaptive_shots_min
    if args.adaptive_shots_max is not None: cfg.adaptive_shots_max = args.adaptive_shots_max
    if args.seed is not None:   cfg.seed = args.seed
    if args.mw_max_trials is not None: cfg.mw_max_trials = args.mw_max_trials
    if args.mw_workers is not None: cfg.mw_workers = args.mw_workers
    if args.max_cores is not None: cfg.mw_workers = min(cfg.mw_workers, max(1, int(args.max_cores)))
    if args.no_symmetry: cfg.mw_use_symmetry = False
    if args.p_meas_factor is not None:
        cfg.p_meas_factor = args.p_meas_factor
        if cfg.p_meas_factor != 1.0:
            cfg.mw_f0_override = None   # symmetric onset pin invalid under asymmetric noise → recompute
    if args.full_dem:
        cfg.mw_single_sector = False
        cfg.mw_f0_override = None     # the exact-onset pin is single-sector-only → run the search
    if args.split_p_low is not None: cfg.split_p_low = args.split_p_low
    if args.split_n_levels is not None: cfg.split_n_levels = args.split_n_levels
    if args.split_chain_steps is not None: cfg.split_chain_steps = args.split_chain_steps
    if args.split_n_seeds is not None: cfg.split_n_seeds = args.split_n_seeds

    outdir = args.outdir if not args.smoke else args.outdir.with_name(args.outdir.name + "_smoke")

    if args.split_only:
        outdir.mkdir(parents=True, exist_ok=True)
        tech2 = _load_json(outdir / "distance.json")
        tech3 = run_technique_iii(cfg, tech2, outdir)
        print(f"\n[split-only] done. P at p_low={cfg.split_p_low}: {tech3['P_logical'][-1]:.3e}")
        return

    run_all(cfg, outdir, do_onset=args.onset_scan, do_split=not args.no_split, do_plot=args.plot,
            do_decoder_conv=args.decoder_conv)


if __name__ == "__main__":
    main()
