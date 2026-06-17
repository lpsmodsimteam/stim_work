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
from typing import Dict, List, Optional

import numpy as np

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

from bb_code_sim import BBCodeSimulator, BB_72_12_6, RelayBPDecoder  # noqa: E402
from surface_code_sim import ErrorModel  # noqa: E402
from importance_sampling import (  # noqa: E402
    FailureSpectrum,
    ImportanceSamplingResult,
    _parse_dem,
    _expand,
    _sample_failures_at_weight,
    fit_failure_spectrum,
    logical_error_rate_from_ansatz,
)


# =================================================================================
# Configuration — production vs smoke. Both drive the SAME code paths in run_all;
# only the numbers differ, so --smoke is a faithful dry run of the multi-hour job.
# =================================================================================
@dataclass
class Config:
    label: str

    # circuit / noise
    p_ref: float = 0.003                 # circuit is built at this physical error rate
    rounds: int = BB_72_12_6.distance    # syndrome rounds (= d = 6 for the memory experiment)

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
    # 500 (was 3000): the IS sweep dominates the runtime, and the Technique-I ansatz pools
    # ~45 weights so it tolerates the larger per-weight binomial noise; only the raw IS points
    # get wider error bars. Override with --shots for tighter raw points.
    shots_per_weight: int = 500
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

    # Technique III — splitting cross-check
    split_p_high: float = 0.006
    split_p_low: float = 0.001
    split_n_levels: int = 6
    # Nudged up from 8/2000/500. The BB(6) diagnostic gave across-seed log_ratios_se≈0.45 at
    # 6 seeds / 500 steps. Crucially we raise the *seeds-to-chain-length ratio*: for a target
    # split into many disconnected logical sectors, more independent seeds cover the sectors
    # better than longer chains (a chain trapped in one sector can't escape it). So n_seeds
    # grows 4× (8→32) while chain_steps grows only 2× (2000→4000) — ratio ~2× higher.
    # Still within the production band of notebooks/gross_code/SPLITTING.md.
    split_n_seeds: int = 32
    split_chain_steps: int = 4000
    split_burn_in: int = 1000
    split_anchor_shots: int = 3000
    split_min_weight_max_trials: int = 400   # BP-OSD trials for splitting's min-weight seeds

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
            onset_shots=40,
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


def build_circuit(cfg: Config):
    """The BB(6) = [[72,12,6]] memory circuit at p_ref with cfg.rounds syndrome rounds."""
    em = ErrorModel.symmetric(cfg.p_ref)
    return BBCodeSimulator(BB_72_12_6).build_circuit(em, rounds=cfg.rounds)


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


def _spectrum_from_checkpoint(ckpt: dict) -> FailureSpectrum:
    done: Dict[int, int] = {int(w): int(F) for w, F in ckpt["failures_by_weight"].items()}
    shots = int(ckpt["shots_per_weight"])
    weights = [w for w in ckpt["weights_plan"] if w in done]
    return FailureSpectrum(
        weights=weights,
        trials=[shots] * len(weights),
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
        dem_check_action_matrices, compute_distance, find_min_weight_logicals,
        min_weight_fail_count,
    )

    # Resume: Technique II is not cheap and depends only on the circuit (rounds), not on p or
    # shots, so a prior distance.json with the same rounds can be reused verbatim on restart.
    cached = _load_json(outdir / "distance.json")
    if cached is not None and int(cached.get("rounds", -1)) == cfg.rounds:
        print(f"\nTechnique II — reusing cached distance.json "
              f"(D={cached['distance']}, onset w0={cached['onset']}, "
              f"f0={cached['onset_fraction']:.3e})", flush=True)
        return cached

    print("\nTechnique II — min-weight onset (BP-OSD, ~1.2s/decode; prints progress) ...", flush=True)
    circuit = build_circuit(cfg)
    H, A, mult, priors = dem_check_action_matrices(circuit)
    t0 = time.perf_counter()

    print(f"  [II.1] fault distance D over {A.shape[0]} logicals ...", flush=True)
    dr = compute_distance(circuit, osd_order=cfg.mw_osd_order, max_iter=cfg.mw_max_iter,
                          priors=priors, progress=True, workers=cfg.mw_workers)
    D = dr.distance
    print(f"  [II.1] D={D}, onset w0={D // 2}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    t1 = time.perf_counter()
    K_obs = circuit.num_observables
    n_sys = (1 << K_obs) - 1 if (cfg.mw_systematic and K_obs <= 20) else 0
    print(f"  [II.2] L(D) search: {n_sys} systematic + up to {cfg.mw_max_trials} random trials ...",
          flush=True)
    logicals = find_min_weight_logicals(
        circuit, D, max_trials=cfg.mw_max_trials, osd_order=cfg.mw_osd_order,
        max_iter=cfg.mw_max_iter, priors=priors, seed=cfg.seed,
        progress_every=max(max(n_sys, cfg.mw_max_trials) // 40, 1), workers=cfg.mw_workers,
        systematic=cfg.mw_systematic,
    )
    print(f"  [II.2] |L(D)|={len(logicals)}   ({time.perf_counter() - t1:.0f}s)", flush=True)

    print("  [II.3] exact onset fraction f*(D/2) via Proposition 1 ...", flush=True)
    fails, n_exp = min_weight_fail_count(H, A, logicals, mult)
    half = D // 2
    log_choose = gammaln(n_exp + 1) - gammaln(half + 1) - gammaln(n_exp - half + 1)
    f_star = float(fails / np.exp(log_choose))

    out = {
        "rounds": cfg.rounds,
        "distance": int(D),
        "onset": int(half),                    # w0 = D/2
        "n_min_logicals": int(len(logicals)),
        "fail_count": int(fails),
        "n_expanded": int(n_exp),
        "onset_fraction": f_star,              # f0 = f*(D/2)
    }
    print(f"  D={D}, onset w0={half}, |L(D)|={len(logicals)}, f0=f*(D/2)={f_star:.3e}   "
          f"(total {time.perf_counter() - t0:.0f}s)")
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
        print(f"[is] resuming: {len(failures_by_weight)}/{len(weights_plan)} weights done")
    else:
        failures_by_weight = {}

    remaining = [w for w in weights_plan if w not in failures_by_weight]

    # Heavy setup (DEM parse + decoder) — only if there's sampling left to do.
    if remaining:
        print("[is] building circuit + decoder ...", flush=True)
        circuit = build_circuit(cfg)
        probs, det_mat, obs_mat = _parse_dem(circuit)
        col_to_mech, q_base, _ = _expand(probs, None)
        n_expanded = int(col_to_mech.shape[0])
        decoder = make_decoder(cfg)
        decoder.setup(circuit)
        print(f"[is] {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
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
            "seed": cfg.seed,
            "weights_plan": weights_plan,
            "failures_by_weight": {str(w): int(F) for w, F in sorted(failures_by_weight.items())},
        })

    t0 = time.perf_counter()
    for i, w in enumerate(remaining):
        rng = np.random.default_rng([cfg.seed, w])   # per-weight RNG → resume-order-independent
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, cfg.shots_per_weight, decoder, rng)
        failures_by_weight[w] = F
        save()  # checkpoint BEFORE moving on — a crash here loses nothing already saved
        done = len(failures_by_weight)
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (len(remaining) - i - 1) / rate if rate > 0 else float("nan")
        print(f"[is] w={w:>4}: F/T={F:>4}/{cfg.shots_per_weight} = {F/cfg.shots_per_weight:.3f}"
              f"   ({done}/{len(weights_plan)}, {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

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
    if cfg.pin_onset and tech2 is not None:
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
    """Multi-seeded Metropolis splitting estimate over an overlapping p-ladder (cross-check)."""
    from splitting import splitting_estimate

    print("\nTechnique III — multi-seeded splitting cross-check ...", flush=True)
    circuit = build_circuit(cfg)
    distance = int(tech2["distance"]) if tech2 is not None else BB_72_12_6.distance
    res = splitting_estimate(
        circuit, make_decoder(cfg),
        p_ref=cfg.p_ref, p_high=cfg.split_p_high, p_low=cfg.split_p_low,
        n_levels=cfg.split_n_levels, n_seeds=cfg.split_n_seeds,
        chain_steps=cfg.split_chain_steps, burn_in=cfg.split_burn_in,
        anchor_shots=cfg.split_anchor_shots, distance=distance,
        min_weight_max_trials=cfg.split_min_weight_max_trials, seed=cfg.seed,
    )
    out = {
        "p_ladder": np.asarray(res.p_ladder).tolist(),
        "P_logical": np.asarray(res.P_logical).tolist(),
        "P_logical_se": np.asarray(res.P_logical_se).tolist(),
    }
    print(f"  ladder {res.p_ladder[0]:.2e}..{res.p_ladder[-1]:.2e}, "
          f"P {np.asarray(res.P_logical)[0]:.2e}..{np.asarray(res.P_logical)[-1]:.2e}")
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
    ax.set_title(f"BB(6) [[72,12,6]] — Fig. 10 reproduction (Relay-BP, {cfg.label})")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "bb6_fig10.png", dpi=150)
    print(f"  wrote {outdir/'bb6_fig10.png'}")


# ================================== driver =======================================
def run_all(cfg: Config, outdir: pathlib.Path, *, do_onset: bool = False,
            do_split: bool = True, do_plot: bool = False) -> dict:
    """Full pipeline. Returns a dict of every result (also written to disk)."""
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.label}] repo={REPO_ROOT}\n[{cfg.label}] outdir={outdir}")
    print(f"[{cfg.label}] p_ref={cfg.p_ref}, rounds={cfg.rounds}, "
          f"shots/weight={cfg.shots_per_weight}, seed={cfg.seed}")
    _atomic_write_json(outdir / "config.json", dataclasses.asdict(cfg))

    weights_plan = list(cfg.weights)
    if do_onset:
        onset = onset_scan(cfg, outdir)
        if onset.get("last_zero") is not None:
            weights_plan = list(range(onset["last_zero"], onset["w_hi"] + 1))
            print(f"  → onset-derived WEIGHTS {weights_plan[0]}..{weights_plan[-1]}")

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

    if do_plot:
        print("\nplotting ...")
        make_plot(cfg, is_result, tech1, tech3, outdir)

    print(f"\n[{cfg.label}] done.")
    return {"tech1": tech1, "tech2": tech2, "tech3": tech3, "is": is_result}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent / "bb6_fig10_out")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end dry run (fast Relay settings, ~20 shots) — exercises "
                         "every code path the production run hits, in well under a minute")
    ap.add_argument("--rounds", type=int, default=None, help="override syndrome rounds (default d=6)")
    ap.add_argument("--p-ref", type=float, default=None, help="override circuit build rate")
    ap.add_argument("--shots", type=int, default=None, help="override shots per weight")
    ap.add_argument("--seed", type=int, default=None, help="override base RNG seed")
    ap.add_argument("--mw-max-trials", type=int, default=None,
                    help="cap Technique-II L(D) search trials (lower = faster, less complete L(D); "
                         "production default 2000)")
    ap.add_argument("--mw-workers", type=int, default=None,
                    help="parallelize Technique-II BP-OSD decodes across N processes "
                         "(default 1 = serial; try 24 on this machine)")
    ap.add_argument("--onset-scan", action="store_true", help="run onset scan to size WEIGHTS")
    ap.add_argument("--no-split", action="store_true", help="skip Technique III (splitting)")
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
    if args.rounds is not None: cfg.rounds = args.rounds
    if args.p_ref is not None:  cfg.p_ref = args.p_ref
    if args.shots is not None:  cfg.shots_per_weight = args.shots
    if args.seed is not None:   cfg.seed = args.seed
    if args.mw_max_trials is not None: cfg.mw_max_trials = args.mw_max_trials
    if args.mw_workers is not None: cfg.mw_workers = args.mw_workers
    if args.max_cores is not None: cfg.mw_workers = min(cfg.mw_workers, max(1, int(args.max_cores)))

    outdir = args.outdir if not args.smoke else args.outdir.with_name(args.outdir.name + "_smoke")
    run_all(cfg, outdir, do_onset=args.onset_scan, do_split=not args.no_split, do_plot=args.plot)


if __name__ == "__main__":
    main()
