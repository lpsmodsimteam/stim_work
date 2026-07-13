#!/usr/bin/env python3
"""Config-driven, cluster-runnable fail-fast experiment engine (arXiv:2511.15177).

One experiment = (code x circuit x decoder x techniques x budget), specified by a YAML config and
run by ``python -m experiment_runner --config <cfg.yaml>``. Each run writes per-technique JSON + a
combined ``result.npz`` to its own ``outdir`` under ``runs/`` and is checkpointed after **every
weight**, so a crashed/requeued SLURM task resumes with at most one weight of lost work.

This generalises the proven BB(6) Figure-10 driver (``experiments/bravyi/bb6_fig10_sweep.py``) into a
code/circuit/decoder-agnostic runner via three registries:

  * CODES            — bivariate-bicycle codes (bb6, bb144 active; bb18, bb288 seams)
  * CIRCUIT_BUILDERS — circuit kinds (memory active; lpu_x1/lpu_z1 wired; automorphism/joint_pauli stubs)
  * DECODERS         — decoders (relay active; bposd, pymatching seams)

The three "fail-fast" techniques are library calls:
  * Technique I   — failure-spectrum ansatz fit + extrapolation   (importance_sampling.py)
  * Technique II  — exact min-weight onset (D, w0, f0)            (min_weight.py)
  * Technique III — multi-seeded / replica-exchange splitting      (splitting.py)

Adding a dimension is a one-line registry entry (codes/decoders) or one builder in
``gross_code_lpu_tdg.py`` + a CIRCUIT_BUILDERS entry (circuits). See ``experiments/README.md``.
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

# Force UTF-8 stdout/stderr: progress lines print mu/->/<- which the default Windows console
# encoding (cp1252) chokes on — that would crash a multi-hour run. No-op where unsupported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from scipy.special import gammaln

from bb_code_sim import (BBCodeParams, BBCodeSimulator, BB_18_4_4, BB_72_12_6, BB_144_12_12,
                         BB_288_12_18, RelayBPDecoder, BPOSDDecoder, BBPyMatchingDecoder,
                         NOISE_CHANNEL_PREDICATES, filter_noise_channel)
from surface_code_sim import ErrorModel
from importance_sampling import (
    FailureSpectrum, ImportanceSamplingResult, _parse_dem, _expand, _sample_failures_at_weight,
    fit_failure_spectrum, logical_error_rate_from_ansatz, predict_failure_fraction,
    shots_to_hit_failures,
)
from repo_paths import RUNS


# =================================================================================
# Registries — the extension points (add a code/decoder/circuit here)
# =================================================================================
CODES: Dict[str, Tuple[BBCodeParams, str]] = {
    "bb6":   (BB_72_12_6,   "BB(6)=[[72,12,6]]"),
    "bb144": (BB_144_12_12, "BB(12)=[[144,12,12]]"),
    # seams (verified n/k; not P0):
    "bb18":  (BB_18_4_4,    "[[18,4,4]]"),
    "bb288": (BB_288_12_18, "[[288,12,18]] two-gross"),
}


def _build_memory_circuit(cfg: "Config"):
    em = ErrorModel(p_phys=cfg.p_ref, p_meas=cfg.p_ref * cfg.p_meas_factor)
    return BBCodeSimulator(cfg.code).build_circuit(em, rounds=cfg.rounds)


def _build_lpu_circuit(cfg: "Config", operator: str):
    # The LPU (tour-de-gross) circuits are gross-code-specific (hardcoded 288-qubit layout).
    import gross_code_lpu_tdg as tdg
    em = ErrorModel(p_phys=cfg.p_ref, p_meas=cfg.p_ref * cfg.p_meas_factor)
    fn = tdg.build_logical_x1_circuit if operator == "X1" else tdg.build_logical_z1_circuit
    return fn(em, C=cfg.lpu_C, d_init=cfg.lpu_d_init)


def _circuit_stub(kind: str):
    def _stub(cfg: "Config"):
        raise NotImplementedError(
            f"circuit kind {kind!r} is a registered seam — implement build_{kind}_circuit in "
            f"src/gross_code_lpu_tdg.py and wire it into CIRCUIT_BUILDERS.")
    return _stub


CIRCUIT_BUILDERS = {
    "memory":       _build_memory_circuit,
    "lpu_x1":       lambda cfg: _build_lpu_circuit(cfg, "X1"),
    "lpu_z1":       lambda cfg: _build_lpu_circuit(cfg, "Z1"),
    "automorphism": _circuit_stub("automorphism"),
    "joint_pauli":  _circuit_stub("joint_pauli"),
}


def _make_relay(cfg: "Config") -> RelayBPDecoder:
    return RelayBPDecoder(
        gamma0=cfg.relay_gamma0, pre_iter=cfg.relay_pre_iter, num_sets=cfg.relay_num_sets,
        set_max_iter=cfg.relay_set_max_iter,
        gamma_dist_interval=(cfg.relay_gamma_lo, cfg.relay_gamma_hi),
        stop_nconv=cfg.relay_stop_nconv,
    )


DECODERS = {
    "relay":      _make_relay,
    "bposd":      lambda cfg: BPOSDDecoder(osd_order=cfg.mw_osd_order),
    "pymatching": lambda cfg: BBPyMatchingDecoder(),
}


# =================================================================================
# Configuration — production vs smoke. Both drive the SAME code paths, so --smoke
# is a faithful dry run of the full job. YAML configs overlay onto this dataclass.
# =================================================================================
@dataclass
class Config:
    label: str = "production"

    # --- what to run (the registry selectors) ---
    code_name: str = "bb6"                # key into CODES
    experiment: str = "memory"            # key into CIRCUIT_BUILDERS
    decoder_name: str = "relay"           # key into DECODERS
    # which techniques (and helpers) to run, in dependency order. "IS" feeds "I".
    techniques: List[str] = field(default_factory=lambda: ["II", "IS", "I", "III"])
    split_method: str = "multiseed"       # Technique III: "multiseed" (paper Alg.2/3) or "replica"
    outdir: Optional[str] = None          # default: runs/<bucket>/<code>/<experiment>

    # --- circuit / noise ---
    p_ref: float = 0.003
    p_meas_factor: float = 1.0            # p_meas = p_meas_factor * p_phys
    rounds: Optional[int] = None          # default: code distance
    lpu_C: int = 10                       # LPU repeated-measurement rounds (lpu_* experiments)
    lpu_d_init: int = 12
    # Five-channel budget campaigns: isolate ONE channel (noise_channel) or drop one channel,
    # keeping the rest (ablate_channel = leave-one-out marginal). Mutually exclusive; applied to
    # the built circuit AFTER the experiment builder — the full-noise path (both None) returns the
    # builder's circuit untouched. Keys = bb_code_sim.NOISE_CHANNEL_PREDICATES minus 'idle' (use
    # the gate_idle/meas_idle split; bare 'idle' would break the five-channel partition).
    noise_channel: Optional[str] = None
    ablate_channel: Optional[str] = None

    # --- Relay-BP decoder (paper Sec 2.4 BB(6) settings in production) ---
    relay_gamma0: float = 0.125
    relay_pre_iter: int = 80
    relay_num_sets: int = 600
    relay_set_max_iter: int = 60
    relay_gamma_lo: float = -0.24
    relay_gamma_hi: float = 0.66
    relay_stop_nconv: int = 6

    # --- importance-sampling sweep (Technique I input) ---
    weights: List[int] = field(default_factory=lambda: list(range(2, 60)))
    weights_explicit: bool = False
    shots_per_weight: int = 500
    shots_by_weight: Optional[Dict[int, int]] = None
    adaptive: bool = False                # 'hit N failures/weight', sweep high->low, predict f(w)
    adaptive_failures: int = 50
    adaptive_shots_min: int = 4
    adaptive_shots_max: int = 20000
    adaptive_predict_window: int = 3
    weight_stride: int = 1
    seed: int = 42

    # --- reweight / extrapolation target grid ---
    p_lo: float = 1e-4
    p_hi: float = 1e-2
    n_p: int = 30

    # --- Technique I — ansatz ---
    ansatz_model: str = "f5"
    pin_onset: bool = True

    # --- Technique II — min-weight search ---
    mw_osd_order: int = 10
    mw_max_iter: int = 200
    mw_max_trials: int = 2000
    mw_workers: int = 1
    mw_systematic: bool = True
    mw_use_symmetry: bool = True
    mw_single_sector: bool = True
    mw_sector_type: int = 0
    mw_f0_override: Optional[float] = None   # bb6 exact pin lives in the bb6 YAML, not the default
    mw_w0_override: Optional[int] = None
    # Decimation (paper §4.2): fix supp(g) odd-weight + decode the reduced low-weight problem, instead
    # of decoding the high-weight [H;g]. Essential for hard searches (e.g. BB(18) two-gross weight-18).
    mw_decimate: bool = False
    mw_decimate_max_odd: int = 3
    # Fault restriction (paper §4.2): search a circuit with FEWER QEC cycles (e.g. 2) for the
    # min-weight logicals — far fewer fault columns, so the search is tractable. The found logicals
    # are valid for the full circuit (a subset). Technique-II ONLY; the rest of the run uses cfg.rounds.
    mw_search_rounds: Optional[int] = None

    # --- Technique III — splitting ---
    split_p_high: float = 0.006
    split_p_low: float = 0.003
    split_n_levels: int = 20
    split_anchor_shots: int = 3000
    # multiseed (paper Alg.2/3) knobs
    split_L: int = 12
    split_M: int = 3
    split_T_init: int = 100_000
    split_eps: float = 0.25
    split_ladder: str = "eq18"
    # replica-exchange knobs (split_method="replica")
    split_re_walkers: int = 6
    split_re_local_steps: int = 6
    split_re_sweeps: int = 200
    split_re_burn_in: int = 60

    # --- onset scan ---
    onset_shots: int = 200
    onset_coarse_step: int = 1

    @property
    def code(self) -> BBCodeParams:
        return CODES[self.code_name][0]

    @property
    def code_label(self) -> str:
        return CODES[self.code_name][1]

    @property
    def p_targets(self) -> np.ndarray:
        return np.logspace(np.log10(self.p_lo), np.log10(self.p_hi), self.n_p)

    def resolved_outdir(self) -> pathlib.Path:
        if self.outdir:
            return pathlib.Path(self.outdir)
        suffix = (f"__iso_{self.noise_channel}" if self.noise_channel
                  else f"__abl_{self.ablate_channel}" if self.ablate_channel else "")
        return RUNS / "framework" / self.code_name / (self.experiment + suffix)

    def __post_init__(self):
        if self.rounds is None:
            self.rounds = self.code.distance
        if self.noise_channel and self.ablate_channel:
            raise SystemExit("noise_channel and ablate_channel are mutually exclusive "
                             "(isolate one channel OR leave one out, not both)")
        valid_channels = sorted(set(NOISE_CHANNEL_PREDICATES) - {"idle"})
        for key in (self.noise_channel, self.ablate_channel):
            if key is not None and key not in valid_channels:
                raise SystemExit(
                    f"unknown noise channel {key!r}; valid: {valid_channels} "
                    "('idle' is the union of gate_idle+meas_idle — use the split channels)")

    @classmethod
    def smoke(cls, **over) -> "Config":
        """Tiny config: every technique runs end-to-end in well under a minute. Fast (not paper-
        accuracy) Relay settings + a handful of shots/weights/chain-steps."""
        base = dict(
            label="smoke", shots_per_weight=20, weights=list(range(2, 11)), n_p=6,
            relay_gamma0=0.1, relay_pre_iter=20, relay_num_sets=20, relay_set_max_iter=20,
            relay_stop_nconv=5, mw_max_trials=50, mw_systematic=False,
            split_p_high=0.01, split_p_low=0.005, split_n_levels=2, split_anchor_shots=80,
            split_L=2, split_M=2, split_T_init=300, split_ladder="geom",
            split_re_walkers=2, split_re_local_steps=2, split_re_sweeps=4, split_re_burn_in=2,
            onset_shots=40,
        )
        base.update(over)
        return cls(**base)


# ============================ decoder / circuit seams ============================
def make_decoder(cfg: Config):
    try:
        return DECODERS[cfg.decoder_name](cfg)
    except KeyError:
        raise SystemExit(f"unknown decoder {cfg.decoder_name!r}; choices: {sorted(DECODERS)}")


def build_circuit(cfg: Config):
    try:
        circ = CIRCUIT_BUILDERS[cfg.experiment](cfg)
    except KeyError:
        raise SystemExit(f"unknown experiment {cfg.experiment!r}; choices: {sorted(CIRCUIT_BUILDERS)}")
    # Channel isolation/ablation for budget campaigns. Full-noise path (both None) returns the
    # builder's circuit object untouched — guaranteed byte-identical to the pre-extension runner.
    if cfg.noise_channel is None and cfg.ablate_channel is None:
        return circ
    if cfg.noise_channel:
        return filter_noise_channel(circ, cfg.noise_channel)
    drop = NOISE_CHANNEL_PREDICATES[cfg.ablate_channel]
    return filter_noise_channel(circ, lambda i, p, n: not drop(i, p, n))


def _technique_ii_circuit(cfg: Config):
    """Circuit for the Technique-II min-weight search — fault-restricted to cfg.mw_search_rounds QEC
    cycles when set (paper §4.2 "fault restrictions"), else the full cfg.rounds circuit. The reduced
    circuit has far fewer fault columns (tractable search); its weight-D logicals are valid for the
    full circuit. Memory experiments only."""
    if cfg.mw_search_rounds and cfg.experiment == "memory":
        return build_circuit(dataclasses.replace(cfg, rounds=int(cfg.mw_search_rounds)))
    return build_circuit(cfg)


# ================================ checkpoint I/O ==================================
def _atomic_write_json(path: pathlib.Path, obj: dict, *, retries: int = 8) -> None:
    """Write JSON to a temp file then rename (atomic). Retries the rename on Windows
    PermissionError (AV/indexer holding the dest), so a transient lock can't kill a long run."""
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
    return ImportanceSamplingResult(p_values=p_arr, P_logical=P_logical, P_logical_se=np.sqrt(var),
                                    spectrum=spectrum)


def _shots_for(cfg: Config, w: int) -> int:
    if cfg.shots_by_weight:
        t = cfg.shots_by_weight.get(int(w))
        if t is not None:
            return int(t)
    return int(cfg.shots_per_weight)


def _parse_shots_by_weight(spec: str) -> Dict[int, int]:
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
    tbw: Dict[int, int] = {int(w): int(t) for w, t in ckpt.get("trials_by_weight", {}).items()}
    shots = int(ckpt.get("shots_per_weight", 0))
    weights = [w for w in ckpt["weights_plan"] if w in done]
    return FailureSpectrum(
        weights=weights, trials=[tbw.get(w, shots) for w in weights],
        failures=[done[w] for w in weights], n_expanded=int(ckpt["n_expanded"]),
        q_base=float(ckpt["q_base"]), p_ref=float(ckpt["p_ref"]),
    )


# ============================ Section 0: onset scan ===============================
ONSET_SEED_STREAM = 1000


def onset_scan(cfg: Config, outdir: pathlib.Path) -> dict:
    """Locate the failure onset and size a contiguous WEIGHTS block (checkpointed, deterministic)."""
    print("Section 0 — onset scan", flush=True)
    circuit = build_circuit(cfg)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, None)
    n_expanded = int(col_to_mech.shape[0])
    decoder = make_decoder(cfg)
    decoder.setup(circuit)

    d = cfg.code.distance
    mu = n_expanded * q_base * (cfg.p_targets / cfg.p_ref)
    mu_max = float(mu.max())
    w_hi = int(np.ceil(mu_max + 4.0 * np.sqrt(mu_max)))
    print(f"  N_expanded={n_expanded}, q_base={q_base:.5f}, mu(p): {mu.min():.1f}..{mu_max:.1f}, "
          f"w_hi={w_hi}", flush=True)

    ckpt_path = outdir / "onset.spectrum.json"
    ck = _load_json(ckpt_path)
    if ck is not None and (int(ck.get("shots", -1)) != cfg.onset_shots or int(ck.get("seed", -1)) != cfg.seed):
        ck = None
    cache: Dict[int, int] = {int(w): int(F) for w, F in ck["failures_by_weight"].items()} if ck else {}

    def scan(w: int) -> int:
        w = int(w)
        if w in cache:
            return cache[w]
        rng = np.random.default_rng([cfg.seed, ONSET_SEED_STREAM, w])
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, cfg.onset_shots, decoder, rng)
        cache[w] = F
        _atomic_write_json(ckpt_path, {"shots": cfg.onset_shots, "seed": cfg.seed,
                                       "failures_by_weight": {str(k): int(v) for k, v in sorted(cache.items())}})
        print(f"    w={w:>4}: F/T = {F:>3}/{cfg.onset_shots} = {F/cfg.onset_shots:.3f}", flush=True)
        return F

    onset: Optional[int] = None
    w = max(d // 2, 2)
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
    """Circuit fault distance D, onset w0=ceil(D/2), and exact onset fraction f0 (even D)."""
    from min_weight import (dem_check_action_matrices, single_sector_dem, compute_distance,
                            find_min_weight_logicals, min_weight_fail_count, expanded_logical_count,
                            build_circuit_translation_perms)

    # Exact onset pin (e.g. bb6 reference enumeration): skip the BP-OSD search.
    if cfg.mw_f0_override is not None:
        w0 = int(cfg.mw_w0_override)
        out = {"rounds": cfg.rounds, "single_sector": bool(cfg.mw_single_sector),
               "sector_type": int(cfg.mw_sector_type) if cfg.mw_single_sector else None,
               "distance": 2 * w0, "onset": w0, "onset_fraction": float(cfg.mw_f0_override),
               "exact_pin": True}
        print(f"\nTechnique II — exact onset pin: D={2*w0}, w0={w0}, f0={cfg.mw_f0_override:.3e}", flush=True)
        _atomic_write_json(outdir / "distance.json", out)
        return out

    cached = _load_json(outdir / "distance.json")
    if (cached is not None and int(cached.get("rounds", -1)) == cfg.rounds
            and bool(cached.get("single_sector", False)) == cfg.mw_single_sector
            and cached.get("search_rounds", None) == cfg.mw_search_rounds):
        print(f"\nTechnique II — reusing cached distance.json (D={cached['distance']}, "
              f"w0={cached['onset']})", flush=True)
        return cached

    sector = cfg.mw_sector_type if cfg.mw_single_sector else None
    mode = f"single-sector (type {sector})" if cfg.mw_single_sector else "full both-sector"
    restrict = f", faults restricted to {cfg.mw_search_rounds} QEC cycles" if cfg.mw_search_rounds else ""
    print(f"\nTechnique II — min-weight onset, {mode} DEM{restrict} "
          f"(BP-OSD{'+decimation' if cfg.mw_decimate else ''}; prints progress) ...", flush=True)
    circuit = _technique_ii_circuit(cfg)

    det_coords = None
    if cfg.mw_single_sector:
        H, A, mult, priors, det_coords = single_sector_dem(circuit, detector_type=cfg.mw_sector_type)
    else:
        H, A, mult, priors = dem_check_action_matrices(circuit)
    n_comp = H.shape[1]
    n_exp = int(mult.sum())
    print(f"  N(compressed)={n_comp}, N(expanded)={n_exp}", flush=True)
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
        try:
            sym_perms = build_circuit_translation_perms(circuit, H, l=cfg.code.l, m=cfg.code.m,
                                                        det_coords=det_coords, verbose=True)
            print(f"  [II.2] {len(sym_perms)} toric perms ready", flush=True)
        except (KeyError, ValueError) as exc:
            print(f"  [II.2] WARNING: toric sym build failed ({exc}); no symmetry", flush=True)
            sym_perms = None

    print(f"  [II.2] L(D) search: {n_sys} systematic + up to {cfg.mw_max_trials} random ...", flush=True)
    logicals, search_trace = find_min_weight_logicals(
        circuit, D, max_trials=cfg.mw_max_trials, osd_order=cfg.mw_osd_order, max_iter=cfg.mw_max_iter,
        priors=priors, seed=cfg.seed, progress_every=max(max(n_sys, cfg.mw_max_trials) // 40, 1),
        workers=cfg.mw_workers, systematic=cfg.mw_systematic, symmetry_perms=sym_perms, sector=sector,
        decimate=cfg.mw_decimate, decimate_max_odd=cfg.mw_decimate_max_odd, return_trace=True)
    ld_comp = len(logicals)
    _atomic_write_json(outdir / "search_convergence.json", {
        "trace": [[int(t), int(c)] for (t, c) in search_trace], "n_systematic": int(n_sys),
        "max_trials": int(cfg.mw_max_trials), "single_sector": bool(cfg.mw_single_sector),
        "distance": int(D), "final": int(ld_comp)})
    ld_exp = expanded_logical_count(logicals, mult)
    print(f"  [II.2] |L(D)| compressed={ld_comp}, expanded={ld_exp:.4g}   "
          f"({time.perf_counter() - t1:.0f}s)", flush=True)

    half = (D + 1) // 2
    if logicals and D % 2 == 0:
        print("  [II.3] exact onset fraction f*(D/2) via Proposition 1 ...", flush=True)
        fails, n_exp = min_weight_fail_count(H, A, logicals, mult)
        log_choose = gammaln(n_exp + 1) - gammaln(half + 1) - gammaln(n_exp - half + 1)
        f_star = float(fails / np.exp(log_choose))
    else:
        n_exp = int(mult.sum())
        fails, f_star = 0, None
        why = "no weight-D logicals found" if not logicals else f"D={D} odd (Prop.1 is even-D only)"
        print(f"  [II.3] {why} -> onset fraction unpinned (f0=None)", flush=True)

    out = {"rounds": cfg.rounds, "search_rounds": cfg.mw_search_rounds,
           "single_sector": bool(cfg.mw_single_sector),
           "sector_type": int(cfg.mw_sector_type) if cfg.mw_single_sector else None, "method": "search",
           "distance": int(D), "distance_is_bound": True, "onset": int(half), "n_compressed": int(n_comp),
           "n_expanded": int(n_exp), "n_min_logicals": int(ld_comp), "n_min_logicals_expanded": int(ld_exp),
           "fail_count": int(fails), "onset_fraction": f_star}
    f0s = f"{f_star:.3e}" if f_star is not None else "None"
    print(f"  D<={D}, w0={half}, |L(D)|>={ld_comp}, f0={f0s}   ({time.perf_counter() - t0:.0f}s)")
    _atomic_write_json(outdir / "distance.json", out)
    return out


# ============================ IS sweep (checkpointed) ============================
def run_is_sweep(cfg: Config, outdir: pathlib.Path, weights_plan: List[int]) -> ImportanceSamplingResult:
    """Sample the failure spectrum f(w) over weights_plan, checkpointing after every weight."""
    ckpt_path = outdir / "spectrum.json"
    ckpt = _load_json(ckpt_path)
    if ckpt is not None:
        if (ckpt.get("weights_plan") != weights_plan or int(ckpt.get("seed", -1)) != cfg.seed):
            raise SystemExit(f"existing checkpoint {ckpt_path} has a different plan/seed. Delete it to "
                             f"start fresh, or restore parameters to resume.")
        failures_by_weight: Dict[int, int] = {int(w): int(F) for w, F in ckpt["failures_by_weight"].items()}
        trials_by_weight: Dict[int, int] = {int(w): int(t) for w, t in ckpt.get("trials_by_weight", {}).items()}
        if not trials_by_weight:
            _sc = int(ckpt.get("shots_per_weight", cfg.shots_per_weight))
            trials_by_weight = {w: _sc for w in failures_by_weight}
        print(f"[is] resuming: {len(failures_by_weight)}/{len(weights_plan)} weights done")
    else:
        failures_by_weight = {}
        trials_by_weight = {}

    remaining = [w for w in weights_plan if w not in failures_by_weight]
    if remaining:
        print("[is] building circuit + decoder ...", flush=True)
        circuit = build_circuit(cfg)
        if cfg.mw_single_sector:
            from min_weight import single_sector_dem
            H, A, mult, probs, _ = single_sector_dem(circuit, detector_type=cfg.mw_sector_type)
            det_mat = H.T.astype(bool); obs_mat = A.T.astype(bool)
            q_base = float(np.asarray(_parse_dem(circuit)[0]).min())
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
            print(f"[is] full DEM: N_expanded={n_expanded}, q_base={q_base:.5f}", flush=True)
    else:
        assert ckpt is not None
        n_expanded, q_base = int(ckpt["n_expanded"]), float(ckpt["q_base"])

    def save() -> None:
        _atomic_write_json(ckpt_path, {
            "n_expanded": int(n_expanded), "q_base": float(q_base), "p_ref": cfg.p_ref,
            "shots_per_weight": cfg.shots_per_weight,
            "trials_by_weight": {str(w): int(t) for w, t in sorted(trials_by_weight.items())},
            "seed": cfg.seed, "weights_plan": weights_plan,
            "failures_by_weight": {str(w): int(F) for w, F in sorted(failures_by_weight.items())}})

    order = sorted(remaining, reverse=True) if cfg.adaptive else remaining
    t0 = time.perf_counter()
    for i, w in enumerate(order):
        rng = np.random.default_rng([cfg.seed, w])
        if cfg.adaptive:
            measured_f = {ww: failures_by_weight[ww] / trials_by_weight[ww]
                          for ww in failures_by_weight if trials_by_weight.get(ww)}
            f_pred = predict_failure_fraction(measured_f, w, cfg.adaptive_predict_window)
            T_w = shots_to_hit_failures(f_pred, cfg.adaptive_failures, cfg.adaptive_shots_min,
                                        cfg.adaptive_shots_max)
        else:
            T_w = _shots_for(cfg, w)
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, T_w, decoder, rng)
        failures_by_weight[w] = F
        trials_by_weight[w] = T_w
        save()
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0.0
        eta = (len(order) - i - 1) / rate if rate > 0 else float("nan")
        print(f"[is] w={w:>4}: F/T={F:>4}/{T_w} = {F/T_w:.3f}"
              + (f"  (f_pred={f_pred:.2g})" if cfg.adaptive else "")
              + f"   ({len(failures_by_weight)}/{len(weights_plan)}, {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    if remaining:
        save()
    final = _load_json(ckpt_path)
    assert final is not None
    return reweight_spectrum(_spectrum_from_checkpoint(final), cfg.p_targets)


# ============================ Technique I: ansatz ===============================
def run_technique_i(cfg: Config, is_result: ImportanceSamplingResult, tech2: Optional[dict],
                    outdir: pathlib.Path) -> dict:
    """Fit the failure-spectrum ansatz and extrapolate LER(p). Pins w0/f0 from Technique II when
    that determined the onset fraction exactly (even D); otherwise fits them."""
    K = build_circuit(cfg).detector_error_model(decompose_errors=False).num_observables
    w0 = f0 = None
    if cfg.pin_onset and tech2 is not None and tech2.get("onset_fraction") is not None:
        w0, f0 = float(tech2["onset"]), float(tech2["onset_fraction"])
    print(f"\nTechnique I — ansatz '{cfg.ansatz_model}', K={K}, pinned w0={w0}, f0={f0}", flush=True)
    try:
        fit = fit_failure_spectrum(is_result.spectrum, K=K, model=cfg.ansatz_model, w0=w0, f0=f0)
        P_ext = logical_error_rate_from_ansatz(fit, list(cfg.p_targets))
    except ValueError as e:
        print(f"  [WARN] ansatz fit failed — {e}; skipping extrapolation (IS/splitting unaffected).")
        out = {"model": cfg.ansatz_model, "K": int(K), "error": str(e), "P_ext": None,
               "pinned": bool(cfg.pin_onset and tech2 is not None)}
        _atomic_write_json(outdir / "ansatz_fit.json", out)
        return out
    out = {"model": cfg.ansatz_model, "K": int(K), "params": {k: float(v) for k, v in fit.params.items()},
           "n_points": int(fit.n_points), "cost": float(fit.cost),
           "pinned": bool(cfg.pin_onset and tech2 is not None), "P_ext": P_ext.tolist()}
    print("  params: " + ", ".join(f"{k}={v:.3g}" for k, v in fit.params.items())
          + f"   (n_points={fit.n_points}, cost={fit.cost:.3g})")
    _atomic_write_json(outdir / "ansatz_fit.json", out)
    return out


# ============================ Technique III: splitting ==========================
def run_technique_iii(cfg: Config, tech2: Optional[dict], outdir: pathlib.Path) -> dict:
    """Splitting cross-check. split_method='multiseed' -> faithful paper Alg.2/3 (BAR + adaptive
    precision + sequential warm-start); 'replica' -> replica-exchange (parallel tempering)."""
    from splitting import multi_seeded_split_estimate, replica_exchange_estimate

    distance = int(tech2["distance"]) if tech2 is not None else cfg.code.distance
    circuit = build_circuit(cfg)
    print(f"\nTechnique III — splitting ({cfg.split_method}) ...", flush=True)
    if cfg.split_method == "multiseed":
        res, diag = multi_seeded_split_estimate(
            circuit, make_decoder(cfg), p_ref=cfg.p_ref, p_high=cfg.split_p_high, p_low=cfg.split_p_low,
            L=cfg.split_L, M=cfg.split_M, T_init=cfg.split_T_init, eps=cfg.split_eps,
            ladder=cfg.split_ladder, n_levels=cfg.split_n_levels, distance=distance,
            anchor_shots=cfg.split_anchor_shots, single_sector=cfg.mw_single_sector,
            sector=cfg.mw_sector_type, seed=cfg.seed)
        out = {"method": "multiseed", "p_ladder": np.asarray(res.p_ladder).tolist(),
               "P_logical": np.asarray(res.P_logical).tolist(),
               "P_logical_se": np.asarray(res.P_logical_se).tolist(),
               "mean_weight": diag["mean_weight"], "T_per_level": diag["T_per_level"],
               "sigma_plus_delta": diag["sigma_plus_delta"]}
    else:
        res, diag = replica_exchange_estimate(
            circuit, make_decoder(cfg), p_ref=cfg.p_ref, p_high=cfg.split_p_high, p_low=cfg.split_p_low,
            n_levels=cfg.split_n_levels, n_walkers=cfg.split_re_walkers,
            local_steps=cfg.split_re_local_steps, n_sweeps=cfg.split_re_sweeps,
            burn_in=cfg.split_re_burn_in, anchor_shots=cfg.split_anchor_shots, distance=distance,
            seed=cfg.seed, single_sector=cfg.mw_single_sector, sector=cfg.mw_sector_type)
        out = {"method": "replica", "p_ladder": np.asarray(res.p_ladder).tolist(),
               "P_logical": np.asarray(res.P_logical).tolist(),
               "P_logical_se": np.asarray(res.P_logical_se).tolist(),
               "swap_accept": list(diag["swap_accept"]), "mean_weight": list(diag["mean_weight"])}
    P = np.asarray(res.P_logical)
    print(f"  ladder {res.p_ladder[0]:.2e}..{res.p_ladder[-1]:.2e}, P {P[0]:.2e}..{P[-1]:.2e}")
    _atomic_write_json(outdir / "splitting.json", out)
    return out


# ================================== driver =======================================
def run_all(cfg: Config, outdir: pathlib.Path) -> dict:
    """Run the techniques listed in cfg.techniques (in dependency order) + write result.npz."""
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{cfg.label}] code={cfg.code_label}, experiment={cfg.experiment}, decoder={cfg.decoder_name}")
    print(f"[{cfg.label}] techniques={cfg.techniques}, outdir={outdir}")
    print(f"[{cfg.label}] p_ref={cfg.p_ref}, rounds={cfg.rounds}, seed={cfg.seed}")
    _atomic_write_json(outdir / "config.json", config_to_dict(cfg))

    want = set(cfg.techniques)
    weights_plan = list(cfg.weights)
    if "onset" in want and not cfg.weights_explicit:
        onset = onset_scan(cfg, outdir)
        if onset.get("last_zero") is not None:
            weights_plan = list(range(onset["last_zero"], onset["w_hi"] + 1))
    if cfg.weight_stride > 1:               # sample every Nth weight; the ansatz interpolates the rest
        weights_plan = sorted(weights_plan)[::cfg.weight_stride]

    tech2 = run_technique_ii(cfg, outdir) if "II" in want else None
    # Technique I needs the IS spectrum; sweep if either IS or I requested.
    is_result = run_is_sweep(cfg, outdir, weights_plan) if (want & {"IS", "I"}) else None
    tech1 = run_technique_i(cfg, is_result, tech2, outdir) if ("I" in want and is_result) else None
    tech3 = run_technique_iii(cfg, tech2, outdir) if "III" in want else None

    np.savez(
        outdir / "result.npz",
        p_values=is_result.p_values if is_result else np.array([]),
        is_P_logical=is_result.P_logical if is_result else np.array([]),
        is_P_logical_se=is_result.P_logical_se if is_result else np.array([]),
        ansatz_p=cfg.p_targets if tech1 else np.array([]),
        ansatz_P=np.asarray(tech1["P_ext"]) if (tech1 and tech1.get("P_ext")) else np.array([]),
        distance=(tech2 or {}).get("distance", 0), onset=(tech2 or {}).get("onset", 0),
        onset_fraction=(tech2 or {}).get("onset_fraction") or np.nan,
        split_p=np.asarray(tech3["p_ladder"]) if tech3 else np.array([]),
        split_P=np.asarray(tech3["P_logical"]) if tech3 else np.array([]),
    )
    print(f"\n[{cfg.label}] done -> {outdir/'result.npz'}")
    return {"tech1": tech1, "tech2": tech2, "tech3": tech3, "is": is_result}


# ============================ YAML config loading ===============================
def config_to_dict(cfg: Config) -> dict:
    """Serializable view of a Config (fields only; properties like code/code_label excluded)."""
    return dataclasses.asdict(cfg)


def load_config(path: pathlib.Path) -> Config:
    """Build a Config from a YAML file: keys must match Config dataclass fields."""
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    # YAML-friendly convenience: weights_range: [lo, hi] -> weights = range(lo, hi+1) (explicit).
    wr = data.pop("weights_range", None)
    if wr is not None:
        data["weights"] = list(range(int(wr[0]), int(wr[1]) + 1))
        data.setdefault("weights_explicit", True)
    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise SystemExit(f"{path}: unknown config keys {sorted(unknown)}; valid: {sorted(valid)}")
    # JSON/YAML maps have string keys; restore int keys for shots_by_weight.
    if isinstance(data.get("shots_by_weight"), dict):
        data["shots_by_weight"] = {int(k): int(v) for k, v in data["shots_by_weight"].items()}
    return Config(**data)


def _apply_cpu_cap(n: Optional[int]) -> None:
    """Cap thread pools BEFORE relay/BLAS initialise (RAYON for relay_bp, OMP/BLAS for numpy)."""
    if n is None:
        env_n = os.environ.get("SLURM_CPUS_PER_TASK")
        n = int(env_n) if env_n else None
    if n is None:
        return
    n = max(1, int(n))
    os.environ["RAYON_NUM_THREADS"] = str(n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)
    print(f"[cfg] capping parallelism to {n} threads (RAYON/OMP/BLAS)")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path, help="YAML experiment config (keys = Config fields)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end dry run (<~1 min)")
    ap.add_argument("--cpus", type=int, default=None, help="cap thread pools (default: $SLURM_CPUS_PER_TASK)")
    ap.add_argument("--outdir", type=pathlib.Path, default=None, help="override outdir")
    ap.add_argument("--techniques", type=str, default=None, help="comma list, e.g. 'II,IS,I,III'")
    ap.add_argument("--code", type=str, default=None, help=f"override code ({sorted(CODES)})")
    ap.add_argument("--split-method", type=str, default=None, choices=["multiseed", "replica"])
    args = ap.parse_args(argv)

    _apply_cpu_cap(args.cpus)

    if args.config:
        cfg = load_config(args.config)
        if args.smoke:
            # overlay smoke budgets but keep the config's code/experiment/decoder selectors
            sm = Config.smoke(code_name=cfg.code_name, experiment=cfg.experiment,
                              decoder_name=cfg.decoder_name, techniques=cfg.techniques,
                              split_method=cfg.split_method, mw_single_sector=cfg.mw_single_sector,
                              mw_f0_override=cfg.mw_f0_override, mw_w0_override=cfg.mw_w0_override,
                              # Carry the Technique-II PRODUCTION path (decimation + fault
                              # restriction): without these a bb288 smoke searches the full
                              # rounds=18 circuit — unboundedly heavier than the real config's
                              # restricted search, the opposite of a smoke.
                              mw_decimate=cfg.mw_decimate, mw_decimate_max_odd=cfg.mw_decimate_max_odd,
                              mw_search_rounds=cfg.mw_search_rounds,
                              noise_channel=cfg.noise_channel, ablate_channel=cfg.ablate_channel,
                              outdir=cfg.outdir)
            cfg = sm
    elif args.smoke:
        cfg = Config.smoke()
    else:
        ap.error("need --config <yaml> (or --smoke for the default tiny run)")

    if args.code:
        cfg.code_name = args.code
        cfg.__post_init__()
    if args.techniques:
        cfg.techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
    if args.split_method:
        cfg.split_method = args.split_method
    if args.outdir:
        cfg.outdir = str(args.outdir)

    outdir = pathlib.Path(cfg.outdir) if cfg.outdir else cfg.resolved_outdir()
    if args.smoke:
        outdir = outdir.with_name(outdir.name + "_smoke")
    run_all(cfg, outdir)


if __name__ == "__main__":
    main()
