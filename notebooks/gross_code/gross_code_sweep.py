#!/usr/bin/env python3
"""Crash-resilient [[144,12,12]] gross-code IS sweep (script form of gross_code_sweep.ipynb).

The notebook runs three weight-stratified importance-sampling sweeps (bare memory,
LPU X̄₁, LPU Z̄₁) over the gross code. Each sweep is ~len(WEIGHTS)*SHOTS_PER_WEIGHT
RelayBP decodes and the LPU circuits cost ~3x the memory circuit, so a full pass is
far too slow to babysit in a notebook cell.

This script does the same computation but checkpoints after **every weight**: the
per-weight failure count is appended to a JSON checkpoint on disk before moving on.
If the machine crashes, re-running resumes from the first unsampled weight — at most
one weight (shots_per_weight decodes) of work is lost.

Determinism / resumability: each (circuit, weight) draws from its own RNG seeded by
(SEED, circuit_index, weight), so the result is independent of where a resume picks
up. This differs numerically from the notebook's single RNG stream but is an equally
valid IS estimate. The reweighting that turns a FailureSpectrum into P_logical(p)
mirrors importance_sampling.importance_sample exactly.

Usage (from anywhere in the repo):
    .venv/bin/python notebooks/gross_code/gross_code_sweep.py
    .venv/bin/python notebooks/gross_code/gross_code_sweep.py --outdir /tmp/sweep --plot
    .venv/bin/python notebooks/gross_code/gross_code_sweep.py --circuits memory   # subset
    .venv/bin/python notebooks/gross_code/gross_code_sweep.py --distance          # + Technique II

Outputs (under --outdir, default notebooks/gross_code/sweep_out/):
    <name>.spectrum.json   incremental per-weight checkpoint (resume source of truth)
    <name>.result.npz      reweighted P_logical(p) + spectrum, written once a circuit finishes
    sweep_results.npz      combined arrays for all finished circuits
    ansatz_fits.json       Technique-I ansatz parameters + extrapolated LER
    *.png                  plots (only with --plot)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np


# --- locate repo src/ regardless of CWD (mirrors the notebook's cell 0) ----------
def _add_src_to_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "src" / "bb_code_sim.py").exists():
            sys.path.insert(0, str(cand / "src"))
            return cand
    raise RuntimeError(f"could not locate repo src/ starting from {here}")


REPO_ROOT = _add_src_to_path()

from scipy.special import gammaln  # noqa: E402

from bb_code_sim import BBCodeSimulator, BB_144_12_12, RelayBPDecoder  # noqa: E402
from surface_code_sim import ErrorModel  # noqa: E402
import gross_code_lpu_tdg as tdg  # noqa: E402
from importance_sampling import (  # noqa: E402
    FailureSpectrum,
    ImportanceSamplingResult,
    _parse_dem,
    _expand,
    _sample_failures_at_weight,
    fit_failure_spectrum,
    logical_error_rate_from_ansatz,
)


# ============================ parameters (mirror the notebook) ====================
SEED = 42
P_REF = 0.005
# WEIGHTS must be a CONTIGUOUS block bracketing the failure onset and the dominant
# binomial mass mu(p)=N_expanded*q(p); see the notebook's Section 0 for the rationale.
WEIGHTS = list(range(1, 190))                 # contiguous fault weights 1..189
SHOTS_PER_WEIGHT = 150                         # samples per weight per circuit
P_TARGETS = np.logspace(-3.5, -2.0, 30)        # reweight to ~3.2e-4 .. 1e-2

# LPU parameters
C, D_INIT = 10, 12

# Ansatz (Technique I)
ANSATZ_MODEL = "f3"

# Circuit build order is fixed so per-weight seeding (SEED, circuit_index, w) is stable.
CIRCUIT_ORDER = ["memory", "x1", "z1"]


def build_circuit(name: str):
    """Build one of the three sweep circuits at p_ref. Kept lazy: building the LPU
    circuits is non-trivial, and a memory-only run shouldn't pay for them."""
    em_ref = ErrorModel.symmetric(P_REF)
    if name == "memory":
        return BBCodeSimulator(BB_144_12_12).build_circuit(em_ref, rounds=BB_144_12_12.distance)
    if name == "x1":
        return tdg.build_logical_x1_circuit(em_ref, C=C, d_init=D_INIT)
    if name == "z1":
        return tdg.build_logical_z1_circuit(em_ref, C=C, d_init=D_INIT)
    raise ValueError(f"unknown circuit {name!r}")


# ================================ checkpoint I/O ==================================
def _atomic_write_json(path: pathlib.Path, obj: dict) -> None:
    """Write JSON to a temp file and rename, so a crash mid-write can't corrupt the
    checkpoint (rename is atomic on the same filesystem)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _checkpoint_path(outdir: pathlib.Path, name: str) -> pathlib.Path:
    return outdir / f"{name}.spectrum.json"


def _load_checkpoint(path: pathlib.Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


# ================================ reweighting ====================================
def reweight_spectrum(spectrum: FailureSpectrum, p_values: np.ndarray) -> ImportanceSamplingResult:
    """P_logical(p) from an accumulated failure spectrum.

    Mirrors importance_sampling.importance_sample's reweighting exactly so the script
    and the library agree; factored out here because the library only exposes it
    bundled with sampling, and we sample/checkpoint ourselves.
    """
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
        p_values=p_arr,
        P_logical=P_logical,
        P_logical_se=np.sqrt(var),
        spectrum=spectrum,
    )


def _spectrum_from_checkpoint(ckpt: dict) -> FailureSpectrum:
    """Build a FailureSpectrum from the completed weights in a checkpoint, in plan order."""
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


# ============================ Section 0: onset scan ==============================
ONSET_SEED_STREAM = 1000  # distinct RNG stream id so scan draws never collide with sweeps


def onset_scan(
    outdir: pathlib.Path,
    shots: int,
    seed: int,
    coarse_step: int,
    fine_step: int = 1,
) -> dict:
    """Locate the failure onset on the (cheapest) memory circuit and size WEIGHTS.

    f(w) is zero until w approaches the circuit fault distance, then turns on. The
    notebook scanned this on a wide ``linspace`` grid (~18-wide steps), which only
    bracketed the onset to a broad interval. Here we go in two phases:

      1. coarse pass upward in steps of ``coarse_step`` until the first weight with
         any observed failure (brackets the onset between ``prev`` and ``hit``);
      2. fine pass over (prev, hit] in steps of ``fine_step`` (default 1) to pin the
         exact onset = first weight with F>0.

    WEIGHTS is then suggested as a contiguous block starting at the **last zero**
    weight (onset-1) — the largest weight that still showed zero failures — and
    running up through the dominant binomial mass μ(p)=N·q(p) over the target grid
    (plus a few σ of margin).

    Per-weight checkpointed into ``onset.spectrum.json`` and deterministic (each
    weight seeded by (seed, ONSET_SEED_STREAM, w)), so a crashed scan resumes exactly.
    """
    print("Section 0 — onset scan on the memory circuit", flush=True)
    circuit = build_circuit("memory")
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, None)
    n_expanded = int(col_to_mech.shape[0])
    decoder = RelayBPDecoder()
    decoder.setup(circuit)

    d = BB_144_12_12.distance
    mu = n_expanded * q_base * (P_TARGETS / P_REF)        # dominant weight center per p
    mu_max = float(mu.max())
    # WEIGHTS upper end: cover the top-of-grid dominant mass plus ~4σ of binomial spread.
    w_hi = int(np.ceil(mu_max + 4.0 * np.sqrt(mu_max)))
    print(f"  N_expanded={n_expanded}, q_base={q_base:.5f}")
    print(f"  dominant μ(p)=N·q over target grid: {mu.min():.1f} .. {mu_max:.1f}")
    print(f"  paper heuristic critical weight ~ d·ln(1/q_base) = {d*np.log(1/q_base):.0f}")
    print(f"  WEIGHTS upper end w_hi = ceil(μ_max + 4√μ_max) = {w_hi}", flush=True)

    ckpt_path = outdir / "onset.spectrum.json"
    ck = _load_checkpoint(ckpt_path)
    if ck is not None and (int(ck.get("shots", -1)) != shots or int(ck.get("seed", -1)) != seed):
        print("  (existing onset checkpoint had different shots/seed — rescanning fresh)")
        ck = None
    cache: Dict[int, int] = {int(w): int(F) for w, F in ck["failures_by_weight"].items()} if ck else {}

    def scan(w: int) -> int:
        w = int(w)
        if w in cache:
            return cache[w]
        rng = np.random.default_rng([seed, ONSET_SEED_STREAM, w])
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, shots, decoder, rng)
        cache[w] = F
        _atomic_write_json(ckpt_path, {
            "shots": shots, "seed": seed,
            "failures_by_weight": {str(k): int(v) for k, v in sorted(cache.items())},
        })
        print(f"    w={w:>4}: F/T = {F:>3}/{shots} = {F/shots:.3f}", flush=True)
        return F

    # Phase 1 — coarse bracket: walk up until the first weight with a failure.
    prev_zero = max(d - coarse_step, 2)   # start a bit below the distance; onset is >= ~d/...
    hit: Optional[int] = None
    w = prev_zero
    while w <= w_hi:
        if scan(w) > 0:
            hit = w
            break
        prev_zero = w
        w += coarse_step
    if hit is None:
        print(f"\n  no failures observed up to w_hi={w_hi} at {shots} shots — "
              f"increase --onset-shots or the scan ceiling; using full WEIGHTS instead.")
        result = {"onset": None, "last_zero": None, "w_hi": w_hi, "n_expanded": n_expanded,
                  "q_base": q_base, "mu_max": mu_max,
                  "spectrum": {str(k): v for k, v in sorted(cache.items())},
                  "suggested_weights": [WEIGHTS[0], WEIGHTS[-1]]}
        _atomic_write_json(outdir / "onset.json", result)
        return result

    # Phase 2 — refine (prev_zero, hit] at fine_step to pin the exact onset.
    onset = hit
    for wf in range(prev_zero + fine_step, hit + 1, fine_step):
        if scan(wf) > 0:
            onset = wf
            break

    last_zero = onset - 1          # largest weight that still showed zero failures
    suggested = list(range(last_zero, w_hi + 1))
    print(f"\n  onset (first failure)   w* = {onset}")
    print(f"  last zero-failure weight   = {last_zero}  ← WEIGHTS starts here")
    print(f"  suggested WEIGHTS = {last_zero}..{w_hi}  ({len(suggested)} contiguous weights)")

    result = {"onset": int(onset), "last_zero": int(last_zero), "w_hi": int(w_hi),
              "n_expanded": n_expanded, "q_base": float(q_base), "mu_max": mu_max,
              "coarse_step": coarse_step, "fine_step": fine_step, "shots": shots,
              "spectrum": {str(k): int(v) for k, v in sorted(cache.items())},
              "suggested_weights": [int(last_zero), int(w_hi)]}
    _atomic_write_json(outdir / "onset.json", result)
    return result


# ============================ per-circuit sweep ===================================
def run_circuit_sweep(
    name: str,
    outdir: pathlib.Path,
    weights_plan: List[int],
    shots_per_weight: int,
    seed: int,
) -> ImportanceSamplingResult:
    """Sample the failure spectrum for one circuit, checkpointing after every weight.

    Resumes from an existing checkpoint if its plan/shots/seed match the request.
    """
    circuit_index = CIRCUIT_ORDER.index(name)
    ckpt_path = _checkpoint_path(outdir, name)
    ckpt = _load_checkpoint(ckpt_path)

    if ckpt is not None:
        mismatch = (
            ckpt.get("weights_plan") != weights_plan
            or int(ckpt.get("shots_per_weight", -1)) != shots_per_weight
            or int(ckpt.get("seed", -1)) != seed
        )
        if mismatch:
            raise SystemExit(
                f"[{name}] existing checkpoint {ckpt_path} was made with a different "
                f"plan/shots/seed. Delete it to start fresh, or restore the original "
                f"parameters to resume. (have weights "
                f"{ckpt.get('weights_plan',[None])[0]}..{ckpt.get('weights_plan',[None])[-1]}, "
                f"shots={ckpt.get('shots_per_weight')}, seed={ckpt.get('seed')})"
            )
        failures_by_weight: Dict[int, int] = {int(w): int(F) for w, F in ckpt["failures_by_weight"].items()}
        print(f"[{name}] resuming: {len(failures_by_weight)}/{len(weights_plan)} weights already done")
    else:
        failures_by_weight = {}

    remaining = [w for w in weights_plan if w not in failures_by_weight]
    if not remaining:
        print(f"[{name}] all {len(weights_plan)} weights already sampled — skipping to reweight")
        assert ckpt is not None  # remaining empty ⇒ a checkpoint must have populated failures
        spectrum = _spectrum_from_checkpoint(ckpt)
        return reweight_spectrum(spectrum, P_TARGETS)

    # Heavy setup (DEM parse + expand + decoder) — done once, only if there's work left.
    print(f"[{name}] building circuit + decoder ...", flush=True)
    circuit = build_circuit(name)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, None)
    n_expanded = int(col_to_mech.shape[0])
    decoder = RelayBPDecoder()
    decoder.setup(circuit)
    print(
        f"[{name}] {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
        f"N_expanded={n_expanded}, q_base={q_base:.5f}",
        flush=True,
    )

    def save() -> None:
        _atomic_write_json(
            ckpt_path,
            {
                "name": name,
                "circuit_index": circuit_index,
                "n_expanded": n_expanded,
                "q_base": q_base,
                "p_ref": P_REF,
                "shots_per_weight": shots_per_weight,
                "seed": seed,
                "weights_plan": weights_plan,
                "failures_by_weight": {str(w): int(F) for w, F in sorted(failures_by_weight.items())},
            },
        )

    t_start = time.perf_counter()
    for i, w in enumerate(remaining):
        # Per-(circuit, weight) RNG: makes each weight a self-contained reproducible
        # unit, so resume order never affects the result.
        rng = np.random.default_rng([seed, circuit_index, w])
        F = _sample_failures_at_weight(det_mat, obs_mat, col_to_mech, w, shots_per_weight, decoder, rng)
        failures_by_weight[w] = F
        save()  # checkpoint BEFORE moving on — crash here loses nothing already saved

        done = len(failures_by_weight)
        elapsed = time.perf_counter() - t_start
        rate = (i + 1) / elapsed
        eta = (len(remaining) - i - 1) / rate if rate > 0 else float("nan")
        print(
            f"[{name}] w={w:>4}: F/T={F:>4}/{shots_per_weight} = {F/shots_per_weight:.3f}"
            f"   ({done}/{len(weights_plan)} weights, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
            flush=True,
        )

    print(f"[{name}] sampling complete in {time.perf_counter()-t_start:.0f}s", flush=True)
    final_ckpt = _load_checkpoint(ckpt_path)
    assert final_ckpt is not None  # we just wrote it in the loop above
    spectrum = _spectrum_from_checkpoint(final_ckpt)
    result = reweight_spectrum(spectrum, P_TARGETS)

    # Persist the finished circuit's reweighted curve alongside the checkpoint.
    np.savez(
        outdir / f"{name}.result.npz",
        p_values=result.p_values,
        P_logical=result.P_logical,
        P_logical_se=result.P_logical_se,
        weights=np.array(spectrum.weights),
        trials=np.array(spectrum.trials),
        failures=np.array(spectrum.failures),
        n_expanded=spectrum.n_expanded,
        q_base=spectrum.q_base,
        p_ref=spectrum.p_ref,
    )
    return result


# ============================ Technique I: ansatz ================================
def run_ansatz(results: Dict[str, ImportanceSamplingResult], outdir: pathlib.Path) -> dict:
    """Fit the failure-spectrum ansatz to each circuit and extrapolate LER (cheap)."""
    K = build_circuit("memory").detector_error_model(decompose_errors=False).num_observables
    out: dict = {"model": ANSATZ_MODEL, "K": int(K), "a": 1 - 2.0 ** -K, "fits": {}}
    print(f"\nTechnique I — ansatz '{ANSATZ_MODEL}', K={K}, saturation a={1 - 2.0**-K:.6f}")
    for name, res in results.items():
        try:
            fit = fit_failure_spectrum(res.spectrum, K=K, model=ANSATZ_MODEL)
            P_ext = logical_error_rate_from_ansatz(fit, list(P_TARGETS))
            out["fits"][name] = {
                "params": {k: float(v) for k, v in fit.params.items()},
                "n_points": int(fit.n_points),
                "cost": float(fit.cost),
                "P_ext": P_ext.tolist(),
            }
            print(f"  {name:7s}: " + ", ".join(f"{k}={v:.3g}" for k, v in fit.params.items())
                  + f"   (n_points={fit.n_points})")
        except ValueError as e:
            out["fits"][name] = {"error": str(e)}
            print(f"  {name:7s}: ansatz fit failed — {e}")
    _atomic_write_json(outdir / "ansatz_fits.json", out)
    return out


# ============================ Technique II: distance =============================
def run_distance(outdir: pathlib.Path) -> dict:
    """Min-weight circuit fault distance + optimal onset on the memory circuit (slow)."""
    from min_weight import compute_distance

    print("\nTechnique II — computing circuit fault distance (BP-OSD, may take minutes) ...", flush=True)
    dr = compute_distance(build_circuit("memory"), osd_order=10, max_iter=200)
    out = {"distance": int(dr.distance), "onset": int(dr.onset),
           "per_logical_weight": [int(x) for x in dr.per_logical_weight]}
    print(f"  fault distance D = {dr.distance} (upper bound for large codes), onset w0* = {dr.onset}")
    _atomic_write_json(outdir / "distance.json", out)
    return out


# ================================== plotting =====================================
def make_plots(results: Dict[str, ImportanceSamplingResult], ansatz: Optional[dict], outdir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {"memory": "Memory (d=12 rounds)", "x1": f"LPU X̄₁ (C={C}, d_init={D_INIT})",
             "z1": f"LPU Z̄₁ (C={C}, d_init={D_INIT})"}
    color = {"memory": "steelblue", "x1": "tomato", "z1": "seagreen"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, res in results.items():
        lo = np.maximum(res.P_logical - res.P_logical_se, 1e-15)
        hi = res.P_logical + res.P_logical_se
        ax.fill_between(res.p_values, lo, hi, color=color[name], alpha=0.25)
        ax.plot(res.p_values, res.P_logical, "-", color=color[name], label=label[name], lw=2)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="50% (random)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Physical error rate $p$"); ax.set_ylabel("Logical error rate (importance-sampled)")
    ax.set_title("[[144,12,12]] Gross Code — LER vs $p$ via IS (RelayBP)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "ler_vs_p.png", dpi=150)
    print(f"  wrote {outdir/'ler_vs_p.png'}")

    if ansatz is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name, res in results.items():
            ax.plot(res.p_values, res.P_logical, "o", color=color[name], alpha=0.5, ms=4,
                    label=f"{label[name]} (reweighted)")
            fit = ansatz["fits"].get(name, {})
            if "P_ext" in fit:
                ax.plot(P_TARGETS, fit["P_ext"], "-", color=color[name], lw=2,
                        label=f"{name} (ansatz {ansatz['model']})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Physical error rate $p$"); ax.set_ylabel("Logical error rate")
        ax.set_title(f"[[144,12,12]] — ansatz-extrapolated LER ({ansatz['model']})")
        ax.legend(fontsize=8, ncol=2); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "ler_ansatz.png", dpi=150)
        print(f"  wrote {outdir/'ler_ansatz.png'}")


# ==================================== main =======================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent / "sweep_out",
                    help="directory for checkpoints + results (default: notebooks/gross_code/sweep_out)")
    ap.add_argument("--circuits", nargs="+", choices=CIRCUIT_ORDER, default=CIRCUIT_ORDER,
                    help="subset of circuits to sweep (default: all three)")
    ap.add_argument("--z1-max-weight", type=int, default=None,
                    help="cap z1's fault-weight ceiling (drops its expensive high-weight tail). "
                         "z1's WEIGHTS plan is truncated to <= this value; the other circuits "
                         "keep the full plan. The z1 curve is then only trustworthy up to the p "
                         "where the dominant binomial mass mu(p)=N*q(p) stays under the cap.")
    ap.add_argument("--shots", type=int, default=SHOTS_PER_WEIGHT, help="shots per weight")
    ap.add_argument("--seed", type=int, default=SEED, help="base RNG seed")
    ap.add_argument("--no-ansatz", action="store_true", help="skip Technique-I ansatz fit")
    ap.add_argument("--distance", action="store_true", help="also run Technique-II distance (slow)")
    ap.add_argument("--onset-scan", action="store_true",
                    help="run Section-0 onset scan first and derive WEIGHTS from it")
    ap.add_argument("--onset-shots", type=int, default=50, help="shots per weight in the onset scan")
    ap.add_argument("--onset-coarse-step", type=int, default=6,
                    help="coarse bracketing step for the onset scan (refined at step 1)")
    ap.add_argument("--plot", action="store_true", help="save PNG plots")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"repo root: {REPO_ROOT}")
    print(f"outdir:    {args.outdir}")
    print(f"circuits:  {args.circuits}")

    # WEIGHTS: module default, or derived from the onset scan when --onset-scan is set.
    weights = list(WEIGHTS)
    if args.onset_scan:
        onset = onset_scan(args.outdir, args.onset_shots, args.seed, args.onset_coarse_step)
        if onset["last_zero"] is not None:
            weights = list(range(onset["last_zero"], onset["w_hi"] + 1))
            print(f"  → using onset-derived WEIGHTS {weights[0]}..{weights[-1]} for the sweeps\n")
        else:
            print(f"  → onset scan inconclusive; falling back to default WEIGHTS\n")

    print(f"weights:   {weights[0]}..{weights[-1]} ({len(weights)}), shots/weight={args.shots}, seed={args.seed}")
    print(f"p targets: {len(P_TARGETS)} pts, {P_TARGETS[0]:.2e}..{P_TARGETS[-1]:.2e}\n")

    results: Dict[str, ImportanceSamplingResult] = {}
    for name in CIRCUIT_ORDER:        # iterate in canonical order for stable seeding
        if name in args.circuits:
            wplan = weights
            if name == "z1" and args.z1_max_weight is not None:
                wplan = [w for w in weights if w <= args.z1_max_weight]
                print(f"[z1] capping weight ceiling at {args.z1_max_weight}: "
                      f"plan {wplan[0]}..{wplan[-1]} ({len(wplan)} weights, tail dropped)")
            results[name] = run_circuit_sweep(name, args.outdir, wplan, args.shots, args.seed)

    # Combined arrays for downstream plotting/analysis.
    combined = {}
    for name, res in results.items():
        combined[f"{name}_P_logical"] = res.P_logical
        combined[f"{name}_P_logical_se"] = res.P_logical_se
    combined["p_values"] = P_TARGETS
    np.savez(args.outdir / "sweep_results.npz", **combined)
    print(f"\nwrote {args.outdir/'sweep_results.npz'}")

    ansatz = None
    if not args.no_ansatz:
        ansatz = run_ansatz(results, args.outdir)
    if args.distance:
        run_distance(args.outdir)
    if args.plot:
        print("\nplotting ...")
        make_plots(results, ansatz, args.outdir)

    print("\ndone.")


if __name__ == "__main__":
    main()
