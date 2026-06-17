#!/usr/bin/env python3
"""Step-3 validation: run improved min-weight logical search with systematic enumeration.

Runs Technique II on the Bravyi BB(6) circuit with systematic=True (all 2^K-1 = 4095
syndrome classes enumerated before random trials). Reports the final |L(D)| count to
compare against the old random-only result (~188 from the previous session).

Usage:
    python notebooks/bb_code/run_mw_improved.py [--workers N] [--max-trials N] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

import numpy as np
from scipy.special import gammaln

from bb_code_sim import build_bb_circuit, BB_72_12_6
from surface_code_sim import ErrorModel
from min_weight import (
    dem_check_action_matrices, compute_distance,
    find_min_weight_logicals, min_weight_fail_count,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-trials", type=int, default=2000)
    ap.add_argument("--outdir", type=pathlib.Path,
                    default=_HERE / "bb6_fig10_out_mw")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--p-ref", type=float, default=0.003)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    em = ErrorModel(p_phys=args.p_ref, p_meas=args.p_ref)
    circuit = build_bb_circuit(BB_72_12_6, em, args.rounds, idle_noise=True)
    H, A, mult, priors = dem_check_action_matrices(circuit)
    N_exp = int(mult.sum())
    K = circuit.num_observables
    n_sys = (1 << K) - 1
    print(f"BB(6) circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
          f"N_expanded={N_exp}, K={K}", flush=True)
    print(f"Workers={args.workers}, systematic={n_sys} syndrome classes, "
          f"random max_trials={args.max_trials}", flush=True)

    t0 = time.perf_counter()
    print("\n[II.1] Computing distance ...", flush=True)
    D = compute_distance(circuit, osd_order=10, max_iter=200, priors=priors,
                         progress=True, workers=args.workers).distance
    print(f"[II.1] D={D}, onset w0={D // 2}   ({time.perf_counter()-t0:.0f}s)", flush=True)

    t1 = time.perf_counter()
    print(f"\n[II.2] L(D) search: {n_sys} systematic + {args.max_trials} random, "
          f"workers={args.workers} ...", flush=True)
    logicals = find_min_weight_logicals(
        circuit, D,
        max_trials=args.max_trials,
        osd_order=10, max_iter=200,
        priors=priors, seed=42,
        progress_every=max((n_sys + args.max_trials) // 40, 1),
        workers=args.workers,
        systematic=True,
    )
    print(f"[II.2] |L(D)|={len(logicals)}   ({time.perf_counter()-t1:.0f}s)", flush=True)

    t2 = time.perf_counter()
    print("\n[II.3] Computing exact onset fraction f*(D/2) ...", flush=True)
    fails, n_exp2 = min_weight_fail_count(H, A, logicals, mult)
    half = D // 2
    log_choose = gammaln(n_exp2 + 1) - gammaln(half + 1) - gammaln(n_exp2 - half + 1)
    f_star = fails / np.exp(log_choose)
    print(f"[II.3] |F(D/2)|={fails:.4g}  (paper Table-2: 3.83e8)  "
          f"f*(D/2)={f_star:.3e}   ({time.perf_counter()-t2:.0f}s)", flush=True)

    result = {
        "distance": D, "onset": half, "n_min_logicals": len(logicals),
        "fail_count": fails, "n_expanded": n_exp2, "onset_fraction": f_star,
        "rounds": args.rounds, "workers": args.workers, "max_trials": args.max_trials,
        "systematic": True, "elapsed_s": time.perf_counter() - t0,
    }
    out = args.outdir / "distance.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}", flush=True)
    print(f"\nTotal elapsed: {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"\nSummary: D={D}, |L(D)|={len(logicals)} "
          f"(old random-only: ~188), |F(D/2)|={fails:.3e} "
          f"(paper: 3.83e8), ratio={fails/3.83e8:.3f}", flush=True)
