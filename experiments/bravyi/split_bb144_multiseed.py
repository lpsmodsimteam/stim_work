"""Paper-faithful multi-seeded splitting for bb144 single-sector (arXiv:2511.15177 Alg.2/3 + §5.3).

Unlike split_bb144_better.py (the replica-exchange "bridge fix", which is the paper's *future-work*
idea and under-estimates), this runs the paper's ACTUAL method:
  * §5.3 multi-seeded warm-start: L Monte-Carlo-sampled TYPICAL failing seeds at p_0; each lower-p
    chain initialised from the adjacent higher-p chain's final failing config.  No min-weight seeds
    (the paper's BB footnote says min-weight-logical seeding is exactly what made BB runs fail).
  * Algorithm 2: BAR ratio estimator + adaptive sigma+Delta precision controller (eps/sqrt(t)).
  * Algorithm 3: single-random-bit Metropolis with decoder failing-set test.

Goal: see whether the faithful method closes the gap to the IS-ansatz curve
(runs/bravyi/bb12/bb144_adaptive_1e6) that the replica-exchange variant could not.

COMPUTE NOTE: the paper used T_init=1e6 for BB(12) and reported it needs "a large amount of compute
time". This pure-Python inner loop makes a full converged bb144 run a multi-hour-to-day job. Start
with --pilot, watch the per-level mean-weight descend and sigma+Delta meet the budget, then scale
T_init / L up for production.

    --pilot : coarse geom ladder, small L/M/T_init — a fast behavioural sanity check.
"""
import sys, json, time, argparse
import numpy as np
from bb_code_sim import BB_144_12_12, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from splitting import multi_seeded_split_estimate
from repo_paths import RUNS

P_REF = 0.003
OUTDIR = RUNS / "bravyi" / "bb12" / "bb144_split_multiseed"


def _dec():
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def _keep_awake(enable):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x40 if enable else 0))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="coarse/quick behavioural sanity check")
    ap.add_argument("--ladder", choices=["geom", "eq18"], default="geom",
                    help="geom (n-levels) or paper Eq.18 spacing (dense; ~100+ levels for bb144)")
    ap.add_argument("--levels", type=int, default=None, help="n_levels for geom ladder")
    ap.add_argument("--tinit", type=int, default=None, help="initial chain length per level")
    ap.add_argument("--L", type=int, default=None, help="number of MC typical seeds")
    ap.add_argument("--M", type=int, default=None, help="chains per seed")
    ap.add_argument("--phigh", type=float, default=0.006)
    ap.add_argument("--plow", type=float, default=None)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        levels, tinit, L, M = 12, 20_000, 4, 2
        plow, out, tag = 0.0015, OUTDIR / "splitting_pilot.json", "PILOT"
    else:
        levels, tinit, L, M = 44, 200_000, 12, 3
        plow, out, tag = 0.0004, OUTDIR / "splitting.json", "FULL"
    levels = args.levels or levels
    tinit = args.tinit or tinit
    L = args.L or L; M = args.M or M
    plow = args.plow or plow

    circuit = build_bb_circuit(BB_144_12_12, ErrorModel(p_phys=P_REF, p_meas=P_REF),
                               rounds=12, idle_noise=True)

    print(f"[{tag}] faithful multi-seeded splitting: ladder={args.ladder} "
          f"{args.phigh:.0e}->{plow:.0e} (levels={levels}), L={L}, M={M}, T_init={tinit} ...", flush=True)
    _keep_awake(True); t0 = time.time()
    try:
        res, diag = multi_seeded_split_estimate(
            circuit, _dec(), p_ref=P_REF, p_high=args.phigh, p_low=plow,
            L=L, M=M, T_init=tinit, eps=0.25, ladder=args.ladder, n_levels=levels,
            distance=11, anchor_shots=4000, single_sector=True, sector=0, seed=42)
    finally:
        _keep_awake(False)
    dt = time.time() - t0
    pl = np.asarray(res.p_ladder); P = np.asarray(res.P_logical); SE = np.asarray(res.P_logical_se)
    out.write_text(json.dumps({
        "method": "multi_seeded_split_estimate (arXiv:2511.15177 Alg.2/3 + 5.3)",
        "p_ladder": pl.tolist(), "P_logical": P.tolist(), "P_logical_se": SE.tolist(),
        "diagnostics": {"mean_weight": diag["mean_weight"], "T_per_level": diag["T_per_level"],
                        "sigma_plus_delta": diag["sigma_plus_delta"], "P_high": diag["P_high"]},
        "params": {"L": L, "M": M, "T_init": tinit, "ladder": args.ladder, "n_levels": levels,
                   "p_high": args.phigh, "p_low": plow, "eps": 0.25}}, indent=2))
    print(f"  done ({dt:.0f}s): ladder {pl[0]:.2e}..{pl[-1]:.2e}, P {P[0]:.2e}..{P[-1]:.2e} (+/-{SE[-1]:.1e})",
          flush=True)
    print(f"  mean weight {diag['mean_weight'][0]:.1f} -> {diag['mean_weight'][-1]:.1f}; "
          f"T/level {min(diag['T_per_level'])}..{max(diag['T_per_level'])}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
