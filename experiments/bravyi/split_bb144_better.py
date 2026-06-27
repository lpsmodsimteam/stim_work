"""Better-params replica-exchange splitting for bb144 single-sector — the "bridge fix".

Targets the new adaptive-IS curve (runs/bravyi/bb12/bb144_adaptive_1e6). Fixes the original run's
0.14-swap bottleneck (the empty w~16-34 weight band that stranded single-flip mixing) with:
  * intermediate-weight seeds harvested at crossover rates (seed_p_grid),
  * weight-matched rung placement (replica_exchange_estimate's new seeding),
  * a ladder focused on the crossover (6e-3 -> 1.2e-3, ~3x denser there than the old 6e-3->1e-4),
  * more walkers (10 vs 4) and sweeps.

The deep tail below p_low is a clean q^w0 slope you extrapolate; we spend the compute on the
crossover where the absolute height was biased.

    --pilot : quick validation (4 walkers, 20 sweeps, NO min-weight seed search) ~1h. Confirms the
              former bottleneck rung now swaps healthily and measures per-sweep time BEFORE the
              full ~13h run.
"""
import sys, json, time, argparse
import numpy as np
from bb_code_sim import BB_144_12_12, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from min_weight import single_sector_dem, find_min_weight_logicals
from splitting import replica_exchange_estimate
from repo_paths import RUNS

P_REF = 0.003
P_HIGH, P_LOW, NLEV = 0.006, 0.0004, 44          # span the ~7e-4 cliff into the onset region, denser
GAP_WEIGHTS = list(range(12, 78, 6))             # inflate min-weight seeds to these weights (bridge w~7-90)
OUTDIR = RUNS / "bravyi" / "bb12" / "bb144_split_better"


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
    ap.add_argument("--pilot", action="store_true",
                    help="quick validation: 4 walkers, 20 sweeps, no min-weight search (~1h)")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    circuit = build_bb_circuit(BB_144_12_12, ErrorModel(p_phys=P_REF, p_meas=P_REF), rounds=12, idle_noise=True)

    # Min-weight seeds are the inflation base; the partial search finds none for bb144, so use the
    # systematic search but CACHE it (paid once, reused by later runs / the other mode).
    cache = OUTDIR / "mw_supports.json"
    if cache.exists():
        supports = [frozenset(s) for s in json.loads(cache.read_text())]
        print(f"[seeds] reusing {len(supports)} cached min-weight logicals", flush=True)
    else:
        H, A, mult, probs, _ = single_sector_dem(circuit, detector_type=0)
        print("[seeds] systematic min-weight logical search (workers=24, ~45min) ...", flush=True)
        t = time.time()
        supports = list(find_min_weight_logicals(
            circuit, 11, max_trials=200, osd_order=10, max_iter=200, priors=probs,
            seed=42, progress_every=500, workers=24, systematic=True, sector=0))
        print(f"[seeds] found {len(supports)} min-weight logicals in {time.time()-t:.0f}s", flush=True)
        cache.write_text(json.dumps([sorted(int(c) for c in s) for s in supports]))
    if args.pilot:
        nwalk, nsweep, burn = 4, 20, 5
        out, tag = OUTDIR / "splitting_pilot.json", "PILOT"
    else:
        nwalk, nsweep, burn = 10, 110, 40
        out, tag = OUTDIR / "splitting.json", "FULL"

    print(f"[{tag}] bridge-fix replica-exchange: ladder {P_HIGH:.0e}->{P_LOW:.0e} x{NLEV}, "
          f"walkers={nwalk}, sweeps={nsweep}, gap_weights={GAP_WEIGHTS}, n_seeds={len(supports)} ...",
          flush=True)
    _keep_awake(True); t0 = time.time()
    try:
        temper, diag = replica_exchange_estimate(
            circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
            n_walkers=nwalk, local_steps=4, n_sweeps=nsweep, burn_in=burn, anchor_shots=4000,
            distance=11, seed=42, single_sector=True, sector=0,
            mw_supports=supports, gap_weights=GAP_WEIGHTS)
    finally:
        _keep_awake(False)
    dt = time.time() - t0
    pl = np.asarray(temper.p_ladder); tP = np.asarray(temper.P_logical); tSE = np.asarray(temper.P_logical_se)
    out.write_text(json.dumps({
        "tempered": {"p_ladder": pl.tolist(), "P_logical": tP.tolist(), "P_logical_se": tSE.tolist()},
        "diagnostics": {"swap_accept": diag["swap_accept"], "mean_weight": diag["mean_weight"]},
        "seeded": True, "n_seeds": len(supports), "gap_weights": GAP_WEIGHTS,
        "params": {"n_walkers": nwalk, "n_sweeps": nsweep, "n_levels": NLEV,
                   "p_high": P_HIGH, "p_low": P_LOW}}, indent=2))
    print(f"  done ({dt:.0f}s): ladder {pl[0]:.2e}..{pl[-1]:.2e}, P {tP[0]:.2e}..{tP[-1]:.2e}", flush=True)
    print(f"  swap-accept {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f}  "
          f"(former bottleneck was 0.14); mean weight {diag['mean_weight'][0]:.1f} -> {diag['mean_weight'][-1]:.1f}",
          flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
