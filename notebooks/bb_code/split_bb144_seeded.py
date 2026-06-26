"""Replica-exchange splitting for bb144 single-sector, WITH min-weight onset seeds.

The seed-less run underestimated the low-p LER badly (chains stuck at mean weight ~27 while the onset
is w=6). The fix: seed the low-q chains with min-weight logical configs so they can reach the onset.
The seed search is slow ONLY because min_weight_logical_seeds runs find_min_weight_logicals
single-threaded; here we run it PARALLEL (24 workers, ~minutes) up front and pass the result via
mw_supports. Trimmed sweep settings (4 walkers, 3 local, 80 sweeps) keep total runtime ~6-7h.
"""
import sys, json, pathlib, time
import numpy as np
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from bb_code_sim import BB_144_12_12, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from min_weight import single_sector_dem, find_min_weight_logicals
from splitting import replica_exchange_estimate

P_REF, P_HIGH, P_LOW, NLEV = 0.003, 0.006, 1e-4, 40
OUT = _HERE / "bb144_curve_hi" / "splitting.json"   # overwrites the seed-less result


def _dec():
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def _keep_awake(enable):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 | 0x40 if enable else 0))
    except Exception:
        pass


def main():
    circuit = build_bb_circuit(BB_144_12_12, ErrorModel(p_phys=P_REF, p_meas=P_REF), rounds=12, idle_noise=True)
    H, A, mult, probs, det_coords = single_sector_dem(circuit, detector_type=0)

    # --- PARALLEL min-weight seed search (the part that was the 12h bottleneck single-threaded) ---
    print("[seeds] parallel min-weight logical search (workers=24) ...", flush=True)
    t = time.time()
    supports = find_min_weight_logicals(
        circuit, 11, max_trials=200, osd_order=10, max_iter=200, priors=probs,
        seed=42, progress_every=500, workers=24, systematic=True, sector=0)
    print(f"[seeds] found {len(supports)} min-weight logicals in {time.time()-t:.0f}s", flush=True)

    print(f"tempered (replica exchange, SINGLE-SECTOR, SEEDED), ladder {P_HIGH:.0e}->{P_LOW:.0e}, "
          f"{NLEV} levels ...", flush=True)
    _keep_awake(True)
    t0 = time.time()
    try:
        temper, diag = replica_exchange_estimate(
            circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
            n_walkers=4, local_steps=3, n_sweeps=80, burn_in=30, anchor_shots=2000,
            distance=11, seed=42, single_sector=True, sector=0,
            mw_supports=list(supports))   # <-- the onset seeds
    finally:
        _keep_awake(False)
    dt = time.time() - t0
    pl = np.asarray(temper.p_ladder); tP = np.asarray(temper.P_logical); tSE = np.asarray(temper.P_logical_se)
    out = {"tempered": {"p_ladder": pl.tolist(), "P_logical": tP.tolist(), "P_logical_se": tSE.tolist()},
           "diagnostics": {"swap_accept": diag["swap_accept"], "mean_weight": diag["mean_weight"]},
           "seeded": True, "n_seeds": len(supports)}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  done ({dt:.0f}s): ladder {pl[0]:.2e}..{pl[-1]:.2e}, P {tP[0]:.2e}..{tP[-1]:.2e}", flush=True)
    print(f"  swap-accept {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f}; "
          f"mean weight {diag['mean_weight'][0]:.1f} (hi-q) -> {diag['mean_weight'][-1]:.1f} (lo-q)", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
