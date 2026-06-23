"""Technique-III replica-exchange splitting on the FULL both-sector DEM (near-threshold cross-check).

The full-DEM analog of split_crosscheck.py. Runs the **tempered** replica-exchange estimate only —
the over/under sequential bracket needs single-sector min-weight seeds, and the tempered estimate is
the accurate one anyway. Reduced settings: full-DEM decodes are ~10-50x slower than the single Z-sector,
so this is a near-threshold (6e-3 -> 3e-3) run of ~tens of minutes rather than the deep single-sector run.

Writes bb6_fulldem_curve/splitting.json {tempered, diagnostics}; bb6_report consumes it (the bracket
band is optional and simply omitted here). replica_exchange_estimate batches decodes via the relay
decoder's own threads (no process pool), so no __main__ guard is required, but we keep one anyway.
"""
import sys, json, pathlib, time
import numpy as np
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from bb_code_sim import BB_72_12_6, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from splitting import replica_exchange_estimate

P_REF, P_HIGH, P_LOW, NLEV = 0.003, 0.006, 0.003, 12


def _dec():
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def main():
    circuit = build_bb_circuit(BB_72_12_6, ErrorModel(p_phys=P_REF, p_meas=P_REF), rounds=6, idle_noise=True)
    print("tempered (replica exchange, full both-sector DEM) ...", flush=True)
    t0 = time.time()
    temper, diag = replica_exchange_estimate(
        circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
        n_walkers=6, local_steps=6, n_sweeps=200, burn_in=60, anchor_shots=2000,
        distance=6, seed=42, single_sector=False)        # full DEM (sector=None internally)
    dt = time.time() - t0
    pl = np.asarray(temper.p_ladder); tP = np.asarray(temper.P_logical); tSE = np.asarray(temper.P_logical_se)
    out = {"tempered": {"p_ladder": pl.tolist(), "P_logical": tP.tolist(), "P_logical_se": tSE.tolist()},
           "diagnostics": {"swap_accept": diag["swap_accept"], "mean_weight": diag["mean_weight"]}}
    (_HERE / "bb6_fulldem_curve" / "splitting.json").write_text(json.dumps(out, indent=2))
    print(f"  done ({dt:.0f}s): ladder {pl[0]:.2e}..{pl[-1]:.2e}, P {tP[0]:.2e}..{tP[-1]:.2e}", flush=True)
    print(f"  swap-accept {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f}; "
          f"mean weight {diag['mean_weight'][0]:.1f} (hi-q) -> {diag['mean_weight'][-1]:.1f} (lo-q)", flush=True)
    print("wrote bb6_fulldem_curve/splitting.json")


if __name__ == "__main__":
    main()
