"""Technique-III REPLICA-EXCHANGE splitting for BB(12)=[[144,12,12]], SINGLE-SECTOR (Z-type).

The bb144 analog of split_fulldem.py. Uses replica_exchange_estimate (parallel tempering with swaps
between adjacent ladder rates) — the method that actually mixes for codes with many inequivalent
logical operators (the gross code), unlike the driver's swap-less splitting_estimate. Single-sector
matches the IS sweep's representation (same N / q_base) and is far faster than the full both-sector
DEM. Ladder 6e-3 -> 1e-4 over 40 levels (deep, to 1e-4). Prints [tempering] swap-accept progress.

Writes bb144_curve_hi/splitting.json {tempered, diagnostics}; bb6_report consumes it next to the
hi-weight IS sweep + ansatz. A keep-awake guard holds the system on (this run does not checkpoint).
"""
import sys, json, pathlib, time
import numpy as np
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from bb_code_sim import BB_144_12_12, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from splitting import replica_exchange_estimate

P_REF, P_HIGH, P_LOW, NLEV = 0.003, 0.006, 1e-4, 40   # deep run: anchor 6e-3 -> 1e-4
OUT = _HERE / "bb144_curve_hi" / "splitting.json"


def _dec():
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def _keep_awake(enable):
    try:
        import ctypes
        flags = 0x80000000 | (0x00000001 | 0x00000040 if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def main():
    circuit = build_bb_circuit(BB_144_12_12, ErrorModel(p_phys=P_REF, p_meas=P_REF),
                               rounds=12, idle_noise=True)
    print(f"tempered (replica exchange, SINGLE-SECTOR), BB(12)=[[144,12,12]], "
          f"ladder {P_HIGH:.0e}->{P_LOW:.0e}, {NLEV} levels ...", flush=True)
    _keep_awake(True)
    t0 = time.time()
    try:
        # Settings trimmed from (6 walkers, 6 local, 200 sweeps) -> (4,3,80) so the run finishes in
        # ~6-8h instead of ~44h: each sweep costs n_walkers*(L+1)*local_steps decodes, and the bb144
        # single-sector decode is heavy. The 40-level ladder is KEPT (fine rungs -> accurate ratios to
        # 1e-4); the cut is to the within-level mixing/averaging, so the estimate is noisier but still
        # a valid cross-check. swap-accept in the diagnostics shows whether it mixed.
        temper, diag = replica_exchange_estimate(
            circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
            n_walkers=4, local_steps=3, n_sweeps=80, burn_in=30, anchor_shots=2000,
            distance=11, seed=42, single_sector=True, sector=0,   # single Z-sector (type 0)
            mw_supports=[])   # skip the slow distance=11 BP-OSD min-weight seed search; MC seeds suffice
    finally:
        _keep_awake(False)
    dt = time.time() - t0
    pl = np.asarray(temper.p_ladder); tP = np.asarray(temper.P_logical); tSE = np.asarray(temper.P_logical_se)
    out = {"tempered": {"p_ladder": pl.tolist(), "P_logical": tP.tolist(), "P_logical_se": tSE.tolist()},
           "diagnostics": {"swap_accept": diag["swap_accept"], "mean_weight": diag["mean_weight"]}}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  done ({dt:.0f}s): ladder {pl[0]:.2e}..{pl[-1]:.2e}, P {tP[0]:.2e}..{tP[-1]:.2e}", flush=True)
    print(f"  swap-accept {min(diag['swap_accept']):.2f}..{max(diag['swap_accept']):.2f}; "
          f"mean weight {diag['mean_weight'][0]:.1f} (hi-q) -> {diag['mean_weight'][-1]:.1f} (lo-q)", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
