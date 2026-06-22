"""Technique-III splitting cross-check of the Figure-10 curve, on the single-sector representation.

Produces three estimates over the same near-threshold ladder (p=0.006 -> 0.003):
  * tempered  — replica_exchange_estimate (balanced proposal + parallel tempering): the ACCURATE
    estimate (weight-distribution mixes correctly), agrees with Technique-I across the range.
  * over / under — the two one-sided SEQUENTIAL splitting runs (min-weight seeds overshoot,
    high-weight MC seeds undershoot). Their [min,max] is a validation BRACKET the tempered
    estimate should lie inside.

Writes bb6_fig10_curve/splitting.json (tempered + bracket + diagnostics + per-rung vs Technique-I).
Figures are made by bb6_report.py.
"""
import sys, json, pathlib, time
import numpy as np
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from bb_code_sim import BB_72_12_6, build_bb_circuit, RelayBPDecoder
from surface_code_sim import ErrorModel
from min_weight import single_sector_dem, find_min_weight_logicals, build_circuit_translation_perms
from splitting import replica_exchange_estimate, splitting_estimate

P_REF, P_HIGH, P_LOW, NLEV = 0.003, 0.006, 0.003, 16

def _dec():  # decoder for the cross-check chains: lighter than the curve's 600 legs, which is fine
    # near threshold (validated: a num_sets=100 cross-check agrees with the 600-leg curve to ~25%).
    return RelayBPDecoder(gamma0=0.125, pre_iter=80, num_sets=100, set_max_iter=60, stop_nconv=6)


def main():
    circuit = build_bb_circuit(BB_72_12_6, ErrorModel(p_phys=P_REF, p_meas=P_REF), rounds=6, idle_noise=True)
    H, A, mult, probs, dc = single_sector_dem(circuit, detector_type=0)
    perms = build_circuit_translation_perms(None, H, det_coords=dc, verbose=False)
    print("min-weight seeds (systematic + symmetry) ...", flush=True)
    t0 = time.time()
    mw_supports = find_min_weight_logicals(None, 6, matrices=(H, A, probs), systematic=True,
                                           max_trials=0, symmetry_perms=perms, workers=20)
    print(f"  {len(mw_supports)} seeds ({time.time()-t0:.0f}s)", flush=True)

    print("tempered (replica exchange) ...", flush=True); t0 = time.time()
    temper, diag = replica_exchange_estimate(
        circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
        n_walkers=8, local_steps=8, n_sweeps=300, burn_in=100, anchor_shots=3000,
        distance=6, seed=42, single_sector=True, sector=0)
    print(f"  tempered done ({time.time()-t0:.0f}s)", flush=True)

    print("bracket: over (min-weight seeds) ...", flush=True); t0 = time.time()
    over = splitting_estimate(
        circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
        n_seeds=16, chain_steps=4000, burn_in=1000, anchor_shots=3000, distance=6, seed=42,
        single_sector=True, sector=0, use_min_weight_seeds=True, mw_supports=mw_supports)
    print(f"  over done ({time.time()-t0:.0f}s)", flush=True)
    print("bracket: under (high-weight MC seeds) ...", flush=True); t0 = time.time()
    under = splitting_estimate(
        circuit, _dec(), p_ref=P_REF, p_high=P_HIGH, p_low=P_LOW, n_levels=NLEV,
        n_seeds=16, chain_steps=4000, burn_in=1000, anchor_shots=3000, distance=6, seed=42,
        single_sector=True, sector=0, use_min_weight_seeds=False)
    print(f"  under done ({time.time()-t0:.0f}s)", flush=True)

    # Compare to the Technique-I ansatz curve.
    npz = np.load(_HERE / "bb6_fig10_curve" / "bb6_fig10.npz")
    ap, aP = npz["ansatz_p"], npz["ansatz_P"]
    pl = np.asarray(temper.p_ladder); tP = np.asarray(temper.P_logical); tSE = np.asarray(temper.P_logical_se)
    oP = np.asarray(over.P_logical); uP = np.asarray(under.P_logical)
    lo = np.minimum(oP, uP); hi = np.maximum(oP, uP)
    rows = []
    print("\n  p          tempered (SE)        ansatz      ratio   bracket[lo,hi]")
    for k, pv in enumerate(pl):
        aL = float(aP[np.argmin(abs(ap - pv))]); r = tP[k] / aL if aL > 0 else float("nan")
        inside = lo[k] <= tP[k] <= hi[k]
        regime = "valid" if (0.5 <= r <= 2.0 and inside) else "check"
        rows.append({"p": float(pv), "tempered_P": float(tP[k]), "tempered_SE": float(tSE[k]),
                     "ansatz_P": aL, "over_P": float(oP[k]), "under_P": float(uP[k]),
                     "bracket_lo": float(lo[k]), "bracket_hi": float(hi[k]),
                     "ratio": float(r), "inside_bracket": bool(inside), "regime": regime})
        print(f"  {pv:.3e}  {tP[k]:.3e} (±{tSE[k]:.0e})  {aL:.3e}  {r:.2f}   "
              f"[{lo[k]:.2e},{hi[k]:.2e}]{'' if inside else '  OUT'}")
    out = {"tempered": {"p_ladder": pl.tolist(), "P_logical": tP.tolist(), "P_logical_se": tSE.tolist()},
           "bracket": {"p_ladder": pl.tolist(), "lo": lo.tolist(), "hi": hi.tolist()},
           "diagnostics": {"swap_accept": diag["swap_accept"], "mean_weight": diag["mean_weight"]},
           "compare": rows}
    (_HERE / "bb6_fig10_curve" / "splitting.json").write_text(json.dumps(out, indent=2))
    print("\nwrote bb6_fig10_curve/splitting.json")


if __name__ == "__main__":
    main()
