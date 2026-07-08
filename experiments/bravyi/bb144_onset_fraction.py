#!/usr/bin/env python3
"""Exact Technique-II onset fraction f*(w0) for a bb144 single-sector channel, offline.

The inline II.3 step in ``bb6_fig10_sweep.py`` (Proposition 1, :func:`min_weight_fail_count`)
enumerates *every* weight-D/2 restriction of *every* min-weight logical — ~C(D, D/2)·|L(D)|
subsets. For the cz channel (D=22, |L(D)|=17856) that is ~1.26e10 restrictions grouped by a
936-bit syndrome, which OOMs a 64 GB box (this is what crashed the run). The driver's budget
guard now leaves f0 unpinned there; this script computes what IS tractable *offline*, reusing
the L(D) the driver persists (``logicals_LD.npz``) so no second 12-min search is needed.

What it does, memory-bounded (safe to run alongside the IS sweep):
  1. Rebuild the *identical* single-sector DEM (same Config + build_circuit as the driver).
  2. Load L(D) from ``<outdir>/logicals_LD.npz`` (validated against the flags and run config).
  3. Stream the weight-w0 restrictions into a packed-int set, ABORTING if |R| exceeds --cap
     (so it never blows memory). If it finishes under the cap, |R| is exact and we run the exact
     Proposition-1 fail count (``min_weight.fail_count_from_restrictions``) → f*(w0). If it
     aborts, we report |L(D)|, |R| > cap, and the rigorous upper bound
     f*(w0) ≤ (C(D,w0)·|L(D)|) / C(N_exp, w0) — the paper's bounds regime.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.special import gammaln

# Sibling driver (same directory, on sys.path when run as a script); importing it also puts
# the repo's src/ on sys.path, so the library imports below work without a local bootstrap.
import bb6_fig10_sweep as drv
from bb_code_sim import BB_144_12_12
from min_weight import fail_count_from_restrictions, single_sector_dem, expanded_logical_count

PACK_BITS = 14  # columns are packed 14 bits each -> requires N < 16384


def _pack(cols) -> int:
    """Pack a sorted restriction into one big-int key (PACK_BITS bits per column index)."""
    k = 0
    for c in cols:
        k = (k << PACK_BITS) | int(c)
    return k


def _unpack(key: int, w: int) -> tuple:
    """Inverse of :func:`_pack`: recover the w column indices from a packed key."""
    mask = (1 << PACK_BITS) - 1
    return tuple((key >> (PACK_BITS * j)) & mask for j in range(w - 1, -1, -1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=pathlib.Path, required=True,
                    help="run dir containing logicals_LD.npz")
    ap.add_argument("--noise-model", default="cz", choices=["full", "cz", "meas", "prep", "idle"])
    ap.add_argument("--sector", type=int, default=0)
    ap.add_argument("--cap", type=int, default=80_000_000,
                    help="abort the |R| enumeration above this many unique restrictions (memory guard)")
    args = ap.parse_args()

    cfg = drv.Config.production()
    cfg.code, cfg.code_label = BB_144_12_12, "BB(12)=[[144,12,12]]"
    cfg.rounds = 12
    cfg.noise_model = None if args.noise_model == "full" else args.noise_model
    cfg.mw_single_sector = True
    cfg.mw_sector_type = args.sector

    # Guard against rebuilding a DEM that doesn't match the run that produced logicals_LD.npz:
    # the npz stores the sector, and the run's config.json stores its noise model.
    run_cfg = args.outdir / "config.json"
    if run_cfg.exists():
        run_model = json.loads(run_cfg.read_text()).get("noise_model") or "full"
        if run_model != args.noise_model:
            raise SystemExit(f"{args.outdir} was produced with noise_model={run_model!r}, "
                             f"not --noise-model {args.noise_model!r}")

    print(f"[onset] rebuilding {cfg.code_label} single-sector (type {args.sector}) DEM, "
          f"noise={args.noise_model} ...", flush=True)
    circuit = drv.build_circuit(cfg)
    H, A, mult, _priors, _ = single_sector_dem(circuit, detector_type=args.sector)
    N = H.shape[1]
    n_exp = int(mult.sum())
    assert N < (1 << PACK_BITS), f"N={N} exceeds {PACK_BITS}-bit packing"

    npz = np.load(args.outdir / "logicals_LD.npz")
    if bool(npz["single_sector"]) and int(npz["sector"]) != args.sector:
        raise SystemExit(f"logicals_LD.npz was built for sector {int(npz['sector'])}, "
                         f"not --sector {args.sector}")
    supports = npz["supports"]
    D = int(npz["distance"])
    if supports.size == 0:
        print("[onset] L(D) is empty — nothing to do."); return
    logicals = [frozenset(int(c) for c in row) for row in supports]
    ld = len(logicals)
    w0 = (D + 1) // 2            # ceil(D/2): correct for even AND odd D
    ld_exp = expanded_logical_count(logicals, mult)
    naive_total = comb(D, w0) * ld
    log_choose = gammaln(n_exp + 1) - gammaln(w0 + 1) - gammaln(n_exp - w0 + 1)
    print(f"[onset] D={D} ({'even' if D % 2 == 0 else 'odd'}), w0={w0}, |L(D)|={ld} "
          f"(exp {ld_exp:.4g}), N_exp={n_exp}", flush=True)
    print(f"[onset] naive restriction budget C({D},{w0})*|L(D)| = {naive_total:.4g}; cap={args.cap:.3g}",
          flush=True)

    if D % 2 != 0:
        print("[onset] odd D — this tool covers the even-D Proposition-1 path; use the odd-D "
              "(Appendix A.6) machinery for this channel.")
        return

    # Stream unique weight-w0 restrictions into a packed-int set, aborting above the cap.
    seen: set[int] = set()
    t0 = time.perf_counter()
    aborted = False
    for i, s in enumerate(logicals):
        cols = sorted(s)
        for r in combinations(cols, w0):
            seen.add(_pack(r))
        if len(seen) > args.cap:
            aborted = True
            print(f"[onset] |R| exceeded cap after {i + 1}/{ld} logicals "
                  f"(|R|>{len(seen):.4g}); stopping.", flush=True)
            break
        if (i + 1) % 500 == 0:
            print(f"[onset]   {i + 1}/{ld} logicals, |R| so far={len(seen):.4g} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    out = {"noise_model": args.noise_model, "sector": args.sector, "distance": D, "onset": w0,
           "n_min_logicals": ld, "n_min_logicals_expanded": float(ld_exp), "n_expanded": n_exp,
           "naive_restrictions": float(naive_total)}

    if aborted:
        f0_upper = float(naive_total / np.exp(log_choose))
        out.update({"exact": False, "unique_restrictions_lower_bound": len(seen),
                    "onset_fraction_upper_bound": f0_upper, "onset_fraction": None})
        print(f"[onset] EXACT f0 infeasible at this scale (|R|>{args.cap:.3g}). "
              f"Rigorous upper bound f*({w0}) <= {f0_upper:.4e}. "
              f"Report |L(D)|>={ld} as a bound (paper convention).", flush=True)
    else:
        uniq_R = len(seen)
        print(f"[onset] |R| (exact unique restrictions) = {uniq_R} "
              f"({time.perf_counter() - t0:.0f}s) — running exact Proposition-1 fail count ...",
              flush=True)
        fails = fail_count_from_restrictions(H, A, mult, (_unpack(k, w0) for k in seen))
        f0 = float(fails / np.exp(log_choose))
        out.update({"exact": True, "unique_restrictions": uniq_R, "fail_count": fails,
                    "onset_fraction": f0})
        print(f"[onset] EXACT: |F({w0})|={fails}, f*({w0})={f0:.6e}", flush=True)

    dst = args.outdir / "onset_fraction.json"
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[onset] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
