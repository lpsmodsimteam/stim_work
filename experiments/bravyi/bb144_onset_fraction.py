#!/usr/bin/env python3
"""Exact Technique-II onset fraction f*(w0) for a bb144 single-sector channel, offline.

The inline II.3 step in ``bb6_fig10_sweep.py`` (Proposition 1, :func:`min_weight_fail_count`)
enumerates *every* weight-D/2 restriction of *every* min-weight logical — ~C(D, D/2)·|L(D)|
subsets. For the cz channel (D=22, |L(D)|=17856) that is ~1.26e10 restrictions grouped by a
936-bit syndrome, which OOMs a 64 GB box (this is what crashed the run). The driver now skips it
above a feasibility cap and leaves f0 unpinned; this script computes what IS tractable *offline*,
reusing the L(D) the driver persists (``logicals_LD.npz``) so no second 12-min search is needed.

What it does, memory-bounded (safe to run alongside the IS sweep):
  1. Rebuild the *identical* cz single-sector DEM (same Config + build_circuit as the driver).
  2. Load L(D) from ``<outdir>/logicals_LD.npz``.
  3. Stream the weight-w0 restrictions into a packed-int set, ABORTING if |R| exceeds --cap
     (so it never blows memory). If it finishes under the cap, |R| is exact and we run the exact
     Proposition-1 fail count → f*(w0). If it aborts, we report |L(D)|, |R| > cap, and the rigorous
     upper bound f*(w0) ≤ (C(D,w0)·|L(D)|) / C(N_exp, w0) — the paper's bounds regime.

``--self-test`` validates the packed exact fail count against the library's
:func:`min_weight_fail_count` on a small synthetic (H, A) and exits.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from itertools import combinations
from math import comb

import numpy as np


def _add_src(here: pathlib.Path) -> pathlib.Path:
    for cand in [here.parent, *here.parents]:
        if (cand / "src" / "bb_code_sim.py").exists():
            sys.path.insert(0, str(cand / "src"))
            sys.path.insert(0, str(cand / "experiments" / "bravyi"))
            return cand
    raise RuntimeError(f"could not locate repo src/ from {here}")


REPO = _add_src(pathlib.Path(__file__).resolve())
from scipy.special import gammaln  # noqa: E402


def _pack(cols) -> int:
    """Pack a sorted restriction (columns < 2^14) into one big int key. Columns must be < 16384."""
    k = 0
    for c in cols:
        k = (k << 14) | int(c)
    return k


def _fail_count_packed(H, A, mult, unique_restrictions):
    """Exact Proposition-1 |F| from an explicit set of unique restrictions (packed as (sorted-tuple)).

    Mirrors min_weight.min_weight_fail_count but takes the already-deduped restrictions so the caller
    controls memory. ``unique_restrictions`` is an iterable of sorted int tuples (all the same length w0).
    """
    H = H.astype(np.uint8); A = A.astype(np.uint8)
    mult = np.asarray(mult, dtype=np.int64)
    by_sigma: dict[bytes, dict[bytes, int]] = {}
    for r in unique_restrictions:
        idx = np.asarray(r, dtype=np.int64)
        rho = int(np.prod(mult[idx]))
        sig = np.packbits((H[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        act = np.packbits((A[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        d = by_sigma.setdefault(sig, {})
        d[act] = d.get(act, 0) + rho
    fails = 0
    for d in by_sigma.values():
        s = list(d.values())
        fails += sum(s) - max(s)
    return int(fails)


def _self_test() -> None:
    """Validate _fail_count_packed against the library min_weight_fail_count on a small case."""
    from min_weight import min_weight_fail_count
    rng = np.random.default_rng(0)
    # Small synthetic: N=14 columns, weight-4 logicals (w0=2). Build a few even-weight nullspace-ish
    # supports by hand; correctness only needs the two routines to agree on the SAME logical set.
    M, K, N = 6, 2, 14
    H = (rng.random((M, N)) < 0.4).astype(np.uint8)
    A = (rng.random((K, N)) < 0.4).astype(np.uint8)
    mult = rng.integers(1, 4, size=N).astype(np.int64)
    D = 4
    logicals = [frozenset(map(int, rng.choice(N, size=D, replace=False))) for _ in range(20)]
    logicals = [s for s in logicals if len(s) == D]
    ref, _ = min_weight_fail_count(H, A, logicals, mult)
    uniq = {tuple(sorted(r)) for s in logicals for r in combinations(sorted(s), D // 2)}
    got = _fail_count_packed(H, A, mult, uniq)
    assert got == ref, f"self-test FAILED: packed={got} vs library={ref}"
    print(f"[self-test] OK: packed fail count == library ({got}) on {len(logicals)} synthetic logicals")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="validate the exact counter and exit")
    ap.add_argument("--outdir", type=pathlib.Path, help="run dir containing logicals_LD.npz")
    ap.add_argument("--noise-model", default="cz", choices=["full", "cz", "meas", "prep", "idle"])
    ap.add_argument("--sector", type=int, default=0)
    ap.add_argument("--cap", type=int, default=80_000_000,
                    help="abort the |R| enumeration above this many unique restrictions (memory guard)")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.outdir is None:
        ap.error("--outdir is required unless --self-test")

    import bb6_fig10_sweep as drv
    from bb_code_sim import BB_144_12_12
    from min_weight import single_sector_dem, expanded_logical_count

    cfg = drv.Config.production()
    cfg.code, cfg.code_label = BB_144_12_12, "BB(12)=[[144,12,12]]"
    cfg.rounds = 12
    cfg.noise_model = None if args.noise_model == "full" else args.noise_model
    cfg.mw_single_sector = True
    cfg.mw_sector_type = args.sector

    print(f"[onset] rebuilding {cfg.code_label} single-sector (type {args.sector}) DEM, "
          f"noise={args.noise_model} ...", flush=True)
    circuit = drv.build_circuit(cfg)
    H, A, mult, priors, _ = single_sector_dem(circuit, detector_type=args.sector)
    N = H.shape[1]
    n_exp = int(mult.sum())
    assert N < (1 << 14), f"N={N} exceeds 14-bit packing"

    npz = np.load(args.outdir / "logicals_LD.npz")
    supports = npz["supports"]
    D = int(npz["distance"])
    if supports.size == 0:
        print("[onset] L(D) is empty — nothing to do."); return
    logicals = [frozenset(int(c) for c in row) for row in supports]
    ld = len(logicals)
    w0 = D // 2 if D % 2 == 0 else (D + 1) // 2
    ld_exp = expanded_logical_count(logicals, mult)
    per_logical = comb(D, w0)
    naive_total = per_logical * ld
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
        restr = (tuple((k >> (14 * j)) & 0x3FFF for j in range(w0 - 1, -1, -1)) for k in seen)
        fails = _fail_count_packed(H, A, mult, restr)
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
