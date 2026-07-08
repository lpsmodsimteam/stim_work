"""
Technique II — computing/bounding min-weight properties (arXiv:2511.15177 §4).

For a QEC system with detector check matrix H (M×N) and logical action matrix
A (K×N) derived from a Stim circuit's detector error model:

  * ``compute_distance``  — the circuit fault distance D = min{|l| : Hl=0, Al≠0}
    and the optimal onset weight w0* = ceil(D/2) (§4.1).
  * ``find_min_weight_logicals`` — the set L(D) of weight-D logical bitstrings (§4.2).
  * ``min_weight_fail_count`` / ``optimal_onset_fraction`` — the exact onset failure
    fraction f*(D/2) = |F(D/2)| / C(N, D/2) for even D, via Proposition 1 (§4.3).

The decoder used for distance/logical search is ldpc's BP-OSD, which exposes the
fault-space correction (``osdw_decoding``). Because BP-OSD is not a guaranteed
min-weight decoder, results are exact for small codes and upper bounds (on D) /
lower bounds (on |L(D)|) otherwise — matching the paper's Table 2 conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import stim
from ldpc.bposd_decoder import BpOsdDecoder

from importance_sampling import _expand, _parse_dem


def build_circuit_translation_perms(
    circuit: stim.Circuit,
    H: np.ndarray,
    l: int = 6,
    m: int = 6,
    *,
    verify: bool = True,
    verbose: bool = True,
    det_coords: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> List[np.ndarray]:
    """Build all l*m DEM mechanism permutations for toric translations T_(a,b).

    Returns a list of l*m arrays, shape (N,): perms[a*m+b][j] is the index of
    the mechanism that fault j maps to under T_(a,b).

    Matches mechanisms by H-column detector support (no observable matching).
    This works because BB codes have no H-column collisions — each fault fires a
    unique detector set. Observable validity of translated logicals is guaranteed
    by the toric code automorphism property.

    ``det_coords`` maps each row of ``H`` to its detector coordinate ``(type, s, c)``.
    Pass the renumbered mapping from :func:`single_sector_dem` when ``H`` is a
    single-sector matrix (its rows are a renumbered detector subset, not the circuit's
    full detector indexing). Defaults to ``circuit.get_detector_coordinates()`` (the
    full both-sector DEM), where row index == detector id.

    Raises ValueError on H-column collisions, KeyError if the circuit lacks
    exact Z_l × Z_m toric symmetry.
    """
    n_det = H.shape[0]
    N = H.shape[1]

    if det_coords is None:
        det_coords = circuit.get_detector_coordinates()
    coord_to_det: Dict[Tuple[int, int, int], int] = {}
    for det_id, coords in det_coords.items():
        coord_to_det[(int(coords[0]), int(coords[1]), int(coords[2]))] = int(det_id)

    if verbose:
        print("  [toric sym] building H-only signature index ...", flush=True)
    H_cols: List[FrozenSet[int]] = [frozenset(np.flatnonzero(H[:, j]).tolist()) for j in range(N)]
    h_sig: Dict[FrozenSet[int], int] = {}
    for j in range(N):
        if H_cols[j] in h_sig:
            raise ValueError(
                f"H-column collision at mechanisms {h_sig[H_cols[j]]} and {j}. "
                "Cannot build toric permutation."
            )
        h_sig[H_cols[j]] = j
    if verbose:
        print(f"  [toric sym] {N} unique H-signatures", flush=True)

    mech_perms: List[np.ndarray] = []
    for a in range(l):
        for b in range(m):
            det_perm = np.empty(n_det, dtype=np.int32)
            for det_id, coords in det_coords.items():
                tc, s, c = int(coords[0]), int(coords[1]), int(coords[2])
                i_s, j_s = divmod(s, m)
                s_new = ((i_s + a) % l) * m + ((j_s + b) % m)
                tkey = (tc, s_new, c)
                if tkey not in coord_to_det:
                    raise KeyError(
                        f"Translated detector {(tc, s, c)} -> {tkey} not found. "
                        f"Circuit lacks Z_{l}xZ_{m} toric symmetry."
                    )
                det_perm[det_id] = coord_to_det[tkey]

            mech_perm = np.empty(N, dtype=np.int32)
            for j in range(N):
                trans_h = frozenset(int(det_perm[d]) for d in H_cols[j])
                if trans_h not in h_sig:
                    raise KeyError(
                        f"Translated H-support of mechanism {j} under T_({a},{b}) not found."
                    )
                mech_perm[j] = h_sig[trans_h]
            mech_perms.append(mech_perm)

            if verify and (a, b) == (1, 0):
                if not np.array_equal(H[np.ix_(det_perm, mech_perm)], H):
                    raise AssertionError("T_(1,0) H-symmetry check failed.")
                if verbose:
                    print("  [toric sym] T_(1,0) verified OK", flush=True)

    if verbose:
        print(f"  [toric sym] {l*m} translation perms built", flush=True)
    return mech_perms


def single_sector_dem(
    circuit: stim.Circuit,
    detector_type: int = 0,
    q_base: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int, int]]]:
    """Single-CSS-sector merged DEM — the paper's 'Z-type decoding' representation.

    arXiv:2511.15177 decodes a single CSS sector: only detectors of one type (the
    Z-checks, ``detector_type=0``) are used. Fault mechanisms that differ only in their
    *other*-sector detector support are merged into one column, combining probabilities
    by the exact independent-channel XOR rule ``p = (1 - Π(1-2 p_i)) / 2``. For BB(6)
    this reproduces the paper's compressed Ñ=2233 and expanded N≈46224, whereas the full
    both-sector DEM inflates both (16164 / 68940).

    Returns ``(H, A, mult, probs, det_coords)``:
      H          : (n_sector × N_merged) uint8 sector-detector parity-check
      A          : (K × N_merged) uint8 observable-action matrix
      mult       : per-column expansion multipliers m_j = max(round(p/q_base), 1)
      probs      : merged per-column probabilities (BP error channel)
      det_coords : {renumbered_row -> (type, s, c)} for the kept sector detectors;
                   pass to :func:`build_circuit_translation_perms` as ``det_coords=``.
    """
    from collections import defaultdict

    probs_full, det_mat, obs_mat = _parse_dem(circuit)
    coords_all = circuit.get_detector_coordinates()
    n_det_full = det_mat.shape[1]
    sector_mask = np.array(
        [int(coords_all[d][0]) == detector_type for d in range(n_det_full)], dtype=bool
    )
    sector_dets = np.flatnonzero(sector_mask)
    row_of = {int(d): i for i, d in enumerate(sector_dets)}
    n_sector = len(sector_dets)
    K = obs_mat.shape[1]

    # Merge by (sector-detector support, observable support); combine probabilities by
    # the XOR rule via the running product Π(1-2 p_i) → p_merged = (1 - Π)/2.
    prod_term: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = defaultdict(lambda: 1.0)
    for j in range(det_mat.shape[0]):
        dsig = tuple(int(d) for d in np.flatnonzero(det_mat[j] & sector_mask))
        osig = tuple(int(o) for o in np.flatnonzero(obs_mat[j]))
        if not dsig and not osig:
            continue  # invisible to this sector (no sector detector, no observable)
        prod_term[(dsig, osig)] *= (1.0 - 2.0 * probs_full[j])

    cols = list(prod_term.keys())
    N_merged = len(cols)
    H = np.zeros((n_sector, N_merged), dtype=np.uint8)
    A = np.zeros((K, N_merged), dtype=np.uint8)
    probs = np.empty(N_merged, dtype=float)
    for col, key in enumerate(cols):
        dsig, osig = key
        probs[col] = (1.0 - prod_term[key]) / 2.0
        for d in dsig:
            H[row_of[d], col] = 1
        for o in osig:
            A[o, col] = 1

    if q_base is None:
        q_base = float(probs_full.min())
    mult = np.maximum(np.round(probs / q_base).astype(np.int64), 1)

    det_coords = {
        row_of[int(d)]: (
            int(coords_all[int(d)][0]), int(coords_all[int(d)][1]), int(coords_all[int(d)][2])
        )
        for d in sector_dets
    }
    return H, A, mult, probs, det_coords


def expanded_logical_count(logical_supports, multipliers) -> int:
    """Expanded |L(D)| = Σ_S Π_{j∈S} mult_j over compressed weight-D logical supports.

    Each compressed logical (a weight-D subset S of merged columns) corresponds to
    Π_{j∈S} mult_j logicals in the expanded representation — one interchangeable copy
    per merged column. Matches the paper's expanded |L(D)| convention (BB(6): 6.01×10¹²).
    Python big-ints avoid overflow.
    """
    mult = np.asarray(multipliers)
    total = 0
    for s in logical_supports:
        prod = 1
        for j in s:
            prod *= int(mult[j])
        total += prod
    return int(total)


# ---------------------------------------------------------------------------
# Anchored weight-6 MITM — per-anchor / per-anchor-pair-chunk canonical enumeration (module-level
# so it is picklable under the Windows 'spawn' start method for the optional process pool).
#
# The detector bitmasks (col_h), observable bitmasks (col_a) and per-column min-detector (mindet)
# are large Python-int lists shared once via the pool initializer; each worker enumerates the
# canonical weight-6 logicals for a slice of one anchor's seed pairs and returns only the small
# result set. The (anchor, pair-slice) tasks are independent, so the merged union is complete.
# ---------------------------------------------------------------------------
_MITM: Dict[str, object] = {}


def _mitm_init(col_h, col_a, mindet, n_det, weight, dmax) -> None:
    _MITM.clear()
    _MITM.update(col_h=col_h, col_a=col_a, mindet=mindet, n_det=n_det, weight=int(weight),
                 dmax=int(dmax), _cache={})


def _mitm_prepare(d0: int):
    """Per-anchor scratch (cols0, det_cols, hmap), cached so repeated anchor-pair chunks of the
    same ``d0`` build it only once in a given worker process."""
    cache = _MITM["_cache"]
    if d0 in cache:
        return cache[d0]
    col_h = _MITM["col_h"]; mindet = _MITM["mindet"]; n_det = _MITM["n_det"]
    cols0 = [c for c in range(len(mindet)) if mindet[c] == d0]
    det_cols: List[List[int]] = [[] for _ in range(n_det)]
    hmap: Dict[int, int] = {}
    for c in range(len(mindet)):
        if mindet[c] < d0:
            continue
        h = col_h[c]
        hmap[h] = c                     # column supports are distinct -> unique key
        x = h
        while x:
            det_cols[(x & -x).bit_length() - 1].append(c)
            x &= x - 1
    deg = [len(dc) for dc in det_cols]  # per-detector candidate count (for most-constrained pivot)
    cache[d0] = (cols0, det_cols, hmap, deg)
    return cache[d0]


def _mitm_chunk(task) -> Set[FrozenSet[int]]:
    """Enumerate weight-``w`` logicals for a slice ``[lo, hi)`` of anchor d0's ordered anchor pairs.

    A weight-``w`` logical S (⊕col_h = 0, ⊕col_a ≠ 0) with global-min detector d0 has ≥2 columns
    touching d0 — and any column touching d0 has min-detector == d0 (none touch a lower one), so
    those columns lie in ``cols0`` (min-detector == d0). We seed every anchor *pair* (a, b) ⊂ cols0
    (cancelling d0) and complete the remaining ``w-2`` columns drawn from the columns with
    min-detector ≥ d0 (so additional cols0 columns are permitted — this folds the |S0|=2,4,…=w
    cases into the same branch). The completion is a ``(w-3)``-deep *pivot chain* plus a final
    hash-closed column: at each step the residual XOR ``T`` the still-unchosen columns must produce
    is nonzero, so its lowest set bit ``lowbit(T)`` is touched by an odd (hence ≥1) number of them —
    the next column is enumerated only from ``det_cols[lowbit(T)]``, never the full O(N²) pair list;
    after ``w-1`` columns are placed the residual equals exactly one column's support, fixed by a
    ``col_h`` hash-map lookup. (w=6 reduces to the original pair + c1,c2,c3 + hash close.)

    Splitting an anchor's pair list into ``[lo, hi)`` slices lets the big anchors (large ``cols0``)
    be load-balanced across all workers instead of one anchor bounding the wall-time.

    Completeness: the chain misses S only if some intermediate residual hits 0 with columns still
    to place — i.e. a proper prefix of S is itself a null cycle. Its prefix (⊕col_a ≠ 0) or its
    complement (⊕col_a = 0 ⇒ complement carries S's nonzero action) is then a logical of weight
    < w = D, contradicting the distance. So for a true weight-D logical no intermediate residual is
    0 and S is always found; ``frozenset`` dedup absorbs the orderings of the completion columns.
    """
    d0, lo, hi = task
    CH = _MITM["col_h"]; CA = _MITM["col_a"]; w = _MITM["weight"]; dmax = _MITM["dmax"]
    cols0, det_cols, hmap, deg = _mitm_prepare(d0)
    hget = hmap.get
    res: Set[FrozenSet[int]] = set()
    if len(cols0) < 2:
        return res
    add = res.add
    n_pivot = w - 3                      # det_cols-loop columns before the single hash-closed column

    def _pivot_det(T):
        """Most-constrained-variable pivot: among the set detectors of ``T`` return the one touched
        by the fewest candidate columns (smallest ``det_cols``). Any set bit must be cleared by a
        remaining column, so branching on the scarcest one is valid (completeness) and minimizes the
        fan-out — collapsing effective branching from ~mean-degree to a handful. Ties → lowest."""
        x = T
        bd = (x & -x).bit_length() - 1
        bl = deg[bd]
        x &= x - 1
        while x and bl > 1:
            d = (x & -x).bit_length() - 1
            if deg[d] < bl:
                bl = deg[d]; bd = d
            x &= x - 1
        return bd

    def _extend(T, chosen, axor, plies):
        """Recurse over pivot columns (``plies`` left) then hash-close the final column.

        ``chosen`` is the tuple of completion columns picked so far (anchor pair carried in the
        closure as a/b); ``T`` is the running detector residual; ``axor`` the running observable XOR
        of the anchor pair + ``chosen``. Picks the next column from the most-constrained set detector.

        Branch-and-bound: the still-missing ``plies+1`` columns (``plies`` pivots + 1 hash-close)
        must XOR to ``T``; a GF(2) XOR of k columns has popcount ≤ k·dmax, so ``T`` with
        ``popcount(T) > (plies+1)·dmax`` is unreachable and the subtree is pruned.
        """
        if T.bit_count() > (plies + 1) * dmax:           # B&B: residual too heavy to clear
            return
        cand = det_cols[_pivot_det(T)]
        if plies == 1:                                   # last pivot column, then hash close
            for c in cand:
                if c == a or c == b or c in chosen:
                    continue
                cw = hget(T ^ CH[c])
                if cw is None or cw == a or cw == b or cw == c or cw in chosen:
                    continue
                if (axor ^ CA[c] ^ CA[cw]) != 0:
                    add(frozenset((a, b, c, cw, *chosen)))
            return
        lim = plies * dmax                               # bound for the child (plies-1 pivots + close)
        for c in cand:                                   # interior pivot column
            if c == a or c == b or c in chosen:
                continue
            newT = T ^ CH[c]
            if newT == 0 or newT.bit_count() > lim:      # closed early, or too heavy for the child
                continue
            _extend(newT, chosen + (c,), axor ^ CA[c], plies - 1)

    # Map a flat pair index range [lo,hi) onto (i, j) upper-triangular pairs of cols0 without
    # materializing the whole list.
    import itertools as _it
    pair_iter = _it.islice(_it.combinations(cols0, 2), lo, hi)
    for a, b in pair_iter:
        Tfull = CH[a] ^ CH[b]
        if Tfull == 0:
            continue
        if n_pivot == 1:                                 # w == 4: pair + 1 pivot + hash close
            ax0 = CA[a] ^ CA[b]
            for c in det_cols[(Tfull & -Tfull).bit_length() - 1]:
                if c == a or c == b:
                    continue
                cw = hget(Tfull ^ CH[c])
                if cw is None or cw == a or cw == b or cw == c:
                    continue
                if (ax0 ^ CA[c] ^ CA[cw]) != 0:
                    add(frozenset((a, b, c, cw)))
        else:
            _extend(Tfull, (), CA[a] ^ CA[b], n_pivot)
    return res


def _mitm_iter_progress(iterable, *, total, desc, verbose):
    """Yield from ``iterable`` while showing progress.

    A long weight-12 enumeration would otherwise run silently for minutes/hours. When attached to
    an interactive terminal we use a ``tqdm`` bar; under a redirected log (the overnight case) tqdm
    carriage-returns are noise, so we fall back to a flushed line every ~30s. ``verbose=False``
    passes through untouched.
    """
    if not verbose:
        yield from iterable
        return
    import sys as _sys
    is_tty = bool(getattr(_sys.stdout, "isatty", lambda: False)())
    if is_tty:
        try:
            from tqdm import tqdm
            for x in tqdm(iterable, total=total, desc=desc, unit="task"):
                yield x
            return
        except Exception:
            pass
    import time as _t
    t0 = _t.time(); last = t0
    for i, x in enumerate(iterable, 1):
        yield x
        now = _t.time()
        if now - last >= 30.0 or i == total:
            print(f"    {desc}: {i}/{total} tasks ({now - t0:.0f}s)", flush=True)
            last = now


def exact_min_weight_logicals_mitm(
    H: np.ndarray,
    A: np.ndarray,
    det_coords: Dict[int, Tuple[int, int, int]],
    *,
    l: int = 6,
    m: int = 6,
    weight: int = 6,
    seed: int = 12345,
    verbose: bool = True,
) -> Set[FrozenSet[int]]:
    """Exact complete weight-``weight`` logical enumeration via anchored pivot-chain MITM.

    Generalizes ``bb6_exact_enum_mitm.py`` to ANY DEM representation (single-sector or full
    both-sector): finds every weight-``weight`` column subset S of H with ``H·1_S = 0`` and
    ``A·1_S ≠ 0`` (the minimum-weight logical operators), then expands by the ``l*m`` toric
    translation permutations. Returns a set of frozensets of column indices (compressed).

    ``det_coords`` maps each ROW of H to ``(type, s, c)`` (from :func:`single_sector_dem`, or
    built from ``circuit.get_detector_coordinates()`` for the full DEM). Detectors are remapped
    internally into ``(type, c, s)``-sorted order so each ``l*m`` toric orbit is a contiguous
    block and the canonical anchors are ``block*l*m`` — independent of the input numbering.

    Algorithm (any even ``weight`` = D): each min-weight logical is canonicalized by its
    global-minimum detector, which an ``lm``-shift toric translation maps onto one of the
    ``n_det/lm`` canonical anchors ``block*lm`` — so only those anchors are searched, then the
    results are expanded by the ``lm`` translations. At an anchor d0 the logical is an anchor *pair*
    (two columns whose min-detector is d0) plus ``weight-2`` completion columns found by a
    detector-pivot chain (see :func:`_mitm_chunk`): a ``(weight-3)``-deep chain then one hash-closed
    column. The pivot chain enumerates each completion column only from the columns touching the
    current residual's lowest detector — tens of candidates — instead of an O(N²) pair table, which
    made the full DEM (N≈16k) intractable. Cost grows as ``anchor_pairs · avg_deg^(weight-4)``, so
    the independent anchors are run across a process pool when the work is large (weight-12 always).

    ``seed`` is retained for signature compatibility but is now unused: the pivot chain closes each
    candidate with an *exact* ``col_h`` lookup, so the probabilistic GF(2) hash of the original
    (and its collision guard) is no longer needed.
    """
    import os, time
    if weight < 4:
        raise ValueError(f"weight must be an integer >= 4 (got {weight})")
    # NB: weight may be odd. A codeword needs ≥2 columns sharing its global-min detector (GF(2)
    # parity of *detectors*), which holds for any total weight — circuit DEMs with measurement
    # noise have odd-weight space-time logicals (e.g. the 144 single-sector circuit distance is 11).
    n_det, N = H.shape
    lm = l * m
    if n_det % lm != 0:
        raise ValueError(f"n_det={n_det} not divisible by l*m={lm}; toric orbit blocks ill-defined")

    # Remap detector rows into (type, c, s) order: each (type,c) orbit becomes a contiguous block
    # of lm rows with s as the in-block phase, so the canonical anchors are block*lm regardless of
    # the input detector numbering (single_sector_dem and the full DEM number detectors differently).
    order = sorted(range(n_det), key=lambda d: (int(det_coords[d][0]), int(det_coords[d][2]),
                                                int(det_coords[d][1])))
    Hr = H[order, :]
    coords_r = {i: det_coords[order[i]] for i in range(n_det)}
    for b in range(n_det // lm):
        blk = [coords_r[b * lm + k] for k in range(lm)]
        if len({(t, c) for (t, s, c) in blk}) != 1 or len({s for (t, s, c) in blk}) != lm:
            raise ValueError("toric orbit blocks malformed after remap (need lm phases per (type,c))")

    # Per-column detector / observable bitmasks (Python big-ints) — vectorized per column from the
    # sparse CSC structure so N≈16k columns build in well under a second.
    from scipy.sparse import csc_matrix
    Hs = csc_matrix(Hr); As = csc_matrix(A.astype(np.uint8))
    col_h = [0] * N
    col_a = [0] * N
    for j in range(N):
        h = 0
        for d in Hs.indices[Hs.indptr[j]:Hs.indptr[j + 1]]:
            h |= (1 << int(d))
        col_h[j] = h
        a = 0
        for r in As.indices[As.indptr[j]:As.indptr[j + 1]]:
            a |= (1 << int(r))
        col_a[j] = a

    mindet = [((col_h[j] & -col_h[j]).bit_length() - 1) if col_h[j] else -1 for j in range(N)]
    dmax = max((h.bit_count() for h in col_h if h), default=1)   # heaviest column, for the B&B bound

    results: Set[FrozenSet[int]] = set()
    t0 = time.time()
    canon = [b * lm for b in range(n_det // lm)]

    # Per-anchor column counts (cols whose min-detector == d0), used to size anchor-pair work.
    from math import comb as _comb
    cnt0 = [0] * n_det
    for c in range(N):
        if mindet[c] >= 0:
            cnt0[mindet[c]] += 1

    # Build (d0, lo, hi) tasks: each is a slice of anchor d0's C(|cols0|,2) ordered pairs. The big
    # anchors are split into several chunks so all workers stay busy (one huge anchor would otherwise
    # bound the wall-time). Chunk size targets a few× the worker count of total tasks.
    anchor_pairs = sum(_comb(cnt0[d0], 2) for d0 in canon)
    ncpu = os.cpu_count() or 1
    target_tasks = max(4 * ncpu, 1)
    chunk = max(1, anchor_pairs // target_tasks)
    tasks: List[Tuple[int, int, int]] = []
    for d0 in canon:
        npair = _comb(cnt0[d0], 2)
        for lo in range(0, npair, chunk):
            tasks.append((d0, lo, min(lo + chunk, npair)))

    # Run across a process pool when the search is large enough to amortize the (Windows 'spawn')
    # process + pickle overhead. Cost grows with anchor_pairs and, super-linearly, with the
    # pivot-chain branching ~ the mean columns-per-detector (N / n_det). The serial path is identical
    # (used for small problems such as the single-sector DEM, where spawning would dominate) and for
    # nested/daemon contexts where a child pool is disallowed. NOTE: when a *caller's* script triggers
    # the pool it must guard its entry with ``if __name__ == "__main__":`` (the multiprocessing
    # 'spawn' rule); otherwise each spawned worker re-executes the script.
    import multiprocessing as _mp
    avg_deg = N / max(n_det, 1)
    cost = anchor_pairs * avg_deg ** max(weight - 4, 1)   # ~leaf evals; weight-6 → avg_deg^2 as before
    use_pool = cost > 5_000_000 and ncpu > 1 and not _mp.current_process().daemon
    desc = f"[mitm] w{weight} anchors"
    if verbose:
        print(f"  [mitm] weight-{weight} enumeration: {len(tasks)} tasks, anchor_pairs={anchor_pairs}, "
              f"avg_deg={avg_deg:.1f}, {'pool' if use_pool else 'serial'} ...", flush=True)
    if use_pool:
        try:
            nproc = min(len(tasks), ncpu)
            with _mp.Pool(nproc, initializer=_mitm_init,
                          initargs=(col_h, col_a, mindet, n_det, weight, dmax)) as pool:
                for r in _mitm_iter_progress(pool.imap_unordered(_mitm_chunk, tasks),
                                             total=len(tasks), desc=desc, verbose=verbose):
                    results |= r
        except (OSError, ValueError, RuntimeError, ImportError):
            use_pool = False
    if not use_pool:
        _mitm_init(col_h, col_a, mindet, n_det, weight, dmax)
        for t in _mitm_iter_progress(tasks, total=len(tasks), desc=desc, verbose=verbose):
            results |= _mitm_chunk(t)

    if verbose:
        print(f"  [mitm] {len(results)} canonical weight-{weight} logicals ({time.time() - t0:.0f}s)", flush=True)

    perms = build_circuit_translation_perms(None, Hr, l=l, m=m, det_coords=coords_r, verbose=False)
    full: Set[FrozenSet[int]] = set()
    for L in results:
        for p in perms:
            full.add(frozenset(int(p[c]) for c in L))
    if verbose:
        print(f"  [mitm] |L(D)| = {len(full)} after {len(perms)}-shift expansion", flush=True)
    return full


def dem_check_action_matrices(
    circuit: stim.Circuit,
    sector: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, A, multipliers, probs) from the circuit's detector error model.

    H : (M×N) uint8 detector-flip parity-check; columns are fault mechanisms.
    A : (K×N) uint8 logical-observable action matrix.
    multipliers : per-mechanism expansion counts m_j (≥1) for the expanded rep.
    probs : raw DEM mechanism probabilities (used as the BP error channel).

    ``sector`` (e.g. 0 for Z-type decoding) restricts to a single CSS detector sector
    via :func:`single_sector_dem` — the paper's representation. ``None`` (default) keeps
    the full both-sector DEM (backward compatible).
    """
    if sector is not None:
        H, A, mult, probs, _ = single_sector_dem(circuit, detector_type=sector)
        return H, A, mult, probs
    probs, det_mat, obs_mat = _parse_dem(circuit)
    H = det_mat.T.astype(np.uint8)
    A = obs_mat.T.astype(np.uint8)
    _, _, mult = _expand(probs, None)
    return H, A, mult, probs


def _bposd(H: np.ndarray, priors, osd_order: int, max_iter: int) -> BpOsdDecoder:
    return BpOsdDecoder(
        H.astype(np.uint8),
        error_channel=list(priors),
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=osd_order,
    )


# ---------------------------------------------------------------------------
# Optional process-parallel BP-OSD workers.
#
# ldpc's BpOsdDecoder is single-threaded per decode, but the distance decodes (one
# per logical observable) and the L(D)-search trials are independent, so they
# parallelise across processes. Workers build their own decoder from matrices shared
# once via the pool initializer; only small results are returned. These must be
# module-level (picklable) to work under the Windows 'spawn' start method.
# ---------------------------------------------------------------------------
_MW: Dict[str, object] = {}


def _mw_init(H, A, priors, osd_order, max_iter) -> None:
    _MW.update(H=H, A=A, priors=priors, osd_order=osd_order, max_iter=max_iter, M=H.shape[0])


def _mw_forcing_correction(extra_row: np.ndarray) -> np.ndarray:
    """OSD-W correction for H stacked with one forcing row whose check is fired."""
    H = _MW["H"]; M = _MW["M"]
    dec = _bposd(np.vstack([H, extra_row[None, :]]), _MW["priors"], _MW["osd_order"], _MW["max_iter"])
    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    dec.decode(syndrome)
    return np.asarray(dec.osdw_decoding, dtype=np.uint8)


def _mw_distance_task(i: int) -> Tuple[int, np.ndarray]:
    corr = _mw_forcing_correction(_MW["A"][i])
    return int(corr.sum()), corr


def _mw_logical_trial_task(args) -> Optional[Tuple[int, tuple]]:
    trial, base_seed = args
    A = _MW["A"]; K = A.shape[0]
    rng = np.random.default_rng([base_seed, trial])     # deterministic per trial
    coeffs = rng.integers(0, 2, size=K)
    if not coeffs.any():
        coeffs[rng.integers(K)] = 1
    g = (coeffs @ A) % 2
    corr = _mw_forcing_correction(g)
    H = _MW["H"]
    if (H @ corr % 2).any() or not (A @ corr % 2).any():
        return None
    return int(corr.sum()), tuple(int(x) for x in np.flatnonzero(corr))


def _mw_systematic_task(mask: int) -> Optional[Tuple[int, tuple]]:
    """Decode the GF(2) combination of logical generators given by bitmask.

    Mask bit i set means include generator i in the combination. All 2^K − 1
    nonzero masks cover every coset of the logical group exactly once, so this
    fully exhausts the syndrome space for L(D) search.
    """
    A = _MW["A"]; K = A.shape[0]
    coeffs = np.array([(mask >> i) & 1 for i in range(K)], dtype=np.uint8)
    g = (coeffs @ A) % 2
    corr = _mw_forcing_correction(g)
    H = _MW["H"]
    if (H @ corr % 2).any() or not (A @ corr % 2).any():
        return None
    return int(corr.sum()), tuple(int(x) for x in np.flatnonzero(corr))


def _decimated_correction(H, priors, osd_order, max_iter, g, rng, max_odd):
    """Decimation (paper §4.2): instead of decoding ``[H; g]`` (the appended logical-forcing row
    ``g`` is usually HIGH weight, which degrades BP-OSD), FIX the bits in ``supp(g)`` to a low
    odd-weight assignment so the g-check is unsatisfied, then decode the REDUCED, now-low-weight
    problem on the remaining columns. The combined solution is a nontrivial logical:
    ``H@corr=0`` by construction; and ``g@corr = odd = 1`` with ``g=h+l`` (``h`` in rowspace(H),
    so ``h@corr=0``) forces ``l@corr=1`` — a nonzero logical action.

    Returns the full correction vector (uint8, length N) or None if degenerate.
    """
    priors = np.asarray(priors, dtype=float)
    M, N = H.shape
    supp_g = np.flatnonzero(g)
    if supp_g.size == 0 or supp_g.size >= N:
        return None
    odds = [k for k in range(1, max_odd + 1, 2) if k <= supp_g.size]
    k = int(rng.choice(odds)) if odds else 1
    set_pos = supp_g[rng.choice(supp_g.size, size=k, replace=False)]
    x = np.zeros(N, dtype=np.uint8)
    x[set_pos] = 1
    sigma = (H.astype(np.int64) @ x.astype(np.int64)) % 2     # syndrome induced by the fixed bits
    unfixed = np.setdiff1d(np.arange(N, dtype=np.int64), supp_g, assume_unique=True)
    dec = _bposd(H[:, unfixed], priors[unfixed], osd_order, max_iter)
    dec.decode(sigma.astype(np.uint8))
    x[unfixed] = np.asarray(dec.osdw_decoding, dtype=np.uint8)  # unfixed entries were 0
    return x


def _decimated_trial(H, A, priors, osd_order, max_iter, rng, max_odd):
    """One decimated min-weight-logical trial: g = h + l with h ~ rowspace(H), l ~ rowspace(A)\\{0}
    (random h varies supp(g) between trials). Returns a valid logical correction or None."""
    M, N = H.shape
    K = A.shape[0]
    l_coeffs = rng.integers(0, 2, size=K)
    if not l_coeffs.any():
        l_coeffs[rng.integers(K)] = 1
    h_coeffs = rng.integers(0, 2, size=M)
    g = ((h_coeffs @ H) + (l_coeffs @ A)) % 2
    corr = _decimated_correction(H, priors, osd_order, max_iter, g, rng, max_odd)
    if corr is None or (H @ corr % 2).any() or not (A @ corr % 2).any():
        return None
    return corr


def _mw_decimated_trial_task(args) -> Optional[Tuple[int, tuple]]:
    """Parallel-worker wrapper for a decimated trial (uses the pool-shared _MW matrices)."""
    trial, base_seed, max_odd = args
    rng = np.random.default_rng([base_seed, 7777, trial])      # distinct stream from the base search
    corr = _decimated_trial(_MW["H"], _MW["A"], _MW["priors"], _MW["osd_order"], _MW["max_iter"],
                            rng, max_odd)
    if corr is None:
        return None
    return int(corr.sum()), tuple(int(x) for x in np.flatnonzero(corr))


def _mw_coset_enum_task(args) -> List[tuple]:
    """Enumerate weight-D logicals in one coset via column-exclusion DFS.

    The systematic search returns a single min-weight representative per coset, but a
    coset may hold several distinct weight-D logicals (within-coset multiplicity). This
    explores them: decode the forced syndrome; for each weight-D logical found, branch by
    *removing* each of its columns and re-decoding, so subsequent solutions must avoid that
    column. Bounded by ``budget`` decodes per coset. Columns are physically dropped (not
    just down-weighted) so each solution provably avoids the excluded set.
    """
    mask, D, budget = args
    H = _MW["H"]; A = _MW["A"]; base = np.asarray(_MW["priors"], dtype=float)
    osd_order = _MW["osd_order"]; max_iter = _MW["max_iter"]
    K = A.shape[0]; N = H.shape[1]
    coeffs = np.array([(mask >> i) & 1 for i in range(K)], dtype=np.uint8)
    g = (coeffs @ A) % 2
    found: Set[tuple] = set()
    seen_excl: Set[frozenset] = set()
    stack: List[frozenset] = [frozenset()]
    decodes = 0
    while stack and decodes < budget:
        excl = stack.pop()
        if excl in seen_excl:
            continue
        seen_excl.add(excl)
        kept = np.array([c for c in range(N) if c not in excl], dtype=np.int64)
        Hk = H[:, kept]; gk = g[kept]
        if not gk.any():
            continue  # no column carries this logical action once excluded
        M = Hk.shape[0]
        dec = _bposd(np.vstack([Hk, gk[None, :]]), base[kept], osd_order, max_iter)
        syndrome = np.zeros(M + 1, dtype=np.uint8); syndrome[M] = 1
        dec.decode(syndrome)
        decodes += 1
        corr = np.asarray(dec.osdw_decoding, dtype=np.uint8)
        if int(corr.sum()) != D:
            continue
        supp = tuple(int(kept[i]) for i in np.flatnonzero(corr))
        full = np.zeros(N, dtype=np.uint8); full[list(supp)] = 1
        if (H @ full % 2).any() or not (A @ full % 2).any():
            continue
        if supp in found:
            continue
        found.add(supp)
        for c in supp:                      # branch: exclude each column in turn
            stack.append(excl | {c})
    return list(found)


@dataclass
class DistanceResult:
    distance: int
    onset: int                       # ceil(D/2)
    per_logical_weight: List[int]    # min-weight logical found for each observable i
    witnesses: List[np.ndarray]      # the corresponding fault bitstrings


def compute_distance(
    circuit: Optional[stim.Circuit] = None,
    *,
    osd_order: int = 10,
    max_iter: int = 200,
    priors=None,
    progress: bool = False,
    workers: int = 1,
    sector: Optional[int] = None,
    matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> DistanceResult:
    """Circuit fault distance D and optimal onset ceil(D/2) (paper §4.1, BCG+24).

    For each logical i, append row A[i] to H, decode the syndrome that forces the
    appended check to fire → a fault bitstring nontrivial on logical i. D is the
    minimum weight over i. Exact if BP-OSD returns min-weight; otherwise an upper
    bound on D. ``sector`` restricts to a single CSS detector sector (see
    :func:`dem_check_action_matrices`).

    ``matrices=(H, A, probs)`` decodes pre-built matrices directly (e.g. the Bravyi
    reference single-sector construction), bypassing ``dem_check_action_matrices`` and
    the need for a Stim circuit.
    """
    if matrices is not None:
        H, A, probs = matrices
        H = np.asarray(H, dtype=np.uint8); A = np.asarray(A, dtype=np.uint8)
    else:
        H, A, _, probs = dem_check_action_matrices(circuit, sector=sector)
    M, N = H.shape
    K = A.shape[0]
    if priors is None:
        priors = probs
    if K == 0:
        raise ValueError("circuit has no logical observables")

    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    weights: List[int] = []
    witnesses: List[np.ndarray] = []

    if workers and workers > 1 and K > 1:
        import os
        from multiprocessing import Pool
        nproc = min(int(workers), K, os.cpu_count() or 1)
        with Pool(nproc, initializer=_mw_init, initargs=(H, A, priors, osd_order, max_iter)) as pool:
            results = pool.map(_mw_distance_task, range(K))
        for i, (w, corr) in enumerate(results):
            # Sanity: a valid logical satisfies H·corr = 0 and is nontrivial on A.
            if (H @ corr % 2).any() or not (A @ corr % 2).any():
                raise RuntimeError(f"decoder returned a non-logical correction for observable {i}")
            weights.append(int(w))
            witnesses.append(corr)
        if progress:
            print(f"      distance [parallel x{nproc}]: weights {weights}, min {min(weights)}", flush=True)
    else:
        for i in range(K):
            Hi = np.vstack([H, A[i : i + 1, :]])
            dec = _bposd(Hi, priors, osd_order, max_iter)
            dec.decode(syndrome)
            corr = np.asarray(dec.osdw_decoding, dtype=np.uint8)
            # Sanity: a valid logical satisfies H·corr = 0 and is nontrivial on A.
            if (H @ corr % 2).any() or not (A @ corr % 2).any():
                raise RuntimeError(f"decoder returned a non-logical correction for observable {i}")
            weights.append(int(corr.sum()))
            witnesses.append(corr)
            if progress:
                print(f"      distance: logical {i+1}/{K} → weight {weights[-1]} "
                      f"(running min {min(weights)})", flush=True)

    D = min(weights)
    onset = (D + 1) // 2
    return DistanceResult(distance=D, onset=onset, per_logical_weight=weights, witnesses=witnesses)


def find_min_weight_logicals(
    circuit: Optional[stim.Circuit] = None,
    D: Optional[int] = None,
    *,
    max_trials: int = 2000,
    patience: int = 300,
    osd_order: int = 10,
    max_iter: int = 200,
    priors=None,
    seed: Optional[int] = None,
    progress_every: int = 0,
    workers: int = 1,
    systematic: bool = True,
    symmetry_perms: Optional[List[np.ndarray]] = None,
    sector: Optional[int] = None,
    matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    return_trace: bool = False,
    decimate: bool = False,
    decimate_max_odd: int = 3,
):
    """Search for L(D), the set of weight-D logical bitstrings (paper §4.2).

    ``decimate=True`` (paper §4.2 "Faster search by decimating the appended row"): for the random
    trials, build g = h + l (h ~ rowspace(H), l ~ rowspace(A)\\{0}), FIX supp(g) to a low odd-weight
    assignment, and decode the reduced low-weight problem on the remaining columns — instead of
    decoding the high-weight ``[H; g]`` directly (which degrades BP-OSD). The paper found this
    essential for hard cases (e.g. BB(18): weight-18 logicals found only with decimation).
    ``decimate_max_odd`` bounds the odd number of supp(g) bits set to 1 (sampled from {1,3,..}).
    Decimation applies to the random-trial phase; the systematic coset sweep is unchanged.

    ``return_trace=True`` additionally returns a saturation trace ``[(cumulative_trials,
    |L(D)| found), ...]`` recorded each time the count grows (systematic then random phases) —
    for the search-convergence plot, which should plateau once all min-weight logicals are found.
    Returns ``found`` (a set) normally, or ``(found, trace)`` when ``return_trace`` is set.

    ``systematic=True`` (default) exhaustively enumerates all 2^K − 1 nonzero GF(2)
    combinations of the K logical generators before any random trials, fully covering
    every syndrome class at least once. For K=12 this is 4095 extra decodes (~10 min
    with workers=8 at ~1.2s each). Disable with ``systematic=False`` for the old
    random-only behaviour.

    ``progress_every`` (>0) prints a status line every that-many trials.
    ``workers`` > 1 runs the (independent) trials across a process pool. The parallel
    path uses per-trial deterministic seeding and runs the full ``max_trials`` —
    ``patience`` early-stop does not apply (the trials are dispatched up front).

    ``symmetry_perms`` (optional): list of permutation arrays, each of shape (N,),
    mapping mechanism indices to their symmetry-equivalent counterparts (e.g., the
    36 toric translations from build_circuit_translation_perms in
    bb6_bitflip_comparison.py). When provided, each newly found logical is expanded
    by all permutations immediately, dramatically reducing the trials needed.
    The caller must ensure permutations preserve validity (i.e., the code has the
    exact symmetry encoded in the permutations).
    """
    if matrices is not None:
        H, A, probs = matrices
        H = np.asarray(H, dtype=np.uint8); A = np.asarray(A, dtype=np.uint8)
    else:
        H, A, _, probs = dem_check_action_matrices(circuit, sector=sector)
    M, N = H.shape
    K = A.shape[0]
    if priors is None:
        priors = probs
    if D is None:
        D = compute_distance(circuit, osd_order=osd_order, max_iter=max_iter,
                             priors=priors, workers=workers, sector=sector,
                             matrices=matrices).distance

    n_systematic = (1 << K) - 1 if (systematic and K <= 20) else 0

    def _expand_by_sym(support: FrozenSet[int], found_set: Set[FrozenSet[int]]) -> None:
        """Add all symmetry images of support to found_set (in-place)."""
        if symmetry_perms is None:
            return
        for sp in symmetry_perms:
            ts = frozenset(int(sp[m]) for m in support)
            found_set.add(ts)

    if workers and workers > 1:
        import os
        from multiprocessing import Pool
        nproc = min(int(workers), os.cpu_count() or 1)
        base_seed = 0 if seed is None else int(seed)
        found_p: Set[FrozenSet[int]] = set()
        trace: List[Tuple[int, int]] = []

        with Pool(nproc, initializer=_mw_init, initargs=(H, A, priors, osd_order, max_iter)) as pool:
            # Phase 1: systematic sweep of all 2^K - 1 syndrome classes.
            if n_systematic > 0:
                if progress_every:
                    print(f"      L(D) search [systematic x{nproc}]: "
                          f"{n_systematic} syndrome classes ...", flush=True)
                for n, res in enumerate(
                    pool.imap_unordered(_mw_systematic_task, range(1, n_systematic + 1), chunksize=16), 1
                ):
                    if res is not None and res[0] == D:
                        support = frozenset(res[1])
                        if support not in found_p:
                            found_p.add(support)
                            _expand_by_sym(support, found_p)
                            trace.append((n, len(found_p)))
                    if progress_every and n % progress_every == 0:
                        print(f"      L(D) systematic: {n}/{n_systematic}, "
                              f"|L(D)|={len(found_p)}", flush=True)
                if progress_every:
                    print(f"      L(D) systematic done: |L(D)|={len(found_p)}", flush=True)

            # Phase 2: random trials for max_trials additional attempts.
            if max_trials > 0:
                if decimate:
                    task_fn = _mw_decimated_trial_task
                    tasks = [(t, base_seed, decimate_max_odd) for t in range(max_trials)]
                else:
                    task_fn = _mw_logical_trial_task
                    tasks = [(t, base_seed) for t in range(max_trials)]
                for n, res in enumerate(pool.imap_unordered(task_fn, tasks, chunksize=4), 1):
                    if res is not None and res[0] == D:
                        support = frozenset(res[1])
                        if support not in found_p:
                            found_p.add(support)
                            _expand_by_sym(support, found_p)
                            trace.append((n_systematic + n, len(found_p)))
                    if progress_every and n % progress_every == 0:
                        print(f"      L(D) random [parallel x{nproc}]: {n}/{max_trials}, "
                              f"|L(D)|={len(found_p)}", flush=True)

        return (found_p, trace) if return_trace else found_p

    rng = np.random.default_rng(seed)
    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    found: Set[FrozenSet[int]] = set()
    trace: List[Tuple[int, int]] = []
    no_new = 0

    # Systematic phase (serial).
    if n_systematic > 0:
        if progress_every:
            print(f"      L(D) search [systematic]: {n_systematic} syndrome classes ...", flush=True)
        for n, mask in enumerate(range(1, n_systematic + 1), 1):
            coeffs = np.array([(mask >> i) & 1 for i in range(K)], dtype=np.uint8)
            g = (coeffs @ A) % 2
            dec = _bposd(np.vstack([H, g[None, :]]), priors, osd_order, max_iter)
            dec.decode(syndrome)
            corr = np.asarray(dec.osdw_decoding, dtype=np.uint8)
            is_logical = not (H @ corr % 2).any() and (A @ corr % 2).any()
            if is_logical and int(corr.sum()) == D:
                support = frozenset(np.flatnonzero(corr).tolist())
                if support not in found:
                    found.add(support)
                    _expand_by_sym(support, found)
                    trace.append((n, len(found)))
            if progress_every and n % progress_every == 0:
                print(f"      L(D) systematic: {n}/{n_systematic}, |L(D)|={len(found)}", flush=True)

    for trial in range(max_trials):
        if progress_every and trial > 0 and trial % progress_every == 0:
            print(f"      L(D) search: trial {trial}/{max_trials}, "
                  f"|L(D)|={len(found)} found, no-new streak {no_new}/{patience}", flush=True)
        if decimate:
            corr = _decimated_trial(H, A, priors, osd_order, max_iter, rng, decimate_max_odd)
        else:
            coeffs = rng.integers(0, 2, size=K)
            if not coeffs.any():
                coeffs[rng.integers(K)] = 1
            g = (coeffs @ A) % 2
            dec = _bposd(np.vstack([H, g[None, :]]), priors, osd_order, max_iter)
            dec.decode(syndrome)
            c = np.asarray(dec.osdw_decoding, dtype=np.uint8)
            corr = c if (not (H @ c % 2).any() and (A @ c % 2).any()) else None
        if corr is not None and int(corr.sum()) == D:
            support = frozenset(np.flatnonzero(corr).tolist())
            if support not in found:
                found.add(support)
                _expand_by_sym(support, found)
                trace.append((n_systematic + trial + 1, len(found)))
                no_new = 0
                continue
        no_new += 1
        if no_new >= patience:
            break
    return (found, trace) if return_trace else found


def find_all_min_weight_logicals(
    circuit: Optional[stim.Circuit] = None,
    D: Optional[int] = None,
    *,
    matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    sector: Optional[int] = None,
    budget_per_coset: int = 40,
    symmetry_perms: Optional[List[np.ndarray]] = None,
    workers: int = 8,
    priors=None,
    osd_order: int = 10,
    max_iter: int = 200,
    progress_every: int = 0,
) -> Set[FrozenSet[int]]:
    """Exhaustive(-ish) L(D): every coset + within-coset multiplicity (paper's fault
    restrictions). Decodes all 2^K−1 cosets; per coset, :func:`_mw_coset_enum_task`
    branches by column-exclusion to pull out *all* weight-D logicals it can reach within
    ``budget_per_coset`` decodes. Each find is expanded by ``symmetry_perms`` (the Z₆×Z₆
    toric shifts), so only orbit representatives need be discovered.

    Use when the single-representative-per-coset search (:func:`find_min_weight_logicals`)
    saturates below the true |L(D)| — i.e. cosets hold multiple weight-D logicals not
    related by the symmetry group. Pass ``matrices=(H, A, probs)`` for the reference
    construction, or a ``circuit`` (+optional ``sector``) for the Stim path.
    """
    import os
    from multiprocessing import Pool

    if matrices is not None:
        H, A, probs = matrices
        H = np.asarray(H, dtype=np.uint8); A = np.asarray(A, dtype=np.uint8)
    else:
        H, A, _, probs = dem_check_action_matrices(circuit, sector=sector)
    if priors is None:
        priors = probs
    K = A.shape[0]
    if D is None:
        D = compute_distance(matrices=(H, A, priors), workers=workers,
                             osd_order=osd_order, max_iter=max_iter).distance
    n_sys = (1 << K) - 1

    found: Set[FrozenSet[int]] = set()

    def _expand(supp: tuple) -> None:
        s = frozenset(supp)
        if s in found:
            return
        found.add(s)
        if symmetry_perms is not None:
            for p in symmetry_perms:
                found.add(frozenset(int(p[c]) for c in supp))

    nproc = min(int(workers), os.cpu_count() or 1)
    tasks = [(m, D, budget_per_coset) for m in range(1, n_sys + 1)]
    with Pool(nproc, initializer=_mw_init, initargs=(H, A, priors, osd_order, max_iter)) as pool:
        for n, res in enumerate(pool.imap_unordered(_mw_coset_enum_task, tasks, chunksize=8), 1):
            for supp in res:
                _expand(supp)
            if progress_every and n % progress_every == 0:
                print(f"      coset-enum: {n}/{n_sys}, |L(D)|={len(found)}", flush=True)
    return found


def min_weight_fail_count(
    H: np.ndarray,
    A: np.ndarray,
    logical_supports,
    multipliers: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Exact |F(D/2)| for even D via Proposition 1 (paper §4.3).

    F(D/2) is the set of weight-D/2 failing errors under a max-class min-weight
    decoder, and equals the failing subset of L(D)|_{D/2} (the weight-D/2
    restrictions of the min-weight logicals). We enumerate those restrictions,
    partition them by (syndrome σ=H·r, action a=A·r) in the expanded representation
    (each restriction r contributes ρ(r)=Π_{j∈r} m_j copies), and for each σ the
    largest-action class succeeds while the rest fail.

    Returns (|F(D/2)|, N_expanded). Requires all logicals to share weight D (even).
    """
    H = H.astype(np.uint8)
    A = A.astype(np.uint8)
    N = H.shape[1]
    if multipliers is None:
        multipliers = np.ones(N, dtype=np.int64)
    multipliers = np.asarray(multipliers, dtype=np.int64)

    supports = [frozenset(s) for s in logical_supports]
    if not supports:
        raise ValueError("no logicals provided")
    D = len(next(iter(supports)))
    if any(len(s) != D for s in supports):
        raise ValueError("all logicals must have the same weight D")
    if D % 2 != 0:
        raise ValueError("Proposition-1 exact onset requires even D (see Appendix A.6 for odd D)")
    half = D // 2

    # Unique weight-D/2 restrictions of the min-weight logicals.
    restrictions: Set[FrozenSet[int]] = set()
    for s in supports:
        for r in combinations(sorted(s), half):
            restrictions.add(frozenset(r))

    # Group restrictions by syndrome σ, accumulating expanded-rep set sizes per action a.
    # sigma_key -> { action_key -> summed multiplicity }
    by_sigma: Dict[bytes, Dict[bytes, int]] = {}
    for r in restrictions:
        idx = np.fromiter(r, dtype=np.int64, count=len(r))
        rho = int(np.prod(multipliers[idx]))
        sig = np.packbits((H[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        act = np.packbits((A[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        by_sigma.setdefault(sig, {})
        by_sigma[sig][act] = by_sigma[sig].get(act, 0) + rho

    # For each syndrome, the max-class action succeeds; all other classes fail.
    fails = 0
    for action_sizes in by_sigma.values():
        sizes = list(action_sizes.values())
        fails += sum(sizes) - max(sizes)

    N_expanded = int(multipliers.sum())
    return fails, N_expanded


def find_weight_logicals_mitm(
    H: np.ndarray,
    A: np.ndarray,
    weight: int,
    *,
    max_choose: int = 5_000_000,
) -> Set[FrozenSet[int]]:
    """Exact, complete enumeration of all weight-``weight`` (EVEN) logical supports via half-MITM.

    Every weight-``weight`` logical splits into two disjoint weight-``weight/2`` halves with equal
    detector syndrome (``H·ℓ=0``) and differing logical action (``A·ℓ≠0``). We enumerate all
    weight-``weight/2`` column subsets, bucket them by syndrome, and pair up disjoint halves whose
    actions differ. Exact and complete, but the enumeration is ``C(N, weight/2)`` subsets, so this is
    intended for SMALL systems (e.g. the [[18,4,4]] tutorial, where it supplies ``L(D+1)`` for the
    odd-``D`` onset). Raises if the enumeration would exceed ``max_choose`` subsets.

    Returns a set of frozenset column supports, each of size ``weight``.
    """
    from math import comb
    H = np.asarray(H, dtype=np.uint8); A = np.asarray(A, dtype=np.uint8)
    M, N = H.shape; K = A.shape[0]
    if weight % 2 != 0:
        raise ValueError(f"half-MITM enumeration needs even weight (got {weight})")
    half = weight // 2
    if comb(N, half) > max_choose:
        raise ValueError(f"C({N},{half})={comb(N, half)} exceeds max_choose={max_choose}; "
                         f"this exact enumerator is for small systems only")
    powM = 1 << np.arange(M, dtype=object)
    powK = 1 << np.arange(K, dtype=object)
    bsig = [int(x) for x in (H.T.astype(object) @ powM)]   # per-column detector syndrome (int mask)
    bact = [int(x) for x in (A.T.astype(object) @ powK)]   # per-column logical action (int mask)
    groups: Dict[int, List[Tuple[FrozenSet[int], int]]] = {}
    for combo in combinations(range(N), half):
        sig = 0; act = 0
        for c in combo:
            sig ^= bsig[c]; act ^= bact[c]
        groups.setdefault(sig, []).append((frozenset(combo), act))
    logicals: Set[FrozenSet[int]] = set()
    for lst in groups.values():
        if len({a for _, a in lst}) < 2:        # need two differing actions to make a logical
            continue
        for i in range(len(lst)):
            pi, ai = lst[i]
            for j in range(i + 1, len(lst)):
                pj, aj = lst[j]
                if aj != ai and pi.isdisjoint(pj):
                    logicals.add(pi | pj)
    return logicals


def min_weight_fail_count_odd(
    H: np.ndarray,
    A: np.ndarray,
    logicals_D,
    logicals_Dp1,
    multipliers: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Exact |F(w0)| for ODD D via Appendix A.6 of arXiv:2511.15177, where w0 = (D+1)/2.

    For odd D the onset weight is w0 = ⌈D/2⌉ = (D+1)/2 and BOTH L(D) and L(D+1) contribute failures:

      (i)  Every weight-w0 restriction of L(D) fails: if r ⊂ ℓ ∈ L(D) then ℓ∖r has weight (D−1)/2 < w0,
           so a min-weight decoder prefers that lighter, different-class correction. We count them all.
      (ii) Weight-w0 restrictions of L(D+1) — after removing any that also appear in L(D)|_{w0} — fail
           unless their class wins the max-class vote among same-syndrome weight-w0 errors (the
           Section-4.3 procedure: per syndrome, the largest-action class succeeds, the rest fail; an
           A-way tie gives success probability 1/A, i.e. ``sum − max`` fails in expectation).

    With complete L(D) and L(D+1) this is exact; with partial (sampled) sets it is a lower bound on the
    number of min-weight fails. Returns (|F(w0)|, N_expanded). See :func:`min_weight_fail_count` for the
    even-D case (Proposition 1).
    """
    H = H.astype(np.uint8); A = A.astype(np.uint8)
    N = H.shape[1]
    if multipliers is None:
        multipliers = np.ones(N, dtype=np.int64)
    multipliers = np.asarray(multipliers, dtype=np.int64)

    sup_D = [frozenset(s) for s in logicals_D]
    if not sup_D:
        raise ValueError("no weight-D logicals provided")
    D = len(next(iter(sup_D)))
    if any(len(s) != D for s in sup_D):
        raise ValueError("all L(D) supports must have the same weight D")
    if D % 2 == 0:
        raise ValueError("min_weight_fail_count_odd is for odd D; use min_weight_fail_count (Prop. 1) for even D")
    w0 = (D + 1) // 2

    def rho(cols) -> int:
        return int(np.prod(multipliers[list(cols)]))

    # (i) every weight-w0 restriction of L(D) fails (lighter complement in another class)
    R_D: Set[FrozenSet[int]] = {frozenset(r) for s in sup_D for r in combinations(sorted(s), w0)}
    fails = sum(rho(r) for r in R_D)

    # (ii) weight-w0 restrictions of L(D+1), minus L(D)|_{w0}, scored by the max-class vote
    R_Dp1: Set[FrozenSet[int]] = {frozenset(r) for s in logicals_Dp1 for r in combinations(sorted(s), w0)}
    by_sigma: Dict[bytes, Dict[bytes, int]] = {}
    for r in (R_Dp1 - R_D):
        idx = np.fromiter(r, dtype=np.int64, count=len(r))
        sig = np.packbits((H[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        act = np.packbits((A[:, idx].sum(axis=1) % 2).astype(np.uint8)).tobytes()
        by_sigma.setdefault(sig, {})
        by_sigma[sig][act] = by_sigma[sig].get(act, 0) + rho(idx)
    for action_sizes in by_sigma.values():
        sizes = list(action_sizes.values())
        fails += sum(sizes) - max(sizes)

    N_expanded = int(multipliers.sum())
    return fails, N_expanded


@dataclass
class OnsetResult:
    distance: int
    onset: int                 # w0 = ceil(D/2): D/2 for even D, (D+1)/2 for odd D
    n_min_logicals: int        # |L(D)| found
    fail_count: int            # |F(w0)|
    n_expanded: int            # N in the expanded representation
    onset_fraction: float      # f*(w0) = |F(w0)| / C(N, w0)
    n_min_logicals_Dp1: Optional[int] = None   # |L(D+1)| used for the odd-D onset (None for even D)


def optimal_onset_fraction(
    circuit: stim.Circuit,
    *,
    distance: Optional[int] = None,
    logicals: Optional[Set[FrozenSet[int]]] = None,
    logicals_Dp1: Optional[Set[FrozenSet[int]]] = None,
    osd_order: int = 10,
    max_iter: int = 200,
    max_trials: int = 2000,
    seed: Optional[int] = None,
    sector: Optional[int] = None,
    mitm_max_choose: int = 5_000_000,
) -> OnsetResult:
    """Exact optimal onset fraction f*(w0), w0 = ⌈D/2⌉, for even OR odd circuit distance.

    Convenience wrapper: computes D (if not given), searches L(D) (if not given), then evaluates the
    optimal min-weight fail count |F(w0)| and f*(w0) = |F(w0)| / C(N, w0). The result anchors the
    Technique-I ansatz fit via ``w0=onset`` and ``f0=onset_fraction``.

    * **Even D** — Proposition 1 (§4.3) via :func:`min_weight_fail_count`; onset weight w0 = D/2.
    * **Odd D** — Appendix A.6 via :func:`min_weight_fail_count_odd`; onset weight w0 = (D+1)/2. This
      additionally needs ``L(D+1)``, enumerated exactly with :func:`find_weight_logicals_mitm` (or pass
      ``logicals_Dp1``). The MITM enumerates ``C(N, w0)`` subsets, so the odd-D path is for SMALL
      systems; ``mitm_max_choose`` guards the cost.

    ``sector`` restricts to a single CSS detector sector (see :func:`dem_check_action_matrices`).
    With complete logical sets the result is exact; with partial sets it is a lower bound on |F(w0)|.
    """
    H, A, mult, priors = dem_check_action_matrices(circuit, sector=sector)
    if distance is None:
        distance = compute_distance(circuit, osd_order=osd_order, max_iter=max_iter,
                                    priors=priors, sector=sector).distance
    if logicals is None:
        logicals = find_min_weight_logicals(
            circuit, distance, max_trials=max_trials, osd_order=osd_order,
            max_iter=max_iter, priors=priors, seed=seed, sector=sector,
        )
    from scipy.special import gammaln
    if distance % 2 == 0:
        w0 = distance // 2
        fails, n_exp = min_weight_fail_count(H, A, logicals, mult)
        n_Dp1 = None
    else:
        w0 = (distance + 1) // 2
        if logicals_Dp1 is None:
            logicals_Dp1 = find_weight_logicals_mitm(H, A, distance + 1, max_choose=mitm_max_choose)
        fails, n_exp = min_weight_fail_count_odd(H, A, logicals, logicals_Dp1, mult)
        n_Dp1 = len(logicals_Dp1)
    # C(N_expanded, w0) via log-gamma to avoid overflow on large N.
    log_choose = gammaln(n_exp + 1) - gammaln(w0 + 1) - gammaln(n_exp - w0 + 1)
    f_star = fails / np.exp(log_choose)
    return OnsetResult(
        distance=distance,
        onset=w0,
        n_min_logicals=len(logicals),
        fail_count=fails,
        n_expanded=n_exp,
        onset_fraction=float(f_star),
        n_min_logicals_Dp1=n_Dp1,
    )
