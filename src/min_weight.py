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
) -> Set[FrozenSet[int]]:
    """Search for L(D), the set of weight-D logical bitstrings (paper §4.2).

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
                    if progress_every and n % progress_every == 0:
                        print(f"      L(D) systematic: {n}/{n_systematic}, "
                              f"|L(D)|={len(found_p)}", flush=True)
                if progress_every:
                    print(f"      L(D) systematic done: |L(D)|={len(found_p)}", flush=True)

            # Phase 2: random trials for max_trials additional attempts.
            if max_trials > 0:
                tasks = [(t, base_seed) for t in range(max_trials)]
                for n, res in enumerate(pool.imap_unordered(_mw_logical_trial_task, tasks, chunksize=4), 1):
                    if res is not None and res[0] == D:
                        support = frozenset(res[1])
                        if support not in found_p:
                            found_p.add(support)
                            _expand_by_sym(support, found_p)
                    if progress_every and n % progress_every == 0:
                        print(f"      L(D) random [parallel x{nproc}]: {n}/{max_trials}, "
                              f"|L(D)|={len(found_p)}", flush=True)

        return found_p

    rng = np.random.default_rng(seed)
    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    found: Set[FrozenSet[int]] = set()
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
            if progress_every and n % progress_every == 0:
                print(f"      L(D) systematic: {n}/{n_systematic}, |L(D)|={len(found)}", flush=True)

    for trial in range(max_trials):
        if progress_every and trial > 0 and trial % progress_every == 0:
            print(f"      L(D) search: trial {trial}/{max_trials}, "
                  f"|L(D)|={len(found)} found, no-new streak {no_new}/{patience}", flush=True)
        coeffs = rng.integers(0, 2, size=K)
        if not coeffs.any():
            coeffs[rng.integers(K)] = 1
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
                no_new = 0
                continue
        no_new += 1
        if no_new >= patience:
            break
    return found


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


@dataclass
class OnsetResult:
    distance: int
    onset: int                 # D/2 for even D
    n_min_logicals: int        # |L(D)| found
    fail_count: int            # |F(D/2)|
    n_expanded: int            # N in the expanded representation
    onset_fraction: float      # f*(D/2) = |F(D/2)| / C(N, D/2)


def optimal_onset_fraction(
    circuit: stim.Circuit,
    *,
    distance: Optional[int] = None,
    logicals: Optional[Set[FrozenSet[int]]] = None,
    osd_order: int = 10,
    max_iter: int = 200,
    max_trials: int = 2000,
    seed: Optional[int] = None,
    sector: Optional[int] = None,
) -> OnsetResult:
    """Exact optimal onset fraction f*(D/2) for a circuit with even distance (§4.3).

    Convenience wrapper: computes D (if not given), searches L(D) (if not given),
    then evaluates |F(D/2)| via :func:`min_weight_fail_count`. The result anchors
    the Technique-I ansatz fit via ``w0=onset`` and ``f0=onset_fraction``. ``sector``
    restricts to a single CSS detector sector (see :func:`dem_check_action_matrices`).
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
    fails, n_exp = min_weight_fail_count(H, A, logicals, mult)
    half = distance // 2
    # C(N_expanded, D/2) via log-gamma to avoid overflow on large N.
    from scipy.special import gammaln
    log_choose = gammaln(n_exp + 1) - gammaln(half + 1) - gammaln(n_exp - half + 1)
    f_star = fails / np.exp(log_choose)
    return OnsetResult(
        distance=distance,
        onset=half,
        n_min_logicals=len(logicals),
        fail_count=fails,
        n_expanded=n_exp,
        onset_fraction=float(f_star),
    )
