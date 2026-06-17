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


def dem_check_action_matrices(
    circuit: stim.Circuit,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, A, multipliers, probs) from the circuit's detector error model.

    H : (M×N) uint8 detector-flip parity-check; columns are fault mechanisms.
    A : (K×N) uint8 logical-observable action matrix.
    multipliers : per-mechanism expansion counts m_j (≥1) for the expanded rep.
    probs : raw DEM mechanism probabilities (used as the BP error channel).
    """
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


@dataclass
class DistanceResult:
    distance: int
    onset: int                       # ceil(D/2)
    per_logical_weight: List[int]    # min-weight logical found for each observable i
    witnesses: List[np.ndarray]      # the corresponding fault bitstrings


def compute_distance(
    circuit: stim.Circuit,
    *,
    osd_order: int = 10,
    max_iter: int = 200,
    priors=None,
    progress: bool = False,
    workers: int = 1,
) -> DistanceResult:
    """Circuit fault distance D and optimal onset ceil(D/2) (paper §4.1, BCG+24).

    For each logical i, append row A[i] to H, decode the syndrome that forces the
    appended check to fire → a fault bitstring nontrivial on logical i. D is the
    minimum weight over i. Exact if BP-OSD returns min-weight; otherwise an upper
    bound on D.
    """
    H, A, _, probs = dem_check_action_matrices(circuit)
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
    circuit: stim.Circuit,
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
    """
    H, A, _, probs = dem_check_action_matrices(circuit)
    M, N = H.shape
    K = A.shape[0]
    if priors is None:
        priors = probs
    if D is None:
        D = compute_distance(circuit, osd_order=osd_order, max_iter=max_iter,
                             priors=priors, workers=workers).distance

    n_systematic = (1 << K) - 1 if (systematic and K <= 20) else 0

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
                        found_p.add(frozenset(res[1]))
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
                        found_p.add(frozenset(res[1]))
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
                found.add(frozenset(np.flatnonzero(corr).tolist()))
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
                no_new = 0
                continue
        no_new += 1
        if no_new >= patience:
            break
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
) -> OnsetResult:
    """Exact optimal onset fraction f*(D/2) for a circuit with even distance (§4.3).

    Convenience wrapper: computes D (if not given), searches L(D) (if not given),
    then evaluates |F(D/2)| via :func:`min_weight_fail_count`. The result anchors
    the Technique-I ansatz fit via ``w0=onset`` and ``f0=onset_fraction``.
    """
    H, A, mult, priors = dem_check_action_matrices(circuit)
    if distance is None:
        distance = compute_distance(circuit, osd_order=osd_order, max_iter=max_iter, priors=priors).distance
    if logicals is None:
        logicals = find_min_weight_logicals(
            circuit, distance, max_trials=max_trials, osd_order=osd_order,
            max_iter=max_iter, priors=priors, seed=seed,
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
