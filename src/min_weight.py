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
) -> Set[FrozenSet[int]]:
    """Search for L(D), the set of weight-D logical bitstrings (paper §4.2).

    Repeatedly appends a random nonzero combination of A's rows to H, decodes the
    forcing syndrome, and keeps weight-D logicals. Stops after ``patience``
    consecutive trials with no new logical, or ``max_trials`` total. Returns each
    logical as a frozenset of fault indices (its support). For larger codes this
    converges to a lower bound on |L(D)| (Fig. 6).
    """
    H, A, _, probs = dem_check_action_matrices(circuit)
    M, N = H.shape
    K = A.shape[0]
    if priors is None:
        priors = probs
    if D is None:
        D = compute_distance(circuit, osd_order=osd_order, max_iter=max_iter, priors=priors).distance

    rng = np.random.default_rng(seed)
    syndrome = np.zeros(M + 1, dtype=np.uint8)
    syndrome[M] = 1
    found: Set[FrozenSet[int]] = set()
    no_new = 0
    for _ in range(max_trials):
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
