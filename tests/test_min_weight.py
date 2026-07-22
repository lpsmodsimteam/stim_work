"""Tests for Technique II — min-weight properties (arXiv:2511.15177 §4)."""
import itertools
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from surface_code_sim import SurfaceCodeSimulator, ErrorModel
from min_weight import (compute_distance, min_weight_fail_count, find_min_weight_logicals,
                        dem_check_action_matrices)


def test_decimation_finds_valid_min_weight_logicals():
    """Decimation (paper §4.2) must return VALID weight-D logicals (H@x=0, A@x!=0, |x|=D)."""
    circ = SurfaceCodeSimulator(distance=3).build_circuit(ErrorModel.symmetric(0.02), rounds=2)
    H, A, mult, probs = dem_check_action_matrices(circ)
    N = H.shape[1]
    D = compute_distance(matrices=(H, A, probs)).distance
    res = find_min_weight_logicals(matrices=(H, A, probs), D=D, max_trials=150,
                                   systematic=False, decimate=True, workers=1, seed=0)
    assert len(res) >= 1, "decimation found no weight-D logicals"
    for s in res:
        v = np.zeros(N, dtype=np.uint8); v[list(s)] = 1
        assert len(s) == D
        assert not (H @ v % 2).any()    # respects all stabilizer checks
        assert (A @ v % 2).any()        # nontrivial logical action


# --- brute-force references (tiny codes only) -------------------------------

def _brute_distance_and_logicals(H, A):
    N = H.shape[1]
    best = None
    for bits in itertools.product([0, 1], repeat=N):
        e = np.array(bits, np.uint8)
        if e.sum() and not (H @ e % 2).any() and (A @ e % 2).any():
            best = e.sum() if best is None else min(best, e.sum())
    L = set()
    if best is not None:
        for bits in itertools.product([0, 1], repeat=N):
            e = np.array(bits, np.uint8)
            if e.sum() == best and not (H @ e % 2).any() and (A @ e % 2).any():
                L.add(frozenset(np.flatnonzero(e).tolist()))
    return best, L


def _brute_fail_count(H, A, D):
    """|F(D/2)| by definition: among weight-D/2 errors whose syndrome has min weight
    D/2, the max-class action succeeds and the rest fail."""
    N = H.shape[1]
    half = D // 2
    groups = {}
    for bits in itertools.product([0, 1], repeat=N):
        e = np.array(bits, np.uint8)
        if e.sum() != half:
            continue
        sig = (H @ e % 2).tobytes()
        act = (A @ e % 2).tobytes()
        groups.setdefault(sig, {})
        groups[sig][act] = groups[sig].get(act, 0) + 1
    fails = 0
    for sig, acts in groups.items():
        sigv = np.frombuffer(sig, np.uint8)
        # skip syndromes correctable below weight D/2
        correctable = any(
            ((H @ _e % 2) == sigv).all()
            for w in range(half)
            for _e in (_onehot(N, combo) for combo in itertools.combinations(range(N), w))
        )
        if correctable:
            continue
        sizes = list(acts.values())
        fails += sum(sizes) - max(sizes)
    return fails


def _onehot(N, combo):
    e = np.zeros(N, np.uint8)
    for i in combo:
        e[i] = 1
    return e


def test_prop1_matches_brute_force_on_small_codes():
    """min_weight_fail_count (Proposition 1) == direct max-class min-weight decoder."""
    validated = 0
    for seed in range(400):
        rs = np.random.default_rng(seed)
        H = rs.integers(0, 2, size=(3, 8), dtype=np.uint8)
        A = rs.integers(0, 2, size=(1, 8), dtype=np.uint8)
        D, L = _brute_distance_and_logicals(H, A)
        if D is None or D < 2 or D % 2 != 0 or not L:
            continue
        prop, n_exp = min_weight_fail_count(H, A, L)
        assert prop == _brute_fail_count(H, A, D)
        assert n_exp == H.shape[1]  # multipliers default to 1
        validated += 1
        if validated >= 6:
            break
    assert validated >= 6, "did not find enough small even-distance test codes"


def test_min_weight_fail_count_rejects_odd_distance():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    A = np.array([[1, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="even D"):
        min_weight_fail_count(H, A, [frozenset({0, 1, 2})])  # D=3 (odd)


def test_distance_surface_d3():
    circ = SurfaceCodeSimulator(distance=3).build_circuit(ErrorModel.symmetric(0.01), rounds=3)
    res = compute_distance(circ, osd_order=20, max_iter=400)
    assert res.distance == 3
    assert res.onset == 2  # ceil(D/2); matches the ansatz fit's auto-found w0
    # every witness is a genuine min-weight logical
    from min_weight import dem_check_action_matrices
    H, A, _, _ = dem_check_action_matrices(circ)
    for w in res.witnesses:
        assert int(w.sum()) == 3
        assert not (H @ w % 2).any() and (A @ w % 2).any()


def test_fail_count_from_restrictions_matches_enumerating_wrapper():
    """The shared Prop-1 vote helper must agree with min_weight_fail_count on the same logicals.

    (This replaces the old bb144_onset_fraction.py --self-test; 100 is the frozen golden value
    for this seed-0 synthetic, so a semantics drift in either routine fails loudly.)
    """
    from itertools import combinations
    from min_weight import fail_count_from_restrictions

    rng = np.random.default_rng(0)
    M, K, N = 6, 2, 14
    H = (rng.random((M, N)) < 0.4).astype(np.uint8)
    A = (rng.random((K, N)) < 0.4).astype(np.uint8)
    mult = rng.integers(1, 4, size=N).astype(np.int64)
    D = 4
    logicals = [frozenset(map(int, rng.choice(N, size=D, replace=False))) for _ in range(20)]

    ref, _ = min_weight_fail_count(H, A, logicals, mult)
    uniq = {tuple(sorted(r)) for s in logicals for r in combinations(sorted(s), D // 2)}
    assert fail_count_from_restrictions(H, A, mult, uniq) == ref == 100


def test_fail_count_budget_guard_raises():
    """Over-budget enumerations must raise BEFORE committing memory (even and odd variants)."""
    from min_weight import InfeasibleEnumerationError, min_weight_fail_count_odd

    H = np.zeros((2, 30), dtype=np.uint8)
    A = np.ones((1, 30), dtype=np.uint8)
    logicals_even = [frozenset(range(i, i + 4)) for i in range(0, 24, 4)]      # weight-4
    with pytest.raises(InfeasibleEnumerationError):
        min_weight_fail_count(H, A, logicals_even, max_restrictions=3)
    fails, _ = min_weight_fail_count(H, A, logicals_even, max_restrictions=10**6)  # under budget: runs

    logicals_odd = [frozenset(range(i, i + 3)) for i in range(0, 24, 3)]       # weight-3 (odd)
    with pytest.raises(InfeasibleEnumerationError):
        min_weight_fail_count_odd(H, A, logicals_odd, logicals_even, max_restrictions=3)


def test_bb_72_4_8_registry_params():
    """[[72,4,8]]: same polynomials as BB_18_4_4 at (l,m)=(6,6) must give n=72, k=4 (d=8 is the
    exact split-MITM result recorded in bb_code_sim.py; too slow to re-verify here)."""
    from bb_code_sim import BB_18_4_4, BB_72_4_8, build_parity_checks, _gf2_rank

    assert (BB_72_4_8.a_exps, BB_72_4_8.b_exps) == (BB_18_4_4.a_exps, BB_18_4_4.b_exps)
    HX, HZ = build_parity_checks(BB_72_4_8)
    n = HX.shape[1]
    assert n == 72
    assert not (HX @ HZ.T % 2).any()                      # CSS commutation
    assert n - _gf2_rank(HX) - _gf2_rank(HZ) == 4         # k = 4


def test_symmetry_prune_matches_full_sweep():
    """Symmetry-pruned systematic L(D) sweep (one functional per translation orbit) must
    reproduce the full 2^K-1 sweep's L(D) EXACTLY on a toric BB code. Guards the ~orbit-size
    speedup used by run_technique_ii (mw_symmetry_prune)."""
    from bb_code_sim import BBCodeParams, BBCodeSimulator
    from min_weight import build_circuit_translation_perms, symmetry_orbit_representatives

    P = BBCodeParams(l=3, m=3, a_exps=[(1, 0), (0, 0), (0, 2)],
                     b_exps=[(0, 1), (0, 0), (2, 0)], distance=4)
    circ = BBCodeSimulator(P).build_circuit(ErrorModel.symmetric(0.005), rounds=2)
    H, A, mult, priors = dem_check_action_matrices(circ, sector=None)
    perms = build_circuit_translation_perms(circ, H, l=3, m=3, verbose=False)
    D = compute_distance(circ, priors=priors, sector=None).distance

    reps = symmetry_orbit_representatives(H, A, perms)
    assert 0 < len(reps) < (1 << A.shape[0]) - 1, "orbit reps should be a strict subset of all classes"

    kw = dict(D=D, max_trials=0, priors=priors, sector=None, symmetry_perms=perms, workers=1)
    L_full = find_min_weight_logicals(circ, **kw)
    L_prun = find_min_weight_logicals(circ, systematic_masks=reps, **kw)
    assert L_prun == L_full and len(L_full) > 0, "pruned systematic sweep != full sweep L(D)"
