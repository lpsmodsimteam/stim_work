"""Tests for Technique II — min-weight properties (arXiv:2511.15177 §4)."""
import itertools
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from surface_code_sim import SurfaceCodeSimulator, ErrorModel
from min_weight import compute_distance, min_weight_fail_count


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
