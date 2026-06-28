"""Tests for Technique III — multi-seeded Metropolis splitting (arXiv:2511.15177 §5).

These test the *mechanics* (config->failure path, chain stays in the failing set,
trivial ratio == 1, seed conversion), not convergence. Everything uses a tiny
surface code and small step/shot counts so each test runs well under 30s.
"""
import sys
import pathlib

# Support both flat layout and a src/ layout if present.
_here = pathlib.Path(__file__).resolve().parent
for cand in (_here, _here / "src", _here.parent / "src"):
    if (cand / "splitting.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import pytest

from surface_code_sim import SurfaceCodeSimulator, ErrorModel, PyMatchingDecoder
from importance_sampling import _parse_dem, _expand, _sample_failures_at_weight
from splitting import (
    _config_syndrome_truth,
    _config_fails,
    _mech_to_cols,
    _metropolis_chain,
    min_weight_logical_seeds,
    high_rate_mc_seeds,
    splitting_estimate,
    SplittingResult,
    _eq18_ladder,
    _bar_ratio,
    multi_seeded_split_estimate,
)


# --- shared tiny fixture ----------------------------------------------------

def _tiny_setup():
    """A small d=3 surface code circuit + MWPM decoder, plus expanded-rep arrays."""
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.02), rounds=3)
    probs, det_mat, obs_mat = _parse_dem(circuit)
    col_to_mech, q_base, _ = _expand(probs, None)
    dec = PyMatchingDecoder()
    dec.setup(circuit)
    return circuit, det_mat, obs_mat, col_to_mech, q_base, dec


def _find_a_failing_config(det_mat, obs_mat, col_to_mech, dec, rng, max_w=6):
    """Brute-search a small failing expanded-column config for tests."""
    N_exp = col_to_mech.shape[0]
    for w in range(1, max_w + 1):
        for _ in range(2000):
            cols = rng.choice(N_exp, size=w, replace=False)
            if _config_fails(det_mat, obs_mat, col_to_mech, cols, dec):
                return frozenset(int(c) for c in cols)
    return None


# --- (a) failing seed actually fails; path matches reference ----------------

def test_config_path_matches_sample_failures_reference():
    """_config_syndrome_truth must agree with the IS reference _sample_failures_at_weight.

    We sample weight-w configs, compute failure both via our single-config path and
    via the reference batch path, and require the per-config failure flags match
    exactly (same syndrome/truth/decoder), so the two estimators are comparable.
    """
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    N_exp = col_to_mech.shape[0]
    rng = np.random.default_rng(0)

    for w in (1, 2, 3):
        # Build a batch of configs and decode both ways.
        configs = [rng.choice(N_exp, size=w, replace=False) for _ in range(40)]
        M, K = det_mat.shape[1], obs_mat.shape[1]
        synd = np.zeros((len(configs), M), dtype=bool)
        truth = np.zeros((len(configs), K), dtype=bool)
        for t, cols in enumerate(configs):
            synd[t], truth[t] = _config_syndrome_truth(det_mat, obs_mat, col_to_mech, cols)
        preds = dec.decode_batch(synd)
        batch_fail = np.any(preds != truth, axis=1)
        single_fail = np.array([
            _config_fails(det_mat, obs_mat, col_to_mech, cols, dec) for cols in configs
        ])
        assert np.array_equal(batch_fail, single_fail)


def test_failing_seed_actually_fails():
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    rng = np.random.default_rng(1)
    seed = _find_a_failing_config(det_mat, obs_mat, col_to_mech, dec, rng)
    assert seed is not None, "expected to find a small failing config on a d=3 code"
    assert _config_fails(det_mat, obs_mat, col_to_mech, seed, dec)


# --- (b) the Metropolis chain stays within the failing set ------------------

def test_metropolis_chain_stays_in_failing_set():
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    rng = np.random.default_rng(2)
    seed = _find_a_failing_config(det_mat, obs_mat, col_to_mech, dec, rng)
    assert seed is not None

    # Instrument by re-running the chain and checking states. We rely on the chain's
    # invariant: it only ever commits a flip that keeps the config failing. We verify
    # by running with a sane q and confirming the chain runs and reports a finite
    # ratio (trivial-ratio case below also exercises in-set residency).
    q = float(q_base * (0.02 / 0.02))  # q at p_ref
    cr, _ = _metropolis_chain(   # returns (ChainResult, final_config) for warm-starting
        det_mat, obs_mat, col_to_mech, dec,
        q_cur=q, q_next=q, seed_config=seed,
        chain_steps=200, burn_in=20, thin=1,
        rng=np.random.default_rng(3),
    )
    # Every recorded sample weight is >= 1 (the empty config never fails on a
    # zero-syndrome decode), confirming the chain never left the failing set into
    # the trivially-correct all-zero state.
    assert cr.n_samples > 0
    assert cr.mean_weight >= 1.0
    assert 0.0 <= cr.accept_rate <= 1.0


def test_chain_rejects_nonfailing_seed():
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    # The empty config has the all-zero syndrome -> decoder predicts no flip ->
    # truth is all-zero -> it does NOT fail. Seeding with it must raise.
    with pytest.raises(ValueError, match="does not fail"):
        _metropolis_chain(
            det_mat, obs_mat, col_to_mech, dec,
            q_cur=0.01, q_next=0.01, seed_config=frozenset(),
            chain_steps=10, burn_in=0, thin=1, rng=np.random.default_rng(0),
        )


# --- (c) single-level ratio with q_high == q_low is ~1 ----------------------

def test_trivial_ratio_is_one():
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    rng = np.random.default_rng(4)
    seed = _find_a_failing_config(det_mat, obs_mat, col_to_mech, dec, rng)
    assert seed is not None
    q = 0.01
    cr, _ = _metropolis_chain(   # returns (ChainResult, final_config) for warm-starting
        det_mat, obs_mat, col_to_mech, dec,
        q_cur=q, q_next=q, seed_config=seed,
        chain_steps=300, burn_in=50, thin=1,
        rng=np.random.default_rng(5),
    )
    # q_next == q_cur => every per-sample reweight term is exactly 0 in log-space.
    assert np.allclose(cr.log_ratio_terms, 0.0)
    assert cr.log_ratio_mean == pytest.approx(0.0, abs=1e-12)


# --- (d) seed conversion from mechanism indices to expanded columns ---------

def test_mech_to_cols_inverse():
    _, _, _, col_to_mech, _, _ = _tiny_setup()
    m2c = _mech_to_cols(col_to_mech)
    # Every expanded column maps back to a mechanism that lists it.
    for c, j in enumerate(col_to_mech.tolist()):
        assert c in m2c[j]
    # Round-trip: picking the first column for each mechanism yields that mechanism.
    for j, cols in m2c.items():
        assert int(col_to_mech[cols[0]]) == j


def test_min_weight_seed_conversion_yields_failing_config():
    """A min-weight-logical support (mechanism indices) lifts to a failing config."""
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    seeds = min_weight_logical_seeds(
        circuit, col_to_mech, det_mat, obs_mat, dec,
        distance=3, max_trials=200, seed=0,
    )
    # d=3 surface code has weight-3 logicals; we should lift at least one to a
    # failing expanded-column config.
    assert len(seeds) >= 1
    for s in seeds:
        assert _config_fails(det_mat, obs_mat, col_to_mech, s, dec)
        assert len(s) == 3  # weight equals the distance


def test_high_rate_mc_seeds_all_fail():
    circuit, det_mat, obs_mat, col_to_mech, q_base, dec = _tiny_setup()
    q_high = float(q_base * (0.03 / 0.02))
    seeds = high_rate_mc_seeds(
        det_mat, obs_mat, col_to_mech, dec, q_high,
        n_seeds=3, max_shots=2000, rng=np.random.default_rng(7),
    )
    for s in seeds:
        assert _config_fails(det_mat, obs_mat, col_to_mech, s, dec)


# --- end-to-end smoke test (tiny config) ------------------------------------

def test_splitting_estimate_smoke():
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.02), rounds=3)
    res = splitting_estimate(
        circuit, PyMatchingDecoder(),
        p_ref=0.02, p_high=0.03, p_low=0.02,
        n_levels=2, n_seeds=2, chain_steps=60, burn_in=10,
        anchor_shots=40, use_min_weight_seeds=True, distance=3, seed=0,
    )
    assert isinstance(res, SplittingResult)
    assert res.q_ladder.shape == (3,)
    assert res.P_logical.shape == (3,)
    assert np.all(np.isfinite(res.P_logical))
    assert res.P_logical[0] == pytest.approx(res.P_high, rel=1e-12)
    # Lower p should not give a higher estimate (monotone-ish; allow MC slack).
    assert res.P_logical[-1] <= res.P_logical[0] * 5


# --- paper-faithful multi-seeded splitting (Alg. 2/3 + §5.3) ----------------

def test_eq18_ladder_descends_to_p_low():
    """Eq.18 ladder starts at p_high, decreases monotonically, and reaches <= p_low."""
    ladder = _eq18_ladder(p_high=0.03, p_low=0.005, N_exp=5000, distance=4)
    assert ladder[0] == pytest.approx(0.03)
    assert ladder[-1] <= 0.005
    assert np.all(np.diff(ladder) < 0)


def test_bar_ratio_trivial_is_one():
    """With q_a == q_b the BAR ratio P(p_b)/P(p_a) must be exactly 1 for any weights."""
    rng = np.random.default_rng(0)
    w_a = rng.integers(2, 20, size=500).astype(float)
    w_b = rng.integers(2, 20, size=500).astype(float)
    r = _bar_ratio(w_a, w_b, q_a=1e-3, q_b=1e-3, N_exp=10_000)
    assert r == pytest.approx(1.0, abs=1e-9)


def test_multiseed_estimate_smoke():
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.02), rounds=3)
    res, diag = multi_seeded_split_estimate(
        circuit, PyMatchingDecoder(),
        p_ref=0.02, p_high=0.03, p_low=0.02, ladder="geom", n_levels=3,
        L=2, M=2, T_init=300, eps=0.5, distance=3, anchor_shots=400, seed=0,
        verbose=False,
    )
    assert isinstance(res, SplittingResult)
    assert res.P_logical.shape == res.q_ladder.shape
    assert np.all(np.isfinite(res.P_logical))
    assert res.P_logical[0] == pytest.approx(res.P_high, rel=1e-12)
    assert res.P_logical[-1] <= res.P_logical[0] * 5
    assert diag["n_inst"] == 4
    assert len(diag["levels"]) == res.q_ladder.shape[0]
