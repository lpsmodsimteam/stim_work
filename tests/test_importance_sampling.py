"""Tests for the failure-spectrum ansatz / extrapolation (arXiv:2511.15177, Technique I)."""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from importance_sampling import (
    FailureSpectrum,
    AnsatzFit,
    failure_spectrum_ansatz,
    fit_failure_spectrum,
    logical_error_rate_from_ansatz,
)


def test_ansatz_zero_below_onset():
    w = np.arange(0, 10)
    f = failure_spectrum_ansatz(w, w0=5, f0=1e-2, a=0.5, model="f3", gamma=4.0)
    assert np.all(f[:5] == 0.0)
    assert np.all(f[5:] > 0.0)


def test_ansatz_value_at_onset_is_f0():
    # At w = w0 the power term is 1, so f(w0) = a(1 - exp(-f0/a)) ≈ f0 for small f0.
    for model, extra in [("f2", {}), ("f3", {"gamma": 4.0}),
                         ("f5", {"gamma1": 4.0, "gamma2": 1.0, "wc": 12.0})]:
        f0 = 1e-3
        val = float(failure_spectrum_ansatz(5, w0=5, f0=f0, a=0.5, model=model, **extra))
        assert val == pytest.approx(f0, rel=1e-2)


def test_ansatz_monotonic_and_saturates():
    w = np.arange(5, 400)
    a = 1.0 - 2.0 ** -3
    f = failure_spectrum_ansatz(w, w0=5, f0=1e-3, a=a, model="f3", gamma=5.0)
    assert np.all(np.diff(f) >= -1e-12)      # monotonically increasing
    assert f[-1] == pytest.approx(a, rel=1e-3)  # saturates at a = 1 - 2^-K


def test_fit_recovers_synthetic_spectrum():
    # Draw a known f3 spectrum, sample binomial failures, refit, recover params.
    rng = np.random.default_rng(0)
    w0_true, f0_true, gamma_true, a = 5.0, 1e-2, 4.0, 0.5
    weights = list(range(5, 31))
    T = 200_000
    f_true = failure_spectrum_ansatz(np.array(weights, float), w0=w0_true,
                                     f0=f0_true, a=a, model="f3", gamma=gamma_true)
    failures = [int(rng.binomial(T, min(p, 1.0))) for p in f_true]
    spec = FailureSpectrum(
        weights=weights, trials=[T] * len(weights), failures=failures,
        n_expanded=10_000, q_base=1e-3, p_ref=5e-3,
    )
    # Pin w0 (it is known here); recover f0 and gamma.
    fit = fit_failure_spectrum(spec, K=1, model="f3", w0=w0_true)
    assert fit.params["f0"] == pytest.approx(f0_true, rel=0.15)
    assert fit.params["gamma"] == pytest.approx(gamma_true, rel=0.15)


def test_P_ansatz_matches_brute_transform():
    # logical_error_rate_from_ansatz must equal a direct math.comb transform.
    fit = AnsatzFit(
        model="f3", params={"w0": 3.0, "f0": 0.1, "gamma": 3.0}, free_names=[],
        a=0.5, param_cov=None, n_expanded=40, q_base=1.0, p_ref=1.0,
        n_points=0, cost=0.0,
    )
    N, b = 40, 1.0
    p_values = np.array([0.01, 0.05, 0.1])
    got = logical_error_rate_from_ansatz(fit, p_values, b=b)

    for p, g in zip(p_values, got):
        q = p / b
        brute = sum(
            float(failure_spectrum_ansatz(w, w0=3.0, f0=0.1, a=0.5, model="f3", gamma=3.0))
            * math.comb(N, w) * q ** w * (1 - q) ** (N - w)
            for w in range(3, N + 1)
        )
        assert g == pytest.approx(brute, rel=1e-9)


def test_adaptive_early_stop_skips_deep_zero_bins():
    """stop_after_zero_bins: an empty max-budget bin ends the descent; lower weights are omitted
    from the spectrum (equivalent to sampling them: zero-failure bins reweight to exactly 0).
    Surface d=5 + MWPM: min-weight decoding makes f(1)=f(2)=0 exactly (onset w0=3)."""
    from surface_code_sim import SurfaceCodeSimulator, ErrorModel, PyMatchingDecoder
    from importance_sampling import importance_sample_adaptive

    circ = SurfaceCodeSimulator(distance=5).build_circuit(ErrorModel.symmetric(0.01), rounds=2)
    res = importance_sample_adaptive(circ, PyMatchingDecoder(), p_ref=0.01, p_values=[0.01],
                                     weights=list(range(1, 9)), target_failures=20,
                                     shots_min=4, shots_max=200,
                                     stop_after_zero_bins=1, seed=0)
    ws = list(res.spectrum.weights)
    assert 1 not in ws                       # skipped: descent stopped at the first empty clamp bin
    assert any(F > 0 for F in res.spectrum.failures)      # the sampled part saw real failures
    assert res.P_logical[0] > 0
