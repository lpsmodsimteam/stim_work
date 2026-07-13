"""lambda_analysis: synthetic-spectrum unit tests + regression against the K=4 methods cache.

This module is the campaign's gate arithmetic (G1/G4a read its outputs), so the tests pin its
behavior analytically where a closed form exists and against the validated K=4 notebook cache
where one doesn't.
"""
import json
import math
import pathlib

import numpy as np
import pytest

from importance_sampling import FailureSpectrum, reweight_spectrum
from lambda_analysis import (fill_spectrum, pool_spectra, reweight_filled, rw_stats, eps_stats,
                             per_round, cycles_of, mass_window_p_max, zero_bin_fraction,
                             inv_lambda_stats, crossing_p, pseudo_threshold, lambda_curve,
                             threshold_from_crossing, verdict, lambda_decomposition, load_run)

K4_CACHE = pathlib.Path(__file__).resolve().parents[1] / "runs" / "error_model_comparison_18_4_4"


def flat_spec(N=40, q=0.002, p_ref=0.01, f=0.25, T=1000, weights=None):
    """Spectrum with constant failure fraction f on the given weights (default all 1..N)."""
    ws = list(weights) if weights is not None else list(range(1, N + 1))
    return FailureSpectrum(weights=ws, trials=[T] * len(ws),
                           failures=[int(round(f * T))] * len(ws),
                           n_expanded=N, q_base=q, p_ref=p_ref)


# ------------------------------ closed forms ---------------------------------
def test_rw_stats_closed_form():
    # constant f over ALL weights: LER(p) = f * (1 - (1-q(p))^N)
    N, q, f = 40, 0.002, 0.25
    s = flat_spec(N=N, q=q, f=f)
    for p in (0.002, 0.01, 0.04):
        qq = q * p / s.p_ref
        v, se, head = rw_stats(s, p)
        assert v == pytest.approx(f * (1.0 - (1.0 - qq) ** N), rel=1e-12)
        assert se > 0 and head == pytest.approx(0.0, abs=1e-15)   # no zero bins -> no headroom


def test_headroom_counts_zero_bins():
    s = flat_spec(N=10, f=0.2, T=100)
    z = FailureSpectrum(weights=s.weights, trials=s.trials,
                        failures=[0] + s.failures[1:],            # empty the w=1 bin
                        n_expanded=s.n_expanded, q_base=s.q_base, p_ref=s.p_ref)
    v, _, head = rw_stats(z, 0.001)
    v3 = FailureSpectrum(weights=s.weights, trials=s.trials, failures=[3] + s.failures[1:],
                         n_expanded=s.n_expanded, q_base=s.q_base, p_ref=s.p_ref)
    assert v + head == pytest.approx(rw_stats(v3, 0.001)[0], rel=1e-12)


def test_fill_spectrum_recovers_strided_mass():
    # q chosen so the binomial mass sits at mu = N*q = 24 — deep inside the stride-2 tail
    full = flat_spec(N=60, q=0.4, f=0.3)
    strided = flat_spec(N=60, q=0.4, f=0.3, weights=[1, 2, 3, 4, 5] + list(range(6, 61, 2)))
    p_hi = 0.01
    raw = float(reweight_spectrum(strided, [p_hi]).P_logical[0])
    filled = float(reweight_filled(strided, [p_hi])[0])
    exact = float(reweight_spectrum(full, [p_hi]).P_logical[0])
    assert filled == pytest.approx(exact, rel=1e-9)     # constant f: fill is exact
    assert raw < 0.75 * exact                            # unfilled undercounts badly
    # contiguous input passes through unchanged
    assert fill_spectrum(full).weights == full.weights


def test_pool_spectra_adds_counts_and_rejects_mismatch():
    a, b = flat_spec(T=100), flat_spec(T=300)
    pooled = pool_spectra(a, b)
    assert pooled.trials[0] == 400 and pooled.failures[0] == a.failures[0] + b.failures[0]
    with pytest.raises(ValueError, match="different expanded"):
        pool_spectra(a, flat_spec(q=0.003))


def test_per_round_and_eps_stats():
    assert per_round(0.19, 2) == pytest.approx(1 - math.sqrt(0.81))
    s = flat_spec()
    L, seL, _ = rw_stats(s, 0.01)
    e, seE, _ = eps_stats(s, 0.01, 4)
    assert e == pytest.approx(1 - (1 - L) ** 0.25, rel=1e-12)
    assert seE < seL                                     # per-round transform contracts errors


def test_mass_window_and_zero_bins():
    s = flat_spec(N=40, q=0.002, p_ref=0.01)
    p_max = mass_window_p_max(s)
    mu = s.n_expanded * s.q_base * p_max / s.p_ref
    assert mu + 4 * math.sqrt(mu) == pytest.approx(max(s.weights), rel=1e-9)
    z = FailureSpectrum(weights=[1, 2, 3, 4], trials=[10] * 4, failures=[0, 0, 1, 5],
                        n_expanded=40, q_base=0.002, p_ref=0.01)
    assert zero_bin_fraction(z) == 0.5


# ------------------------------ ratios / crossings ----------------------------
def test_crossings_analytic():
    pg = np.geomspace(1e-4, 1e-1, 200)
    y1, y2 = 3.0 * pg**2, 10.0 * pg**3                   # cross at p = 0.3
    assert crossing_p(pg, y1, y2) is None                # 0.3 outside the grid
    y2 = 300.0 * pg**3                                   # cross at p = 0.01
    assert crossing_p(pg, y1, y2) == pytest.approx(0.01, rel=1e-3)
    assert pseudo_threshold(pg, 100 * pg**2) == pytest.approx(0.01, rel=1e-3)


def test_lambda_curve_masks_and_threshold_refuses_extrapolation():
    small = flat_spec(N=30, q=0.002, f=0.4)
    large = flat_spec(N=90, q=0.002, f=0.4)
    pg = np.geomspace(1e-4, 1.0, 60)                     # deliberately absurd top
    p, lam, es, el = lambda_curve(small, large, pg, 2, 4)
    assert p.max() <= min(mass_window_p_max(small), mass_window_p_max(large)) * (1 + 1e-9)
    assert np.all(np.isfinite(lam))
    out = threshold_from_crossing(small, large, pg, 2, 4)
    assert out["p_max_valid"] == pytest.approx(float(p.max()))
    assert (out["p_th"] is None) == out["bounded"]


def test_verdicts_and_decomposition():
    assert verdict(1.0, 0.6, 0.2, 2.0) == "~0 within 2σ (noise)"
    assert verdict(1.0, 0.1, -0.5, 2.0) == "sign not robust to zero-bin truncation"
    assert verdict(-1.0, 0.1, -2.0, -0.5) == "solid"
    # decomposition arithmetic: shares of identical ablations are identical, sum reported
    small, large = flat_spec(N=30, f=0.4, T=4000), flat_spec(N=90, f=0.1, T=4000)
    abl_s, abl_l = flat_spec(N=30, f=0.2, T=4000), flat_spec(N=90, f=0.02, T=4000)
    d = lambda_decomposition(small, large, lambda ch: abl_s, lambda ch: abl_l,
                             ["a", "b"], 0.003, 2, 4)
    assert d["rows"][0]["contribution"] == pytest.approx(d["rows"][1]["contribution"])
    assert d["sum_contributions"] == pytest.approx(2 * d["rows"][0]["contribution"])
    assert 1.0 / d["inv_full"] == pytest.approx(d["lambda_full"])


# ------------------------------ run adapter -----------------------------------
def test_load_run_roundtrip(tmp_path):
    ckpt = {"n_expanded": 40, "q_base": 0.002, "p_ref": 0.01, "shots_per_weight": 0,
            "seed": 42, "weights_plan": [2, 3, 4, 5],
            "trials_by_weight": {"2": 100, "3": 100, "4": 100},
            "failures_by_weight": {"2": 1, "3": 7, "4": 30}}      # w=5 not done yet (mid-run)
    cfg = {"experiment": "memory", "rounds": 6, "p_ref": 0.01, "lpu_C": 10,
           "weights": [2, 3, 4, 5]}
    (tmp_path / "spectrum.json").write_text(json.dumps(ckpt))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    run = load_run(tmp_path)
    assert run.spectrum.weights == [2, 3, 4]              # mid-run checkpoint loads cleanly
    assert run.cycles == 6 and run.done_fraction == pytest.approx(0.75)
    e, se, head = run.eps(0.003)
    assert 0 < e < 1 and se > 0 and head >= 0
    lpu = load_run.__wrapped__ if hasattr(load_run, "__wrapped__") else None  # noqa: F841
    assert cycles_of({"experiment": "lpu_x1", "lpu_C": 10, "rounds": 12}) == 10
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")


@pytest.mark.skipif(not (K4_CACHE / "tech1__full_symmetric.json").exists(),
                    reason="K=4 methods cache not present (runs/ is gitignored)")
def test_k4_lambda_regression():
    """Reproduce the validated K=4 notebook's Λ_full(5e-4) from the cached spectra."""
    def spec_of(name):
        r = json.loads((K4_CACHE / f"{name}.json").read_text(encoding="utf-8"))["result"]
        return FailureSpectrum(**r["spectrum"])
    inv = inv_lambda_stats(spec_of("tech1__full_symmetric"), spec_of("tech1_72__full_symmetric"),
                           5e-4, rounds_small=2, rounds_large=4)
    lam = 1.0 / inv.value
    # value pinned from the 2026-07-09 report execution (Λ_full = 102 ± 53); regression band
    # is deliberately tight — the cache is frozen data, so only CODE changes can move this.
    assert lam == pytest.approx(102.0, rel=0.02)
    assert 1.0 / (inv.value + 2 * inv.se) < 60           # ±σ is large: the known noise floor
    assert inv.lo <= inv.value <= inv.hi
