"""Smoke tests for the BB(6) Figure-10 reproduction job (experiments/bravyi/bb6_fig10_sweep.py).

These run the *entire* multi-hour pipeline at tiny scale (Config.smoke): onset scan,
Technique II (min-weight), the checkpointed IS sweep, Technique I (ansatz), Technique III
(splitting), and plotting. The point is to catch errors in any code path BEFORE the real
multi-hour production run is launched — not to check accuracy.

They need the real decoder stack (relay_bp + ldpc); if either is missing the whole module
skips, so the suite stays green on machines without the Rust-built relay_bp.

Run:  python -m pytest tests/test_bb6_fig10.py -v
"""
import pathlib
import sys

import numpy as np
import pytest

# Technique III needs relay_bp (Rust); Technique II needs ldpc (BP-OSD). Skip all if absent.
pytest.importorskip("relay_bp", reason="relay_bp not installed (build from pinned commit)")
pytest.importorskip("ldpc", reason="ldpc not installed")

# Make the production driver importable (conftest already put src/ on the path).
_BB6_DIR = pathlib.Path(__file__).parent.parent / "experiments" / "bravyi"
sys.path.insert(0, str(_BB6_DIR))
import bb6_fig10_sweep as bb6  # noqa: E402


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Run the full pipeline once at smoke scale; share the results across tests."""
    cfg = bb6.Config.smoke()
    outdir = tmp_path_factory.mktemp("bb6_smoke")
    results = bb6.run_all(cfg, outdir, do_onset=True, do_split=True, do_plot=True)
    return cfg, outdir, results


# --------------------------------------------------------------------------- config
def test_smoke_config_is_tiny():
    cfg = bb6.Config.smoke()
    assert cfg.label == "smoke"
    assert cfg.shots_per_weight <= 50
    assert len(cfg.weights) <= 20
    # smoke must use the SAME relay knobs the decoder factory reads (so the path is real)
    dec = bb6.make_decoder(cfg)
    assert dec._num_sets == cfg.relay_num_sets
    assert dec._gamma0 == cfg.relay_gamma0


def test_circuit_and_decoder_roundtrip():
    cfg = bb6.Config.smoke()
    circuit = bb6.build_circuit(cfg)
    # BB(6) = [[72,12,6]] → 12 logical observables, distance-6 memory experiment.
    assert circuit.num_observables == 12
    dec = bb6.make_decoder(cfg)
    dec.setup(circuit)
    sampler = circuit.compile_detector_sampler(seed=0)
    det, obs = sampler.sample(16, separate_observables=True)
    pred = dec.decode_batch(det)
    assert pred.shape == obs.shape


# ----------------------------------------------------------------- Technique II
def test_technique_ii_distance_and_onset(smoke_run):
    _, _, results = smoke_run
    t2 = results["tech2"]
    assert t2["distance"] == 6              # BB(6) circuit fault distance
    assert t2["onset"] == 3                 # w0 = D/2 (even D → Proposition-1 onset)
    assert 0.0 < t2["onset_fraction"] < 1.0


# ----------------------------------------------------------------- IS sweep + reweight
def test_is_sweep_spectrum_and_curve(smoke_run):
    cfg, _, results = smoke_run
    res = results["is"]
    # The fixture runs with do_onset=True, so the plan is the onset-derived contiguous
    # block (NOT cfg.weights). Check it is a non-empty run of consecutive weights.
    w = list(res.spectrum.weights)
    assert len(w) > 0
    assert w == list(range(w[0], w[-1] + 1)), "IS weights must be a contiguous block"
    assert res.P_logical.shape == (cfg.n_p,)
    assert np.all(np.isfinite(res.P_logical))
    assert np.all(res.P_logical >= 0.0)
    assert res.spectrum.n_expanded > 0


def test_checkpoint_resume_is_deterministic(tmp_path):
    """Running the sweep twice in one dir resumes from the checkpoint and gives identical
    results — the property the multi-hour job relies on after a crash/restart."""
    cfg = bb6.Config.smoke()
    plan = list(cfg.weights)
    first = bb6.run_is_sweep(cfg, tmp_path, plan)
    assert (tmp_path / "bb6.spectrum.json").exists()
    second = bb6.run_is_sweep(cfg, tmp_path, plan)   # all weights already done → pure resume
    assert first.spectrum.failures == second.spectrum.failures
    np.testing.assert_array_equal(first.P_logical, second.P_logical)


def test_checkpoint_plan_mismatch_raises(tmp_path):
    """A checkpoint made with a different plan/shots/seed must refuse to silently resume."""
    cfg = bb6.Config.smoke()
    bb6.run_is_sweep(cfg, tmp_path, list(cfg.weights))
    with pytest.raises(SystemExit):
        bb6.run_is_sweep(cfg, tmp_path, [w + 100 for w in cfg.weights])  # different plan


def test_reweight_shapes_no_decoder():
    """reweight_spectrum is pure numpy — exercise it directly (no decoder needed)."""
    from importance_sampling import FailureSpectrum
    spec = FailureSpectrum(weights=[2, 3, 4], trials=[10, 10, 10], failures=[0, 1, 3],
                           n_expanded=500, q_base=1e-3, p_ref=0.003)
    p = np.logspace(-4, -2, 7)
    out = bb6.reweight_spectrum(spec, p)
    assert out.P_logical.shape == (7,)
    assert np.all(np.isfinite(out.P_logical)) and np.all(out.P_logical >= 0)


# ----------------------------------------------------------------- Technique I
def test_technique_i_ansatz(smoke_run):
    cfg, _, results = smoke_run
    t1 = results["tech1"]
    P_ext = np.asarray(t1["P_ext"])
    assert P_ext.shape == (cfg.n_p,)
    assert np.all(np.isfinite(P_ext)) and np.all(P_ext > 0)
    # f3 ansatz exposes w0, f0, gamma; pinned from Technique II in smoke (pin_onset=True)
    assert {"w0", "f0", "gamma"} <= set(t1["params"])
    assert t1["pinned"] is True
    assert t1["params"]["w0"] == pytest.approx(results["tech2"]["onset"])


# ----------------------------------------------------------------- Technique III
def test_technique_iii_splitting(smoke_run):
    cfg, _, results = smoke_run
    t3 = results["tech3"]
    assert len(t3["p_ladder"]) == cfg.split_n_levels + 1
    assert len(t3["P_logical"]) == cfg.split_n_levels + 1
    assert np.all(np.isfinite(t3["P_logical"]))


# ----------------------------------------------------------------- artifacts
def test_all_output_files_written(smoke_run):
    _, outdir, _ = smoke_run
    for name in ["config.json", "onset.json", "distance.json", "bb6.spectrum.json",
                 "ansatz_fit.json", "splitting.json", "bb6_fig10.npz", "bb6_fig10.png"]:
        assert (outdir / name).exists(), f"missing output artifact {name}"
