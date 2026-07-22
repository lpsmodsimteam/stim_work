"""Smoke tests for the config-driven fail-fast experiment framework (experiment_runner.py).

These exercise the registries, YAML loading, and a tiny end-to-end run of each P0 config's code
paths — so a config typo or a broken seam is caught locally before the cluster. Budgets are smoke
(not accuracy): every technique runs, returns finite shapes, and writes its outputs.
"""
import pathlib
import sys

# Support both flat and src/ layouts (mirrors the other tests).
_here = pathlib.Path(__file__).resolve().parent
for cand in (_here, _here / "src", _here.parent / "src"):
    if (cand / "experiment_runner.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np

import experiment_runner as er

REPO = pathlib.Path(er.__file__).resolve().parent.parent
CONFIGS = REPO / "experiments" / "configs"


def test_registries_present():
    assert {"bb6", "bb144", "bb18", "bb288"} <= set(er.CODES)
    assert {"memory", "lpu_x1", "lpu_z1", "automorphism", "joint_pauli"} <= set(er.CIRCUIT_BUILDERS)
    assert {"relay", "bposd", "pymatching"} <= set(er.DECODERS)


def test_load_p0_configs():
    for name in ("bb6_memory", "bb144_memory", "bb288_memory"):
        cfg = er.load_config(CONFIGS / f"{name}.yaml")
        assert cfg.code_name in er.CODES
        assert cfg.experiment in er.CIRCUIT_BUILDERS
        assert cfg.decoder_name in er.DECODERS
        assert cfg.techniques  # non-empty
        assert cfg.code.distance > 0          # the code property resolves
        assert cfg.rounds == cfg.code.distance  # __post_init__ default


def test_bb288_construction():
    """Two-gross code resolves to [[288,12,18]] (exponents -> n/k self-check)."""
    cfg = er.load_config(CONFIGS / "bb288_memory.yaml")
    assert cfg.code_name == "bb288"
    assert 2 * cfg.code.l * cfg.code.m == 288
    assert cfg.code.distance == 18
    assert cfg.rounds == 18


def test_bb288_technique_ii_augmentations():
    """bb288 enables the paper's §4.2 search accelerations (decimation + 2-cycle fault restriction)."""
    cfg = er.load_config(CONFIGS / "bb288_memory.yaml")
    assert cfg.mw_decimate is True
    assert cfg.mw_search_rounds == 2


def test_fault_restriction_shrinks_technique_ii_circuit():
    """mw_search_rounds builds a smaller Technique-II circuit (fewer QEC cycles -> fewer detectors),
    while the rest of the run still uses cfg.rounds."""
    full = er.Config.smoke(code_name="bb144", experiment="memory")
    restricted = er.Config.smoke(code_name="bb144", experiment="memory", mw_search_rounds=2)
    c_full = er._technique_ii_circuit(full)
    c_restr = er._technique_ii_circuit(restricted)
    assert c_restr.num_detectors < c_full.num_detectors
    assert restricted.rounds == full.code.distance      # the experiment rounds are untouched


def test_weights_range_expands():
    cfg = er.load_config(CONFIGS / "bb6_memory.yaml")
    assert cfg.weights == list(range(2, 61))
    assert cfg.weights_explicit is True


def test_unknown_config_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("code_name: bb6\nnot_a_field: 1\n")
    try:
        er.load_config(bad)
        assert False, "expected SystemExit on unknown key"
    except SystemExit as e:
        assert "not_a_field" in str(e)


def test_lpu_circuit_kinds_build():
    """The former automorphism/joint_pauli stubs are now real builders wired to
    the cfg.lpu_* fields; tiny C/d_init keeps this seconds-scale."""
    for kind, expect_k in (("joint_pauli", 12), ("automorphism", 12)):
        cfg = er.Config.smoke(code_name="bb144", experiment=kind,
                              lpu_C=2, lpu_d_init=2)
        circuit = er.build_circuit(cfg)
        assert circuit.num_observables == expect_k
        assert circuit.num_detectors > 0


def test_joint_pauli_config_knobs():
    """lpu_include_memory_obs drops the 11 memory observables; lpu_shift routes
    the automorphism (None -> builder default 'y')."""
    cfg = er.Config.smoke(code_name="bb144", experiment="joint_pauli",
                          lpu_C=2, lpu_d_init=2, lpu_include_memory_obs=False)
    assert er.build_circuit(cfg).num_observables == 1
    cfg_x = er.Config.smoke(code_name="bb144", experiment="automorphism",
                            lpu_C=2, lpu_d_init=2, lpu_shift="x")
    assert er.build_circuit(cfg_x).num_observables == 12


def test_bb6_smoke_all_techniques(tmp_path):
    """BB(6) is small + has the exact onset pin, so all 3 techniques run fast end-to-end."""
    cfg = er.Config.smoke(code_name="bb6", experiment="memory",
                          techniques=["II", "IS", "I", "III"], split_method="multiseed",
                          mw_f0_override=2.3239e-5, mw_w0_override=3)
    out = er.run_all(cfg, tmp_path / "bb6")
    assert (tmp_path / "bb6" / "result.npz").exists()
    assert (tmp_path / "bb6" / "config.json").exists()
    assert out["tech2"]["distance"] == 6
    assert out["tech3"] is not None and np.all(np.isfinite(out["tech3"]["P_logical"]))
    assert out["is"] is not None and np.all(np.isfinite(out["is"].P_logical))


def test_bb144_smoke_is_and_split(tmp_path):
    """BB(12): skip the slow Technique-II distance search; exercise the IS + multiseed-split paths
    on the large single-sector DEM."""
    cfg = er.Config.smoke(code_name="bb144", experiment="memory",
                          techniques=["IS", "III"], split_method="multiseed")
    out = er.run_all(cfg, tmp_path / "bb144")
    assert (tmp_path / "bb144" / "result.npz").exists()
    assert out["is"] is not None and np.all(np.isfinite(out["is"].P_logical))
    assert out["tech3"] is not None and np.all(np.isfinite(out["tech3"]["P_logical"]))
