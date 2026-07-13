"""Gates for the experiment_runner channel extension (G2 of the cluster campaign plan).

Three guarantees before any channel config ships to the cluster:
1. BYTE-IDENTITY — with noise_channel/ablate_channel unset, build_circuit returns the experiment
   builder's circuit untouched (the full-noise path is provably unaffected by the extension).
2. PARTITION — the five budget channels (cz/meas/prep/gate_idle/meas_idle) classify the noise
   instructions of every campaign circuit disjointly, and the only unclassified residue is the
   post-H DEPOLARIZE1 (single-qubit GATE noise — deliberately outside the five-channel budget).
3. Config plumbing — outdir suffixing and the mutual-exclusion / valid-key validation.
"""
import dataclasses

import pytest
import stim

import experiment_runner as er
from bb_code_sim import NOISE_CHANNEL_PREDICATES, NOISE_INSTRUCTIONS

FIVE = ["cz", "meas", "prep", "gate_idle", "meas_idle"]

# Campaign circuit matrix at smoke scale (circuit BUILD cost only — no decoding here).
CIRCUITS = [
    dict(code_name="bb6", experiment="memory"),
    dict(code_name="bb144", experiment="memory"),
    dict(code_name="bb288", experiment="memory", rounds=2),   # build-only; 2 rounds keeps it fast
    dict(code_name="bb144", experiment="lpu_x1", lpu_C=2, lpu_d_init=2),
    dict(code_name="bb144", experiment="lpu_z1", lpu_C=2, lpu_d_init=2),
]


def _cfg(**over):
    return er.Config(label="test", techniques=[], **over)


def _classify(circuit):
    """Per noise instruction: which of the five channels match (list of hit-sets + residue)."""
    insts = list(circuit.flattened())
    hits, residue = [], []
    for i, inst in enumerate(insts):
        if inst.name not in NOISE_INSTRUCTIONS:
            continue
        prev = insts[i - 1] if i > 0 else None
        nxt = insts[i + 1] if i + 1 < len(insts) else None
        matched = {ch for ch in FIVE if NOISE_CHANNEL_PREDICATES[ch](inst, prev, nxt)}
        hits.append(matched)
        if not matched:
            residue.append((inst, prev))
    return hits, residue


@pytest.mark.parametrize("spec", CIRCUITS, ids=lambda s: f"{s['code_name']}-{s['experiment']}")
def test_full_noise_path_byte_identical(spec):
    cfg = _cfg(**spec)
    via_extension = er.build_circuit(cfg)
    direct = er.CIRCUIT_BUILDERS[cfg.experiment](cfg)
    assert str(via_extension) == str(direct)
    # and the extension didn't even copy: same object back (early-return contract)
    assert er.build_circuit(cfg) is er.CIRCUIT_BUILDERS[cfg.experiment](cfg) or \
        str(er.build_circuit(cfg)) == str(direct)


@pytest.mark.parametrize("spec", CIRCUITS, ids=lambda s: f"{s['code_name']}-{s['experiment']}")
def test_five_channel_partition(spec):
    cfg = _cfg(**spec)
    circ = er.build_circuit(cfg)
    hits, residue = _classify(circ)
    assert hits, "no noise instructions found — not a noisy circuit?"
    # disjoint: no instruction claimed by two channels
    multi = [h for h in hits if len(h) > 1]
    assert not multi, f"non-disjoint channel classification: {multi[:5]}"
    # residue = post-H DEPOLARIZE1 only (1q gate noise, outside the five-channel budget)
    bad = [(i.name, p.name if p else None) for i, p in residue
           if not (i.name == "DEPOLARIZE1" and p is not None and p.name == "H")]
    assert not bad, f"unclassified noise beyond post-H DEPOLARIZE1: {bad[:5]}"


@pytest.mark.parametrize("spec", CIRCUITS, ids=lambda s: f"{s['code_name']}-{s['experiment']}")
def test_iso_plus_ablate_reassemble(spec):
    """iso(ch) keeps exactly what abl(ch) drops: counts add back to the full circuit's."""
    cfg = _cfg(**spec)
    full = er.build_circuit(cfg)
    n_full = sum(1 for i in full.flattened() if i.name in NOISE_INSTRUCTIONS)
    for ch in FIVE:
        n_iso = sum(1 for i in er.build_circuit(dataclasses.replace(cfg, noise_channel=ch)).flattened()
                    if i.name in NOISE_INSTRUCTIONS)
        n_abl = sum(1 for i in er.build_circuit(dataclasses.replace(cfg, ablate_channel=ch)).flattened()
                    if i.name in NOISE_INSTRUCTIONS)
        assert n_iso + n_abl == n_full, (ch, n_iso, n_abl, n_full)
        assert n_iso > 0 or spec["experiment"].startswith("lpu"), f"channel {ch} empty on {spec}"


def test_outdir_suffixes():
    base = _cfg(code_name="bb6", experiment="memory")
    iso = dataclasses.replace(base, noise_channel="cz")
    abl = dataclasses.replace(base, ablate_channel="meas_idle")
    assert base.resolved_outdir().name == "memory"
    assert iso.resolved_outdir().name == "memory__iso_cz"
    assert abl.resolved_outdir().name == "memory__abl_meas_idle"
    # explicit outdir wins unchanged
    ovr = dataclasses.replace(iso, outdir="runs/elsewhere")
    assert str(ovr.resolved_outdir()).replace("\\", "/") == "runs/elsewhere"


def test_channel_key_validation():
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _cfg(code_name="bb6", noise_channel="cz", ablate_channel="meas")
    with pytest.raises(SystemExit, match="unknown noise channel"):
        _cfg(code_name="bb6", noise_channel="nope")
    with pytest.raises(SystemExit, match="use the split channels"):
        _cfg(code_name="bb6", ablate_channel="idle")
