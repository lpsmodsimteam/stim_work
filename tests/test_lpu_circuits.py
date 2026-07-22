"""Tests for the five tour-de-gross LPU circuit builders (gross_code_lpu_tdg).

Covers the X̄₁ / Z̄₁ half-LPU measurements, the idle memory baseline, and the
two full builders added for the Ȳ₁ in-module measurement (full LPU, uniform
convention) and the shift automorphism:

  (a) zero-noise determinism — at ErrorModel(0,0) no detector fires and no
      observable flips over 100 detector-sampler shots;
  (b) DEM construction — detector_error_model(decompose_errors=False) builds
      (the Ȳ₁ deformed code is non-CSS, so no X/Z decomposition), with pinned
      M (detectors) and K (observables);
  (c) stim.TableauSimulator logical checks — the Ȳ₁ circuit leaves the state
      in a Ȳ₁ eigenstate consistent with the noiseless MPP reference, with
      the commuting Z-type logicals unchanged (after the m_e Pauli-frame
      correction); one shift instruction's swap layers conjugate the logical
      Paulis exactly as the GF(2)-recomputed action matrix says;
  (d) RelayBPDecoder().setup succeeds on both new circuits.

Budgets: everything here is p=0 sampling, DEM builds, or tiny (C≤3) circuits —
seconds-scale, per the repo's ≤30 s simulation rule.
"""
import pathlib
import sys

# Support both flat and src/ layouts (mirrors the other tests).
_here = pathlib.Path(__file__).resolve().parent
for cand in (_here, _here / "src", _here.parent / "src"):
    if (cand / "gross_code_lpu_tdg.py").exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import pytest
import stim

import gross_code_lpu_tdg as tdg
from bb_code_sim import RelayBPDecoder, _gf2_solve
from surface_code_sim import ErrorModel

EM0 = ErrorModel(p_phys=0.0, p_meas=0.0)
EM = ErrorModel(p_phys=1e-3, p_meas=1e-3)

# name -> (builder(em), expected M, expected K) at production defaults.
BUILDERS = {
    "x1": (lambda em: tdg.build_logical_x1_circuit(em), 4658, 1),
    "z1": (lambda em: tdg.build_logical_z1_circuit(em), 4638, 1),
    "idle": (lambda em: tdg.build_idle_memory_circuit(em), 1872, 12),
    "joint_pauli": (lambda em: tdg.build_joint_pauli_circuit(em), 5600, 12),
    "automorphism": (lambda em: tdg.build_automorphism_circuit(em), 7776, 12),
}


# ------------------------- (a) zero-noise determinism -------------------------
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_zero_noise_determinism(name):
    builder, _, _ = BUILDERS[name]
    circuit = builder(EM0)
    dets, obs = circuit.compile_detector_sampler().sample(
        100, separate_observables=True)
    assert int(dets.sum()) == 0, f"{name}: detectors fired at p=0"
    assert int(obs.sum()) == 0, f"{name}: observables flipped at p=0"


# ------------------------------ (b) DEM builds --------------------------------
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_dem_builds_with_expected_m_and_k(name):
    builder, M, K = BUILDERS[name]
    circuit = builder(EM)
    dem = circuit.detector_error_model(decompose_errors=False)
    print(f"{name}: M={dem.num_detectors} K={dem.num_observables} "
          f"mechanisms={sum(1 for i in dem if i.type == 'error')}")
    assert dem.num_detectors == M
    assert dem.num_observables == K


def test_y1_dem_has_no_same_syndrome_different_action_mechanisms():
    """Regression for the Ȳ₁ framing bug: no two single-fault DEM mechanisms
    may share a detector signature while acting differently on the
    observables — such pairs are informationally unresolvable and put an
    irreducible floor under f(1) for any decoder.  The broken framing had
    46 pairs straddling the mid-circuit reference MPP (one member flips
    reference AND gauged outcome, the other only the gauged outcome) and,
    with the MPP moved to t=0 alone, 58 pairs at the unanchored first noisy
    round (mid-round Z-type faults vs same-X-check partners).  Fixed by the
    t=0 MPP anchor + the noiseless encoding round; scan the production DEM
    (p=0.005 symmetric, idle_noise=True — the diagnosed configuration) and
    assert zero groups."""
    em = ErrorModel.symmetric(0.005)
    circuit = tdg.build_joint_pauli_circuit(em, idle_noise=True)
    dem = circuit.detector_error_model(decompose_errors=False)
    groups = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        det, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        groups.setdefault(tuple(sorted(det)), set()).add(tuple(sorted(obs)))
    bad = [k for k, v in groups.items() if len(v) > 1]
    assert not bad, (
        f"{len(bad)} same-syndrome/different-action mechanism groups "
        f"(first: detectors {bad[0]} -> actions {groups[bad[0]]})")


def test_idle_noise_variants_still_deterministic():
    """The idle_pool threading must not break p=0 determinism."""
    for circuit in (
        tdg.build_joint_pauli_circuit(EM0, C=2, d_init=2, idle_noise=True),
        tdg.build_automorphism_circuit(EM0, C=2, d_init=2, idle_noise=True),
    ):
        dets, obs = circuit.compile_detector_sampler().sample(
            50, separate_observables=True)
        assert int(dets.sum()) == 0 and int(obs.sum()) == 0


def test_automorphism_shift_x_route():
    """The B-route (δ=x: L via Z-checks, R via X-checks) is also deterministic."""
    circuit = tdg.build_automorphism_circuit(EM0, shift="x", C=2, d_init=2)
    dets, obs = circuit.compile_detector_sampler().sample(
        50, separate_observables=True)
    assert int(dets.sum()) == 0 and int(obs.sum()) == 0


# --------------------- (c) TableauSimulator logical checks --------------------
def test_y1_measurement_projects_onto_y1_eigenstate():
    """Run the p=0 Ȳ₁ circuit up to (excluding) the final transversal data
    readout: the state must be a Ȳ₁ eigenstate whose eigenvalue matches the
    noiseless MPP reference AND the last round's 24-record vertex product, and
    every commuting Z-type logical must still be +1 after the m_e Pauli-frame
    correction of the return edge readout."""
    C, d_init = 3, 2
    circuit = tdg.build_joint_pauli_circuit(EM0, C=C, d_init=d_init)

    # Measurement layout of the builder (absolute record indices):
    #   1 MPP (the reference is anchored BEFORE any noise — record 0), then
    #   1 noiseless encoding round + d_init noisy bare rounds of 144, then
    #   C LPU rounds of 187 = [X 72][Z 72][vertex 24][cycle 19], then the 47
    #   edge readouts, then (1 + d_init) bare rounds of 144, then the 144
    #   final data.
    mpp_idx = 0
    lpu_base = 1 + (d_init + 1) * 144
    v_start_last = lpu_base + (C - 1) * 187 + 144
    edge_start = lpu_base + C * 187
    edge_all = sorted(tdg.EDGE_QUBIT.values())

    # Execute everything before the final data M — the LAST "M" instruction
    # (at p=0 stim fuses each round's adjacent X-anc/Z-anc M's, so matching on
    # target count would stop at the first bare round instead).
    insts = list(circuit.flattened())
    last_m = max(i for i, inst in enumerate(insts) if inst.name == "M")
    sim = stim.TableauSimulator()
    for inst in insts[:last_m]:
        sim.do(inst)
    rec = sim.current_measurement_record()
    assert len(rec) == circuit.num_measurements - 144

    m_mpp = bool(rec[mpp_idx])
    m_vertex = bool(np.bitwise_xor.reduce(
        [int(rec[v_start_last + i]) for i in range(24)]))
    assert m_vertex == m_mpp, "last-round vertex product must equal the MPP bit"

    # Ȳ₁ eigenstate consistent with the MPP outcome.
    n = sim.num_qubits
    yq = 96  # x⁴R
    ps = stim.PauliString(n)
    ps[yq] = 2  # Y
    for q in np.where(tdg._op_vec(tdg.P, tdg.Q))[0]:
        if int(q) != yq:
            ps[int(q)] = 1  # X
    for q in np.where(tdg._op_vec(tdg.Z1_L, tdg.Z1_R))[0]:
        if int(q) != yq:
            ps[int(q)] = 3  # Z
    exp = sim.peek_observable_expectation(ps)
    assert exp != 0, "state must be a Ȳ₁ eigenstate"
    assert exp == (-1 if m_mpp else +1)

    # Commuting Z-type logicals unchanged (mod the m_e correction records).
    _, recipe = tdg._y1_observable_recipe()
    assert len(recipe) == 11
    for w, e_list in recipe:
        pz = stim.PauliString(n)
        for q in np.where(w)[0]:
            pz[int(q)] = 3
        frame = bool(np.bitwise_xor.reduce(
            [int(rec[edge_start + edge_all.index(e)]) for e in e_list]
        )) if e_list else False
        exp_k = sim.peek_observable_expectation(pz)
        assert exp_k != 0, "commuting logical must stay sharp"
        assert exp_k == (-1 if frame else +1), (
            "commuting Z-logical changed by the Ȳ₁ measurement")


def _class_decomposition(v, logs, H):
    """GF(2) coefficients of v over the canonical logical basis mod stabilizers."""
    basis = np.vstack([np.vstack(logs), H]).astype(np.uint8).T
    x = _gf2_solve(basis, v.astype(np.uint8))
    assert x is not None
    return x[:12]


@pytest.mark.parametrize("shift", ["y", "x"])
def test_automorphism_conjugation_matches_logical_action(shift):
    """Conjugating the 12 X̄/Z̄ pairs through one shift instruction's swap
    layers (the unitary part — the MRs act on vacated |0⟩ qubits) equals the
    GF(2)-recomputed logical action matrix; the action has order 6."""
    t1, t2, t4, t5 = tdg._shift_swap_layers(shift)
    swap = stim.Circuit()
    for layer in (t1, t2, t4, t5):
        swap.append("CX", [q for pair in layer for q in pair])

    perm = tdg._shift_data_perm(shift)
    perm_inv = np.zeros(tdg.N_DATA, dtype=np.int64)
    perm_inv[perm] = np.arange(tdg.N_DATA)
    log_Z, log_X = tdg._tdg_logical_ops()
    Az, Ax = tdg.shift_logical_action(shift)

    n = tdg.N_GROSS
    for logs, H, A, pauli in ((log_Z, tdg.H_Z, Az, 3), (log_X, tdg.H_X, Ax, 1)):
        for k in range(12):
            ps = stim.PauliString(n)
            supp = np.where(logs[k])[0]
            for q in supp:
                ps[int(q)] = pauli
            pushed = ps.after(swap)
            assert pushed.sign == +1
            # Data support permutes exactly.  The move-through-|0⟩ hop is an
            # isometry, not a permutation unitary: Z-type operators may leave
            # Z "dust" on check ancillas — harmless, because the ancillas are
            # |0⟩ (Z eigenvalue +1) both before the instruction and at the
            # t6 MR.  X/Y support on ancillas would be a real error.
            for q in range(n):
                want = pauli if (q < tdg.N_DATA and logs[k][perm_inv[q]]) else 0
                got = pushed[q]
                if q < tdg.N_DATA:
                    assert got == want
                else:
                    assert got in (0, 3), "X/Y dust on a check ancilla"
            # Class decomposition of the pushed operator = action-matrix row.
            v = np.zeros(tdg.N_DATA, dtype=np.uint8)
            v[perm[supp]] = 1
            assert np.array_equal(_class_decomposition(v, logs, H), A[k])

    # Order-6 (paper A.2): δ⁶ acts as the logical identity, δ itself doesn't.
    eye = np.eye(12, dtype=np.uint8)
    assert not np.array_equal(Az, eye)
    Az6, Ax6 = tdg.shift_logical_action(shift, power=6)
    assert np.array_equal(Az6, eye) and np.array_equal(Ax6, eye)
    assert np.array_equal((Ax @ Az.T) % 2, eye)


# --------------------------- (d) Relay-BP decoder -----------------------------
@pytest.mark.parametrize("build", [
    lambda: tdg.build_joint_pauli_circuit(EM, C=2, d_init=2),
    lambda: tdg.build_automorphism_circuit(EM, C=2, d_init=2),
], ids=["joint_pauli", "automorphism"])
def test_relay_bp_setup(build):
    circuit = build()
    decoder = RelayBPDecoder()
    decoder.setup(circuit)  # must not raise (non-CSS joint DEM for Ȳ₁)
