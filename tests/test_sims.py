import numpy as np
import pytest

from surface_code_sim import (
    CodeType, ErrorModel, SurfaceCodeSimulator, PyMatchingDecoder, SimulationResult
)
from bb_code_sim import (
    BB_72_12_6, build_parity_checks, find_logical_ops,
    build_bb_circuit, BBCodeSimulator, BPOSDDecoder
)
import gross_code_lpu_tdg as tdg
from importance_sampling import importance_sample, ImportanceSamplingResult


# --- Surface code ---

def test_error_model_symmetric():
    em = ErrorModel.symmetric(0.01)
    assert em.p_phys == em.p_meas == 0.01


def test_surface_code_circuit_builds():
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.01), rounds=3)
    assert circuit is not None


def test_surface_code_run_returns_result():
    sim = SurfaceCodeSimulator(distance=3)
    result = sim.run(ErrorModel.symmetric(0.01), rounds=3, shots=20)
    assert isinstance(result, SimulationResult)
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.shots == 20


def test_surface_code_zero_noise():
    sim = SurfaceCodeSimulator(distance=3)
    result = sim.run(ErrorModel(p_phys=0.0, p_meas=0.0), rounds=3, shots=50)
    assert result.logical_error_rate == 0.0


# --- BB code parity checks ---

def test_bb_72_parity_check_shape():
    H_X, H_Z = build_parity_checks(BB_72_12_6)
    n = 2 * BB_72_12_6.l * BB_72_12_6.m  # 72 qubits
    assert H_X.shape[1] == n
    assert H_Z.shape[1] == n


def test_bb_72_css_orthogonality():
    H_X, H_Z = build_parity_checks(BB_72_12_6)
    # H_X @ H_Z^T = 0 mod 2
    assert np.all((H_X @ H_Z.T) % 2 == 0)


def test_bb_72_logical_ops_count():
    H_X, H_Z = build_parity_checks(BB_72_12_6)
    L_Z, L_X = find_logical_ops(H_X, H_Z)
    assert L_Z.shape[0] == 12  # k=12 logical qubits
    assert L_X.shape[0] == 12


def test_bb_circuit_builds():
    circuit = build_bb_circuit(BB_72_12_6, ErrorModel.symmetric(0.01), rounds=2)
    assert circuit is not None


def test_bb_simulator_run():
    sim = BBCodeSimulator(BB_72_12_6)
    result = sim.run(ErrorModel.symmetric(0.01), rounds=2, shots=20)
    assert isinstance(result, SimulationResult)
    assert 0.0 <= result.logical_error_rate <= 1.0
    assert result.shots == 20


def test_bb_zero_noise():
    sim = BBCodeSimulator(BB_72_12_6)
    result = sim.run(ErrorModel(p_phys=0.0, p_meas=0.0), rounds=2, shots=20)
    assert result.logical_error_rate == 0.0


# --- Gross code (TDG convention) ---

def test_tdg_parity_check_shape():
    assert tdg.H_X.shape == (72, 144)
    assert tdg.H_Z.shape == (72, 144)


def test_tdg_css_orthogonality():
    assert np.all((tdg.H_X @ tdg.H_Z.T) % 2 == 0)


def test_tdg_graph_counts():
    assert len(tdg.V_ALL) == 23
    assert len(tdg.E_ALL) == 47
    assert len(tdg.U_ALL) == 19


def test_tdg_total_qubits():
    assert tdg.N_TOTAL_QUBITS == 378


def test_tdg_logical_commutation():
    # X̄₁ and Z̄₁ must anticommute; X̄₁ and Z̄₇ must commute
    assert tdg._symplectic_overlap(tdg.X1_L, tdg.X1_R, tdg.Z1_L, tdg.Z1_R) == 1
    assert tdg._symplectic_overlap(tdg.X1_L, tdg.X1_R, tdg.Z7_L, tdg.Z7_R) == 0


def test_tdg_x1_circuit_zero_noise():
    em = ErrorModel(p_phys=0.0, p_meas=0.0)
    circ = tdg.build_logical_x1_circuit(em, C=2, d_init=2)
    sampler = circ.compile_detector_sampler(seed=0)
    dets, obs = sampler.sample(50, separate_observables=True)
    assert int(dets.sum()) == 0
    assert int(obs.sum()) == 0


def test_tdg_z1_circuit_zero_noise():
    em = ErrorModel(p_phys=0.0, p_meas=0.0)
    circ = tdg.build_logical_z1_circuit(em, C=2, d_init=2)
    sampler = circ.compile_detector_sampler(seed=0)
    dets, obs = sampler.sample(50, separate_observables=True)
    assert int(dets.sum()) == 0
    assert int(obs.sum()) == 0


# --- Importance sampling ---

def test_is_returns_correct_shapes():
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.01), rounds=3)
    result = importance_sample(
        circuit, PyMatchingDecoder(),
        p_ref=0.01,
        p_values=[0.001, 0.005, 0.01],
        weights=[1, 2, 3],
        shots_per_weight=50,
        seed=0,
    )
    assert isinstance(result, ImportanceSamplingResult)
    assert result.P_logical.shape == (3,)
    assert result.P_logical_se.shape == (3,)
    assert result.spectrum.weights == [1, 2, 3]


def test_is_decreasing_with_p():
    # Lower p should give lower P_logical
    sim = SurfaceCodeSimulator(distance=3)
    circuit = sim.build_circuit(ErrorModel.symmetric(0.01), rounds=3)
    result = importance_sample(
        circuit, PyMatchingDecoder(),
        p_ref=0.01,
        p_values=[0.001, 0.005, 0.01],
        weights=list(range(1, 8)),
        shots_per_weight=200,
        seed=0,
    )
    assert result.P_logical[0] < result.P_logical[1] < result.P_logical[2]


def test_is_matches_direct_mc():
    # IS estimate at p_built should agree with direct MC within ~3σ
    sim = SurfaceCodeSimulator(distance=3)
    em = ErrorModel.symmetric(0.01)
    direct = sim.run(em, rounds=3, shots=5000, seed=0)

    circuit = sim.build_circuit(em, rounds=3)
    result = importance_sample(
        circuit, PyMatchingDecoder(),
        p_ref=0.01,
        p_values=[0.01],
        weights=list(range(1, 10)),
        shots_per_weight=300,
        seed=0,
    )
    diff = abs(result.P_logical[0] - direct.logical_error_rate)
    se = np.hypot(result.P_logical_se[0], direct.logical_error_rate_se)
    assert diff < 4 * se, f"IS={result.P_logical[0]:.4f}, MC={direct.logical_error_rate:.4f}, diff/SE={diff/se:.1f}"


