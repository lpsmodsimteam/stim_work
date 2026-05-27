"""
Bivariate Bicycle (BB) code simulation using Stim + PyMatching.

Based on: Bravyi et al., arXiv:2308.07915
"""

from __future__ import annotations

import numpy as np
import stim
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import pymatching
from ldpc.bposd_decoder import BpOsdDecoder
from surface_code_sim import ErrorModel, Decoder, PyMatchingDecoder, SimulationResult


class BPOSDDecoder(Decoder):
    """
    Belief-propagation + ordered-statistics decoder via ldpc.
    This is the recommended decoder for BB codes (and other LDPC codes)
    because MWPM cannot handle their weight-3 check connectivity.
    """

    def __init__(self, osd_order: int = 10) -> None:
        self._osd_order = osd_order
        self._decoder = None
        self._n_obs: int = 0

    def setup(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(flatten_loops=True)
        n_det = circuit.num_detectors
        n_obs = circuit.num_observables
        self._n_obs = n_obs

        # Build the check matrix H and prior probabilities from the DEM.
        # Each DEM instruction "error(p) D_i D_j ... L_k ..." becomes a column.
        error_mechanisms: list = []
        priors: list = []

        for instruction in dem.flattened():
            if instruction.type == "error":
                p = instruction.args_copy()[0]
                dets = []
                obs_mask = np.zeros(n_obs, dtype=np.uint8)
                for t in instruction.targets_copy():
                    if t.is_relative_detector_id():
                        dets.append(t.val)
                    elif t.is_logical_observable_id():
                        obs_mask[t.val] ^= 1
                error_mechanisms.append((dets, obs_mask, p))
                priors.append(p)

        n_err = len(error_mechanisms)
        H = np.zeros((n_det, n_err), dtype=np.uint8)
        self._obs_matrix = np.zeros((n_obs, n_err), dtype=np.uint8)
        for col, (dets, obs_mask, _) in enumerate(error_mechanisms):
            for d in dets:
                H[d, col] ^= 1
            self._obs_matrix[:, col] = obs_mask

        if n_err == 0 or all(p == 0 for p in priors):
            self._decoder = None
            return

        self._decoder = BpOsdDecoder(
            H,
            error_channel=list(priors),
            max_iter=n_det,
            bp_method="ms",
            ms_scaling_factor=0.625,
            osd_method="osd_cs",
            osd_order=self._osd_order,
        )

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        shots = detection_events.shape[0]
        predictions = np.zeros((shots, self._n_obs), dtype=bool)
        if self._decoder is None:
            return predictions  # zero-noise: no errors, predict all-zero
        for i in range(shots):
            self._decoder.decode(detection_events[i].astype(np.uint8))
            corr = self._decoder.osdw_decoding
            predictions[i] = (self._obs_matrix @ corr) % 2
        return predictions


class RelayBPDecoder(Decoder):
    """
    Relay belief-propagation decoder via relay_bp (Rust).

    Chains multiple DMem-BP runs initialized with the previous run's final
    marginals, breaking symmetry traps that stall standard BP. Supports true
    batched decoding (parallelised in Rust), making it much faster than the
    shot-by-shot BPOSDDecoder loop for BB codes.
    """

    def __init__(
        self,
        gamma0: float = 0.1,
        pre_iter: int = 20,
        num_sets: int = 20,
        set_max_iter: int = 20,
        gamma_dist_interval: tuple = (-0.24, 0.66),
        stop_nconv: int = 5,
        parallel: bool = True,
    ) -> None:
        # Default params are tuned for speed (420 iterations/shot).
        # For accuracy use pre_iter=80, num_sets=100, set_max_iter=60
        # (6080 iterations/shot — ~14x slower but better LER near threshold).
        self._gamma0 = gamma0
        self._pre_iter = pre_iter
        self._num_sets = num_sets
        self._set_max_iter = set_max_iter
        self._gamma_dist_interval = gamma_dist_interval
        self._stop_nconv = stop_nconv
        self._parallel = parallel
        self._observable_decoder = None

    def setup(self, circuit: stim.Circuit) -> None:
        import relay_bp
        from relay_bp.stim import CheckMatrices

        dem = circuit.detector_error_model(flatten_loops=True)
        cm = CheckMatrices.from_dem(dem)
        relay_decoder = relay_bp.RelayDecoderF64(
            cm.check_matrix,
            error_priors=cm.error_priors,
            gamma0=self._gamma0,
            pre_iter=self._pre_iter,
            num_sets=self._num_sets,
            set_max_iter=self._set_max_iter,
            gamma_dist_interval=self._gamma_dist_interval,
            stop_nconv=self._stop_nconv,
        )
        self._observable_decoder = relay_bp.ObservableDecoderRunner(
            relay_decoder, cm.observables_matrix
        )

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._observable_decoder is None:
            raise RuntimeError("Call setup(circuit) before decode_batch.")
        return self._observable_decoder.decode_observables_batch(
            detection_events.astype(np.uint8), parallel=self._parallel, progress_bar=False
        )


class BBPyMatchingDecoder(Decoder):
    """
    MWPM decoder for BB codes. Passes ignore_decomposition_failures=True
    because DEPOLARIZE2 noise on high-connectivity codes generates hyperedges
    that can't be split into graphlike (2-symptom) components.
    """

    def __init__(self) -> None:
        self._matching = None

    def setup(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        self._matching = pymatching.Matching.from_detector_error_model(dem)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._matching is None:
            raise RuntimeError("Call setup(circuit) before decode_batch.")
        result = self._matching.decode_batch(detection_events)
        assert isinstance(result, np.ndarray)
        return result


# ---------------------------------------------------------------------------
# Code parameters
# ---------------------------------------------------------------------------

@dataclass
class BBCodeParams:
    l: int
    m: int
    a_exps: List[Tuple[int, int]]  # monomials in A(x,y), e.g. [(3,0),(0,1),(0,2)]
    b_exps: List[Tuple[int, int]]  # monomials in B(x,y), e.g. [(0,3),(1,0),(2,0)]
    distance: int


BB_72_12_6   = BBCodeParams(l=6,  m=6, a_exps=[(3,0),(0,1),(0,2)], b_exps=[(0,3),(1,0),(2,0)], distance=6)
BB_144_12_12 = BBCodeParams(l=12, m=6, a_exps=[(3,0),(0,1),(0,2)], b_exps=[(0,3),(1,0),(2,0)], distance=12)


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------

def _gf2_rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Row-reduce A over GF(2). Returns (rref_matrix, pivot_cols)."""
    A = A.copy() % 2
    rows, cols = A.shape
    pivot_row = 0
    pivots = []
    for col in range(cols):
        # find pivot in this column at or below pivot_row
        found = -1
        for r in range(pivot_row, rows):
            if A[r, col]:
                found = r
                break
        if found == -1:
            continue
        A[[pivot_row, found]] = A[[found, pivot_row]]
        for r in range(rows):
            if r != pivot_row and A[r, col]:
                A[r] = (A[r] + A[pivot_row]) % 2
        pivots.append(col)
        pivot_row += 1
    return A, pivots


def _gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Basis for the null space of A over GF(2). Rows are null vectors."""
    _, n = A.shape
    rref_A, pivots = _gf2_rref(A)
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    null_vecs = []
    for f in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[f] = 1
        for row_i, p_i in enumerate(pivots):
            vec[p_i] = rref_A[row_i, f]
        null_vecs.append(vec)
    return np.array(null_vecs, dtype=np.uint8) if null_vecs else np.empty((0, n), dtype=np.uint8)


def _gf2_rowspace(A: np.ndarray) -> np.ndarray:
    """Basis for the row space of A over GF(2)."""
    rref, pivots = _gf2_rref(A)
    return rref[:len(pivots)]


def _gf2_rank(A: np.ndarray) -> int:
    _, pivots = _gf2_rref(A)
    return len(pivots)


# ---------------------------------------------------------------------------
# Parity check construction
# ---------------------------------------------------------------------------

def _poly_matrix(l: int, m: int, exps: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build the l*m × l*m circulant matrix over GF(2) for a polynomial
    given by a list of (x-exponent, y-exponent) monomials.

    Row s corresponds to check (i,j) with s = i*m + j.
    Column t corresponds to data qubit (i',j') with t = i'*m + j'.
    Entry [s, t] = 1 iff (i' - i, j' - j) mod (l, m) is one of the monomials.
    """
    n = l * m
    M = np.zeros((n, n), dtype=np.uint8)
    for (ax, ay) in exps:
        for i in range(l):
            for j in range(m):
                s = i * m + j
                t = ((i + ax) % l) * m + ((j + ay) % m)
                M[s, t] ^= 1
    return M


def build_parity_checks(params: BBCodeParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (H_X, H_Z) over GF(2).
      H_X = [A | B]      shape (l*m, 2*l*m)
      H_Z = [B^T | A^T]  shape (l*m, 2*l*m)
    """
    A = _poly_matrix(params.l, params.m, params.a_exps)
    B = _poly_matrix(params.l, params.m, params.b_exps)
    H_X = np.hstack([A, B]).astype(np.uint8)
    H_Z = np.hstack([B.T % 2, A.T % 2]).astype(np.uint8)
    return H_X, H_Z


# ---------------------------------------------------------------------------
# Logical operator finder
# ---------------------------------------------------------------------------

def find_logical_ops(H_X: np.ndarray, H_Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k canonical logical Z/X operator pairs for a CSS code.

    Algorithm:
    1. For each null space, reduce each vector modulo the opposing stabiliser RREF.
       This ensures outputs stay in ker(H_X) / ker(H_Z) respectively.
    2. Collect k linearly independent reduced vectors.
    3. Canonicalise the pairing so that log_Z[i] anti-commutes with log_X[i]
       and commutes with log_X[j≠i], via a GF(2) basis change.

    Returns (log_Z, log_X), each shape (k, n).
    """
    def _reduce_modulo(vectors: np.ndarray, stab_rref: np.ndarray,
                       stab_pivots: List[int]) -> np.ndarray:
        """Reduce each row of `vectors` modulo the RREF of the stabiliser matrix.
        The result stays within the same coset — i.e. the reduced vector is
        congruent to the original modulo rowspace(stab) and is still in ker(H)."""
        out = []
        for v in vectors:
            v = v.copy() % 2
            for row_i, col in enumerate(stab_pivots):
                if v[col]:
                    v = (v + stab_rref[row_i]) % 2
            if v.any():
                out.append(v)
        if not out:
            return np.empty((0, vectors.shape[1]), dtype=np.uint8)
        # De-duplicate via RREF and return independent rows
        rref, pivots = _gf2_rref(np.array(out, dtype=np.uint8))
        return rref[:len(pivots)].copy()

    def _gf2_inv(M: np.ndarray) -> np.ndarray:
        """Invert a square GF(2) matrix via augmented RREF."""
        k = M.shape[0]
        aug = np.hstack([M.copy() % 2, np.eye(k, dtype=np.uint8)])
        rref, _ = _gf2_rref(aug)
        return rref[:, k:].copy()

    null_X = _gf2_nullspace(H_X)
    null_Z = _gf2_nullspace(H_Z)
    stab_Z_rref, pivots_Z = _gf2_rref(H_Z)
    stab_X_rref, pivots_X = _gf2_rref(H_X)

    log_Z = _reduce_modulo(null_X, stab_Z_rref, pivots_Z)
    log_X = _reduce_modulo(null_Z, stab_X_rref, pivots_X)

    # Canonicalise: find basis change so log_Z[i] · log_X[j] = δ_ij
    # New anti-comm = P @ M, so choose P = M^{-1} and replace log_Z.
    M = log_Z @ log_X.T % 2  # (k, k)
    P = _gf2_inv(M)           # P @ M = I
    log_Z = (P @ log_Z) % 2

    return log_Z, log_X


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_bb_circuit(
    params: BBCodeParams,
    error_model: ErrorModel,
    rounds: int,
) -> stim.Circuit:
    """
    Build a noisy Stim circuit for a BB code Z-memory experiment.

    Qubit layout:
      [0,       n)         — data qubits  (n = 2*l*m)
      [n,       n+n_c)     — X-ancilla    (n_c = l*m)
      [n+n_c,   n+2*n_c)   — Z-ancilla
    """
    l, m = params.l, params.m
    n_c = l * m          # number of checks of each type
    n   = 2 * n_c        # data qubits
    x_anc = list(range(n,       n + n_c))
    z_anc = list(range(n + n_c, n + 2 * n_c))

    p = error_model.p_phys
    pm = error_model.p_meas

    A = _poly_matrix(l, m, params.a_exps)
    B = _poly_matrix(l, m, params.b_exps)
    H_X, H_Z = build_parity_checks(params)

    # Precompute connections for each check
    # x_conns[s] = list of data qubit indices that X-check s acts on
    # z_conns[s] = list of data qubit indices that Z-check s acts on
    def _connections(H: np.ndarray) -> List[List[int]]:
        conns = []
        for s in range(n_c):
            conns.append(list(np.where(H[s])[0]))
        return conns

    x_conns = _connections(H_X)
    z_conns = _connections(H_Z)

    # Organise CNOTs into 6 layers (one per monomial in A+B / A^T+B^T)
    # Layer order: a_exps[0], a_exps[1], a_exps[2], b_exps[0], b_exps[1], b_exps[2]
    # For X-checks: ancilla (x_anc[s]) → data qubit
    # For Z-checks: data qubit → ancilla (z_anc[s])
    def _cnot_layers_x():
        layers = []
        for (ax, ay) in params.a_exps:  # left data (col < n_c)
            layer = []
            for i in range(l):
                for j in range(m):
                    s = i * m + j
                    t = ((i + ax) % l) * m + ((j + ay) % m)  # left data qubit
                    layer.append((x_anc[s], t))
            layers.append(layer)
        for (bx, by) in params.b_exps:  # right data (col >= n_c)
            layer = []
            for i in range(l):
                for j in range(m):
                    s = i * m + j
                    t = n_c + ((i + bx) % l) * m + ((j + by) % m)
                    layer.append((x_anc[s], t))
            layers.append(layer)
        return layers

    def _cnot_layers_z():
        # H_Z = [B^T | A^T], so Z-check s is connected via B^T (left) and A^T (right)
        # B^T[s,t]=1 iff B[t,s]=1 iff (t - s) ≡ some b_exp → offset = -b_exp = (l-bx, m-by)
        layers = []
        for (bx, by) in params.b_exps:  # B^T: left data
            layer = []
            for i in range(l):
                for j in range(m):
                    s = i * m + j
                    t = ((i - bx) % l) * m + ((j - by) % m)
                    layer.append((t, z_anc[s]))
            layers.append(layer)
        for (ax, ay) in params.a_exps:  # A^T: right data
            layer = []
            for i in range(l):
                for j in range(m):
                    s = i * m + j
                    t = n_c + ((i - ax) % l) * m + ((j - ay) % m)
                    layer.append((t, z_anc[s]))
            layers.append(layer)
        return layers

    x_layers = _cnot_layers_x()
    z_layers = _cnot_layers_z()

    # Find logical operators for observables
    log_Z, _ = find_logical_ops(H_X, H_Z)
    k = len(log_Z)

    # -----------------------------------------------------------------------
    # Build circuit
    # -----------------------------------------------------------------------
    circuit = stim.Circuit()

    def add_reset_data():
        circuit.append("R", list(range(n)))
        if pm > 0:
            circuit.append("X_ERROR", list(range(n)), pm)

    def add_init_ancilla():
        circuit.append("R", x_anc)
        if pm > 0:
            circuit.append("X_ERROR", x_anc, pm)
        circuit.append("H", x_anc)
        if p > 0:
            circuit.append("DEPOLARIZE1", x_anc, p)
        circuit.append("R", z_anc)
        if pm > 0:
            circuit.append("X_ERROR", z_anc, pm)

    def add_cnot_layers():
        for layer in x_layers:
            targets = [q for pair in layer for q in pair]
            circuit.append("CX", targets)
            if p > 0:
                circuit.append("DEPOLARIZE2", targets, p)
        for layer in z_layers:
            targets = [q for pair in layer for q in pair]
            circuit.append("CX", targets)
            if p > 0:
                circuit.append("DEPOLARIZE2", targets, p)

    def add_measure_x_ancilla():
        if p > 0:
            circuit.append("H", x_anc)
            circuit.append("DEPOLARIZE1", x_anc, p)
        else:
            circuit.append("H", x_anc)
        if pm > 0:
            circuit.append("X_ERROR", x_anc, pm)
        circuit.append("M", x_anc)

    def add_measure_z_ancilla():
        if pm > 0:
            circuit.append("X_ERROR", z_anc, pm)
        circuit.append("M", z_anc)

    def add_reset_ancilla():
        circuit.append("R", z_anc)
        if pm > 0:
            circuit.append("X_ERROR", z_anc, pm)
        circuit.append("R", x_anc)
        if pm > 0:
            circuit.append("X_ERROR", x_anc, pm)
        circuit.append("H", x_anc)
        if p > 0:
            circuit.append("DEPOLARIZE1", x_anc, p)

    # Init
    add_reset_data()
    add_init_ancilla()
    circuit.append("TICK")

    # Syndrome rounds
    for rnd in range(rounds):
        add_cnot_layers()
        circuit.append("TICK")
        add_measure_x_ancilla()
        add_measure_z_ancilla()
        circuit.append("TICK")

        # Detectors
        # X-check detectors: only from round 2 onwards (round 1 is random for Z-memory)
        if rnd == 0:
            # Z-checks are deterministic in round 1 (data starts in |0>)
            for s in range(n_c):
                # rec offset: z-ancilla s was measured at position -(n_c - s) in M block
                # x_anc block: n_c measurements, then z_anc block: n_c measurements
                # z_anc[s] is at offset -(n_c - s) from end of this tick's measurements
                circuit.append("DETECTOR", [stim.target_rec(-(n_c - s))], [0, s, 0])
        else:
            # X-check detectors: current XOR previous
            for s in range(n_c):
                # x_anc[s] in current round: offset -(2*n_c - s) from end of M block
                # x_anc[s] in previous round: that was 2*n_c measurements ago
                cur = -(2 * n_c - s)
                prev = cur - 2 * n_c
                circuit.append("DETECTOR", [stim.target_rec(cur), stim.target_rec(prev)], [1, s, rnd])
            # Z-check detectors
            for s in range(n_c):
                cur = -(n_c - s)
                prev = cur - 2 * n_c
                circuit.append("DETECTOR", [stim.target_rec(cur), stim.target_rec(prev)], [0, s, rnd])

        if rnd < rounds - 1:
            add_reset_ancilla()
            circuit.append("TICK")

    # Final data measurements
    if pm > 0:
        circuit.append("X_ERROR", list(range(n)), pm)
    circuit.append("M", list(range(n)))

    # Final detectors from data measurements (Z-checks only)
    # Z-check s acts on data qubits z_conns[s]; H_Z row s
    # Compare parity of those data measurements with the last Z-ancilla measurement
    for s in range(n_c):
        data_qubits = z_conns[s]
        # data qubit q was measured at offset -(n - q) from end of final M block
        data_recs = [stim.target_rec(-(n - q)) for q in data_qubits]
        # last Z-ancilla[s] measurement was 2*n_c + n measurements ago
        # (after final data M block of size n, before that: x_anc n_c + z_anc n_c)
        last_z_rec = stim.target_rec(-(n + n_c - s))
        circuit.append("DETECTOR", data_recs + [last_z_rec], [0, s, rounds])

    # Observables: logical Z operators
    for i, lz in enumerate(log_Z):
        data_qubits = list(np.where(lz)[0])
        recs = [stim.target_rec(-(n - q)) for q in data_qubits]
        circuit.append("OBSERVABLE_INCLUDE", recs, i)

    return circuit


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BBCodeSimulator:
    """
    Build Stim circuits and run logical idle experiments for a BB code.
    Mirrors the SurfaceCodeSimulator interface.
    """

    def __init__(self, params: BBCodeParams) -> None:
        self.params = params

    def build_circuit(self, error_model: ErrorModel, rounds: int) -> stim.Circuit:
        return build_bb_circuit(self.params, error_model, rounds)

    def run(
        self,
        error_model: ErrorModel,
        rounds: int,
        shots: int,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        circuit = self.build_circuit(error_model, rounds)
        if decoder is None:
            decoder = RelayBPDecoder()
        decoder.setup(circuit)

        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
        predictions = decoder.decode_batch(detection_events)

        logical_errors = np.any(predictions != observable_flips, axis=1)
        n_err = int(np.sum(logical_errors))
        ler   = n_err / shots
        from scipy.stats import beta as _beta
        lo, hi = _beta.interval(0.95, n_err + 0.5, shots - n_err + 0.5)
        ler_se = float((hi - lo) / 2)

        return SimulationResult(
            distance=self.params.distance,
            rounds=rounds,
            error_model=error_model,
            shots=shots,
            num_logical_errors=n_err,
            logical_error_rate=ler,
            logical_error_rate_se=ler_se,
        )

    def sweep_p(
        self,
        p_values: List[float],
        rounds: Optional[int] = None,
        shots: int = 10_000,
        p_meas_factor: float = 1.0,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        if rounds is None:
            rounds = self.params.distance
        return [
            self.run(ErrorModel(p, p * p_meas_factor), rounds=rounds, shots=shots,
                     decoder=decoder, seed=seed)
            for p in p_values
        ]

    def sweep_rounds(
        self,
        round_values: List[int],
        error_model: ErrorModel,
        shots: int = 10_000,
        decoder: Optional[Decoder] = None,
        seed: Optional[int] = None,
    ) -> List[SimulationResult]:
        from tqdm import tqdm
        return [
            self.run(error_model, rounds=r, shots=shots, decoder=decoder, seed=seed)
            for r in tqdm(round_values, desc="sweep_rounds", unit="round")
        ]


# ---------------------------------------------------------------------------
# Multi-code threshold sweep
# ---------------------------------------------------------------------------

def bb_threshold_sweep(
    code_list: List[BBCodeParams],
    p_values: List[float],
    rounds_per_code: Optional[Dict[int, int]] = None,
    shots: int = 10_000,
    p_meas_factor: float = 1.0,
    decoder_cls=RelayBPDecoder,
    seed: Optional[int] = None,
) -> Dict[int, List[SimulationResult]]:
    """
    Sweep physical error rates over a list of BB codes.

    Returns dict mapping code distance → List[SimulationResult].
    """
    all_results: Dict[int, List[SimulationResult]] = {}
    for params in code_list:
        sim    = BBCodeSimulator(params)
        rounds = (rounds_per_code or {}).get(params.distance, params.distance)
        all_results[params.distance] = sim.sweep_p(
            p_values,
            rounds=rounds,
            shots=shots,
            p_meas_factor=p_meas_factor,
            decoder=decoder_cls(),
            seed=seed,
        )
    return all_results
