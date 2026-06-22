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
        self.setup_from_matrices(cm.check_matrix, cm.error_priors, cm.observables_matrix)

    def setup_from_matrices(self, check_matrix, error_priors, observables_matrix) -> None:
        """Configure Relay-BP directly from a detector check matrix, per-mechanism priors,
        and an observable-action matrix — bypassing the Stim DEM.

        Used for the single-CSS-sector representation: the Stim circuit is still the source
        (build it, then ``single_sector_dem`` derives H/A/probs), but decoding runs on the
        projected sector matrices rather than the full both-sector DEM. ``check_matrix`` and
        ``observables_matrix`` may be dense uint8 arrays or scipy sparse; converted to CSR.
        """
        import relay_bp
        from scipy.sparse import csr_matrix
        H = csr_matrix(np.asarray(check_matrix, dtype=np.uint8)) if not hasattr(check_matrix, "tocsr") else check_matrix.tocsr()
        O = csr_matrix(np.asarray(observables_matrix, dtype=np.uint8)) if not hasattr(observables_matrix, "tocsr") else observables_matrix.tocsr()
        relay_decoder = relay_bp.RelayDecoderF64(
            H,
            error_priors=np.asarray(error_priors, dtype=float),
            gamma0=self._gamma0,
            pre_iter=self._pre_iter,
            num_sets=self._num_sets,
            set_max_iter=self._set_max_iter,
            gamma_dist_interval=self._gamma_dist_interval,
            stop_nconv=self._stop_nconv,
        )
        self._observable_decoder = relay_bp.ObservableDecoderRunner(relay_decoder, O)

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


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Return x s.t. A @ x = b (mod 2), or None if no solution exists."""
    m, n = A.shape
    aug = np.hstack([A.copy() % 2, (b.copy() % 2).reshape(-1, 1)])
    rref, pivots = _gf2_rref(aug)
    for row_i in range(len(pivots)):
        if pivots[row_i] == n:
            return None
    for row_i in range(len(pivots), m):
        if rref[row_i, n]:
            return None
    x = np.zeros(n, dtype=np.uint8)
    for row_i, col in enumerate(pivots):
        x[col] = rref[row_i, n]
    return x


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
    idle_noise: bool = True,
) -> stim.Circuit:
    """
    Build a noisy Stim circuit for a BB code Z-memory experiment.

    idle_noise: if True (default), apply DEPOLARIZE1(p) to every qubit left idle during
        each CNOT sub-layer — the "standard circuit noise model" (idle X/Y/Z each p/3,
        cf. arXiv:2511.15177). If False, only active gates are noisy (the older, lighter
        model). Note: enabling idle noise changes the DEM (more fault mechanisms) and hence
        all sampled results, including for the gross [[144,12,12]] code.

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
    # Build circuit — Bravyi et al. depth-7 syndrome schedule (arXiv:2308.07915,
    # github.com/sbravyi/BivariateBicycleCodes). Each syndrome cycle is 8 rounds; in
    # rounds 1-5 every qubit is busy (X-checks act on one data half, Z-checks the other),
    # idle qubits (rounds 0/6/7) pick up DEPOLARIZE1(p) — the standard circuit noise model.
    # x_layers[j]/z_layers[j] are already indexed by Bravyi "direction" j (0-5):
    #   X: 0,1,2 -> data-left via A1,A2,A3;  3,4,5 -> data-right via B1,B2,B3
    #   Z: 0,1,2 -> data-left via B1^T,..;   3,4,5 -> data-right via A1^T,..
    # -----------------------------------------------------------------------
    circuit = stim.Circuit()
    data_all = list(range(n))
    all_q = list(range(n + 2 * n_c))

    # Steering vectors: round -> monomial direction (0-5) or 'idle'.
    sX = ["idle", 1, 4, 3, 5, 0, 2]
    sZ = [3, 5, 0, 1, 2, 4, "idle"]

    # Measurement bookkeeping: absolute index of every measurement appended so far.
    meas_count = 0
    z_meas: List[Dict[int, int]] = []   # z_meas[c][s] = abs index of z_anc[s] in cycle c
    x_meas: List[Dict[int, int]] = []   # x_meas[c][s] = abs index of x_anc[s] in cycle c
    data_meas: Dict[int, int] = {}      # data_meas[q] = abs index of final data measurement

    def rec(abs_idx: int):
        return stim.target_rec(abs_idx - meas_count)

    def add_idle(active: set) -> None:
        if idle_noise and p > 0:
            idle = [q for q in all_q if q not in active]
            if idle:
                circuit.append("DEPOLARIZE1", idle, p)

    def add_cnots(pairs) -> set:
        flat = [q for pair in pairs for q in pair]
        if flat:
            circuit.append("CX", flat)
            if p > 0:
                circuit.append("DEPOLARIZE2", flat, p)
        return set(flat)

    # Initial state prep: data in |0>, Z-ancillas reset (X-ancillas are prepped each round 0).
    circuit.append("R", data_all + z_anc)
    if pm > 0:
        circuit.append("X_ERROR", data_all + z_anc, pm)
    circuit.append("TICK")

    for c in range(rounds):
        zc: Dict[int, int] = {}
        xc: Dict[int, int] = {}

        # round 0 — PrepX (reset + H the X-ancillas); Z-CNOT direction sZ[0]
        circuit.append("R", x_anc)
        if pm > 0:
            circuit.append("X_ERROR", x_anc, pm)
        circuit.append("H", x_anc)
        if p > 0:
            circuit.append("DEPOLARIZE1", x_anc, p)
        active = set(x_anc) | add_cnots(z_layers[sZ[0]])
        add_idle(active)
        circuit.append("TICK")

        # rounds 1-5 — X-CNOT (sX[t]) and Z-CNOT (sZ[t]) in parallel (all qubits busy)
        for t in range(1, 6):
            active = add_cnots(x_layers[sX[t]] + z_layers[sZ[t]])
            add_idle(active)
            circuit.append("TICK")

        # round 6 — measure Z-ancillas; final X-CNOT direction sX[6]
        if pm > 0:
            circuit.append("X_ERROR", z_anc, pm)
        circuit.append("M", z_anc)
        for s in range(n_c):
            zc[s] = meas_count + s
        meas_count += n_c
        z_meas.append(zc)
        active = set(z_anc) | add_cnots(x_layers[sX[6]])
        add_idle(active)
        # Z-check detectors: deterministic in cycle 0 (data |0>), else XOR with previous cycle.
        for s in range(n_c):
            if c == 0:
                circuit.append("DETECTOR", [rec(zc[s])], [0, s, c])
            else:
                circuit.append("DETECTOR", [rec(zc[s]), rec(z_meas[c - 1][s])], [0, s, c])
        circuit.append("TICK")

        # round 7 — measure X-ancillas; PrepZ (reset Z-ancillas); data idle
        circuit.append("H", x_anc)
        if p > 0:
            circuit.append("DEPOLARIZE1", x_anc, p)
        if pm > 0:
            circuit.append("X_ERROR", x_anc, pm)
        circuit.append("M", x_anc)
        for s in range(n_c):
            xc[s] = meas_count + s
        meas_count += n_c
        x_meas.append(xc)
        circuit.append("R", z_anc)
        if pm > 0:
            circuit.append("X_ERROR", z_anc, pm)
        add_idle(set(x_anc) | set(z_anc))   # all data idle
        # X-check detectors: first cycle's X measurement is random, so compare from cycle 1 on.
        if c > 0:
            for s in range(n_c):
                circuit.append("DETECTOR", [rec(xc[s]), rec(x_meas[c - 1][s])], [1, s, c])
        circuit.append("TICK")

    # Final transversal data measurement (Z basis).
    if pm > 0:
        circuit.append("X_ERROR", data_all, pm)
    circuit.append("M", data_all)
    for q in range(n):
        data_meas[q] = meas_count + q
    meas_count += n

    # Final Z-check detectors: reconstruct each Z-stabilizer from data, XOR last z_anc[s].
    for s in range(n_c):
        recs = [rec(data_meas[int(q)]) for q in z_conns[s]] + [rec(z_meas[rounds - 1][s])]
        circuit.append("DETECTOR", recs, [0, s, rounds])

    # Observables: logical Z operators read from the final data measurement.
    for i, lz in enumerate(log_Z):
        recs = [rec(data_meas[int(q)]) for q in np.where(lz)[0]]
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
