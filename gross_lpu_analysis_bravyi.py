"""
Gross Code [[144,12,12]] LPU Analysis Tool

Reference: "Tour de Gross: A Modular Quantum Computer Based on Bivariate
           Bicycle Codes", arXiv:2506.03094, Appendix A.3

The [[144,12,12]] Gross code is a bivariate bicycle code with parameters:
  l=12, m=6, A(x,y)=x^3+y+y^2, B(x,y)=y^3+x+x^2

Physical qubit layout (Stim convention, matching bb_code_sim.py):
  [0,    71]  — 72 data qubits, L-block  (i,j) -> i*m + j
  [72,  143]  — 72 data qubits, R-block  (i,j) -> l*m + i*m + j
  [144, 215]  — 72 X-ancilla            (i,j) -> 2*l*m + i*m + j
  [216, 287]  — 72 Z-ancilla            (i,j) -> 3*l*m + i*m + j

LPU (Logical Processing Unit) components — Appendix A.3 of arXiv:2506.03094:
  V_l, E_l, U_l          — left side, measures <X̄_1, Z̄_7>
  V_r, E_r, U_r          — right side, measures <X̄_7, Z̄_1>
  B_B (bridge data)      — connects left and right sides
  U_b (bridge checks)    — bridge ancilla
  v_Bell (Bell check)    — shared check between left and right

  Total LPU: ~90 qubits  (degree census: Fig. 7b of the paper)

Usage:
  python gross_lpu_analysis.py           — run full analysis + verification
  import gross_lpu_analysis as lpu       — use as library
"""

from __future__ import annotations

import sys
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Gross code global parameters
# ---------------------------------------------------------------------------

L = 12          # torus dimension x
M = 6           # torus dimension y
N_C = L * M     # 72: number of checks of each type (= L-block or R-block data count)
N_DATA = 2 * N_C    # 144 total data qubits
N_ANC  = 2 * N_C    # 144 total ancilla qubits
N_TOTAL = N_DATA + N_ANC  # 288

# A(x,y) = x^3 + y + y^2  — monomials as (x-exponent, y-exponent)
A_EXPS: List[Tuple[int, int]] = [(3, 0), (0, 1), (0, 2)]
# B(x,y) = y^3 + x + x^2
B_EXPS: List[Tuple[int, int]] = [(0, 3), (1, 0), (2, 0)]


# ---------------------------------------------------------------------------
# Qubit index mappings
# ---------------------------------------------------------------------------

def data_L(i: int, j: int) -> int:
    """L-block data qubit at lattice site (i, j). Returns index in [0, 72)."""
    return (i % L) * M + (j % M)

def data_R(i: int, j: int) -> int:
    """R-block data qubit at lattice site (i, j). Returns index in [72, 144)."""
    return N_C + (i % L) * M + (j % M)

def x_check_anc(i: int, j: int) -> int:
    """X-type ancilla qubit at lattice site (i, j). Returns index in [144, 216)."""
    return 2 * N_C + (i % L) * M + (j % M)

def z_check_anc(i: int, j: int) -> int:
    """Z-type ancilla qubit at lattice site (i, j). Returns index in [216, 288)."""
    return 3 * N_C + (i % L) * M + (j % M)

def qubit_type(idx: int) -> str:
    """Return block label for a physical qubit index."""
    if 0 <= idx < N_C:
        return "data_L"
    if N_C <= idx < 2 * N_C:
        return "data_R"
    if 2 * N_C <= idx < 3 * N_C:
        return "x_anc"
    if 3 * N_C <= idx < N_TOTAL:
        return "z_anc"
    return "unknown"

def qubit_coords(idx: int) -> Tuple[str, int, int]:
    """Return (block, i, j) for any physical qubit index."""
    block = qubit_type(idx)
    local = idx % N_C
    return block, local // M, local % M

def check_idx_to_anc(check_s: int, check_type: str) -> int:
    """Convert check index s (0..71) to physical ancilla qubit index."""
    i, j = check_s // M, check_s % M
    return x_check_anc(i, j) if check_type == "X" else z_check_anc(i, j)


# ---------------------------------------------------------------------------
# Tanner graph — checks, edges
# ---------------------------------------------------------------------------

def build_tanner_graph() -> Dict:
    """
    Build the full Tanner graph for the Gross code.

    X-check at (i,j): connects to L-block via A monomials, R-block via B monomials.
    Z-check at (i,j): H_Z=[B^T|A^T], so connects to L-block via -B monomials,
                      R-block via -A monomials (negative mod l,m).

    Returns dict with keys:
      x_check_data[s]  — 6 data qubit indices for X-check s
      z_check_data[s]  — 6 data qubit indices for Z-check s
      data_x_checks[d] — X-check indices containing data qubit d
      data_z_checks[d] — Z-check indices containing data qubit d
      edges            — list of (data_idx, anc_idx) pairs
    """
    x_check_data: List[List[int]] = []
    z_check_data: List[List[int]] = []

    for i in range(L):
        for j in range(M):
            # X-check neighbors
            x_nbrs = (
                [data_L(i + ax, j + ay) for (ax, ay) in A_EXPS] +
                [data_R(i + bx, j + by) for (bx, by) in B_EXPS]
            )
            x_check_data.append(x_nbrs)

            # Z-check neighbors: B^T gives L-block at -(B monomial), A^T gives R-block at -(A monomial)
            z_nbrs = (
                [data_L(i - bx, j - by) for (bx, by) in B_EXPS] +
                [data_R(i - ax, j - ay) for (ax, ay) in A_EXPS]
            )
            z_check_data.append(z_nbrs)

    data_x_checks: List[List[int]] = [[] for _ in range(N_DATA)]
    data_z_checks: List[List[int]] = [[] for _ in range(N_DATA)]
    for s, dq in enumerate(x_check_data):
        for d in dq:
            data_x_checks[d].append(s)
    for s, dq in enumerate(z_check_data):
        for d in dq:
            data_z_checks[d].append(s)

    edges: List[Tuple[int, int]] = []
    for s, dq in enumerate(x_check_data):
        anc = x_check_anc(s // M, s % M)
        for d in dq:
            edges.append((d, anc))
    for s, dq in enumerate(z_check_data):
        anc = z_check_anc(s // M, s % M)
        for d in dq:
            edges.append((d, anc))

    return {
        "x_check_data": x_check_data,
        "z_check_data": z_check_data,
        "data_x_checks": data_x_checks,
        "data_z_checks": data_z_checks,
        "edges": edges,
    }


_GRAPH_CACHE: Optional[Dict] = None

def _graph() -> Dict:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = build_tanner_graph()
    return _GRAPH_CACHE


# ---------------------------------------------------------------------------
# Syndrome extraction CNOT cycles
# ---------------------------------------------------------------------------

def cnot_cycle_schedule() -> List[Dict]:
    """
    Return the 6-layer CNOT schedule for one syndrome extraction cycle.

    The Gross code has 6 monomials total (3 in A, 3 in B), each defining one
    CNOT layer in the syndrome extraction circuit. Layers 0-2 correspond to
    A monomials; layers 3-5 to B monomials.

    Within each layer, every check applies one CNOT to one data qubit:
      X-checks: ancilla -> data  (ancilla is control)
      Z-checks: data -> ancilla  (data is control)

    Each returned dict has:
      'layer':   layer index 0..5
      'monomial': which polynomial term (e.g. "A[0]=(x^3,y^0)")
      'x_pairs': list of (x_anc_idx, data_idx) CNOT pairs for all X-checks
      'z_pairs': list of (data_idx, z_anc_idx) CNOT pairs for all Z-checks
    """
    layers = []

    # Layers 0-2: A monomials
    # X-checks: anc(i,j) -> data_L(i+ax, j+ay)
    # Z-checks: data_R(i,j) -> anc(i+ax, j+ay)   [from A^T acting on R-block]
    for k, (ax, ay) in enumerate(A_EXPS):
        x_pairs = [
            (x_check_anc(i, j), data_L(i + ax, j + ay))
            for i in range(L) for j in range(M)
        ]
        z_pairs = [
            (data_R(i, j), z_check_anc(i + ax, j + ay))
            for i in range(L) for j in range(M)
        ]
        layers.append({
            "layer": k,
            "monomial": f"A[{k}]=(x^{ax},y^{ay})",
            "x_pairs": x_pairs,
            "z_pairs": z_pairs,
        })

    # Layers 3-5: B monomials
    # X-checks: anc(i,j) -> data_R(i+bx, j+by)
    # Z-checks: data_L(i,j) -> anc(i+bx, j+by)   [from B^T acting on L-block]
    for k, (bx, by) in enumerate(B_EXPS):
        x_pairs = [
            (x_check_anc(i, j), data_R(i + bx, j + by))
            for i in range(L) for j in range(M)
        ]
        z_pairs = [
            (data_L(i, j), z_check_anc(i + bx, j + by))
            for i in range(L) for j in range(M)
        ]
        layers.append({
            "layer": 3 + k,
            "monomial": f"B[{k}]=(x^{bx},y^{by})",
            "x_pairs": x_pairs,
            "z_pairs": z_pairs,
        })

    return layers


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def qubits_in_check(check_s: int, check_type: str = "X") -> List[int]:
    """
    Return the 6 data qubit indices for check number check_s (0..71).
    check_type: "X" or "Z"
    Physical ancilla index: x_check_anc(check_s//6, check_s%6) or z_check_anc(...)
    """
    g = _graph()
    return (g["x_check_data"][check_s] if check_type == "X"
            else g["z_check_data"][check_s])

def checks_of_qubit(data_idx: int) -> Dict[str, List[int]]:
    """
    Return {"X": [...], "Z": [...]} check indices (0..71 each) for data_idx (0..143).
    """
    g = _graph()
    return {
        "X": g["data_x_checks"][data_idx],
        "Z": g["data_z_checks"][data_idx],
    }

def edges_of_qubit(data_idx: int) -> List[Tuple[int, int]]:
    """
    Return Tanner graph edges (data_idx, ancilla_idx) for a data qubit.
    3 edges to X-ancilla + 3 edges to Z-ancilla = 6 total.
    """
    g = _graph()
    result = []
    for s in g["data_x_checks"][data_idx]:
        result.append((data_idx, x_check_anc(s // M, s % M)))
    for s in g["data_z_checks"][data_idx]:
        result.append((data_idx, z_check_anc(s // M, s % M)))
    return result

def cycle_participants(layer_idx: int) -> Dict:
    """
    Return the CNOT pairs for syndrome extraction cycle layer 0..5.

    layer_idx 0-2: A monomials (X-anc<->data_L and data_R<->Z-anc)
    layer_idx 3-5: B monomials (X-anc<->data_R and data_L<->Z-anc)
    """
    return cnot_cycle_schedule()[layer_idx]

def data_qubits_in_cycle_layer(layer_idx: int) -> Set[int]:
    """Return the set of data qubit indices that participate in CNOT layer layer_idx."""
    layer = cnot_cycle_schedule()[layer_idx]
    dq: Set[int] = set()
    for _, d in layer["x_pairs"]:
        dq.add(d)
    for d, _ in layer["z_pairs"]:
        dq.add(d)
    return dq

def subgraph_cycles(
    data_qubits: Set[int],
    anc_qubits: Set[int],
) -> List[List[int]]:
    """
    Find the minimum cycle basis of the induced Tanner subgraph.
    Nodes = data_qubits ∪ anc_qubits; edges from the full Tanner graph restricted
    to these nodes.
    Requires networkx.
    """
    if not HAS_NX:
        raise ImportError("pip install networkx  to use subgraph_cycles()")
    g = _graph()
    G = nx.Graph()
    for d in data_qubits:
        G.add_node(d)
    for a in anc_qubits:
        G.add_node(a)
    for d in data_qubits:
        for s in g["data_x_checks"][d]:
            anc = x_check_anc(s // M, s % M)
            if anc in anc_qubits:
                G.add_edge(d, anc)
        for s in g["data_z_checks"][d]:
            anc = z_check_anc(s // M, s % M)
            if anc in anc_qubits:
                G.add_edge(d, anc)
    return list(nx.minimum_cycle_basis(G))


# ---------------------------------------------------------------------------
# GF(2) linear algebra (self-contained, no external deps)
# ---------------------------------------------------------------------------

def _gf2_rref(A: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    A = A.copy() % 2
    rows, cols = A.shape
    pivot_row = 0
    pivots: List[int] = []
    for col in range(cols):
        found = next((r for r in range(pivot_row, rows) if A[r, col]), -1)
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
    return (np.array(null_vecs, dtype=np.uint8) if null_vecs
            else np.empty((0, n), dtype=np.uint8))


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
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


def _reduce_mod(vectors: np.ndarray, stab_rref: np.ndarray,
                stab_pivots: List[int]) -> np.ndarray:
    out = []
    for v in vectors:
        v = v.copy() % 2
        for row_i, pcol in enumerate(stab_pivots):
            if v[pcol]:
                v = (v + stab_rref[row_i]) % 2
        if v.any():
            out.append(v)
    if not out:
        return np.empty((0, vectors.shape[1]), dtype=np.uint8)
    rref, pivots = _gf2_rref(np.array(out, dtype=np.uint8))
    return rref[:len(pivots)].copy()


def build_parity_checks() -> Tuple[np.ndarray, np.ndarray]:
    """Return (H_X, H_Z) for the Gross code. Shapes: (72, 144) each."""
    def poly_mat(exps: List[Tuple[int, int]]) -> np.ndarray:
        M_ = np.zeros((N_C, N_C), dtype=np.uint8)
        for (ax, ay) in exps:
            for i in range(L):
                for j in range(M):
                    s = i * M + j
                    t = ((i + ax) % L) * M + ((j + ay) % M)
                    M_[s, t] ^= 1
        return M_

    A = poly_mat(A_EXPS)
    B = poly_mat(B_EXPS)
    H_X = np.hstack([A, B]).astype(np.uint8)
    H_Z = np.hstack([B.T % 2, A.T % 2]).astype(np.uint8)
    return H_X, H_Z


def compute_logical_operators() -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute all 12 pairs of logical operators (log_Z, log_X).
    Each row is a weight-12 operator on the 144 data qubits.
    Returns (log_Z, log_X), shapes (12, 144).
    """
    H_X, H_Z = build_parity_checks()

    null_X = _gf2_nullspace(H_X)
    stab_Z_rref, pivots_Z = _gf2_rref(H_Z)
    log_Z = _reduce_mod(null_X, stab_Z_rref, pivots_Z)
    k = len(log_Z)

    A_aug = np.vstack([H_Z, log_Z]).astype(np.uint8)
    log_X = np.zeros((k, H_Z.shape[1]), dtype=np.uint8)
    for i in range(k):
        b = np.zeros(H_Z.shape[0] + k, dtype=np.uint8)
        b[H_Z.shape[0] + i] = 1
        x = _gf2_solve(A_aug, b)
        assert x is not None, f"log_X[{i}] has no solution — CSS violation"
        log_X[i] = x

    return log_Z, log_X


# ---------------------------------------------------------------------------
# LPU qubit sets  (arXiv:2506.03094, Appendix A.3)
# ---------------------------------------------------------------------------
#
# The LPU is organized as:
#
#   Left side  (V_l, E_l, U_l, v_Bell)  → measures <X̄_1, Z̄_7>
#   Right side (V_r, E_r, U_r, v_Bell)  → measures <X̄_7, Z̄_1>
#   Bridge     (B_B data, U_b checks)   → enables joint measurement
#
# Degree census from Figure 7(b) of the paper (total ≈ 90 qubits):
#
#   Component | #qubits | degree distribution
#   ----------|---------|---------------------
#   V_l       |   11    | 6: 11
#   E_l       |   18    | 4: 1, 5: 2, 6: 13, 7: 2
#   U_l       |    5    | 4: 2, 5: 3
#   B_B       |   11    | 3: 1, 4: 10
#   U_b       |   11    | 3: 1, 4: 10
#   U_r       |    3    | 4: 2, 5: 1
#   E_r       |   18    | 4: 2, 5: 8, 6: 8
#   V_r       |   11    | 6: 11
#
# The sets below are seeded by the logical operator supports (derivable from
# the math) and should be updated with the exact qubit coordinates from
# Appendix A.3 using set_lpu_component().

_LPU_SETS: Optional[Dict[str, List[int]]] = None


# ---------------------------------------------------------------------------
# Monomial helper
# ---------------------------------------------------------------------------

def _mono(xi: int, yi: int, block: str) -> int:
    """Convert a monomial x^xi * y^yi in the given block ('L' or 'R') to a physical index."""
    i, j = xi % L, yi % M
    return data_L(i, j) if block == "L" else data_R(i, j)


def derive_lpu_from_paper() -> Dict[str, object]:
    """
    Derive all LPU qubit sets from Appendix A.3 of arXiv:2506.03094.

    For the Gross code, the paper selects (Eq. 32):
      p = x^4 + x^5 + x^6y + x^4y^2 + x^5y^4 + x^6y^5
      q = x^3 + x^4 + x^3y + x^3y^2 + x^4y^2 + x^3y^5
      r = 1  + x^8  + xy   + x^9y   + x^3y^4  + x^11y^4
      s = x  + x^9  + x^4y^4 + x^8y^4 + y^5  + x^8y^5

    X̄_1 = X(p,q) supported on pL + qR  (V_l vertices + identified vertex)
    X̄_7 = X(r,s) supported on rL + sR  (V_r vertices + identified vertex)

    Identified vertex (shared bridge point):
      x^4R  = data_R(4,0) = 96   (from G_l / X̄_1 side)
      x^9yL = data_L(9,1) = 55   (from G_r / X̄_7 side)
    These two gross-code data qubits are connected by the Bell-pair check (v_Bell).

    New LPU ancilla qubits (indices 288+):
      288-305  E_l  (18 edge qubits on G_l edges)
      306-310  U_l  (5 cycle-check qubits for G_l)
      311-321  B_B  (11 bridge data qubits)
      322-332  U_b  (11 bridge cycle-check qubits)
      333-335  U_r  (3 cycle-check qubits for G_r)
      336-353  E_r  (18 edge qubits on G_r edges)
      354-355  v_Bell (2 Bell-pair ancilla qubits for the identified vertex)

    Returns a dict with all component qubit index lists plus graph structure.
    """
    # ------------------------------------------------------------------
    # V_l: support of X̄_1 = X(p,q), 12 monomials, minus identified vertex x^4R
    # ------------------------------------------------------------------
    p_L = [_mono(4,0,'L'), _mono(5,0,'L'), _mono(6,1,'L'),
           _mono(4,2,'L'), _mono(5,4,'L'), _mono(6,5,'L')]   # [24,30,37,26,34,41]
    q_R = [_mono(3,0,'R'), _mono(4,0,'R'), _mono(3,1,'R'),
           _mono(3,2,'R'), _mono(4,2,'R'), _mono(3,5,'R')]   # [90,96,91,92,98,95]

    identified_L = _mono(4,0,'R')   # x^4R = 96 — identified with x^9yL
    identified_R = _mono(9,1,'L')   # x^9yL = 55 — identified with x^4R

    V_l = sorted(set(p_L + q_R) - {identified_L})   # 11 gross-code data qubits
    v_bell_data = sorted([identified_L, identified_R]) # [55, 96] — both gross-code qubits

    # ------------------------------------------------------------------
    # V_r: support of X̄_7 = X(r,s), minus identified vertex x^9yL
    # ------------------------------------------------------------------
    r_L = [_mono(0,0,'L'), _mono(8,0,'L'), _mono(1,1,'L'),
           _mono(9,1,'L'), _mono(3,4,'L'), _mono(11,4,'L')]  # [0,48,7,55,22,70]
    s_R = [_mono(1,0,'R'), _mono(9,0,'R'), _mono(4,4,'R'),
           _mono(8,4,'R'), _mono(0,5,'R'), _mono(8,5,'R')]   # [78,126,100,124,77,125]

    V_r = sorted(set(r_L + s_R) - {identified_R})   # 11 gross-code data qubits

    # ------------------------------------------------------------------
    # E_l: 18 new data qubits, one per edge of G_l.
    # An edge exists between γ,δ ∈ V_l ∪ {identified_L} when they share a Z-check.
    # ------------------------------------------------------------------
    g = _graph()
    v_l_full = set(V_l) | {identified_L}   # 12 vertices of G_l
    v_r_full = set(V_r) | {identified_R}   # 12 vertices of G_r

    def _z_check_edges(vertex_set: Set[int]) -> List[Tuple[int, int]]:
        """Find all pairs in vertex_set that share a Z-check (edges of G)."""
        edges: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for s in range(N_C):
            dq = g["z_check_data"][s]
            members = [d for d in dq if d in vertex_set]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    e = (min(members[a], members[b]), max(members[a], members[b]))
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)
        return sorted(edges)

    gl_edges = _z_check_edges(v_l_full)   # should be 18 edges
    gr_edges = _z_check_edges(v_r_full)   # should be 18 edges

    # Assign new physical indices: E_l starts at N_TOTAL (= 288)
    E_l_base = N_TOTAL                              # 288
    E_r_base = E_l_base + len(gl_edges)             # 306
    U_l_base = E_r_base + len(gr_edges)             # 324
    B_B_base = U_l_base + 5                         # 329
    U_b_base = B_B_base + 11                        # 340
    U_r_base = U_b_base + 11                        # 351
    v_Bell_base = U_r_base + 3                      # 354

    E_l = list(range(E_l_base, E_l_base + len(gl_edges)))          # 18 qubits
    E_r = list(range(E_r_base, E_r_base + len(gr_edges)))          # 18 qubits
    U_l = list(range(U_l_base, U_l_base + 5))                      # 5 qubits
    B_B = list(range(B_B_base, B_B_base + 11))                     # 11 qubits
    U_b = list(range(U_b_base, U_b_base + 11))                     # 11 qubits
    U_r = list(range(U_r_base, U_r_base + 3))                      # 3 qubits
    v_Bell = list(range(v_Bell_base, v_Bell_base + 2))              # 2 qubits

    # ------------------------------------------------------------------
    # G_l cycle basis (5 cycles from Appendix A.3, p.49 of arXiv:2506.03094)
    # Each cycle is a sequence of data qubit indices (vertices of G_l).
    # The corresponding U_l[k] is the check qubit for cycle k.
    # ------------------------------------------------------------------
    gl_cycles_monomials = [
        # Cycle 0: x^4L -> x^3y^2R -> x^3yR -> x^5L -> x^4L
        [_mono(4,0,'L'), _mono(3,2,'R'), _mono(3,1,'R'), _mono(5,0,'L')],
        # Cycle 1: x^3y^5R -> x^3R -> x^6y^5L -> x^4y^2L -> x^3y^5R
        [_mono(3,5,'R'), _mono(3,0,'R'), _mono(6,5,'L'), _mono(4,2,'L')],
        # Cycle 2: x^3y^5R -> x^3R -> x^6y^5L -> x^4R -> x^5y^4L -> x^3y^5R
        [_mono(3,5,'R'), _mono(3,0,'R'), _mono(6,5,'L'), _mono(4,0,'R'), _mono(5,4,'L')],
        # Cycle 3: x^4L -> x^3y^2R -> x^6yL -> x^5y^4L -> x^4R -> x^4L
        [_mono(4,0,'L'), _mono(3,2,'R'), _mono(6,1,'L'), _mono(5,4,'L'), _mono(4,0,'R')],
        # Cycle 4: x^3yR -> x^6yL -> x^4y^2R -> x^5L -> x^3yR
        [_mono(3,1,'R'), _mono(6,1,'L'), _mono(4,2,'R'), _mono(5,0,'L')],
    ]

    # ------------------------------------------------------------------
    # G_r cycle basis (3 cycles from Appendix A.3, p.49)
    # ------------------------------------------------------------------
    gr_cycles_monomials = [
        # Cycle 0: x^9yL -> x^8y^4R -> x^8y^5R -> x^11y^4L -> x^9yL
        [_mono(9,1,'L'), _mono(8,4,'R'), _mono(8,5,'R'), _mono(11,4,'L')],
        # Cycle 1: xyL -> xR -> x^4y^4R -> x^3y^4L -> xyL
        [_mono(1,1,'L'), _mono(1,0,'R'), _mono(4,4,'R'), _mono(3,4,'L')],
        # Cycle 2: 1L -> y^5R -> x^9R -> x^9yL -> x^8y^4R -> 1L
        [_mono(0,0,'L'), _mono(0,5,'R'), _mono(9,0,'R'), _mono(9,1,'L'), _mono(8,4,'R')],
    ]

    # ------------------------------------------------------------------
    # Bridge edges: 11 pairs (G_l vertex, G_r vertex) from Eq.(39)
    # B_B[k] is the new data qubit on bridge edge k.
    # U_b[k] is the new check qubit for the cycle created by bridge edge k.
    # ------------------------------------------------------------------
    bridge_edges_monomials = [
        (_mono(6,5,'L'), _mono(11,4,'L')),   # x^6y^5L <-> x^11y^4L
        (_mono(4,2,'L'), _mono(8,5,'R')),    # x^4y^2L <-> x^8y^5R
        (_mono(3,5,'R'), _mono(8,4,'R')),    # x^3y^5R <-> x^8y^4R
        (_mono(5,4,'L'), _mono(0,0,'L')),    # x^5y^4L <-> 1L
        (_mono(6,1,'L'), _mono(1,0,'R')),    # x^6yL   <-> xR
        (_mono(4,2,'R'), _mono(4,4,'R')),    # x^4y^2R <-> x^4y^4R
        (_mono(5,0,'L'), _mono(8,0,'L')),    # x^5L    <-> x^8L
        (_mono(4,0,'L'), _mono(9,0,'R')),    # x^4L    <-> x^9R
        (_mono(3,2,'R'), _mono(0,5,'R')),    # x^3y^2R <-> y^5R
        (_mono(3,1,'R'), _mono(3,4,'L')),    # x^3yR   <-> x^3y^4L
        (_mono(3,0,'R'), _mono(1,1,'L')),    # x^3R    <-> xyL
    ]
    bridge_edges = [(gl_v, gr_v) for gl_v, gr_v in bridge_edges_monomials]

    # Additional bridge cycle through identified vertex (Eq. 40):
    # x^9yL = x^4R -> x^6y^5L -> x^11y^4L -> x^9yL = x^4R
    identified_cycle = [identified_L, _mono(6,5,'L'), _mono(11,4,'L'), identified_R]

    return {
        # --- Gross-code data qubits (indices 0..143) ---
        "V_l":        sorted(V_l),           # 11 data qubits (left side vertex set)
        "V_r":        sorted(V_r),           # 11 data qubits (right side vertex set)
        "v_bell_data": v_bell_data,          # [55, 96]: the two identified-vertex data qubits
        # --- New LPU qubits (indices 288+) ---
        "E_l":        E_l,                   # 18 edge data qubits of G_l
        "U_l":        U_l,                   # 5 cycle-check qubits of G_l
        "B_B":        B_B,                   # 11 bridge data qubits
        "U_b":        U_b,                   # 11 bridge cycle-check qubits
        "U_r":        U_r,                   # 3 cycle-check qubits of G_r
        "E_r":        E_r,                   # 18 edge data qubits of G_r
        "v_Bell":     v_Bell,                # 2 Bell-pair ancilla qubits
        # --- Graph structure ---
        "gl_edges":   gl_edges,              # (data,data) edge pairs of G_l -> E_l index
        "gr_edges":   gr_edges,              # (data,data) edge pairs of G_r -> E_r index
        "gl_cycles":  gl_cycles_monomials,   # 5 cycles (vertex sequences) -> U_l
        "gr_cycles":  gr_cycles_monomials,   # 3 cycles -> U_r
        "bridge_edges": bridge_edges,        # 11 (G_l vertex, G_r vertex) -> B_B
        "identified_cycle": identified_cycle, # cycle through identified vertex (Eq.40)
        # --- Convenience lookups ---
        "E_l_edge_map": {e: E_l[k] for k, e in enumerate(gl_edges)},  # edge -> new qubit
        "E_r_edge_map": {e: E_r[k] for k, e in enumerate(gr_edges)},
        "B_B_edge_map": {e: B_B[k] for k, e in enumerate(bridge_edges)},
        # identified vertex
        "identified_L": identified_L,        # x^4R = 96
        "identified_R": identified_R,        # x^9yL = 55
    }


def _compute_lpu_seeds() -> Dict[str, List[int]]:
    """Build LPU sets directly from Appendix A.3 (no logical-op computation needed)."""
    d = derive_lpu_from_paper()
    return {
        # Gross-code data qubits
        "V_l":        d["V_l"],
        "V_r":        d["V_r"],
        "v_bell_data": d["v_bell_data"],
        # New LPU qubits
        "E_l":        d["E_l"],
        "U_l":        d["U_l"],
        "B_B":        d["B_B"],
        "U_b":        d["U_b"],
        "U_r":        d["U_r"],
        "E_r":        d["E_r"],
        "v_Bell":     d["v_Bell"],
        # Graph structure (for reference)
        "gl_edges":   d["gl_edges"],
        "gr_edges":   d["gr_edges"],
        "gl_cycles":  d["gl_cycles"],
        "gr_cycles":  d["gr_cycles"],
        "bridge_edges": d["bridge_edges"],
        "identified_cycle": d["identified_cycle"],
        "E_l_edge_map": d["E_l_edge_map"],
        "E_r_edge_map": d["E_r_edge_map"],
        "B_B_edge_map": d["B_B_edge_map"],
        "identified_L": d["identified_L"],
        "identified_R": d["identified_R"],
    }


def get_lpu_sets(recompute: bool = False) -> Dict[str, List[int]]:
    """Return the LPU qubit set dict, computing logical ops on first call."""
    global _LPU_SETS
    if _LPU_SETS is None or recompute:
        _LPU_SETS = _compute_lpu_seeds()
    return _LPU_SETS


def lpu_component(name: str) -> List[int]:
    """
    Return physical qubit indices for the named LPU component.
    Seed names: 'V_l_seed', 'V_r_seed', 'lx0_seed', 'lx6_seed',
                'U_l_seed_xanc', 'U_l_seed_zanc', 'U_r_seed_xanc', 'U_r_seed_zanc'
    Paper names (Appendix A.3): 'V_l', 'E_l', 'U_l', 'V_r', 'E_r', 'U_r',
                                 'B_B', 'U_b', 'v_Bell'
    """
    return get_lpu_sets()[name]


def set_lpu_component(name: str, qubit_indices: List[int]) -> None:
    """
    Set a LPU component to the exact qubit indices from Appendix A.3.
    Example:
        set_lpu_component('V_l', [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60])
    """
    get_lpu_sets()[name] = sorted(qubit_indices)


def lpu_all_qubits() -> Set[int]:
    """Return the union of all LPU qubit index sets (gross-code + new LPU qubits)."""
    sets = get_lpu_sets()
    names = ["V_l", "E_l", "U_l", "V_r", "E_r", "U_r", "B_B", "U_b", "v_Bell",
             "v_bell_data"]
    result: Set[int] = set()
    for name in names:
        v = sets.get(name, [])
        if isinstance(v, list):
            result.update(v)
    return result


# ---------------------------------------------------------------------------
# Connectivity diagram
# ---------------------------------------------------------------------------

def adjacency_dict() -> Dict[int, List[int]]:
    """
    Full qubit adjacency for all 288 physical qubits.
    adj[q] = sorted list of Tanner-graph neighbors.
    Data qubits list ancilla neighbors; ancilla list data neighbors.
    """
    g = _graph()
    adj: Dict[int, List[int]] = {q: [] for q in range(N_TOTAL)}
    for d, a in g["edges"]:
        adj[d].append(a)
        adj[a].append(d)
    for q in adj:
        adj[q] = sorted(adj[q])
    return adj


def print_connectivity_summary(subset: Optional[List[int]] = None,
                               show_all: bool = False) -> None:
    """
    Print physical qubit connectivity.

    subset: restrict to these qubit indices (None = all 288)
    show_all: if False (default), truncate long neighbor lists
    """
    g = _graph()
    qubits = subset if subset is not None else list(range(N_TOTAL))

    degree_hist: Dict[int, int] = {}
    header = f"Gross Code [[144,12,12]] — Connectivity"
    if subset:
        header += f" (subset: {len(subset)} qubits)"
    print(f"\n{'='*65}")
    print(header)
    print(f"{'='*65}")
    print(f"{'Index':>6}  {'Block':<10} {'(i,j)':>6}  {'deg':>3}  Neighbors")
    print(f"{'-'*65}")

    for idx in qubits:
        block, i, j = qubit_coords(idx)

        if block in ("data_L", "data_R"):
            xc = g["data_x_checks"][idx]
            zc = g["data_z_checks"][idx]
            anc_nbrs = (
                [x_check_anc(s // M, s % M) for s in xc] +
                [z_check_anc(s // M, s % M) for s in zc]
            )
            deg = len(anc_nbrs)
            nbr_str = str(sorted(anc_nbrs))
            label = block
        elif block == "x_anc":
            s = idx - 2 * N_C
            dq = g["x_check_data"][s]
            deg = len(dq)
            nbr_str = str(sorted(dq))
            label = "x_anc"
        else:  # z_anc
            s = idx - 3 * N_C
            dq = g["z_check_data"][s]
            deg = len(dq)
            nbr_str = str(sorted(dq))
            label = "z_anc"

        degree_hist[deg] = degree_hist.get(deg, 0) + 1
        if not show_all and len(nbr_str) > 55:
            nbr_str = nbr_str[:52] + "..."
        print(f"  {idx:4d}  {label:<10} ({i:2d},{j:1d})  {deg:3d}  {nbr_str}")

    print(f"\nDegree histogram: {dict(sorted(degree_hist.items()))}")


def print_check_enumeration(check_type: str = "X",
                             max_checks: Optional[int] = None) -> None:
    """
    Print all checks with their data qubit members and ancilla index.
    check_type: "X" or "Z"
    """
    n = N_C if max_checks is None else min(max_checks, N_C)
    print(f"\n=== {check_type}-Checks (showing {n}/{N_C}) ===")
    print(f"  {'#':>3}  anc_idx  (i, j)  data qubits (6)")
    for s in range(n):
        i, j = s // M, s % M
        anc = x_check_anc(i, j) if check_type == "X" else z_check_anc(i, j)
        dq  = qubits_in_check(s, check_type)
        print(f"  {s:3d}  {anc:7d}  ({i:2d},{j:1d})  {dq}")
    if max_checks and max_checks < N_C:
        print(f"  ... ({N_C - n} more)")


def print_cycle_schedule() -> None:
    """Print the full 6-layer CNOT syndrome extraction schedule."""
    print("\n=== Syndrome Extraction CNOT Cycle Schedule (6 layers) ===")
    print(f"  Each layer applies 72 CNOT gates simultaneously.")
    print(f"  X-check: ancilla -> data qubit (ancilla is control)")
    print(f"  Z-check: data qubit -> ancilla (data is control)")
    for layer in cnot_cycle_schedule():
        li = layer["layer"]
        mono = layer["monomial"]
        xp = layer["x_pairs"]
        zp = layer["z_pairs"]
        print(f"\n  Layer {li}: {mono}")
        print(f"    X-check pairs (anc, data) [{len(xp)} total]: {xp[:4]} ...")
        print(f"    Z-check pairs (data, anc) [{len(zp)} total]: {zp[:4]} ...")
        # Which data qubits are touched?
        x_data = sorted({d for _, d in xp})
        z_data = sorted({d for d, _ in zp})
        print(f"    Data qubits in X-CNOTs: {len(x_data)} qubits, first 6: {x_data[:6]}")
        print(f"    Data qubits in Z-CNOTs: {len(z_data)} qubits, first 6: {z_data[:6]}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_code_structure() -> bool:
    """
    Verify fundamental Gross code properties.
    Returns True if all checks pass.
    """
    ok = True
    print("\n--- Verification ---")
    g = _graph()

    # Regularity: each data qubit in exactly 3 X-checks and 3 Z-checks
    for d in range(N_DATA):
        nx_ = len(g["data_x_checks"][d])
        nz_ = len(g["data_z_checks"][d])
        if nx_ != 3 or nz_ != 3:
            print(f"  FAIL: data qubit {d} in {nx_} X-checks, {nz_} Z-checks (expected 3 each)")
            ok = False
    print("  PASS: every data qubit in exactly 3 X-checks + 3 Z-checks")

    # Check weight: each check involves 6 data qubits
    for s in range(N_C):
        wx = len(g["x_check_data"][s])
        wz = len(g["z_check_data"][s])
        if wx != 6 or wz != 6:
            print(f"  FAIL: X-check {s} weight={wx}, Z-check {s} weight={wz} (expected 6)")
            ok = False
    print("  PASS: every check has weight 6")

    # Total Tanner graph edges: N_DATA * 6 = 864
    expected_edges = N_DATA * 6
    if len(g["edges"]) != expected_edges:
        print(f"  FAIL: {len(g['edges'])} edges (expected {expected_edges})")
        ok = False
    else:
        print(f"  PASS: {len(g['edges'])} Tanner graph edges (= {N_DATA}×6)")

    # CSS condition: H_X @ H_Z^T = 0 mod 2
    H_X, H_Z = build_parity_checks()
    if np.any((H_X @ H_Z.T) % 2 != 0):
        print("  FAIL: CSS condition H_X · H_Z^T ≠ 0 mod 2")
        ok = False
    else:
        print("  PASS: CSS condition H_X · H_Z^T = 0 (mod 2)")

    # CNOT schedule: each monomial layer touches all N_C data qubits exactly once
    for layer in cnot_cycle_schedule():
        x_data = [d for _, d in layer["x_pairs"]]
        z_data = [d for d, _ in layer["z_pairs"]]
        if len(set(x_data)) != N_C or len(set(z_data)) != N_C:
            print(f"  FAIL: layer {layer['layer']} CNOT schedule has duplicate targets")
            ok = False
    print("  PASS: each CNOT layer touches all 72 checks exactly once")

    status = "All checks passed." if ok else "Some checks FAILED."
    print(f"\n  {status}")
    return ok


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_connectivity(
    subset: Optional[List[int]] = None,
    highlight_lpu: bool = True,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Plot the Tanner graph.

    subset: restrict to these qubit indices; None = all 288 qubits
    highlight_lpu: color LPU qubits in gold (if any have been set)
    """
    if not HAS_NX or not HAS_MPL:
        print("matplotlib + networkx required: pip install matplotlib networkx")
        return

    g = _graph()
    lpu_q = lpu_all_qubits() if highlight_lpu else set()

    node_set = set(subset) if subset is not None else set(range(N_TOTAL))
    G = nx.Graph()
    for d, a in g["edges"]:
        if d in node_set and a in node_set:
            G.add_edge(d, a)
    # Add isolated nodes too
    for n_ in node_set:
        G.add_node(n_)

    color_map = {
        "data_L": "steelblue",
        "data_R": "darkorange",
        "x_anc":  "crimson",
        "z_anc":  "limegreen",
    }
    colors = []
    for node in G.nodes():
        if node in lpu_q:
            colors.append("gold")
        else:
            colors.append(color_map.get(qubit_type(node), "gray"))

    pos = nx.spring_layout(G, seed=42, k=0.3)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(G, pos=pos, node_color=colors, node_size=30,
                     with_labels=False, ax=ax, edge_color="gray",
                     alpha=0.75, width=0.5)

    legend = [
        mpatches.Patch(color="steelblue",  label="Data L-block [0–71]"),
        mpatches.Patch(color="darkorange", label="Data R-block [72–143]"),
        mpatches.Patch(color="crimson",    label="X-ancilla [144–215]"),
        mpatches.Patch(color="limegreen",  label="Z-ancilla [216–287]"),
    ]
    if lpu_q:
        legend.append(mpatches.Patch(color="gold", label="LPU qubit"))
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    title = "Gross Code [[144,12,12]] Tanner Graph"
    if subset:
        title += f" — {len(subset)}-qubit subset"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_torus_layout(
    highlight: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """
    Plot qubits on the 12×6 torus using a flat (i,j) grid.
    Data L-block and R-block are side by side.
    highlight: list of physical qubit indices to mark in red.
    """
    if not HAS_MPL:
        print("matplotlib required: pip install matplotlib")
        return

    hl = set(highlight) if highlight else set()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    titles = ["L-block data (indices 0-71)", "R-block data (indices 72-143)"]
    offsets = [0, N_C]

    for ax, offset, title in zip(axes, offsets, titles):
        # Build a value grid: 1.0 = highlighted (gold), 0.0 = normal (light blue)
        val_grid = np.zeros((L, M))
        for i in range(L):
            for j in range(M):
                if offset + i * M + j in hl:
                    val_grid[i, j] = 1.0

        # Two-tone colormap: light blue for normal, gold for highlighted
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "torus", ["#d0e8f8", "#FFD700"], N=2
        )
        ax.imshow(val_grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for i in range(L):
            for j in range(M):
                idx = offset + i * M + j
                is_hl = idx in hl
                ax.text(j, i, str(idx), ha="center", va="center",
                        fontsize=7.5,
                        color="black",
                        fontweight="bold" if is_hl else "normal")

        ax.set_xticks(range(M))
        ax.set_xticklabels([f"j={j}" for j in range(M)])
        ax.set_yticks(range(L))
        ax.set_yticklabels([f"i={i}" for i in range(L)])
        ax.set_title(title)
        ax.set_xlabel("j  (y-axis)")
        ax.set_ylabel("i  (x-axis)")

    plt.suptitle("Gross Code [[144,12,12]] — Data Qubit Torus Layout (l=12, m=6)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Gross Code [[144,12,12]] LPU Analysis")
    print("Reference: Tour de Gross (arXiv:2506.03094)")
    print("=" * 65)
    print(f"\nCode parameters:")
    print(f"  l={L}, m={M}  =>  l*m = {N_C} checks of each type")
    print(f"  A(x,y) = x^3 + y + y^2   monomials: {A_EXPS}")
    print(f"  B(x,y) = y^3 + x + x^2   monomials: {B_EXPS}")
    print(f"  Data qubits : {N_DATA}  (L-block: 0..{N_C-1},  R-block: {N_C}..{N_DATA-1})")
    print(f"  X-ancilla   : {N_C}   (indices {2*N_C}..{3*N_C-1})")
    print(f"  Z-ancilla   : {N_C}   (indices {3*N_C}..{N_TOTAL-1})")
    print(f"  Total qubits: {N_TOTAL}")

    # --- Verify code structure ---
    verify_code_structure()

    # --- Checks ---
    print_check_enumeration("X", max_checks=6)
    print_check_enumeration("Z", max_checks=6)

    # --- CNOT cycles ---
    print_cycle_schedule()

    # --- Edges of a few data qubits ---
    print("\n=== Tanner graph edges for data qubits 0, 1, 72 ===")
    for d in [0, 1, 72]:
        block, i, j = qubit_coords(d)
        print(f"\n  Q[{d}] {block} ({i},{j}):")
        for data, anc in edges_of_qubit(d):
            ab, ai, aj = qubit_coords(anc)
            print(f"    edge ({data}, {anc})  →  {ab}({ai},{aj})")

    # --- Checks of a data qubit ---
    print("\n=== Checks involving data qubit 0 ===")
    ch = checks_of_qubit(0)
    print(f"  X-checks: {ch['X']}  (ancilla indices {[x_check_anc(s//M, s%M) for s in ch['X']]})")
    print(f"  Z-checks: {ch['Z']}  (ancilla indices {[z_check_anc(s//M, s%M) for s in ch['Z']]})")

    # --- Connectivity summary (first 12 data qubits) ---
    print_connectivity_summary(subset=list(range(12)))

    # --- LPU structure from Appendix A.3 ---
    print("\n=== LPU Qubit Sets (arXiv:2506.03094 Appendix A.3) ===")
    lpu = get_lpu_sets()
    print(f"  V_l  ({len(lpu['V_l'])} gross-code data qubits, X1-bar side): {lpu['V_l']}")
    print(f"  V_r  ({len(lpu['V_r'])} gross-code data qubits, X7-bar side): {lpu['V_r']}")
    print(f"  Identified vertex: x^4R={lpu['identified_L']} <-> x^9yL={lpu['identified_R']}")
    print(f"  E_l  ({len(lpu['E_l'])} new edge qubits): indices {lpu['E_l'][0]}..{lpu['E_l'][-1]}")
    print(f"  U_l  ({len(lpu['U_l'])} cycle checks):   {lpu['U_l']}")
    print(f"  B_B  ({len(lpu['B_B'])} bridge data):    {lpu['B_B']}")
    print(f"  U_b  ({len(lpu['U_b'])} bridge checks):  {lpu['U_b']}")
    print(f"  U_r  ({len(lpu['U_r'])} cycle checks):   {lpu['U_r']}")
    print(f"  E_r  ({len(lpu['E_r'])} new edge qubits): indices {lpu['E_r'][0]}..{lpu['E_r'][-1]}")
    print(f"  v_Bell ({len(lpu['v_Bell'])} Bell ancilla): {lpu['v_Bell']}")
    print(f"  Total LPU qubits: {len(lpu_all_qubits())}  (paper census: 90)")
    print(f"\n  G_l edges ({len(lpu['gl_edges'])}, should be 18):")
    for k, (a, b) in enumerate(lpu['gl_edges']):
        ca, cb = qubit_coords(a), qubit_coords(b)
        print(f"    E_l[{k:2d}]={lpu['E_l'][k]}  ({a},{b})  "
              f"{ca[0]}({ca[1]},{ca[2]}) -- {cb[0]}({cb[1]},{cb[2]})")
    print(f"\n  G_l cycles (U_l):")
    for k, cycle in enumerate(lpu['gl_cycles']):
        print(f"    U_l[{k}]={lpu['U_l'][k]}  {cycle}")
    print(f"\n  G_r edges ({len(lpu['gr_edges'])}, should be 18):")
    print(f"  G_r cycles (U_r):")
    for k, cycle in enumerate(lpu['gr_cycles']):
        print(f"    U_r[{k}]={lpu['U_r'][k]}  {cycle}")
    print(f"\n  Bridge edges (B_B):")
    for k, (gl_v, gr_v) in enumerate(lpu['bridge_edges']):
        print(f"    B_B[{k:2d}]={lpu['B_B'][k]}  {gl_v} <-> {gr_v}")

    print("\n=== Library API Summary ===")
    print("  qubits_in_check(s, 'X'/'Z')       → 6 data qubit indices for check s")
    print("  checks_of_qubit(d)                 → {'X': [...], 'Z': [...]} for data d")
    print("  edges_of_qubit(d)                  → [(d, anc), ...] Tanner graph edges")
    print("  cycle_participants(layer)           → CNOT pairs for layer 0..5")
    print("  data_qubits_in_cycle_layer(layer)  → set of data qubits in that layer")
    print("  subgraph_cycles(data_set, anc_set) → minimum cycle basis (needs networkx)")
    print("  lpu_component('V_l')               → qubit list for LPU component")
    print("  set_lpu_component('V_l', [...])    → set component from Appendix A.3")
    print("  print_connectivity_summary(subset) → text connectivity table")
    print("  adjacency_dict()                   → adj[q] = sorted neighbor list")
    print("  plot_connectivity(subset)          → Tanner graph plot")
    print("  plot_torus_layout(highlight)       → 12×6 torus qubit grid")
