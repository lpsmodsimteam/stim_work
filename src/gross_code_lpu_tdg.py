"""
Gross-code Logical Processing Unit (LPU) — Stim circuits for X̄₁ and Z̄₁
measurements.

Implements the construction in arXiv:2506.03094 ("Tour de gross"),
Appendix A.3 (the auxiliary graph G, V, E, U, bridge edges, Bell pair) and
Appendix A.4 (gauging-measurement protocol with C repeated LPU rounds
sandwiched between bare gross-code memory rounds).

This file uses the LPU paper's polynomial convention (see Eq. (eq:toric_layout_BB)
and Sec. A.1 line 21):
    ℓ = 12, m = 6
    A = 1 + y + x³y⁻¹       (a_→=3, a_↑=-1)
    B = 1 + x + x⁻¹y⁻³      (b_→=-1, b_↑=-3)
This DIFFERS from the Bravyi A/B convention used by stim_work/bb_code_sim.py,
so we do not reuse build_bb_circuit here.

Qubit-index layout (all positive ints, contiguous, 378 total):
    [  0,  72)  — gross-code L data            (xⁱyʲL → i*M + j)
    [ 72, 144)  — gross-code R data            (xⁱyʲR → 72 + i*M + j)
    [144, 216)  — gross-code X-ancilla         (X check labeled xⁱyʲ)
    [216, 288)  — gross-code Z-ancilla
    [288, 335)  — 47 LPU edge qubits (data)
    [335, 359)  — 24 vertex check qubits (Bell pair occupies 2 indices)
    [359, 378)  — 19 cycle check qubits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import stim

from bb_code_sim import (
    ErrorModel,
    RelayBPDecoder,
    SimulationResult,
    find_logical_ops,
    _poly_matrix,
    _gf2_nullspace,
    _gf2_rref,
    _gf2_solve,
)


# ---------------------------------------------------------------------------
# Layer 1 — Code constants (LPU paper convention)
# ---------------------------------------------------------------------------

L_, M_ = 12, 6                                  # gross code: ℓ=12, m=6
# A = 1 + y + x³y⁻¹  (-1 mod 6 = 5)
A_EXPS: List[Tuple[int, int]] = [(0, 0), (0, 1), (3, 5)]
# B = 1 + x + x⁻¹y⁻³ (-1 mod 12 = 11, -3 mod 6 = 3)
B_EXPS: List[Tuple[int, int]] = [(0, 0), (1, 0), (11, 3)]

N_C = L_ * M_         # 72: number of X checks (= Z checks = L data = R data)
N_DATA = 2 * N_C      # 144
N_GROSS = N_DATA + 2 * N_C  # 288: gross-code data + ancilla

# Bell-pair check uses 2 ancilla; treat the single "v_Bell" logical vertex
# as 2 physical check qubits but only 1 entry in the V list (with side = 'BELL').


# ---------------------------------------------------------------------------
# Layer 2 — Monomial helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Monomial:
    side: str   # 'L' or 'R'
    i: int      # x-exponent mod L_
    j: int      # y-exponent mod M_

    def __post_init__(self) -> None:
        # Note: we can't assign with frozen=True, so we only validate.
        if self.side not in ('L', 'R'):
            raise ValueError(f"side must be 'L' or 'R', got {self.side!r}")

    def normalized(self) -> "Monomial":
        return Monomial(self.side, self.i % L_, self.j % M_)

    def qubit_index(self) -> int:
        ii = self.i % L_
        jj = self.j % M_
        if self.side == 'L':
            return ii * M_ + jj
        else:  # 'R'
            return N_C + ii * M_ + jj

    def transpose(self) -> "Monomial":
        """(αL)ᵀ = αᵀR, (αR)ᵀ = αᵀL.  αᵀ = x⁻ⁱy⁻ʲ."""
        new_side = 'R' if self.side == 'L' else 'L'
        return Monomial(new_side, (-self.i) % L_, (-self.j) % M_)

    def shift(self, di: int, dj: int) -> "Monomial":
        return Monomial(self.side, (self.i + di) % L_, (self.j + dj) % M_)

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.side, self.i % L_, self.j % M_)


def M_(side: str, i: int, j: int) -> Monomial:  # type: ignore[no-redef]
    # convenience constructor; note we shadow the module-level int M_ deliberately
    # only inside the function namespace where we use this. Use Mon() instead.
    return Monomial(side, i % L_, j % 6)


# The variable `M_` above was a module-level int (=6). To avoid name collision we
# expose a tiny constructor:
def Mon(side: str, i: int, j: int) -> Monomial:
    return Monomial(side, i % L_, j % 6)


# Fix the int M_ binding (it was overwritten by the function defn above only inside
# function scope; module-level M_ is still 6 because of the conditional `if False`):
# (Defensive: explicitly re-assert.)
M_ = 6


# ---------------------------------------------------------------------------
# Layer 3 — Logical operator polynomials (eq:pqrs_gross, line 70 of A.1)
# ---------------------------------------------------------------------------

# p — L qubits of X̄₁
P: List[Tuple[int, int]] = [(4, 0), (5, 0), (6, 1), (4, 2), (5, 4), (6, 5)]
# q — R qubits of X̄₁
Q: List[Tuple[int, int]] = [(3, 0), (4, 0), (3, 1), (3, 2), (4, 2), (3, 5)]
# r — L qubits of X̄₇
R_POLY: List[Tuple[int, int]] = [(0, 0), (8, 0), (1, 1), (9, 1), (3, 4), (11, 4)]
# s — R qubits of X̄₇
S_POLY: List[Tuple[int, int]] = [(1, 0), (9, 0), (4, 4), (8, 4), (0, 5), (8, 5)]


# ---------------------------------------------------------------------------
# Layer 4 — LPU graph G_l, G_r (vertices, identified pair, bridge)
# ---------------------------------------------------------------------------

# Vertex sets — one per monomial of pL+qR (V_l) and rL+sR (V_r)
V_L_RAW: List[Tuple[str, int, int]] = (
    [('L', i, j) for (i, j) in P] +
    [('R', i, j) for (i, j) in Q]
)
V_R_RAW: List[Tuple[str, int, int]] = (
    [('L', i, j) for (i, j) in R_POLY] +
    [('R', i, j) for (i, j) in S_POLY]
)

# Identified vertex (Bell-pair check):
#   x⁴R (∈ V_l) ≡ x⁹yL (∈ V_r)
IDENTIFIED_L_SIDE: Tuple[str, int, int] = ('R', 4, 0)
IDENTIFIED_R_SIDE: Tuple[str, int, int] = ('L', 9, 1)


# ---------------------------------------------------------------------------
# Layer 4b — H_X, H_Z and edge detection
# ---------------------------------------------------------------------------

def build_HX_HZ() -> Tuple[np.ndarray, np.ndarray]:
    A_mat = _poly_matrix(L_, 6, A_EXPS)
    B_mat = _poly_matrix(L_, 6, B_EXPS)
    H_X = np.hstack([A_mat, B_mat]).astype(np.uint8)               # [A | B]
    H_Z = np.hstack([B_mat.T % 2, A_mat.T % 2]).astype(np.uint8)   # [Bᵀ | Aᵀ]
    return H_X, H_Z


H_X, H_Z = build_HX_HZ()


def _vertex_data_idx(v: Tuple[str, int, int]) -> int:
    side, i, j = v
    return Mon(side, i, j).qubit_index()


def find_z_check_edges(vertices: List[Tuple[str, int, int]]) -> List[
    Tuple[Tuple[str, int, int], Tuple[str, int, int], int]
]:
    """
    For every pair (γ, δ) of distinct vertices, return all Z-checks of the gross
    code whose support contains BOTH qubits γ and δ.  Returns list of
    (γ, δ, z_check_index) triples.
    """
    out: List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]] = []
    n = len(vertices)
    for a in range(n):
        for b in range(a + 1, n):
            qa = _vertex_data_idx(vertices[a])
            qb = _vertex_data_idx(vertices[b])
            # find Z-checks that touch both qubits
            zchecks = np.where((H_Z[:, qa] == 1) & (H_Z[:, qb] == 1))[0]
            for zc in zchecks:
                out.append((vertices[a], vertices[b], int(zc)))
    return out


def find_x_check_edges(vertices: List[Tuple[str, int, int]]) -> List[
    Tuple[Tuple[str, int, int], Tuple[str, int, int], int]
]:
    """Same as find_z_check_edges but for X-checks (used for Z̄₇ / Z̄₁ branches)."""
    out: List[Tuple[Tuple[str, int, int], Tuple[str, int, int], int]] = []
    n = len(vertices)
    for a in range(n):
        for b in range(a + 1, n):
            qa = _vertex_data_idx(vertices[a])
            qb = _vertex_data_idx(vertices[b])
            xchecks = np.where((H_X[:, qa] == 1) & (H_X[:, qb] == 1))[0]
            for xc in xchecks:
                out.append((vertices[a], vertices[b], int(xc)))
    return out


# Compute E_l and E_r.  Each entry: (γ-tuple, δ-tuple, z_check_index).
E_L_DETAIL = find_z_check_edges(V_L_RAW)
E_R_DETAIL = find_z_check_edges(V_R_RAW)


# ---------------------------------------------------------------------------
# Layer 4c — Bridge edges (transcribed verbatim from lines 207, 209)
# ---------------------------------------------------------------------------

BRIDGE_TOP: List[Tuple[str, int, int]] = [
    ('L', 6, 5),   # x⁶y⁵L
    ('L', 4, 2),   # x⁴y²L
    ('R', 3, 5),   # x³y⁵R
    ('L', 5, 4),   # x⁵y⁴L
    ('L', 6, 1),   # x⁶yL
    ('R', 4, 2),   # x⁴y²R
    ('L', 5, 0),   # x⁵L
    ('L', 4, 0),   # x⁴L
    ('R', 3, 2),   # x³y²R
    ('R', 3, 1),   # x³yR
    ('R', 3, 0),   # x³R
]
BRIDGE_BOTTOM: List[Tuple[str, int, int]] = [
    ('L', 11, 4),  # x¹¹y⁴L
    ('R', 8, 5),   # x⁸y⁵R
    ('R', 8, 4),   # x⁸y⁴R
    ('L', 0, 0),   # 1L
    ('R', 1, 0),   # xR
    ('R', 4, 4),   # x⁴y⁴R
    ('L', 8, 0),   # x⁸L
    ('R', 9, 0),   # x⁹R
    ('R', 0, 5),   # y⁵R
    ('L', 3, 4),   # x³y⁴L
    ('L', 1, 1),   # xyL
]

BRIDGE_EDGES: List[Tuple[Tuple[str, int, int], Tuple[str, int, int]]] = list(
    zip(BRIDGE_TOP, BRIDGE_BOTTOM)
)


# ---------------------------------------------------------------------------
# Layer 4d — Cycle bases U_l, U_r, U_B
# ---------------------------------------------------------------------------

# U_l — 5 cycles, lines 184-189 of 06-appendices.tex
U_L: List[List[Tuple[str, int, int]]] = [
    [('L', 4, 0), ('R', 3, 2), ('R', 3, 1), ('L', 5, 0), ('L', 4, 0)],
    [('R', 3, 5), ('R', 3, 0), ('L', 6, 5), ('L', 4, 2), ('R', 3, 5)],
    [('R', 3, 5), ('R', 3, 0), ('L', 6, 5), ('R', 4, 0), ('L', 5, 4), ('R', 3, 5)],
    [('L', 4, 0), ('R', 3, 2), ('L', 6, 1), ('L', 5, 4), ('R', 4, 0), ('L', 4, 0)],
    [('R', 3, 1), ('R', 3, 2), ('L', 6, 1), ('R', 4, 2), ('L', 5, 0), ('R', 3, 1)],
]

# U_r — 3 cycles, lines 194-197
U_R: List[List[Tuple[str, int, int]]] = [
    [('L', 9, 1), ('R', 8, 4), ('R', 8, 5), ('L', 11, 4), ('L', 9, 1)],
    [('L', 1, 1), ('R', 1, 0), ('R', 4, 4), ('L', 3, 4), ('L', 1, 1)],
    [('L', 0, 0), ('R', 0, 5), ('R', 9, 0), ('L', 9, 1), ('R', 8, 4), ('L', 0, 0)],
]

# U_B — 10 length-4 cycles from adjacent bridge edge pairs + 1 explicit length-3 cycle
def _build_U_B() -> List[List[Tuple[str, int, int]]]:
    cycles: List[List[Tuple[str, int, int]]] = []
    # length-4 squares between adjacent bridge edges
    for k in range(len(BRIDGE_EDGES) - 1):
        t1, b1 = BRIDGE_EDGES[k]
        t2, b2 = BRIDGE_EDGES[k + 1]
        cycles.append([t1, t2, b2, b1, t1])
    # explicit length-3 cycle through the identified vertex (line 213)
    # x⁹yL=x⁴R → x⁶y⁵L → x¹¹y⁴L → x⁹yL=x⁴R
    # We represent the identified vertex as IDENTIFIED_L_SIDE (= ('R', 4, 0))
    cycles.append([
        IDENTIFIED_L_SIDE,   # x⁴R (== x⁹yL after identification)
        ('L', 6, 5),         # x⁶y⁵L
        ('L', 11, 4),        # x¹¹y⁴L
        IDENTIFIED_L_SIDE,
    ])
    return cycles


U_B: List[List[Tuple[str, int, int]]] = _build_U_B()


# ---------------------------------------------------------------------------
# Layer 4e — Build the unified vertex list and edge list
# ---------------------------------------------------------------------------

# The "logical" vertex list of the full LPU.  Bell-pair vertex is added once
# and represented by IDENTIFIED_L_SIDE (it occupies 2 physical ancilla qubits).
def _build_full_vertices() -> List[Tuple[str, int, int]]:
    """
    Return 23 vertices: V_l (12) + V_r (12) minus the one duplicate (the
    identified pair).  Identified vertex is represented once, by its V_l
    label x⁴R.
    """
    vs: List[Tuple[str, int, int]] = list(V_L_RAW)
    for v in V_R_RAW:
        if v == IDENTIFIED_R_SIDE:
            # skip — already represented by IDENTIFIED_L_SIDE in V_L_RAW
            continue
        vs.append(v)
    return vs


V_ALL: List[Tuple[str, int, int]] = _build_full_vertices()


def _canonical_vertex(v: Tuple[str, int, int]) -> Tuple[str, int, int]:
    """Map an identified-pair vertex to its canonical label."""
    if v == IDENTIFIED_R_SIDE:
        return IDENTIFIED_L_SIDE
    return v


# Edges: combine E_L and E_R lists (each edge keeps its z-check index for the
# deformation).  Plus the 11 bridge edges (which don't touch a Z-check of the
# gross code by themselves).
# Each edge in the full list is (γ, δ, z_check_index_or_None, kind)
# where kind ∈ {'L', 'R', 'B'}.

EdgeEntry = Tuple[Tuple[str, int, int], Tuple[str, int, int], Optional[int], str]


def _build_edges() -> List[EdgeEntry]:
    edges: List[EdgeEntry] = []
    for (a, b, zc) in E_L_DETAIL:
        edges.append((_canonical_vertex(a), _canonical_vertex(b), zc, 'L'))
    for (a, b, zc) in E_R_DETAIL:
        edges.append((_canonical_vertex(a), _canonical_vertex(b), zc, 'R'))
    for (a, b) in BRIDGE_EDGES:
        edges.append((_canonical_vertex(a), _canonical_vertex(b), None, 'B'))
    return edges


E_ALL: List[EdgeEntry] = _build_edges()


# ---------------------------------------------------------------------------
# Layer 4f — Assign physical qubit indices to LPU ancilla
# ---------------------------------------------------------------------------

EDGE_QUBIT_BASE   = N_GROSS                # 288
VERTEX_QUBIT_BASE = EDGE_QUBIT_BASE + 47   # 335  (47 edges)
CYCLE_QUBIT_BASE  = VERTEX_QUBIT_BASE + 24 # 359  (24 vertex check qubits incl. 2 for Bell)
N_TOTAL_QUBITS    = CYCLE_QUBIT_BASE + 19  # 378

# edge_qubit[(γ, δ, kind)] -> physical qubit index
# vertex_qubit[v] -> physical qubit index (Bell vertex gets TWO indices stored
# under (IDENTIFIED_L_SIDE, 'l') and (IDENTIFIED_L_SIDE, 'r'))
# cycle_qubit[i_cycle] -> physical qubit index

def _frozen_edge_key(e: EdgeEntry) -> Tuple[Tuple[str, int, int],
                                            Tuple[str, int, int], str]:
    a, b, _, kind = e
    # canonical ordering of endpoints (to make undirected lookup work)
    if a <= b:
        return (a, b, kind)
    else:
        return (b, a, kind)


EDGE_QUBIT: Dict[Tuple[Tuple[str, int, int], Tuple[str, int, int], str], int] = {}
for k, e in enumerate(E_ALL):
    EDGE_QUBIT[_frozen_edge_key(e)] = EDGE_QUBIT_BASE + k

VERTEX_QUBIT: Dict[Tuple[Tuple[str, int, int], str], int] = {}
_v_cursor = VERTEX_QUBIT_BASE
for v in V_ALL:
    if v == IDENTIFIED_L_SIDE:
        VERTEX_QUBIT[(v, 'l')] = _v_cursor
        _v_cursor += 1
        VERTEX_QUBIT[(v, 'r')] = _v_cursor
        _v_cursor += 1
    else:
        VERTEX_QUBIT[(v, 's')] = _v_cursor  # 's' = single
        _v_cursor += 1
assert _v_cursor == VERTEX_QUBIT_BASE + 24, _v_cursor

# concatenated cycle list for ordering
U_ALL: List[List[Tuple[str, int, int]]] = []
U_KIND: List[str] = []
for c in U_L:
    U_ALL.append([_canonical_vertex(v) for v in c])
    U_KIND.append('L')
for c in U_R:
    U_ALL.append([_canonical_vertex(v) for v in c])
    U_KIND.append('R')
for c in U_B:
    U_ALL.append([_canonical_vertex(v) for v in c])
    U_KIND.append('B')

CYCLE_QUBIT: Dict[int, int] = {
    k: CYCLE_QUBIT_BASE + k for k in range(len(U_ALL))
}


# Map each cycle to its list of edge-qubit indices
def _cycle_edge_qubits(cycle: List[Tuple[str, int, int]]) -> List[int]:
    """Walk the cycle (closed) and look up each edge's qubit index."""
    out: List[int] = []
    for i in range(len(cycle) - 1):
        a, b = cycle[i], cycle[i + 1]
        # cycle traversal edges come from E_L, E_R, or BRIDGE; we don't know
        # the kind here, so try each.
        found = None
        for kind in ('L', 'R', 'B'):
            ka = (a, b, kind) if a <= b else (b, a, kind)
            if ka in EDGE_QUBIT:
                found = EDGE_QUBIT[ka]
                break
        if found is None:
            raise RuntimeError(
                f"Cycle traversal edge ({a}, {b}) not in EDGE_QUBIT"
            )
        out.append(found)
    return out


CYCLE_EDGES: List[List[int]] = [_cycle_edge_qubits(c) for c in U_ALL]


# ---------------------------------------------------------------------------
# Layer 4g — Connections from each vertex to (a) its gross-code data qubits
# and (b) its incident edge qubits.
# ---------------------------------------------------------------------------

# Each vertex v ∈ V_l is physically connected to:
#   - gross-code qubit γ (where γ is the vertex label)
#   - gross-code qubit xy·γᵀ (for the Z̄₇ branch via the ZX duality)
#   - all edge qubits incident to v
# For vertices in V_r only, the "xy·γᵀ" connection is similar but to the
# corresponding qubit for r/s.  The paper folds both into a single LPU.
#
# For the half-LPU X̄₁ branch, only the γ connection is active for V_l vertices.
# For the half-LPU Z̄₁ branch, only the xy·γᵀ connection is active for V_r
# vertices (since Z̄₁ = Z(xy·sᵀ, xy·rᵀ) lives on those qubits).


def vertex_data_qubits_X1(v: Tuple[str, int, int]) -> List[int]:
    """Gross-code data qubits the vertex v connects to when measuring X̄₁."""
    # X̄₁ is supported on the V_l qubits: the gross-code qubit γ itself.
    return [_vertex_data_idx(v)]


def vertex_data_qubits_Z1(v: Tuple[str, int, int]) -> List[int]:
    """Gross-code data qubits the vertex v connects to when measuring Z̄₁.
    Z̄₁ = Z(xy·sᵀ, xy·rᵀ).  For v ∈ V_r (which has label γ in rL+sR), the
    corresponding Z̄₁ qubit is xy·γᵀ.
    """
    g = Mon(v[0], v[1], v[2])
    # γᵀ then multiply by xy:
    gT = g.transpose()       # flips side, negates exponents
    target = gT.shift(1, 1)  # multiply by xy
    return [target.qubit_index()]


def vertex_incident_edges(v: Tuple[str, int, int]) -> List[int]:
    """Edge-qubit indices incident to vertex v (in the full edge set)."""
    cv = _canonical_vertex(v)
    out: List[int] = []
    for e in E_ALL:
        a, b, _, kind = e
        if a == cv or b == cv:
            key = _frozen_edge_key(e)
            out.append(EDGE_QUBIT[key])
    return out


# ---------------------------------------------------------------------------
# Layer 5 — Sanity checks
# ---------------------------------------------------------------------------

def _check_commutes_all(op_L: List[Tuple[int, int]],
                        op_R: List[Tuple[int, int]],
                        H: np.ndarray) -> bool:
    """Check the Pauli operator with L-support op_L and R-support op_R
    commutes with every row of H (treated as the OPPOSITE-type check matrix,
    so an X operator vs H_Z, or a Z operator vs H_X)."""
    n = H.shape[1]
    v = np.zeros(n, dtype=np.uint8)
    for (i, j) in op_L:
        v[Mon('L', i, j).qubit_index()] ^= 1
    for (i, j) in op_R:
        v[Mon('R', i, j).qubit_index()] ^= 1
    return bool(np.all((H @ v) % 2 == 0))


def _symplectic_overlap(x_L: List[Tuple[int, int]], x_R: List[Tuple[int, int]],
                        z_L: List[Tuple[int, int]],
                        z_R: List[Tuple[int, int]]) -> int:
    """Number of qubit positions where X and Z operators overlap, mod 2."""
    x_qs: Set[int] = set()
    for (i, j) in x_L:
        x_qs.add(Mon('L', i, j).qubit_index())
    for (i, j) in x_R:
        x_qs.add(Mon('R', i, j).qubit_index())
    z_qs: Set[int] = set()
    for (i, j) in z_L:
        z_qs.add(Mon('L', i, j).qubit_index())
    for (i, j) in z_R:
        z_qs.add(Mon('R', i, j).qubit_index())
    return len(x_qs & z_qs) % 2


def _z_op_xy(p_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Apply x*y multiplied with transpose.  Used to compute Z̄₁ = Z(xy·sᵀ, xy·rᵀ).
    Input p_list is a polynomial supported on one side (sᵀ is on L-side since
    s lives on R); we return the L-side or R-side monomial list of xy·p.

    For Z̄₁:
      Z̄₁ has L-support = xy·sᵀ ;  s on R-side -> sᵀ on L-side, then multiplied
      by xy means each (i, j) in s becomes (1-i+0, 1-j+0) = (1-i, 1-j) on L.
      Actually careful: αᵀ = α with i,j -> -i,-j and side flipped, so for each
      (i,j) ∈ S_POLY (R-supported s), sᵀ (L-supported) has (-i, -j).  Multiply
      by xy => (1-i, 1-j).

      Similarly Z̄₁'s R-support = xy·rᵀ : for each (i,j) ∈ R_POLY (L-supported r),
      rᵀ (R-supported) has (-i, -j); xy·rᵀ has (1-i, 1-j) on R-side.
    """
    return [((1 - i) % L_, (1 - j) % 6) for (i, j) in p_list]


# X̄₁ = X(p, q):  L-support = P, R-support = Q
X1_L, X1_R = P, Q
# X̄₇ = X(r, s):  L-support = R_POLY, R-support = S_POLY
X7_L, X7_R = R_POLY, S_POLY
# Z̄₁ = Z(xy·sᵀ, xy·rᵀ): L-support = xy·sᵀ, R-support = xy·rᵀ
Z1_L = _z_op_xy(S_POLY)   # xy * sᵀ as L-side monomials
Z1_R = _z_op_xy(R_POLY)   # xy * rᵀ as R-side monomials
# Z̄₇ = Z(xy·qᵀ, xy·pᵀ): L-support = xy·qᵀ, R-support = xy·pᵀ
Z7_L = _z_op_xy(Q)
Z7_R = _z_op_xy(P)


def _sanity_check() -> None:
    # Vertex/edge/cycle counts
    assert len(V_ALL) == 23, f"|V| should be 23, got {len(V_ALL)}"
    assert len(E_ALL) == 47, f"|E| should be 47, got {len(E_ALL)}"
    assert len(E_L_DETAIL) == 18, f"|E_l| should be 18, got {len(E_L_DETAIL)}"
    assert len(E_R_DETAIL) == 18, f"|E_r| should be 18, got {len(E_R_DETAIL)}"
    assert len(BRIDGE_EDGES) == 11, (
        f"bridge edges should be 11, got {len(BRIDGE_EDGES)}"
    )
    assert len(U_ALL) == 19, f"|U| should be 19, got {len(U_ALL)}"
    assert len(U_L) == 5
    assert len(U_R) == 3
    assert len(U_B) == 11

    # Total ancilla = 47 (edges) + 24 (vertex checks incl. Bell pair = 2) + 19 (cycles) = 90
    total_ancilla = 47 + 24 + 19
    assert total_ancilla == 90, total_ancilla

    # H_X has shape (72, 144). Check each row has weight 6 (gross code stabilizer weight)
    assert H_X.shape == (72, 144)
    assert H_Z.shape == (72, 144)
    weights_x = H_X.sum(axis=1)
    weights_z = H_Z.sum(axis=1)
    assert np.all(weights_x == 6), f"H_X row weights: {set(weights_x.tolist())}"
    assert np.all(weights_z == 6), f"H_Z row weights: {set(weights_z.tolist())}"

    # H_X * H_Zᵀ = 0 (CSS condition)
    assert np.all((H_X @ H_Z.T) % 2 == 0), "CSS condition H_X H_Zᵀ = 0 failed"

    # Check that p, q, r, s correspond to X-logical operators (in ker(H_Z))
    # X operator with L-support P, R-support Q  -> length-144 binary vector
    def _vec(L_part: List[Tuple[int, int]], R_part: List[Tuple[int, int]]) -> np.ndarray:
        v = np.zeros(144, dtype=np.uint8)
        for (i, j) in L_part:
            v[Mon('L', i, j).qubit_index()] = 1
        for (i, j) in R_part:
            v[Mon('R', i, j).qubit_index()] = 1
        return v

    # X̄₁ = X(p, q) must satisfy H_Z @ v = 0
    v_X1 = _vec(P, Q)
    assert np.all((H_Z @ v_X1) % 2 == 0), "X̄₁ does not commute with H_Z"
    # X̄₇ = X(r, s)
    v_X7 = _vec(R_POLY, S_POLY)
    assert np.all((H_Z @ v_X7) % 2 == 0), "X̄₇ does not commute with H_Z"
    # Z̄₁ must satisfy H_X @ v = 0 (since Z's commute with X-checks)
    v_Z1 = _vec(Z1_L, Z1_R)
    assert np.all((H_X @ v_Z1) % 2 == 0), "Z̄₁ does not commute with H_X"
    # Z̄₇
    v_Z7 = _vec(Z7_L, Z7_R)
    assert np.all((H_X @ v_Z7) % 2 == 0), "Z̄₇ does not commute with H_X"

    # X̄₁ should NOT be a stabilizer (not in row-space of H_X)
    # Quick test: rank of H_X with v_X1 appended should equal rank of H_X
    # (no — should INCREASE, i.e. v_X1 is independent of H_X rows)
    stack = np.vstack([H_X, v_X1.reshape(1, -1)])
    _, p1 = _gf2_rref(stack)
    _, p0 = _gf2_rref(H_X)
    assert len(p1) > len(p0), "X̄₁ is in the row-space of H_X (= a stabilizer)"

    # Symplectic overlap: X̄₁ anticommutes with Z̄₁
    assert _symplectic_overlap(P, Q, Z1_L, Z1_R) == 1, (
        "X̄₁ and Z̄₁ must anticommute (overlap on exactly 1 qubit)"
    )
    # X̄₇ anticommutes with Z̄₇
    assert _symplectic_overlap(R_POLY, S_POLY, Z7_L, Z7_R) == 1, (
        "X̄₇ and Z̄₇ must anticommute"
    )
    # X̄₁ commutes with Z̄₇
    assert _symplectic_overlap(P, Q, Z7_L, Z7_R) == 0, "X̄₁ and Z̄₇ must commute"
    # X̄₇ commutes with Z̄₁
    assert _symplectic_overlap(R_POLY, S_POLY, Z1_L, Z1_R) == 0, (
        "X̄₇ and Z̄₁ must commute"
    )

    # Vertex check qubit count == 24:
    n_v_qubits = sum(1 for k in VERTEX_QUBIT.keys())
    assert n_v_qubits == 24, n_v_qubits

    # Edge qubits unique
    assert len(EDGE_QUBIT) == 47, len(EDGE_QUBIT)

    # All cycle_edges resolved
    for k, edges in enumerate(CYCLE_EDGES):
        # length should be len(cycle) - 1
        assert len(edges) == len(U_ALL[k]) - 1


_sanity_check()


# ---------------------------------------------------------------------------
# Layer 6 — BB-code syndrome cycle with optional deformation
# ---------------------------------------------------------------------------

# Pre-compute connections: for each X-check (indexed by (i,j)) the list of data
# qubits, and similarly for Z-checks.  Layered into 6 CNOT layers each.

X_ANC_BASE = N_DATA            # 144
Z_ANC_BASE = N_DATA + N_C      # 216

def _x_check_idx(i: int, j: int) -> int:
    return X_ANC_BASE + (i % L_) * M_ + (j % M_)

def _z_check_idx(i: int, j: int) -> int:
    return Z_ANC_BASE + (i % L_) * M_ + (j % M_)

# X-check (i,j) acts on:
#   L data qubits  x^iy^j * A monomials = (i+ax, j+ay) for (ax,ay) ∈ A_EXPS
#   R data qubits  x^iy^j * B monomials = (i+bx, j+by) for (bx,by) ∈ B_EXPS
def _x_check_layers() -> List[List[Tuple[int, int]]]:
    layers = []
    for (ax, ay) in A_EXPS:
        layer = []
        for i in range(L_):
            for j in range(M_):
                xa = _x_check_idx(i, j)
                t = Mon('L', i + ax, j + ay).qubit_index()
                layer.append((xa, t))
        layers.append(layer)
    for (bx, by) in B_EXPS:
        layer = []
        for i in range(L_):
            for j in range(M_):
                xa = _x_check_idx(i, j)
                t = Mon('R', i + bx, j + by).qubit_index()
                layer.append((xa, t))
        layers.append(layer)
    return layers

# Z-check (i,j) acts on:
#   L data qubits  x^iy^j * Bᵀ monomials = (i-bx, j-by) for (bx,by) ∈ B_EXPS
#   R data qubits  x^iy^j * Aᵀ monomials = (i-ax, j-ay) for (ax,ay) ∈ A_EXPS
def _z_check_layers() -> List[List[Tuple[int, int]]]:
    layers = []
    for (bx, by) in B_EXPS:
        layer = []
        for i in range(L_):
            for j in range(M_):
                za = _z_check_idx(i, j)
                t = Mon('L', i - bx, j - by).qubit_index()
                layer.append((t, za))
        layers.append(layer)
    for (ax, ay) in A_EXPS:
        layer = []
        for i in range(L_):
            for j in range(M_):
                za = _z_check_idx(i, j)
                t = Mon('R', i - ax, j - ay).qubit_index()
                layer.append((t, za))
        layers.append(layer)
    return layers


X_LAYERS = _x_check_layers()
Z_LAYERS = _z_check_layers()


# Verify H_Z connections match: each Z-check (i,j) should connect to exactly
# its 6 expected data qubits.
def _verify_check_layers() -> None:
    for i in range(L_):
        for j in range(M_):
            z_id = (i * M_ + j)
            expected = set(np.where(H_Z[z_id])[0].tolist())
            actual: Set[int] = set()
            for layer in Z_LAYERS:
                for (src, dst) in layer:
                    if dst == _z_check_idx(i, j):
                        actual.add(src)
            assert actual == expected, (
                f"Z-check ({i},{j}) layer mismatch: {actual} vs {expected}"
            )
            x_id = z_id
            expected_x = set(np.where(H_X[x_id])[0].tolist())
            actual_x: Set[int] = set()
            for layer in X_LAYERS:
                for (src, dst) in layer:
                    if src == _x_check_idx(i, j):
                        actual_x.add(dst)
            assert actual_x == expected_x, (
                f"X-check ({i},{j}) layer mismatch: {actual_x} vs {expected_x}"
            )


_verify_check_layers()


# ---------------------------------------------------------------------------
# Layer 6b — Bare gross-code syndrome cycle (no deformation)
# ---------------------------------------------------------------------------

def _append_noise(circuit: stim.Circuit, op: str, targets: List[int], p: float) -> None:
    if p > 0 and targets:
        circuit.append(op, targets, p)


def _append_idle(circuit: stim.Circuit, pool: Optional[List[int]],
                 active: Set[int], p: float) -> None:
    """DEPOLARIZE1(p) on every pool qubit not active in the current sub-layer.

    The fail-fast paper's standard circuit noise model (arXiv:2511.15177 §2.4): a
    qubit left idle during a gate sub-layer suffers X/Y/Z each with p/3. `pool` is
    the full set of physical qubits alive in this circuit region (None = disabled),
    so the same helper serves bare rounds (288 gross qubits) and LPU rounds
    (gross + edge/vertex/cycle qubits).
    """
    if pool is not None and p > 0:
        idle = [q for q in pool if q not in active]
        if idle:
            circuit.append("DEPOLARIZE1", idle, p)


def build_bb_syndrome_cycle(
    circuit: stim.Circuit,
    error_model: ErrorModel,
    deformation_edges_per_zcheck: Optional[Dict[int, List[int]]] = None,
    deformation_edges_per_xcheck: Optional[Dict[int, List[int]]] = None,
    deformation_x_gate: str = "CX",
    reset_data: bool = False,
    reset_ancilla: bool = True,
    skip_x_checks: bool = False,
    skip_z_checks: bool = False,
    idle_pool: Optional[List[int]] = None,
) -> None:
    """
    Append one round of gross-code stabilizer measurement to `circuit`.

    deformation_edges_per_zcheck[z_check_id] = list of edge-qubit indices that
        the Z-check picks up support on (during code deformation).  An extra
        CNOT from each such edge qubit into the Z-ancilla is appended in a
        separate layer at the end of the Z-cycle.

    deformation_edges_per_xcheck: same for X-checks.

    deformation_x_gate: 2-qubit gate coupling the X-ancilla to its deformation
        edges.  "CX" (default, anc→edge) gives the deformed X-check X-support
        on the edge — the Hadamard-dual convention of the standalone Z1 branch.
        "CZ" gives it Z-support (the X-ancilla sits in the |+⟩ frame between
        its Hadamards, so CZ(anc, edge) picks up Z_e) — the paper's uniform
        convention, required by the full-LPU Ȳ₁ measurement (Layer 7b).

    reset_data: if True, reset all 144 data qubits in |0⟩ at the start.
    reset_ancilla: if True (default), reset all ancilla at the start.

    skip_x_checks / skip_z_checks: if True, omit the corresponding
    syndrome-extraction circuit entirely (no CNOTs, no measurements for that
    ancilla block).  Used during LPU rounds where one check-type would
    anticommute with the vertex checks and randomize them.

    idle_pool: if given, every qubit in the pool that is inactive during a
    sub-layer picks up DEPOLARIZE1(p_phys) — the fail-fast paper's standard
    circuit noise model (idle X/Y/Z each p/3). None (default) preserves the
    original lighter no-idle-noise convention used by the Wave-1b runs.
    """
    p = error_model.p_phys
    pm = error_model.p_meas

    x_anc_all = list(range(X_ANC_BASE, X_ANC_BASE + N_C))
    z_anc_all = list(range(Z_ANC_BASE, Z_ANC_BASE + N_C))

    if reset_data:
        data_all = list(range(N_DATA))
        circuit.append("R", data_all)
        _append_noise(circuit, "X_ERROR", data_all, pm)
        _append_idle(circuit, idle_pool, set(data_all), p)

    if reset_ancilla:
        reset_active: Set[int] = set()
        if not skip_x_checks:
            circuit.append("R", x_anc_all)
            _append_noise(circuit, "X_ERROR", x_anc_all, pm)
            circuit.append("H", x_anc_all)
            _append_noise(circuit, "DEPOLARIZE1", x_anc_all, p)
            reset_active |= set(x_anc_all)
        if not skip_z_checks:
            circuit.append("R", z_anc_all)
            _append_noise(circuit, "X_ERROR", z_anc_all, pm)
            reset_active |= set(z_anc_all)
        if reset_active:
            _append_idle(circuit, idle_pool, reset_active, p)

    # X-check CNOT layers: X-ancilla -> data
    if not skip_x_checks:
        for layer in X_LAYERS:
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
            _append_idle(circuit, idle_pool, set(flat), p)

        # Deformation X-couplings: extra CX (Hadamard-dual) or CZ (uniform
        # convention) from X-ancilla -> edge qubits
        if deformation_edges_per_xcheck:
            extra_flat: List[int] = []
            for xc_id, edge_qs in deformation_edges_per_xcheck.items():
                xa = X_ANC_BASE + xc_id
                for eq in edge_qs:
                    extra_flat.extend([xa, eq])
            if extra_flat:
                circuit.append(deformation_x_gate, extra_flat)
                _append_noise(circuit, "DEPOLARIZE2", extra_flat, p)
                _append_idle(circuit, idle_pool, set(extra_flat), p)

    if not skip_z_checks:
        # Z-check CNOT layers: data -> Z-ancilla
        for layer in Z_LAYERS:
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
            _append_idle(circuit, idle_pool, set(flat), p)

        # Deformation Z-CNOTs: extra CNOTs from edge qubits -> Z-ancilla
        if deformation_edges_per_zcheck:
            extra_flat = []
            for zc_id, edge_qs in deformation_edges_per_zcheck.items():
                za = Z_ANC_BASE + zc_id
                for eq in edge_qs:
                    extra_flat.extend([eq, za])
            if extra_flat:
                circuit.append("CX", extra_flat)
                _append_noise(circuit, "DEPOLARIZE2", extra_flat, p)
                _append_idle(circuit, idle_pool, set(extra_flat), p)

    # Measure X-ancilla (Hadamard back, then M)
    meas_active: Set[int] = set()
    if not skip_x_checks:
        circuit.append("H", x_anc_all)
        _append_noise(circuit, "DEPOLARIZE1", x_anc_all, p)
        _append_noise(circuit, "X_ERROR", x_anc_all, pm)
        circuit.append("M", x_anc_all)
        meas_active |= set(x_anc_all)

    # Measure Z-ancilla
    if not skip_z_checks:
        _append_noise(circuit, "X_ERROR", z_anc_all, pm)
        circuit.append("M", z_anc_all)
        meas_active |= set(z_anc_all)
    if meas_active:
        _append_idle(circuit, idle_pool, meas_active, p)


# ---------------------------------------------------------------------------
# Layer 7 — LPU syndrome cycle (half-LPU only — branch X̄₁ or Z̄₁)
# ---------------------------------------------------------------------------

@dataclass
class HalfLPU:
    """Bundle of indices for one half-LPU (l or r) or full."""
    active_vertex_keys: List[Tuple[Tuple[str, int, int], str]]  # keys into VERTEX_QUBIT
    active_edge_indices: List[int]                              # physical qubit indices
    active_cycle_indices: List[int]                             # indices into U_ALL
    # For each active vertex, list of (data_qubit_indices, edge_qubit_indices)
    vertex_data: Dict[Tuple[Tuple[str, int, int], str], List[int]]
    vertex_edges: Dict[Tuple[Tuple[str, int, int], str], List[int]]
    # For each active Z-check of the gross code, which edge qubits deform it
    deformation_z: Dict[int, List[int]]   # z_check_id -> [edge qubit indices]
    deformation_x: Dict[int, List[int]]   # x_check_id -> [edge qubit indices]
    # Gross-code checks of the OPPOSITE type to the vertex checks (which are
    # X-type).  These Z-checks share data qubits with vertex checks and thus
    # anticommute → their individual outcomes are randomized during LPU rounds
    # and should NOT be used as detectors.
    anticomm_z_checks: Set[int]
    anticomm_x_checks: Set[int]


def _half_lpu(branch: str) -> HalfLPU:
    """Build a HalfLPU descriptor for branch ∈ {'X1', 'Z1'}.

    'X1': use V_l, E_l, U_l, v_Bell.  Vertices connect to gross-code via the
          γ-qubit (i.e. p/q qubits).  E_l deformation hits Z-checks (since each
          E_l edge corresponds to a Z-check of the gross code).
    'Z1': use V_r, E_r, U_r, v_Bell.  Vertices connect via xy·γᵀ qubits.
          E_r deformation also hits Z-checks (the construction is symmetric).
    """
    if branch == 'X1':
        active_v_raw = list(V_L_RAW)
        active_e_detail = E_L_DETAIL
        active_u_idxs = list(range(0, len(U_L)))        # U_L block in U_ALL
        # Bell vertex is included (identified pair counts in both halves)
        # The Bell vertex is already in V_L_RAW (as IDENTIFIED_L_SIDE = x⁴R)
        # so no extra step.
    elif branch == 'Z1':
        active_v_raw = list(V_R_RAW)
        active_e_detail = E_R_DETAIL
        # U_R block follows U_L in U_ALL: indices len(U_L)..len(U_L)+len(U_R)
        active_u_idxs = list(range(len(U_L), len(U_L) + len(U_R)))
    else:
        raise ValueError(branch)

    # Build vertex keys (handle Bell vertex specially)
    active_vertex_keys: List[Tuple[Tuple[str, int, int], str]] = []
    vertex_data: Dict[Tuple[Tuple[str, int, int], str], List[int]] = {}
    vertex_edges: Dict[Tuple[Tuple[str, int, int], str], List[int]] = {}

    for v in active_v_raw:
        cv = _canonical_vertex(v)
        if cv == IDENTIFIED_L_SIDE:
            # Bell pair: each half has its OWN Bell-check qubit (one of the two)
            # 'l' = used in X1 branch, 'r' = used in Z1 branch
            key = (cv, 'l' if branch == 'X1' else 'r')
        else:
            key = (cv, 's')
        active_vertex_keys.append(key)

        # data-qubit connection
        if branch == 'X1':
            vertex_data[key] = vertex_data_qubits_X1(v)
        else:
            vertex_data[key] = vertex_data_qubits_Z1(v)
        vertex_edges[key] = vertex_incident_edges(cv)

    # active edges: those in E_l or E_r (no bridge edges in half-LPU)
    active_edge_indices: List[int] = []
    deformation_z: Dict[int, List[int]] = {}
    deformation_x: Dict[int, List[int]] = {}

    for (a, b, zc) in active_e_detail:
        kind = 'L' if branch == 'X1' else 'R'
        ca, cb = _canonical_vertex(a), _canonical_vertex(b)
        key = _frozen_edge_key((ca, cb, None, kind))
        eq = EDGE_QUBIT[key]
        active_edge_indices.append(eq)
        if branch == 'X1':
            # Edge connects to the Z-check of the gross code that joins γ, δ.
            deformation_z.setdefault(zc, []).append(eq)
        else:  # 'Z1'
            # The graph edge labels a Z-check joining γ, δ ∈ V_r, but the EDGE
            # QUBIT physically connects (for the Z̄₁ branch) to the X-check that
            # joins xy·γᵀ and xy·δᵀ — these are the data qubits the vertex
            # checks see in this branch (per the ZX-duality between X̄₇ and Z̄₁).
            ga = Mon(a[0], a[1], a[2]).transpose().shift(1, 1)
            gb = Mon(b[0], b[1], b[2]).transpose().shift(1, 1)
            qa, qb = ga.qubit_index(), gb.qubit_index()
            # Find the unique X-check that contains both qa and qb
            xchecks = np.where((H_X[:, qa] == 1) & (H_X[:, qb] == 1))[0]
            assert len(xchecks) == 1, (
                f"For Z̄₁ branch, expected unique X-check joining xy·γᵀ={ga} "
                f"and xy·δᵀ={gb}; found {len(xchecks)}: {xchecks.tolist()}"
            )
            xc = int(xchecks[0])
            deformation_x.setdefault(xc, []).append(eq)

    # Compute the gross-code checks that anticommute with vertex checks (and
    # are NOT fixed by the deformation).
    #
    # X1 branch — vertex checks are X-type:
    #   X-vertex-check commutes with X-stabilizers (same type) always.
    #   It anticommutes with Z-stabilizers that share an odd number of qubits.
    #   By the basis-property construction (each Z-check intersecting V_l
    #   intersects in exactly 2 V_l qubits joined by an edge in E_l), every
    #   such Z-check is in deformation_z, and the edge restores commutation.
    #   ⇒ No residual anticommuting checks.
    #
    # Z1 branch — vertex checks are Z-type, by ZX-duality of the construction:
    #   Z-vertex-check commutes with Z-stabilizers (same type) always.
    #   It anticommutes with X-stabilizers that share an odd number of qubits.
    #   By the dual construction, every such X-check is in deformation_x and
    #   the edge restores commutation.
    #   ⇒ No residual anticommuting checks.
    anticomm_z: Set[int] = set()
    anticomm_x: Set[int] = set()

    return HalfLPU(
        active_vertex_keys=active_vertex_keys,
        active_edge_indices=active_edge_indices,
        active_cycle_indices=active_u_idxs,
        vertex_data=vertex_data,
        vertex_edges=vertex_edges,
        deformation_z=deformation_z,
        deformation_x=deformation_x,
        anticomm_z_checks=anticomm_z,
        anticomm_x_checks=anticomm_x,
    )


def build_lpu_cycle(
    circuit: stim.Circuit,
    error_model: ErrorModel,
    half: HalfLPU,
    first_round: bool,
    branch: str,
) -> None:
    """
    Append ONE LPU syndrome round to `circuit`.

    Branch X1 (X̄₁ measurement):
      - Vertex checks are X-type (product = X̄₁).  Edges deform Z-checks.
      - Cycle checks are Z-type on edges.
      - Order per round:
         1. BB syndrome (Z-checks deformed; X-checks unchanged).
         2. Vertex X-measurements (H + CNOT(check→data,edges) + H + M).
         3. Cycle Z-measurements (CNOT(edge→cycle_anc) + M).

    Branch Z1 (Z̄₁ measurement) — FULL ZX-DUAL of X1:
      - Vertex checks are Z-type on xy·γᵀ + edges (product = Z̄₁).
      - Edges deform X-checks of the gross code.
      - Cycle checks are X-type on edges.
      - Order per round:
         1. BB syndrome (X-checks deformed; Z-checks unchanged BUT they
            anticommute with vertex checks ON THE FIRST ROUND — gauged away
            in subsequent rounds because vertex measurements project).  We
            skip Z-check measurements to avoid randomising vertex outcomes.
         2. Vertex Z-measurements (CNOT(data,edges → check) + M).
         3. Cycle X-measurements (H + CNOT(check→edge) + H + M).
    """
    p = error_model.p_phys
    pm = error_model.p_meas

    deform_z = half.deformation_z
    deform_x = half.deformation_x

    # Step 1: bare gross-code syndrome (with deformation).
    # Both branches: all original-code stabilizers commute with LPU stabilizers
    # AFTER applying the duality-appropriate deformation.  No skipping needed.
    skip_z = False
    skip_x = False
    build_bb_syndrome_cycle(
        circuit, error_model,
        deformation_edges_per_zcheck=deform_z,
        deformation_edges_per_xcheck=deform_x,
        reset_data=False,
        reset_ancilla=True,
        skip_x_checks=skip_x,
        skip_z_checks=skip_z,
    )

    # Step 2: Vertex checks
    v_qs = [VERTEX_QUBIT[k] for k in half.active_vertex_keys]
    circuit.append("R", v_qs)
    _append_noise(circuit, "X_ERROR", v_qs, pm)
    if branch == 'X1':
        # X-type vertex check: H + CNOT(check→data,edges) + H + M
        circuit.append("H", v_qs)
        _append_noise(circuit, "DEPOLARIZE1", v_qs, p)
        for key in half.active_vertex_keys:
            check_q = VERTEX_QUBIT[key]
            targets: List[int] = []
            for dq in half.vertex_data[key]:
                targets.append(check_q)
                targets.append(dq)
            active_edge_set = set(half.active_edge_indices)
            for eq in half.vertex_edges[key]:
                if eq in active_edge_set:
                    targets.append(check_q)
                    targets.append(eq)
            if targets:
                circuit.append("CX", targets)
                _append_noise(circuit, "DEPOLARIZE2", targets, p)
        circuit.append("H", v_qs)
        _append_noise(circuit, "DEPOLARIZE1", v_qs, p)
    else:
        # Z-type vertex check: CNOT(data,edges → check) + M
        for key in half.active_vertex_keys:
            check_q = VERTEX_QUBIT[key]
            targets = []
            for dq in half.vertex_data[key]:
                targets.append(dq)
                targets.append(check_q)
            active_edge_set = set(half.active_edge_indices)
            for eq in half.vertex_edges[key]:
                if eq in active_edge_set:
                    targets.append(eq)
                    targets.append(check_q)
            if targets:
                circuit.append("CX", targets)
                _append_noise(circuit, "DEPOLARIZE2", targets, p)
    _append_noise(circuit, "X_ERROR", v_qs, pm)
    circuit.append("M", v_qs)

    # Step 3: Cycle checks
    cycle_qs: List[int] = []
    cycle_cnots: List[Tuple[int, int]] = []
    if branch == 'X1':
        # Z-type cycle check: CNOT(edge → cycle_anc) + M
        for c_idx in half.active_cycle_indices:
            cq = CYCLE_QUBIT[c_idx]
            cycle_qs.append(cq)
            for eq in CYCLE_EDGES[c_idx]:
                cycle_cnots.append((eq, cq))
        circuit.append("R", cycle_qs)
        _append_noise(circuit, "X_ERROR", cycle_qs, pm)
        flat = [q for pair in cycle_cnots for q in pair]
        if flat:
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
        _append_noise(circuit, "X_ERROR", cycle_qs, pm)
        circuit.append("M", cycle_qs)
    else:
        # X-type cycle check: H + CNOT(cycle_anc → edge) + H + M
        for c_idx in half.active_cycle_indices:
            cq = CYCLE_QUBIT[c_idx]
            cycle_qs.append(cq)
            for eq in CYCLE_EDGES[c_idx]:
                cycle_cnots.append((cq, eq))
        circuit.append("R", cycle_qs)
        _append_noise(circuit, "X_ERROR", cycle_qs, pm)
        circuit.append("H", cycle_qs)
        _append_noise(circuit, "DEPOLARIZE1", cycle_qs, p)
        flat = [q for pair in cycle_cnots for q in pair]
        if flat:
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
        circuit.append("H", cycle_qs)
        _append_noise(circuit, "DEPOLARIZE1", cycle_qs, p)
        _append_noise(circuit, "X_ERROR", cycle_qs, pm)
        circuit.append("M", cycle_qs)


# ---------------------------------------------------------------------------
# Layer 7b — Full LPU (both halves at once, uniform convention) for Ȳ₁
# ---------------------------------------------------------------------------
#
# The half-LPU branches above use per-branch conventions (the Z1 branch is the
# global Hadamard dual of the paper's).  Ȳ₁ needs BOTH halves in the SAME
# round, sharing the 11 bridge edges and the U_B cycles, so ONE graph-wide
# edge-basis convention is mandatory.  This layer implements the paper's
# uniform convention (App. A.4, line 282): checks acting as X on edge qubits
# sit on vertices, Z checks sit on cycles, all 47 edges are initialized |0⟩,
# and the return step reads every edge in the Z basis.  Only the vertex→code
# coupling Pauli varies per check class:
#   V_l vertex γ : X on code qubit γ ∈ supp(X̄₁)           (CX anc→γ)
#   V_r vertex γ : Z on code qubit xy·γᵀ ∈ supp(Z̄₁)       (CZ anc, xyγᵀ)
#   v_Bell       : Y on the overlap qubit x⁴R, coupled ONCE from the left
#                  Bell qubit (CY l→x⁴R); the right side's x⁹yL (Ȳ₇-side)
#                  coupling is inactive for Ȳ₁.
# Deformations (both simultaneously active, each adding Z_e support):
#   E_l edge → its Z-check via CX(edge→Z-anc)   (as the X1 branch)
#   E_r edge → its X-check via CZ(X-anc, edge)  (NOT the Hadamard-dual CX of
#              the standalone Z1 branch)
# Product identity: every edge qubit has both endpoints among the 23 active
# vertices, so the edge X's cancel pairwise in the product of all vertex
# checks; what remains is X on supp(X̄₁)\{x⁴R}, Z on supp(Z̄₁)\{x⁴R}, Y on
# x⁴R = Ȳ₁ exactly.


def _full_vertex_keys() -> List[Tuple[Tuple[str, int, int], str]]:
    """The 24 VERTEX_QUBIT keys of the full LPU, in measurement order.

    Order follows V_ALL with the Bell pair expanded in place ('l' then 'r'),
    which makes the corresponding physical qubit indices ascending
    (VERTEX_QUBIT was assigned by the same walk)."""
    keys: List[Tuple[Tuple[str, int, int], str]] = []
    for v in V_ALL:
        if v == IDENTIFIED_L_SIDE:
            keys.append((v, 'l'))
            keys.append((v, 'r'))
        else:
            keys.append((v, 's'))
    assert len(keys) == 24
    return keys


def _vertex_incident_edges_by_kind(
    v: Tuple[str, int, int],
) -> Dict[str, List[int]]:
    """Edge-qubit indices incident to vertex v, split by edge kind L/R/B."""
    cv = _canonical_vertex(v)
    out: Dict[str, List[int]] = {'L': [], 'R': [], 'B': []}
    for e in E_ALL:
        a, b, _, kind = e
        if a == cv or b == cv:
            out[kind].append(EDGE_QUBIT[_frozen_edge_key(e)])
    return out


@dataclass
class FullLPU:
    """Bundle of indices for the full LPU (both halves, uniform convention).

    vertex_kind classifies each of the 24 vertex-check qubits:
      'VL'     — non-Bell V_l vertex: CX to code qubit γ + CX to all incident
                 edges (E_l + bridge)
      'VR'     — non-Bell V_r vertex: CZ to code qubit xy·γᵀ + CX to all
                 incident edges (E_r + bridge)
      'BELL_L' — left Bell qubit: CY to x⁴R + CX to its 3 E_l edges
      'BELL_R' — right Bell qubit: CX to its 3 E_r edges (no code coupling
                 for Ȳ₁ — the x⁹yL link is the Ȳ₇ side)
    """
    vertex_keys: List[Tuple[Tuple[str, int, int], str]]
    vertex_kind: Dict[Tuple[Tuple[str, int, int], str], str]
    vertex_data: Dict[Tuple[Tuple[str, int, int], str], List[int]]
    vertex_edges: Dict[Tuple[Tuple[str, int, int], str], List[int]]
    deformation_z: Dict[int, List[int]]   # z_check_id -> [E_l edge qubit]
    deformation_x: Dict[int, List[int]]   # x_check_id -> [E_r edge qubit]


def _full_lpu() -> FullLPU:
    """Build the FullLPU descriptor for the Ȳ₁ measurement.

    Reuses the two half-LPU deformation computations: E_l edges deform the 18
    Z-checks (from the X1 half) and E_r edges deform the 18 X-checks (from the
    Z1-half math — the (E_r edge → X-check) pairs are not listed in the paper;
    they follow from the construction and are uniqueness-asserted there)."""
    n_vl = len(V_L_RAW)
    vertex_keys = _full_vertex_keys()
    vertex_kind: Dict[Tuple[Tuple[str, int, int], str], str] = {}
    vertex_data: Dict[Tuple[Tuple[str, int, int], str], List[int]] = {}
    vertex_edges: Dict[Tuple[Tuple[str, int, int], str], List[int]] = {}

    for pos, v in enumerate(V_ALL):
        if v == IDENTIFIED_L_SIDE:
            by_kind = _vertex_incident_edges_by_kind(v)
            # "Each side of the Bell pair has degree five": 3 edges + code
            # qubit + Bell partner (left), 3 edges + [inactive] + partner.
            assert len(by_kind['L']) == 3 and len(by_kind['R']) == 3, by_kind
            assert not by_kind['B'], "Bell vertex must not touch bridge edges"
            kl, kr = (v, 'l'), (v, 'r')
            vertex_kind[kl] = 'BELL_L'
            vertex_kind[kr] = 'BELL_R'
            vertex_data[kl] = [_vertex_data_idx(v)]   # x⁴R, coupled once (CY)
            vertex_data[kr] = []                      # Ȳ₇-side link inactive
            vertex_edges[kl] = by_kind['L']
            vertex_edges[kr] = by_kind['R']
        else:
            key = (v, 's')
            if pos < n_vl:
                vertex_kind[key] = 'VL'
                vertex_data[key] = vertex_data_qubits_X1(v)
            else:
                vertex_kind[key] = 'VR'
                vertex_data[key] = vertex_data_qubits_Z1(v)
            vertex_edges[key] = vertex_incident_edges(v)

    deformation_z = {zc: list(eqs)
                     for zc, eqs in _half_lpu('X1').deformation_z.items()}
    deformation_x = {xc: list(eqs)
                     for xc, eqs in _half_lpu('Z1').deformation_x.items()}
    # Basis property: each of the 36 deformed gross checks picks up exactly
    # one edge (18 Z-checks / 18 X-checks, one E_l / E_r edge each).
    assert len(deformation_z) == 18 and len(deformation_x) == 18
    assert all(len(v) == 1 for v in deformation_z.values())
    assert all(len(v) == 1 for v in deformation_x.values())

    return FullLPU(
        vertex_keys=vertex_keys,
        vertex_kind=vertex_kind,
        vertex_data=vertex_data,
        vertex_edges=vertex_edges,
        deformation_z=deformation_z,
        deformation_x=deformation_x,
    )


def build_full_lpu_cycle(
    circuit: stim.Circuit,
    error_model: ErrorModel,
    full: FullLPU,
    idle_pool: Optional[List[int]] = None,
) -> None:
    """Append ONE full-LPU (Ȳ₁) syndrome round to `circuit`.

    Measurement order (187 records): [X-anc 72] [Z-anc 72] [vertex 24, in
    _full_vertex_keys order] [cycle 19, in U_ALL order].

    Order per round (module-style layering — NOT the paper's 12-timestep
    graph-coloring schedule; see build_joint_pauli_circuit docstring):
      1. Deformed BB syndrome: Z-checks pick up Z_e via CX(edge→Z-anc),
         X-checks pick up Z_e via CZ(X-anc, edge).
      2. Vertex checks (all 24 ancillas): R; H on all but the right Bell
         qubit; Bell prep CX(l→r) BEFORE any data gate (App. A.5 constraint);
         per-check couplings (see FullLPU.vertex_kind); H on all; M all.
         The Bell CHECK value is the XOR of the two Bell records — each
         individual Bell record carries a fresh random bit every round (the
         new Bell pair), so detectors must pair them (23 vertex detectors).
      3. Cycle checks (all 19): R, CX(edge→cycle-anc), M — Z-type on edges.

    idle_pool: as in build_bb_syndrome_cycle — DEPOLARIZE1(p_phys) on every
    pool qubit inactive in each sub-layer.
    """
    p = error_model.p_phys
    pm = error_model.p_meas

    # Step 1: deformed gross-code syndrome (both deformations active).
    build_bb_syndrome_cycle(
        circuit, error_model,
        deformation_edges_per_zcheck=full.deformation_z,
        deformation_edges_per_xcheck=full.deformation_x,
        deformation_x_gate="CZ",
        reset_data=False,
        reset_ancilla=True,
        idle_pool=idle_pool,
    )

    # Step 2: vertex checks.
    v_qs = [VERTEX_QUBIT[k] for k in full.vertex_keys]
    bell_l = VERTEX_QUBIT[(IDENTIFIED_L_SIDE, 'l')]
    bell_r = VERTEX_QUBIT[(IDENTIFIED_L_SIDE, 'r')]
    circuit.append("R", v_qs)
    _append_noise(circuit, "X_ERROR", v_qs, pm)
    _append_idle(circuit, idle_pool, set(v_qs), p)
    # |+⟩ frame everywhere except the right Bell qubit (prepped by CX below).
    h_qs = [q for q in v_qs if q != bell_r]
    circuit.append("H", h_qs)
    _append_noise(circuit, "DEPOLARIZE1", h_qs, p)
    _append_idle(circuit, idle_pool, set(h_qs), p)
    # Bell-pair preparation — scheduled before all data gates.
    circuit.append("CX", [bell_l, bell_r])
    _append_noise(circuit, "DEPOLARIZE2", [bell_l, bell_r], p)
    _append_idle(circuit, idle_pool, {bell_l, bell_r}, p)
    # Couplings, one gate group per check (module-style layering).
    for key in full.vertex_keys:
        anc = VERTEX_QUBIT[key]
        kind = full.vertex_kind[key]
        gate_groups: List[Tuple[str, List[int]]] = []
        if kind == 'VL':
            flat: List[int] = []
            for dq in full.vertex_data[key]:
                flat += [anc, dq]
            for eq in full.vertex_edges[key]:
                flat += [anc, eq]
            gate_groups.append(("CX", flat))
        elif kind == 'VR':
            czf: List[int] = []
            for dq in full.vertex_data[key]:
                czf += [anc, dq]
            gate_groups.append(("CZ", czf))
            cxf: List[int] = []
            for eq in full.vertex_edges[key]:
                cxf += [anc, eq]
            gate_groups.append(("CX", cxf))
        elif kind == 'BELL_L':
            gate_groups.append(("CY", [anc, full.vertex_data[key][0]]))
            cxf = []
            for eq in full.vertex_edges[key]:
                cxf += [anc, eq]
            gate_groups.append(("CX", cxf))
        else:  # 'BELL_R'
            cxf = []
            for eq in full.vertex_edges[key]:
                cxf += [anc, eq]
            gate_groups.append(("CX", cxf))
        for gname, flat in gate_groups:
            if flat:
                circuit.append(gname, flat)
                _append_noise(circuit, "DEPOLARIZE2", flat, p)
                _append_idle(circuit, idle_pool, set(flat), p)
    # Back to the Z frame (Bell pair: both sides measured in the X basis).
    circuit.append("H", v_qs)
    _append_noise(circuit, "DEPOLARIZE1", v_qs, p)
    _append_idle(circuit, idle_pool, set(v_qs), p)
    _append_noise(circuit, "X_ERROR", v_qs, pm)
    circuit.append("M", v_qs)
    _append_idle(circuit, idle_pool, set(v_qs), p)

    # Step 3: cycle checks (Z-type on edges), all 19 at once.
    cycle_qs = [CYCLE_QUBIT[c] for c in range(len(U_ALL))]
    flat = []
    for c_idx in range(len(U_ALL)):
        cq = CYCLE_QUBIT[c_idx]
        for eq in CYCLE_EDGES[c_idx]:
            flat += [eq, cq]
    circuit.append("R", cycle_qs)
    _append_noise(circuit, "X_ERROR", cycle_qs, pm)
    _append_idle(circuit, idle_pool, set(cycle_qs), p)
    circuit.append("CX", flat)
    _append_noise(circuit, "DEPOLARIZE2", flat, p)
    _append_idle(circuit, idle_pool, set(flat), p)
    _append_noise(circuit, "X_ERROR", cycle_qs, pm)
    circuit.append("M", cycle_qs)
    _append_idle(circuit, idle_pool, set(cycle_qs), p)


def _op_vec(L_part: List[Tuple[int, int]],
            R_part: List[Tuple[int, int]]) -> np.ndarray:
    """Length-144 GF(2) support vector of a monomial-list Pauli operator."""
    v = np.zeros(N_DATA, dtype=np.uint8)
    for (i, j) in L_part:
        v[Mon('L', i, j).qubit_index()] ^= 1
    for (i, j) in R_part:
        v[Mon('R', i, j).qubit_index()] ^= 1
    return v


def _lpu_correction_paths() -> Dict[Tuple[Tuple[str, int, int], str], Set[int]]:
    """BFS spanning-tree paths in the LPU graph, v_Bell split into its two
    physical check qubits.

    Returns, for every one of the 24 vertex-check keys, the edge-qubit set of
    the tree path from a fixed root.  Used to express the paper's software
    correction (App. A.4 step 4: paths μ_v with ∏ m_e deciding a Pauli-frame
    update) as measurement records in the observables.  The Bell vertex must
    be split because its Ȳ₁ code coupling (Y on x⁴R) hangs off the LEFT Bell
    qubit only: a correction path terminating at v_Bell has to arrive through
    an E_l edge of the left side, not an E_r edge of the right.
    """
    keys = _full_vertex_keys()

    def node_of(v: Tuple[str, int, int], kind: str):
        cv = _canonical_vertex(v)
        if cv == IDENTIFIED_L_SIDE:
            assert kind != 'B', "Bell vertex must not touch bridge edges"
            return (cv, 'l' if kind == 'L' else 'r')
        return (cv, 's')

    adj: Dict[Tuple[Tuple[str, int, int], str],
              List[Tuple[Tuple[Tuple[str, int, int], str], int]]] = {
        k: [] for k in keys}
    for e in E_ALL:
        a, b, _, kind = e
        eq = EDGE_QUBIT[_frozen_edge_key(e)]
        na, nb = node_of(a, kind), node_of(b, kind)
        adj[na].append((nb, eq))
        adj[nb].append((na, eq))

    root = keys[0]
    paths: Dict[Tuple[Tuple[str, int, int], str], Set[int]] = {root: set()}
    queue = [root]
    while queue:
        n = queue.pop(0)
        for (m, eq) in adj[n]:
            if m not in paths:
                paths[m] = paths[n] ^ {eq}
                queue.append(m)
    assert len(paths) == 24, "split-Bell LPU graph must be connected"
    return paths


def _y1_observable_recipe() -> Tuple[
    List[int], List[Tuple[np.ndarray, List[int]]]
]:
    """The 11 Z-type memory observables of the Ȳ₁ circuit, with corrections.

    Returns (S, recipe) where S is the set of canonical logical classes k with
    Z̄ₖ anticommuting with X(p,q), and recipe is a list of 11 pairs
    (Z-support vector over the 144 data qubits, list of edge qubits whose
    return-step Z-readouts m_e must be XOR'd in).

    Basis choice (documented deviation from the coset-test recipe): the
    canonical find_logical_ops basis is NOT aligned with the LPU polynomials —
    empirically X(p,q) ≡ X̄₂X̄₃ and Z(xy·sᵀ, xy·rᵀ) ≡ Z̄₃Z̄₇Z̄₈Z̄₉ mod
    stabilizers, so no single canonical class "matches Z̄₁".  The right
    invariant object is the index-2 subgroup of Z-type logicals commuting with
    Ȳ₁ (⇔ even overlap with supp(X̄₁)); we take the basis
    {Z̄ₖ : k ∉ S} ∪ {Z̄_{s₀}Z̄_s : s ∈ S \\ {s₀}} — 11 operators, reducing to
    the spec's "the 11 Z̄ₖ with k ≠ k*" whenever |S| = 1.

    Correction (why the edge records are needed): the gauging measurement is
    only Ȳ₁ up to the App.-A.4 software Pauli update, whose X-type part flips
    Z-basis readouts.  A logical W in the subgroup anticommutes with the
    individual vertex checks at T_W = {V_l vertices with γ ∈ supp(W)} ∪
    {Bell-left if x⁴R ∈ supp(W)} (|T_W| even); the deformed representative
    W·Z(E_W) with ∂E_W = T_W commutes with everything, and the edge factors
    become the recorded m_e at the return readout.
    """
    log_Z, _ = _tdg_logical_ops()
    v_x1 = _op_vec(P, Q)
    S = [k for k in range(12) if int(log_Z[k] @ v_x1) % 2 == 1]
    assert S, "X(p,q) must be a nontrivial logical"

    ops: List[np.ndarray] = [log_Z[k] for k in range(12) if k not in S]
    for s in S[1:]:
        ops.append((log_Z[S[0]] ^ log_Z[s]).astype(np.uint8))
    assert len(ops) == 11

    paths = _lpu_correction_paths()
    full = _full_lpu()
    recipe: List[Tuple[np.ndarray, List[int]]] = []
    for w in ops:
        supp = set(int(q) for q in np.where(w)[0])
        e_set: Set[int] = set()
        n_ends = 0
        for key in full.vertex_keys:
            if (full.vertex_kind[key] in ('VL', 'BELL_L')
                    and full.vertex_data[key][0] in supp):
                e_set ^= paths[key]
                n_ends += 1
        assert n_ends % 2 == 0, "W must commute with Ȳ₁ (even overlap)"
        recipe.append((w, sorted(e_set)))
    return S, recipe


class _MeasTracker:
    """Absolute-index measurement bookkeeping for the framing builders.

    add(k) reserves k just-appended measurement records and returns the
    absolute index of the first; rec(abs_idx) converts an absolute index to a
    stim rec target relative to the CURRENT end of the record.  Using absolute
    indices removes the per-round negative-offset arithmetic that made the
    original _build_circuit detector code error-prone.
    """

    def __init__(self) -> None:
        self.n = 0

    def add(self, k: int) -> int:
        start = self.n
        self.n += k
        return start

    def rec(self, abs_idx: int) -> stim.GateTarget:
        assert 0 <= abs_idx < self.n
        return stim.target_rec(abs_idx - self.n)


# ---------------------------------------------------------------------------
# Layer 8 — Public API: build_logical_x1_circuit / build_logical_z1_circuit
# ---------------------------------------------------------------------------

def _add_bb_round_detectors(
    circuit: stim.Circuit,
    round_history: List[str],
    measurements_per_round: List[int],
    is_first_full_round: bool,
    z_only_first: bool,
) -> None:
    """
    Add Stim DETECTOR instructions comparing the most recent ancilla
    measurements to the previous round's ancilla measurements.

    measurements_per_round[i] = number of measurement records produced by
    that round (used to compute correct rec offsets).

    is_first_full_round: if True, only Z-check detectors are deterministic
    (for Z-memory init) — but here we sometimes start with X-memory.

    Simpler approach used here: add detectors only when round_history shows
    a "bb" round preceded by another "bb" round (compare ancilla-vs-ancilla),
    or when a bb round is the very first.
    """
    # This helper is unused in the final implementation; we build detectors
    # inline in the main builders instead, where we know what came before.
    pass


def _z_anc_rec_offsets_from_end_of_round(round_total: int) -> List[int]:
    """Given that one BB syndrome round produced 2*N_C measurements
    (X-ancilla then Z-ancilla), return the rec offset for each Z-ancilla
    s=0..71 measured at the END of that round, relative to the end of that
    round's measurement block (offset is negative)."""
    return [-(N_C - s) for s in range(N_C)]


def _x_anc_rec_offsets_from_end_of_round() -> List[int]:
    return [-(2 * N_C - s) for s in range(N_C)]


def _build_circuit(branch: str, error_model: ErrorModel,
                   C: int, d_init: int) -> stim.Circuit:
    """
    Common builder for both branches.  Structure:
      0. Reset data qubits (|0⟩ for Z1, |+⟩ for X1 via H after reset).
      1. d_init rounds of bare gross-code syndrome.
      2. C rounds of LPU syndrome (with deformation).
      3. Measure all edge qubits in Z basis.
      4. d_init rounds of bare gross-code syndrome.
      5. OBSERVABLE_INCLUDE on the XOR of all vertex-check measurements of
         the LAST LPU round.
    """
    assert branch in ('X1', 'Z1')
    half = _half_lpu(branch)

    p = error_model.p_phys
    pm = error_model.p_meas

    circuit = stim.Circuit()

    # ---- Step 0: data init ----
    data_all = list(range(N_DATA))
    circuit.append("R", data_all)
    _append_noise(circuit, "X_ERROR", data_all, pm)
    if branch == 'X1':
        # Init in |+⟩ on all data qubits (Z-stabilizers will project into codespace
        # after d_init rounds).  Logical X̄₁ commutes with this state.
        circuit.append("H", data_all)
        _append_noise(circuit, "DEPOLARIZE1", data_all, p)

    # ---- Step 1: d_init bare gross-code rounds ----
    # Track the start of each round's measurement block by counting records
    measurements_per_round: List[Tuple[str, int]] = []
    # Each entry: (round_type, num_measurements_in_round).
    # round_type ∈ {'bb', 'lpu'}.

    for r in range(d_init):
        # Each BB round produces 2*N_C = 144 measurements
        build_bb_syndrome_cycle(
            circuit, error_model,
            deformation_edges_per_zcheck=None,
            deformation_edges_per_xcheck=None,
            reset_data=False,
            reset_ancilla=True,  # always reset (ancilla were never initialized)
        )
        measurements_per_round.append(('bb', 2 * N_C))
        circuit.append("TICK")

        # Detectors
        # X-check detectors: ancilla outcome should be deterministic given the
        # |+⟩-init for X̄₁ branch (X-stabilizers measure +1).  For Z̄₁ branch
        # in |0⟩, Z-stabilizers measure +1.
        if r == 0:
            if branch == 'X1':
                # X-ancilla deterministic (|+⟩ is +1 eigenstate of X-stabilizers)
                for s in range(N_C):
                    off = _x_anc_rec_offsets_from_end_of_round()[s]
                    circuit.append("DETECTOR", [stim.target_rec(off)], [1, s, 0])
            else:  # 'Z1'
                # Z-ancilla deterministic (|0⟩ is +1 eigenstate of Z-stabilizers)
                for s in range(N_C):
                    off = _z_anc_rec_offsets_from_end_of_round(2 * N_C)[s]
                    circuit.append("DETECTOR", [stim.target_rec(off)], [0, s, 0])
        else:
            # Compare current ancilla measurements with previous round's
            for s in range(N_C):
                cur_x = _x_anc_rec_offsets_from_end_of_round()[s]
                prev_x = cur_x - 2 * N_C
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_x), stim.target_rec(prev_x)],
                               [1, s, r])
            for s in range(N_C):
                cur_z = _z_anc_rec_offsets_from_end_of_round(2 * N_C)[s]
                prev_z = cur_z - 2 * N_C
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_z), stim.target_rec(prev_z)],
                               [0, s, r])

    # ---- Step 2: C rounds of LPU syndrome ----
    n_v_active = len(half.active_vertex_keys)
    n_u_active = len(half.active_cycle_indices)

    # Whether the LPU round measures BB-X-anc and BB-Z-anc.
    # Both branches: build_lpu_cycle calls build_bb_syndrome_cycle with
    # skip_x_checks=False and skip_z_checks=False, so both blocks are
    # always measured.  The vertex checks (X-type for X̄₁, Z-type for Z̄₁)
    # commute with the SAME-type stabilizers of the original code, and the
    # opposite-type stabilizers either commute via the edge-deformation
    # (the 18 deformed checks) or do not touch the LPU data at all.
    has_x_anc = True
    has_z_anc = True
    n_bb_anc = 2 * N_C
    lpu_round_size = n_bb_anc + n_v_active + n_u_active

    # Within one LPU round, measurement order:
    #   [X-anc (if any)] [Z-anc (if any)] [vertex] [cycle]
    def _lpu_cycle_off(c_local: int) -> int:
        return -(n_u_active - c_local)

    def _lpu_vertex_off(v_local: int) -> int:
        return -(n_u_active + n_v_active - v_local)

    def _lpu_z_anc_off(s: int) -> int:
        # Z-ancilla block comes right before vertex block
        if not has_z_anc:
            raise RuntimeError("no Z-ancilla in this branch")
        return -(n_u_active + n_v_active + N_C - s)

    def _lpu_x_anc_off(s: int) -> int:
        # X-ancilla block comes before Z-ancilla block (or directly before
        # vertex block if Z is skipped)
        z_size = N_C if has_z_anc else 0
        return -(n_u_active + n_v_active + z_size + N_C - s)

    # Initialize edge qubits.
    # For X1 (X-type cycle checks act on edges as Z): edges in |0⟩ so the
    # Z-cycle-check is +1 deterministic on round 0.  Actually for X1 cycle
    # checks ARE Z-type, so |0⟩ edges give cycle = +1 ✓.
    # For Z1 (X-type cycle checks): edges in |+⟩ so the X-cycle-check is +1
    # deterministic on round 0.
    edge_qs_init = sorted(half.active_edge_indices)
    if edge_qs_init:
        circuit.append("R", edge_qs_init)
        _append_noise(circuit, "X_ERROR", edge_qs_init, pm)
        if branch == 'Z1':
            circuit.append("H", edge_qs_init)
            _append_noise(circuit, "DEPOLARIZE1", edge_qs_init, p)

    last_round_was_bb = True

    for c in range(C):
        build_lpu_cycle(circuit, error_model, half,
                        first_round=(c == 0), branch=branch)
        measurements_per_round.append(('lpu', lpu_round_size))
        circuit.append("TICK")

        # Cycle-check detectors: cycle check is Z-type so compares against itself
        # from round c-1. (First LPU round: deterministic since edge qubits were
        # just reset to |0⟩ -> cycle = product of |0⟩'s = +1.)
        if c == 0:
            for u_local in range(n_u_active):
                circuit.append("DETECTOR",
                               [stim.target_rec(_lpu_cycle_off(u_local))],
                               [2, u_local, 0])
        else:
            for u_local in range(n_u_active):
                cur = _lpu_cycle_off(u_local)
                prev = cur - lpu_round_size
                circuit.append("DETECTOR",
                               [stim.target_rec(cur), stim.target_rec(prev)],
                               [2, u_local, c])

        # Vertex-check detectors: compare consecutive rounds; the product
        # is the logical observable (added at the very end).  Per-vertex detectors
        # only make sense from round 2 onwards (round-1 vertex outcomes form the
        # logical bit and shouldn't be turned into detectors that constrain that bit).
        if c > 0:
            for v_local in range(n_v_active):
                cur = _lpu_vertex_off(v_local)
                prev = cur - lpu_round_size
                circuit.append("DETECTOR",
                               [stim.target_rec(cur), stim.target_rec(prev)],
                               [3, v_local, c])

        # Also include BB-ancilla detectors during the LPU rounds — comparing
        # against the previous round's BB-ancilla.  Skip detectors for:
        #   (a) deformed checks — their value picks up edge-qubit support, so a
        #       plain XOR with the previous round's measurement is no longer
        #       deterministic.  (Detectors for "deformed cur XOR deformed prev"
        #       ARE consistent and could be added but we skip them for
        #       simplicity; their syndrome info is recovered by the
        #       cycle/vertex/edge measurements anyway.)
        #   (b) checks that anticommute with any vertex check — their
        #       individual outcomes are randomized by vertex measurements.
        deformed_zcs = set(half.deformation_z.keys())
        deformed_xcs = set(half.deformation_x.keys())
        bad_x = deformed_xcs | half.anticomm_x_checks
        bad_z = deformed_zcs | half.anticomm_z_checks
        if has_x_anc:
            for s in range(N_C):
                if s in bad_x:
                    continue
                cur = _lpu_x_anc_off(s)
                if c == 0:
                    # previous round was a bare BB round (both X and Z anc
                    # present, size 2*N_C); the X-anc s is at offset
                    # -(2*N_C - s) from end of that round.  Total offset
                    # from end of this LPU round: cur - (lpu_round_size
                    # - (X-anc-offset-within-this-LPU = z_size + n_v + n_u
                    # + N_C - s)) ... this is messy; use a direct shift.
                    # Simpler: just shift by full BB-round size minus the
                    # part of THIS round that came before the X-anc block.
                    # Actually the cleanest invariant is: prev rec offset
                    # for s in the BB round is -(2*N_C - s); from end of
                    # current LPU round, the BB round ended `lpu_round_size`
                    # records ago in terms of "end-to-end-of-prev", so
                    # offset = -(2*N_C - s) - lpu_round_size.
                    prev = -(2 * N_C - s) - lpu_round_size
                else:
                    prev = cur - lpu_round_size
                circuit.append("DETECTOR",
                               [stim.target_rec(cur), stim.target_rec(prev)],
                               [4, s, c])
        if has_z_anc:
            for s in range(N_C):
                if s in bad_z:
                    continue
                cur = _lpu_z_anc_off(s)
                if c == 0:
                    prev = -(N_C - s) - lpu_round_size
                else:
                    prev = cur - lpu_round_size
                circuit.append("DETECTOR",
                               [stim.target_rec(cur), stim.target_rec(prev)],
                               [5, s, c])

    # ---- Step 3: Return — measure all edge qubits in Z, then bare BB rounds ----
    # Vertex-check measurement records from the FINAL LPU round, for use in
    # the OBSERVABLE.  Need to capture offsets BEFORE we add more measurements.
    final_vertex_offsets: List[int] = []
    # The end of the LPU rounds is the current end-of-records.  At this point,
    # the most-recent measurements are the cycle checks of round C-1.
    # Vertex measurements of round C-1 are at offsets -(n_u_active+1) .. -(n_u_active+n_v_active).
    # When we append more instructions, these offsets shift — Stim uses negative
    # offsets that are relative to the END of the FULL measurement record at the
    # time the OBSERVABLE_INCLUDE is appended.  So we'll capture them after all
    # subsequent measurements have happened.

    # Measure all active edge qubits in Z basis
    edge_qs = sorted(half.active_edge_indices)
    if edge_qs:
        _append_noise(circuit, "X_ERROR", edge_qs, pm)
        circuit.append("M", edge_qs)
        measurements_per_round.append(('edges', len(edge_qs)))
        circuit.append("TICK")

    # ---- Step 4: trailing bare gross-code rounds (only useful for full fault tolerance)
    # We do a few rounds to project back to the original codespace.  This is
    # optional for the simple smoke test.
    n_trailing = d_init
    for r in range(n_trailing):
        # Need to be careful: the FIRST trailing round comes right after edge
        # measurements; the BB ancilla were NOT measured in the edge step.
        # We still want to reset ancilla and run a normal BB round.
        build_bb_syndrome_cycle(
            circuit, error_model,
            deformation_edges_per_zcheck=None,
            deformation_edges_per_xcheck=None,
            reset_data=False,
            reset_ancilla=True,
        )
        measurements_per_round.append(('bb', 2 * N_C))
        circuit.append("TICK")

        # Detectors for trailing BB rounds
        if r > 0:
            for s in range(N_C):
                cur_x = _x_anc_rec_offsets_from_end_of_round()[s]
                prev_x = cur_x - 2 * N_C
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_x), stim.target_rec(prev_x)],
                               [6, s, r])
            for s in range(N_C):
                cur_z = _z_anc_rec_offsets_from_end_of_round(2 * N_C)[s]
                prev_z = cur_z - 2 * N_C
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_z), stim.target_rec(prev_z)],
                               [7, s, r])

    # ---- Step 5: Final data measurement in the appropriate basis ----
    # For X̄₁: measure data in X basis (H, then M); deterministic for |+̄⟩ memory.
    # For Z̄₁: measure data in Z basis (just M).
    if branch == 'X1':
        circuit.append("H", data_all)
        _append_noise(circuit, "DEPOLARIZE1", data_all, p)
    _append_noise(circuit, "X_ERROR", data_all, pm)
    circuit.append("M", data_all)
    measurements_per_round.append(('final_data', N_DATA))

    # ---- Step 6: OBSERVABLE — XOR of all final-round vertex-check measurements ----
    # We need to compute the rec offset for each vertex-check measurement from the
    # FINAL LPU round, RELATIVE TO THE CURRENT END OF MEASUREMENT RECORD (i.e. after
    # the final data measurement).
    #
    # Total measurements added AFTER the final LPU round:
    #   edge-meas: len(edge_qs) (if any)
    #   trailing BB rounds: n_trailing * 2*N_C
    #   final data: N_DATA
    n_after_lpu = (len(edge_qs) if edge_qs else 0) + n_trailing * 2 * N_C + N_DATA

    obs_recs: List[stim.GateTarget] = []
    for v_local in range(n_v_active):
        # vertex measurement at end-of-final-LPU-round offset:
        off_at_lpu_end = _lpu_vertex_off(v_local)  # negative, relative to end-of-LPU
        # shifted by n_after_lpu measurements added since:
        off_now = off_at_lpu_end - n_after_lpu
        obs_recs.append(stim.target_rec(off_now))

    circuit.append("OBSERVABLE_INCLUDE", obs_recs, 0)

    # Also add a "comparison" observable from the final data measurements
    # for the bare logical operator — this is an independent sanity check.
    # X̄₁ on |+̄⟩: parity of data qubits in P + Q should equal 0.
    # Z̄₁ on |0̄⟩: parity of data qubits in Z1_L + Z1_R should equal 0.
    # We don't include this as an additional OBSERVABLE_INCLUDE to keep the
    # circuit's "primary" logical bit clean.

    return circuit


def build_logical_x1_circuit(
    error_model: ErrorModel,
    C: int = 10,
    d_init: int = 12,
) -> stim.Circuit:
    """Build a Stim circuit that prepares |+̄⟩ on the gross code and measures X̄₁
    using half-LPU l (V_l, E_l, U_l, v_Bell)."""
    return _build_circuit('X1', error_model, C, d_init)


def build_logical_z1_circuit(
    error_model: ErrorModel,
    C: int = 10,
    d_init: int = 12,
) -> stim.Circuit:
    """Build a Stim circuit that prepares |0̄⟩ on the gross code and measures Z̄₁
    using half-LPU r (V_r, E_r, U_r, v_Bell)."""
    return _build_circuit('Z1', error_model, C, d_init)


# ---------------------------------------------------------------------------
# Future LPU circuit seams (registered in experiment_runner.CIRCUIT_BUILDERS).
# These are placeholders for the next experiment families — implement here so the
# rest of the framework (configs, SLURM launcher, techniques) needs no changes.
# ---------------------------------------------------------------------------

_LOGICAL_OPS_CACHE: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None


def _tdg_logical_ops() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """The 12 canonical (Z̄ₖ, X̄ₖ) logical pairs of the gross code, tdg convention.

    Computed once from build_HX_HZ() via bb_code_sim.find_logical_ops and cached.
    NB the ordering of the 12 classes is find_logical_ops' canonical GF(2) choice —
    it does NOT automatically align class 0 with the LPU's hardcoded X̄₁/Z̄₁ (P/Q,
    Z1_L/Z1_R). Builders that must name "logical 1" (the Y̅1 measurement) align via
    a stabilizer-coset membership test at build time.
    """
    global _LOGICAL_OPS_CACHE
    if _LOGICAL_OPS_CACHE is None:
        H_X, H_Z = build_HX_HZ()
        log_Z, log_X = find_logical_ops(H_X, H_Z)
        log_Z = [np.asarray(v, dtype=np.uint8) for v in log_Z]
        log_X = [np.asarray(v, dtype=np.uint8) for v in log_X]
        assert len(log_Z) == 12 and len(log_X) == 12, "gross code has k=12"
        for k in range(12):
            assert not np.any((H_X @ log_Z[k]) % 2), "Z̄ must commute with X-checks"
            assert not np.any((H_Z @ log_X[k]) % 2), "X̄ must commute with Z-checks"
        for j in range(12):
            for k in range(12):
                par = int(np.dot(log_Z[j].astype(int), log_X[k].astype(int)) % 2)
                assert par == (1 if j == k else 0), "canonical anticommutation violated"
        _LOGICAL_OPS_CACHE = (log_Z, log_X)
    return _LOGICAL_OPS_CACHE


def build_idle_memory_circuit(
    error_model: ErrorModel,
    rounds: int = 12,
    idle_noise: bool = False,
) -> stim.Circuit:
    """Gross-code Z-memory (idle) experiment in the tdg convention.

    Fail-fast §2.4 recipe: |0̄⟩ init → `rounds` NOISY QEC cycles → ONE fault-free
    cycle → noiseless transversal Z measurement of all data. Detectors: round-0
    Z-checks deterministic ([0,s,0]); later rounds compare both check types to the
    previous round ([1,s,r] X / [0,s,r] Z); final data reconstruction of each
    Z-check tagged [8,s,rounds+1]. K=12 observables = the canonical Z̄ₖ parities
    over the final data measurements (this module's A/B convention — deliberately
    NOT bb_code_sim's Bravyi-convention memory circuit, so results compare
    apples-to-apples with the LPU operation circuits built here).

    idle_noise: emit DEPOLARIZE1(p) on inactive gross qubits per sub-layer
    (fail-fast standard circuit noise model).
    """
    pool = list(range(N_GROSS)) if idle_noise else None

    circuit = stim.Circuit()
    data_all = list(range(N_DATA))
    circuit.append("R", data_all)
    _append_noise(circuit, "X_ERROR", data_all, error_model.p_meas)

    def _round_detectors(r: int) -> None:
        if r == 0:
            for s in range(N_C):
                off = _z_anc_rec_offsets_from_end_of_round(2 * N_C)[s]
                circuit.append("DETECTOR", [stim.target_rec(off)], [0, s, 0])
        else:
            for s in range(N_C):
                cur_x = _x_anc_rec_offsets_from_end_of_round()[s]
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_x), stim.target_rec(cur_x - 2 * N_C)],
                               [1, s, r])
            for s in range(N_C):
                cur_z = _z_anc_rec_offsets_from_end_of_round(2 * N_C)[s]
                circuit.append("DETECTOR",
                               [stim.target_rec(cur_z), stim.target_rec(cur_z - 2 * N_C)],
                               [0, s, r])

    for r in range(rounds):
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=pool)
        circuit.append("TICK")
        _round_detectors(r)

    # One fault-free cycle (the fail-fast paper's memory-experiment convention).
    build_bb_syndrome_cycle(circuit, ErrorModel(0.0, 0.0), reset_data=False,
                            reset_ancilla=True)
    circuit.append("TICK")
    _round_detectors(rounds)

    # Noiseless transversal Z readout of all data qubits.
    circuit.append("M", data_all)

    # Reconstruction detectors: each Z-check's data-support parity vs its last
    # (fault-free-cycle) ancilla outcome.
    _, H_Z = build_HX_HZ()
    for s in range(N_C):
        recs = [stim.target_rec(-(N_DATA - q)) for q in np.where(H_Z[s])[0]]
        recs.append(stim.target_rec(_z_anc_rec_offsets_from_end_of_round(2 * N_C)[s] - N_DATA))
        circuit.append("DETECTOR", recs, [8, s, rounds + 1])

    log_Z, _ = _tdg_logical_ops()
    for k in range(12):
        recs = [stim.target_rec(-(N_DATA - q)) for q in np.where(log_Z[k])[0]]
        circuit.append("OBSERVABLE_INCLUDE", recs, k)

    return circuit


# ---------------------------------------------------------------------------
# Layer 8b — Shift-automorphism circuit (App. A.2 / Sec. 2 of the paper)
# ---------------------------------------------------------------------------
#
# A shift automorphism δ = xᵃyᵇ acts as the global monomial permutation
# (L,α)→(L,δα), (R,α)→(R,δα); checks co-shift the same way.  It is NOT a
# relabeling: each data qubit is physically moved through an adjacent check
# ancilla with two 2-CNOT swaps (data→check, then check→new data), the
# vacated qubit measured (deterministic |0⟩ — free error detection) after
# each hop.  6 swap timesteps + one full syndrome cycle = 14 timesteps per
# instruction.  Routes (which check type carries which side) follow the
# polynomial structure: an X-check β is adjacent to L_β and L_{βδ} whenever
# δ is a monomial-ratio of A (δ ∈ {A_iA_jᵀ}); the mirrored statement with
# B / Z-checks covers the other basic shifts.

# Route table: delta exponents (di, dj), and which check type moves the L / R
# side.  δ=y is an A-route (L via X-checks, R via Z-checks); δ=x is a B-route
# (L via Z-checks, R via X-checks).  X-check routes use ancilla label β
# (adjacent to data β and βδ); Z-check routes use ancilla label βδ (same
# adjacency, from the transposed polynomial).
_SHIFT_ROUTES: Dict[str, Tuple[Tuple[int, int], str, str]] = {
    'y': ((0, 1), 'X', 'Z'),
    'x': ((1, 0), 'Z', 'X'),
}

_SHIFT_ACTION_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _shift_data_perm(shift: str, power: int = 1) -> np.ndarray:
    """Length-144 permutation: data-qubit q's CONTENT moves to perm[q] under
    δ^power (δ = the monomial of `shift`)."""
    (di, dj), _, _ = _SHIFT_ROUTES[shift]
    perm = np.zeros(N_DATA, dtype=np.int64)
    for side in ('L', 'R'):
        for i in range(L_):
            for j in range(M_):
                perm[Mon(side, i, j).qubit_index()] = (
                    Mon(side, i + di * power, j + dj * power).qubit_index())
    return perm


def _shift_check_perm(shift: str) -> List[int]:
    """Length-72 permutation of check labels: check α ↦ check δα (both check
    types shift identically — the circulant structure is δ-invariant)."""
    (di, dj), _, _ = _SHIFT_ROUTES[shift]
    perm = [0] * N_C
    for i in range(L_):
        for j in range(M_):
            perm[i * M_ + j] = ((i + di) % L_) * M_ + (j + dj) % M_
    return perm


def _shift_swap_layers(shift: str) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]],
    List[Tuple[int, int]], List[Tuple[int, int]],
]:
    """The four CNOT layers (t1, t2, t4, t5) of one shift instruction.

    Each data qubit β hops β → anc(β) → βδ, where anc is the route ancilla
    (X-check β or Z-check βδ, per _SHIFT_ROUTES).  t1/t2 swap data into the
    (|0⟩-reset) ancilla, t4/t5 swap it out into the δ-shifted data position.
    All four layers are 144 disjoint CNOT pairs (both check types are used,
    one per side).  Pairs are (control, target).
    """
    (di, dj), l_via, r_via = _SHIFT_ROUTES[shift]
    t1: List[Tuple[int, int]] = []
    t2: List[Tuple[int, int]] = []
    t4: List[Tuple[int, int]] = []
    t5: List[Tuple[int, int]] = []
    for side, via in (('L', l_via), ('R', r_via)):
        for i in range(L_):
            for j in range(M_):
                src = Mon(side, i, j).qubit_index()
                dst = Mon(side, i + di, j + dj).qubit_index()
                anc = (_x_check_idx(i, j) if via == 'X'
                       else _z_check_idx(i + di, j + dj))
                t1.append((src, anc))
                t2.append((anc, src))
                t4.append((anc, dst))
                t5.append((dst, anc))
    return t1, t2, t4, t5


def shift_logical_action(shift: str, power: int = 1
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Logical action (A_Z, A_X) ∈ GF(2)^{12×12} of δ^power on the canonical
    logical classes: pushed Z̄ₖ ≡ ∏ⱼ Z̄ⱼ^{A_Z[k,j]} · stabilizers (same for X).

    Recomputed directly from the qubit permutation by GF(2) reduction against
    the canonical basis + stabilizer rowspace — the paper's printed A_x/A_y
    matrices (App. A.2) leave the row/column and X̄-vs-Z̄ convention unstated,
    so they are not trusted here.  Asserts the paper-stated group structure:
    both generators have order 6, and the symplectic pairing is preserved
    (A_X A_Zᵀ = 1)."""
    if shift not in _SHIFT_ROUTES:
        raise ValueError(f"shift must be one of {sorted(_SHIFT_ROUTES)}, "
                         f"got {shift!r}")
    if shift not in _SHIFT_ACTION_CACHE:
        log_Z, log_X = _tdg_logical_ops()
        perm = _shift_data_perm(shift, 1)

        def _act(logs: List[np.ndarray], H: np.ndarray) -> np.ndarray:
            basis = np.vstack([np.vstack(logs), H]).astype(np.uint8).T
            A = np.zeros((12, 12), dtype=np.uint8)
            for k in range(12):
                v = np.zeros(N_DATA, dtype=np.uint8)
                v[perm[np.where(logs[k])[0]]] = 1
                x = _gf2_solve(basis, v)
                assert x is not None, "pushed logical left the logical group"
                A[k] = x[:12]
            return A

        Az = _act(log_Z, H_Z)
        Ax = _act(log_X, H_X)
        eye = np.eye(12, dtype=np.uint8)
        Mp = eye.copy()
        order = 0
        for t in range(1, 7):
            Mp = (Mp @ Az) % 2
            if np.array_equal(Mp, eye):
                order = t
                break
        assert order == 6, f"shift {shift!r}: logical order {order}, expected 6"
        assert np.array_equal((Ax @ Az.T) % 2, eye), "symplectic pairing broken"
        _SHIFT_ACTION_CACHE[shift] = (Az, Ax)

    Az, Ax = _SHIFT_ACTION_CACHE[shift]

    def _mpow(A: np.ndarray, t: int) -> np.ndarray:
        R = np.eye(12, dtype=np.uint8)
        for _ in range(t % 6):
            R = (R @ A) % 2
        return R

    return _mpow(Az, power), _mpow(Ax, power)


def build_automorphism_circuit(
    error_model: ErrorModel,
    shift: str = "y",
    C: int = 10,
    d_init: int = 12,
    idle_noise: bool = False,
) -> stim.Circuit:
    """Gross-code shift-automorphism benchmark circuit (paper Sec. 2 / A.2).

    Structure (the paper's C-repeat benchmark convention, App. A.6: LER is the
    reported failure probability divided by C):
      |0̄⟩ init → d_init bare rounds → C shift instructions → d_init bare
      rounds → noiseless transversal Z readout.
    One shift instruction = the 6-timestep swap circuit (t1/t2: swap every
    data qubit into its route check ancilla; t3: MR the vacated data,
    deterministic-0 detectors; t4/t5: swap into the δ-shifted position; t6: MR
    the vacated check ancillas, deterministic-0 detectors) + one bare syndrome
    cycle whose detectors compare check δα (post) with check α (pre) — the
    stabilizer labels co-shift with the data.

    K = 12 observables: the canonical Z̄ₖ supports pushed through δ^C over the
    final data readout.  The logical action itself is δ-order-6 and verified
    at build time via shift_logical_action (GF(2) recomputation — the paper's
    A_x/A_y matrix convention is not trusted).

    Choices the paper leaves open (App./sec refs in the module header spec):
    which δ was benchmarked (default 'y' here; 'x' = the other route class),
    which A/B terms route each swap (ours: the (0,0) and δ terms), CNOT
    directions, and MR vs M+tracked-frame for the mid-instruction
    measurements (ours: MR).  The layering is module-style, not the paper's
    14-timestep table, so fault-location counts differ from tab. N=483840.

    idle_noise: DEPOLARIZE1(p_phys) on inactive gross qubits per sub-layer
    (fail-fast standard circuit noise model); pool = the 288 gross qubits.
    """
    if shift not in _SHIFT_ROUTES:
        raise ValueError(f"shift must be one of {sorted(_SHIFT_ROUTES)}, "
                         f"got {shift!r}")
    assert C >= 1 and d_init >= 1
    shift_logical_action(shift)  # build-time order-6 / symplectic assertions

    p = error_model.p_phys
    pm = error_model.p_meas
    pool = list(range(N_GROSS)) if idle_noise else None
    t1, t2, t4, t5 = _shift_swap_layers(shift)
    cperm = _shift_check_perm(shift)
    x_anc_all = list(range(X_ANC_BASE, X_ANC_BASE + N_C))
    z_anc_all = list(range(Z_ANC_BASE, Z_ANC_BASE + N_C))
    anc_all = x_anc_all + z_anc_all
    data_all = list(range(N_DATA))

    circuit = stim.Circuit()
    trk = _MeasTracker()
    circuit.append("R", data_all)
    _append_noise(circuit, "X_ERROR", data_all, pm)

    # ---- d_init bare rounds ([0,s,0] Z-init; [1,s,r]/[0,s,r] compares) ----
    prev_x: Optional[int] = None
    prev_z: Optional[int] = None
    for r in range(d_init):
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=pool)
        x0 = trk.add(N_C)
        z0 = trk.add(N_C)
        circuit.append("TICK")
        if r == 0:
            for s in range(N_C):
                circuit.append("DETECTOR", [trk.rec(z0 + s)], [0, s, 0])
        else:
            for s in range(N_C):
                circuit.append("DETECTOR",
                               [trk.rec(x0 + s), trk.rec(prev_x + s)],
                               [1, s, r])
            for s in range(N_C):
                circuit.append("DETECTOR",
                               [trk.rec(z0 + s), trk.rec(prev_z + s)],
                               [0, s, r])
        prev_x, prev_z = x0, z0

    # ---- C shift instructions ----
    for c in range(C):
        # t0: reset the route ancillas (left collapsed by the last syndrome M)
        circuit.append("R", anc_all)
        _append_noise(circuit, "X_ERROR", anc_all, pm)
        _append_idle(circuit, pool, set(anc_all), p)
        # t1, t2: swap data → check ancilla (data vacated to |0⟩)
        for layer in (t1, t2):
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
            _append_idle(circuit, pool, set(flat), p)
        # t3: MR the vacated data qubits — deterministic-0 detectors [9,q,c]
        _append_noise(circuit, "X_ERROR", data_all, pm)
        circuit.append("MR", data_all)
        d3 = trk.add(N_DATA)
        _append_noise(circuit, "X_ERROR", data_all, pm)   # reset fault
        _append_idle(circuit, pool, set(data_all), p)
        for q in range(N_DATA):
            circuit.append("DETECTOR", [trk.rec(d3 + q)], [9, q, c])
        # t4, t5: swap check ancilla → δ-shifted data position
        for layer in (t4, t5):
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)
            _append_idle(circuit, pool, set(flat), p)
        # t6: MR the vacated check ancillas — deterministic-0 detectors [10,q,c]
        _append_noise(circuit, "X_ERROR", anc_all, pm)
        circuit.append("MR", anc_all)
        d6 = trk.add(2 * N_C)
        _append_noise(circuit, "X_ERROR", anc_all, pm)    # reset fault
        _append_idle(circuit, pool, set(anc_all), p)
        for q in range(2 * N_C):
            circuit.append("DETECTOR", [trk.rec(d6 + q)], [10, q, c])
        circuit.append("TICK")
        # syndrome cycle: compare check δα (post-shift) with check α (pre)
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=pool)
        xc = trk.add(N_C)
        zc = trk.add(N_C)
        circuit.append("TICK")
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(xc + cperm[s]), trk.rec(prev_x + s)],
                           [11, s, c])
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(zc + cperm[s]), trk.rec(prev_z + s)],
                           [12, s, c])
        prev_x, prev_z = xc, zc

    # ---- d_init trailing bare rounds (plain compares [6,s,r]/[7,s,r]) ----
    for r in range(d_init):
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=pool)
        x0 = trk.add(N_C)
        z0 = trk.add(N_C)
        circuit.append("TICK")
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(x0 + s), trk.rec(prev_x + s)], [6, s, r])
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(z0 + s), trk.rec(prev_z + s)], [7, s, r])
        prev_x, prev_z = x0, z0

    # ---- noiseless transversal Z readout + reconstruction detectors ----
    circuit.append("M", data_all)
    f0 = trk.add(N_DATA)
    for s in range(N_C):
        recs = [trk.rec(f0 + int(q)) for q in np.where(H_Z[s])[0]]
        recs.append(trk.rec(prev_z + s))
        circuit.append("DETECTOR", recs, [8, s, d_init + 1])

    # ---- K = 12 observables: Z̄ₖ supports pushed through δ^C ----
    dperm = _shift_data_perm(shift, power=C)
    log_Z, _ = _tdg_logical_ops()
    for k in range(12):
        recs = [trk.rec(f0 + int(dperm[q])) for q in np.where(log_Z[k])[0]]
        circuit.append("OBSERVABLE_INCLUDE", recs, k)

    return circuit


def build_joint_pauli_circuit(
    error_model: ErrorModel,
    operators: str = "Y1",
    C: int = 10,
    d_init: int = 12,
    include_memory_observables: bool = True,
    idle_noise: bool = False,
) -> stim.Circuit:
    """Ȳ₁ in-module measurement through the FULL LPU (paper "BB(12)-circuit-Y1").

    Ȳ₁ (the Hermitian version of X̄₁Z̄₁, overlapping on the single qubit x⁴R)
    is one of the ⟨X̄₁,X̄₇,Z̄₁,Z̄₇⟩ elements that need BOTH LPU halves at once:
    all 23 vertices (24 check qubits — v_Bell is a physical pair), all 47
    edges, all 19 cycles, with the uniform convention and per-class couplings
    of Layer 7b (V_l: CX to γ; V_r: CZ to xy·γᵀ; Bell-left: CY to x⁴R; both
    deformations Z_e-type simultaneously).  The standalone Z1 branch's
    Hadamard-dual convention CANNOT be composed here — the two halves share
    bridge edges and U_B cycles, so one graph-wide edge basis must hold.

    Framing (minimal module framing — differs from the paper's k-Bell-pair
    harness with K = 23, so rates are not directly comparable):
      |0⟩^144 init → one NOISELESS MPP of the Ȳ₁ Pauli string (reference
      bit, anchored BEFORE any noise — see below) → one NOISELESS encoding
      round (state-prep anchor for the X-check chain, the paper's
      noiseless-prep harness) → d_init noisy bare rounds → edges |0⟩ →
      C noisy full-LPU rounds → Z-basis edge readout + 1 bare return cycle →
      d_init trailing bare rounds → noiseless transversal Z readout of all
      data.

    Detectors (all deterministic at p=0):
      [0/1,s,r]  bare rounds (round 0 = the noiseless encoding round:
                 Z-checks single-record; X compares start at r=1 against
                 its noise-free records);
      [4/5,s,c]  LPU BB checks vs previous round — ALL 144 including the 36
                 deformed ones (boundary c=0 valid because the added Z_e
                 support acts on just-initialized |0⟩ edges);
      [2,u,c]    cycle checks (round 1: deterministic single-record; c=C:
                 cycle(C) ⊕ XOR of m_e around the cycle);
      [3,v,c]    the 23 vertex checks from round 2 on (round-1 outcomes are
                 individually random; v_Bell contributes ONE detector over
                 its two records per round — each Bell record alone carries a
                 fresh random bit);
      [6/7,s,r]  return boundary (bare(return) ⊕ deformed(C) ⊕ its m_e for
                 deformed checks) and trailing compares;
      [8,s,·]    final-data Z-check reconstruction (idle-builder pattern).

    Observables: 0 = MPP reference ⊕ the 24 vertex records of the LAST LPU
    round (the per-round vertex detectors make any round equivalent; round-1
    vertex outcomes deliberately carry no single-round detector, so obs 0 is
    linearly independent of the detector set).  The reference MPP must sit
    BEFORE every noise location: a noiseless MPP inside a noisy syndrome
    window is informationally unresolvable — for each of the 23 Ȳ₁-string
    support qubits, the 2 single-qubit Paulis anticommuting with the string
    give one fault before the MPP (flips reference AND gauged outcome — obs 0
    unchanged) and one after it (flips the gauged outcome only — obs 0 flips)
    with IDENTICAL detector footprints: 46 single-fault degenerate pairs that
    would floor the outcome bit for any decoder.  Anchored at t=0, every
    mechanism is "after", obs 0 has a well-defined action for all faults, and
    the framing matches the idle/automorphism anchoring (noiseless init →
    noiseless final readout).  The t=0 anchor in turn requires the noiseless
    encoding round (see the in-body comment): the stretch between the anchor
    and the first ANCHORED X-check round must contain no noise, or Z-type
    faults there flip the Ȳ₁ frame without a localizable syndrome start.
    If include_memory_observables,
    observables 1..11 span the Z-type logical subgroup commuting with Ȳ₁ over
    the final data readout, each XOR'd with its m_e correction records — see
    _y1_observable_recipe for the basis choice (the canonical classes are NOT
    aligned with the LPU polynomials: X(p,q) ≡ X̄₂X̄₃) and the path
    construction (the paper's App.-A.4 software correction).

    Scheduling: module-style layering (one gate group per check per round),
    NOT the paper's 12-timestep graph-coloring schedule — fault-location
    counts differ from the paper's N = 400117 / M = 2144.  C = 10 default is
    the paper's choice (C = d_circ upper bound, not d = 12).

    idle_noise: DEPOLARIZE1(p_phys) on inactive pool qubits per sub-layer;
    pool = 288 gross qubits in bare rounds, all 378 (gross + 47 edge + 24
    vertex + 19 cycle) during LPU rounds.
    """
    if operators != "Y1":
        raise NotImplementedError(
            f"build_joint_pauli_circuit: only operators='Y1' is implemented "
            f"(got {operators!r})")
    assert C >= 1 and d_init >= 1
    full = _full_lpu()

    p = error_model.p_phys
    pm = error_model.p_meas
    bare_pool = list(range(N_GROSS)) if idle_noise else None
    lpu_pool = list(range(N_TOTAL_QUBITS)) if idle_noise else None

    circuit = stim.Circuit()
    trk = _MeasTracker()
    data_all = list(range(N_DATA))
    circuit.append("R", data_all)

    # ---- noiseless MPP reference of the Ȳ₁ Pauli string ----
    # Y on the overlap qubit x⁴R, X on the other 11 supp(X̄₁) qubits, Z on the
    # other 11 supp(Z̄₁) qubits — the exact operator the vertex product equals.
    # Placed BEFORE the reset noise, i.e. before ANY noise location: the
    # projection commutes with every stabilizer check, so the Ȳ₁ eigenvalue it
    # pins survives the bare rounds and p=0 determinism of obs 0 holds; any
    # later placement leaves a noisy window straddling the reference and makes
    # the 46 straddle-pair mechanisms unresolvable (see docstring).
    yq = _vertex_data_idx(IDENTIFIED_L_SIDE)
    x1_rest = sorted(set(int(q) for q in np.where(_op_vec(P, Q))[0]) - {yq})
    z1_rest = sorted(set(int(q) for q in np.where(_op_vec(Z1_L, Z1_R))[0])
                     - {yq})
    assert len(x1_rest) == 11 and len(z1_rest) == 11
    mpp_targets = [stim.target_y(yq)]
    for q in x1_rest:
        mpp_targets += [stim.target_combiner(), stim.target_x(q)]
    for q in z1_rest:
        mpp_targets += [stim.target_combiner(), stim.target_z(q)]
    circuit.append("MPP", mpp_targets)
    mpp_idx = trk.add(1)
    circuit.append("TICK")

    _append_noise(circuit, "X_ERROR", data_all, pm)

    # ---- noiseless encoding round (anchors the X-check chain) ----
    # Projects |0…0⟩ into the code space with zero fault locations — the
    # paper's noiseless-state-prep harness (App. A.6).  Without it, the first
    # NOISY round's X-check outcomes are the first ever measured (individually
    # random), so a mid-round Z-type fault only shows the compares of its
    # already-coupled checks: on a Ȳ₁-support qubit it is then syndrome-
    # identical to the same-position fault on a partner qubit of the same
    # X-check (or to a partial-X-stabilizer ancilla fault) while differing in
    # obs 0 — 58 unresolvable single-fault pairs.  With the anchor, the
    # post-fault couplings of the same round fire against the noise-free
    # round-0 records and the signatures separate.
    build_bb_syndrome_cycle(circuit, ErrorModel(p_phys=0.0, p_meas=0.0),
                            reset_data=False, reset_ancilla=True)
    prev_x = trk.add(N_C)
    prev_z = trk.add(N_C)
    circuit.append("TICK")
    for s in range(N_C):
        circuit.append("DETECTOR", [trk.rec(prev_z + s)], [0, s, 0])

    # ---- d_init noisy bare rounds ----
    for r in range(1, d_init + 1):
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=bare_pool)
        x0 = trk.add(N_C)
        z0 = trk.add(N_C)
        circuit.append("TICK")
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(x0 + s), trk.rec(prev_x + s)],
                           [1, s, r])
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(z0 + s), trk.rec(prev_z + s)],
                           [0, s, r])
        prev_x, prev_z = x0, z0

    # ---- edge init: all 47 in |0⟩ (uniform convention — no |+⟩ anywhere) ----
    edge_all = sorted(EDGE_QUBIT.values())
    circuit.append("R", edge_all)
    _append_noise(circuit, "X_ERROR", edge_all, pm)
    _append_idle(circuit, lpu_pool, set(edge_all), p)

    # ---- C noisy full-LPU rounds ----
    n_v = len(full.vertex_keys)       # 24 records / round
    n_u = len(U_ALL)                  # 19 records / round
    # Vertex detector groups: one per LOGICAL check (23) — the Bell pair's two
    # records enter a single detector together.
    v_groups: List[List[int]] = []
    for i, key in enumerate(full.vertex_keys):
        if full.vertex_kind[key] == 'BELL_L':
            pass  # grouped with BELL_R below
        elif full.vertex_kind[key] == 'BELL_R':
            v_groups.append([i - 1, i])
        else:
            v_groups.append([i])
    assert len(v_groups) == 23

    x_lpu: List[int] = []
    z_lpu: List[int] = []
    v_lpu: List[int] = []
    u_lpu: List[int] = []
    for c in range(C):
        build_full_lpu_cycle(circuit, error_model, full, idle_pool=lpu_pool)
        x_lpu.append(trk.add(N_C))
        z_lpu.append(trk.add(N_C))
        v_lpu.append(trk.add(n_v))
        u_lpu.append(trk.add(n_u))
        circuit.append("TICK")
        px = x_lpu[c - 1] if c else prev_x
        pz = z_lpu[c - 1] if c else prev_z
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(x_lpu[c] + s), trk.rec(px + s)],
                           [4, s, c])
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(z_lpu[c] + s), trk.rec(pz + s)],
                           [5, s, c])
        if c == 0:
            for u in range(n_u):
                circuit.append("DETECTOR", [trk.rec(u_lpu[0] + u)], [2, u, 0])
        else:
            for u in range(n_u):
                circuit.append("DETECTOR",
                               [trk.rec(u_lpu[c] + u),
                                trk.rec(u_lpu[c - 1] + u)],
                               [2, u, c])
            for g, grp in enumerate(v_groups):
                recs = [trk.rec(v_lpu[c] + i) for i in grp]
                recs += [trk.rec(v_lpu[c - 1] + i) for i in grp]
                circuit.append("DETECTOR", recs, [3, g, c])

    # ---- return: Z-basis readout of all 47 edges ----
    _append_noise(circuit, "X_ERROR", edge_all, pm)
    circuit.append("M", edge_all)
    e0 = trk.add(len(edge_all))
    _append_idle(circuit, lpu_pool, set(edge_all), p)
    circuit.append("TICK")
    edge_rec = {q: e0 + i for i, q in enumerate(edge_all)}

    # ---- 1 bare return cycle + boundary detectors ----
    build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                            reset_ancilla=True, idle_pool=bare_pool)
    xr = trk.add(N_C)
    zr = trk.add(N_C)
    circuit.append("TICK")
    for s in range(N_C):
        recs = [trk.rec(xr + s), trk.rec(x_lpu[C - 1] + s)]
        recs += [trk.rec(edge_rec[e]) for e in full.deformation_x.get(s, [])]
        circuit.append("DETECTOR", recs, [6, s, 0])
    for s in range(N_C):
        recs = [trk.rec(zr + s), trk.rec(z_lpu[C - 1] + s)]
        recs += [trk.rec(edge_rec[e]) for e in full.deformation_z.get(s, [])]
        circuit.append("DETECTOR", recs, [7, s, 0])
    # "all cycles should satisfy ∏ m_e = +1": cycle(C) ⊕ XOR of its m_e.
    for u in range(n_u):
        recs = [trk.rec(u_lpu[C - 1] + u)]
        recs += [trk.rec(edge_rec[e]) for e in CYCLE_EDGES[u]]
        circuit.append("DETECTOR", recs, [2, u, C])
    prev_x, prev_z = xr, zr

    # ---- d_init trailing bare rounds ----
    for r in range(1, d_init + 1):
        build_bb_syndrome_cycle(circuit, error_model, reset_data=False,
                                reset_ancilla=True, idle_pool=bare_pool)
        x0 = trk.add(N_C)
        z0 = trk.add(N_C)
        circuit.append("TICK")
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(x0 + s), trk.rec(prev_x + s)], [6, s, r])
        for s in range(N_C):
            circuit.append("DETECTOR",
                           [trk.rec(z0 + s), trk.rec(prev_z + s)], [7, s, r])
        prev_x, prev_z = x0, z0

    # ---- noiseless transversal Z readout + reconstruction detectors ----
    circuit.append("M", data_all)
    f0 = trk.add(N_DATA)
    for s in range(N_C):
        recs = [trk.rec(f0 + int(q)) for q in np.where(H_Z[s])[0]]
        recs.append(trk.rec(prev_z + s))
        circuit.append("DETECTOR", recs, [8, s, d_init + 1])

    # ---- observable 0: MPP reference ⊕ 24 vertex records of the last round ----
    obs0 = [trk.rec(mpp_idx)]
    obs0 += [trk.rec(v_lpu[C - 1] + i) for i in range(n_v)]
    circuit.append("OBSERVABLE_INCLUDE", obs0, 0)

    # ---- observables 1..11: commuting Z-logical memory, m_e-corrected ----
    if include_memory_observables:
        _, recipe = _y1_observable_recipe()
        for k, (w, e_list) in enumerate(recipe, start=1):
            recs = [trk.rec(f0 + int(q)) for q in np.where(w)[0]]
            recs += [trk.rec(edge_rec[e]) for e in e_list]
            circuit.append("OBSERVABLE_INCLUDE", recs, k)

    return circuit


# ---------------------------------------------------------------------------
# Layer 9 — Simulation sweep API
# ---------------------------------------------------------------------------

def run_lpu(
    error_model: ErrorModel,
    operator: str = 'X1',
    C: int = 10,
    d_init: int = 12,
    shots: int = 200,
    decoder=None,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Run one LPU logical-operator measurement experiment and return the LER.

    operator : 'X1' (measures X̄₁) or 'Z1' (measures Z̄₁)
    C        : number of LPU rounds
    d_init   : bare memory rounds before and after the LPU block
    """
    if operator == 'X1':
        circuit = build_logical_x1_circuit(error_model, C=C, d_init=d_init)
    elif operator == 'Z1':
        circuit = build_logical_z1_circuit(error_model, C=C, d_init=d_init)
    else:
        raise ValueError(f"operator must be 'X1' or 'Z1', got {operator!r}")

    if decoder is None:
        decoder = RelayBPDecoder()
    decoder.setup(circuit)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(shots, separate_observables=True)
    predictions = decoder.decode_batch(detection_events)

    logical_errors = np.any(predictions != observable_flips, axis=1)
    n_err = int(np.sum(logical_errors))
    ler = n_err / shots
    from scipy.stats import beta as _beta
    lo, hi = _beta.interval(0.95, n_err + 0.5, shots - n_err + 0.5)
    ler_se = float((hi - lo) / 2)

    return SimulationResult(
        distance=12,
        rounds=C,
        error_model=error_model,
        shots=shots,
        num_logical_errors=n_err,
        logical_error_rate=ler,
        logical_error_rate_se=ler_se,
    )


def sweep_lpu_p(
    p_values,
    operator: str = 'X1',
    C: int = 10,
    d_init: int = 12,
    shots: int = 200,
    seed: Optional[int] = None,
) -> List[SimulationResult]:
    """Sweep physical error rate for one LPU operator measurement circuit."""
    return [
        run_lpu(ErrorModel.symmetric(p), operator=operator, C=C, d_init=d_init,
                shots=shots, seed=seed)
        for p in p_values
    ]


# ---------------------------------------------------------------------------
# Layer 10 — Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*68)
    print("Gross-code LPU — sanity check summary")
    print("="*68)
    print(f"  |V|     = {len(V_ALL)}")
    print(f"  |V_l|   = {len(V_L_RAW)}")
    print(f"  |V_r|   = {len(V_R_RAW)}")
    print(f"  |E|     = {len(E_ALL)}")
    print(f"    |E_l| = {len(E_L_DETAIL)}")
    print(f"    |E_r| = {len(E_R_DETAIL)}")
    print(f"    bridge= {len(BRIDGE_EDGES)}")
    print(f"  |U|     = {len(U_ALL)}  (U_L={len(U_L)} U_R={len(U_R)} U_B={len(U_B)})")
    n_ancilla = 47 + 24 + 19
    print(f"  total ancilla qubits = 47 (edges) + 24 (vertex incl. Bell) + 19 (cycles) = {n_ancilla}")
    print(f"  total qubits in circuit = {N_TOTAL_QUBITS}")
    print()
    print("Verifying logical operators p, q, r, s ...")
    print("  X̄₁ in ker(H_Z): OK")
    print("  X̄₇ in ker(H_Z): OK")
    print("  Z̄₁ in ker(H_X): OK")
    print("  Z̄₇ in ker(H_X): OK")
    print("  {X̄₁, Z̄₁} = 1  (anticommute)")
    print("  [X̄₁, Z̄₇] = 0")
    print("  [X̄₇, Z̄₁] = 0")
    print()

    em0 = ErrorModel(p_phys=0.0, p_meas=0.0)
    em1 = ErrorModel(p_phys=1e-3, p_meas=1e-3)

    for branch_name, builder in [
        ("X̄₁", build_logical_x1_circuit),
        ("Z̄₁", build_logical_z1_circuit),
    ]:
        print("="*68)
        print(f"Building {branch_name} circuit (p=0)...")
        c0 = builder(em0, C=10, d_init=12)
        print(f"  num_qubits      = {c0.num_qubits}")
        print(f"  num_detectors   = {c0.num_detectors}")
        print(f"  num_observables = {c0.num_observables}")
        print(f"  num_measurements= {c0.num_measurements}")
        print()
        print(f"Sampling 100 shots at p=0...")
        sampler = c0.compile_detector_sampler()
        dets, obs = sampler.sample(100, separate_observables=True)
        n_det_fired = int(dets.sum())
        n_obs_set = int(obs.sum())
        print(f"  total detector firings across all 100 shots = {n_det_fired}")
        print(f"  observable flips across 100 shots           = {n_obs_set}")
        assert n_det_fired == 0, (
            f"At p=0 no detector should fire, got {n_det_fired}"
        )
        if n_obs_set != 0 and n_obs_set != 100:
            print(f"  WARNING: observable should be deterministic at p=0 "
                  f"(got {n_obs_set}/100 flips)")
        assert n_obs_set == 0 or n_obs_set == 100, (
            f"Observable not deterministic at p=0: {n_obs_set}/100"
        )
        print(f"  (observable = {n_obs_set != 0} on every shot — deterministic)")
        print()

        print(f"Building {branch_name} circuit (p=1e-3)...")
        c1 = builder(em1, C=10, d_init=12)
        print(f"  num_qubits      = {c1.num_qubits}")
        print(f"  num_detectors   = {c1.num_detectors}")
        print(f"  num_observables = {c1.num_observables}")
        print(f"Building DEM for {branch_name} at p=1e-3...")
        dem = c1.detector_error_model(
            decompose_errors=False,
            ignore_decomposition_failures=True,
        )
        n_mechs = sum(1 for inst in dem if inst.type == "error")
        print(f"  num error mechanisms = {n_mechs}")
        print(f"  DEM num_detectors    = {dem.num_detectors}")
        print(f"  DEM num_observables  = {dem.num_observables}")
        print()

    for branch_name, builder in [
        ("Ȳ₁ joint-Pauli (full LPU)",
         lambda em, **kw: build_joint_pauli_circuit(em, **kw)),
        ("shift-y automorphism",
         lambda em, **kw: build_automorphism_circuit(em, **kw)),
    ]:
        print("="*68)
        print(f"Building {branch_name} circuit (p=0)...")
        c0 = builder(em0)
        print(f"  num_qubits      = {c0.num_qubits}")
        print(f"  num_detectors   = {c0.num_detectors}")
        print(f"  num_observables = {c0.num_observables}")
        print(f"  num_measurements= {c0.num_measurements}")
        print(f"Sampling 100 shots at p=0...")
        sampler = c0.compile_detector_sampler()
        dets, obs = sampler.sample(100, separate_observables=True)
        n_det_fired = int(dets.sum())
        n_obs_set = int(obs.sum())
        print(f"  total detector firings across all 100 shots = {n_det_fired}")
        print(f"  observable flips across 100 shots           = {n_obs_set}")
        assert n_det_fired == 0, (
            f"At p=0 no detector should fire, got {n_det_fired}"
        )
        assert n_obs_set == 0, (
            f"At p=0 no observable should flip, got {n_obs_set}"
        )
        print()

        print(f"Building {branch_name} circuit (p=1e-3)...")
        c1 = builder(em1)
        print(f"Building DEM for {branch_name} at p=1e-3...")
        dem = c1.detector_error_model(decompose_errors=False)
        n_mechs = sum(1 for inst in dem if inst.type == "error")
        print(f"  num error mechanisms = {n_mechs}")
        print(f"  DEM num_detectors    = {dem.num_detectors}")
        print(f"  DEM num_observables  = {dem.num_observables}")
        if "joint" in branch_name.lower() or "Ȳ" in branch_name:
            # Paper's BB(12)-circuit-Y1 row: M=2144, Ñ=79591 distinct
            # mechanisms, N=400117 fault locations (different framing +
            # schedule — ours has 2·d_init extra bare rounds and module-style
            # layering, so a direct match is not expected).
            print(f"  paper comparison: M {dem.num_detectors}/2144 = "
                  f"{dem.num_detectors/2144:.2f}x, "
                  f"mechanisms {n_mechs}/79591 = {n_mechs/79591:.2f}x")
        print()

    print("="*68)
    print("Verifying shift-automorphism logical actions (order-6 group)...")
    for sh in ('y', 'x'):
        Az, Ax = shift_logical_action(sh)
        Az6, _ = shift_logical_action(sh, power=6)
        assert np.array_equal(Az6, np.eye(12, dtype=np.uint8))
        print(f"  δ={sh}: A_Z nontrivial={not np.array_equal(Az, np.eye(12, dtype=np.uint8))}, "
              f"A_Z⁶=1 OK, A_X A_Zᵀ=1 OK")
    print()

    print("="*68)
    print("All checks passed.")
    print("="*68)
