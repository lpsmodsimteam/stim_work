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


def build_bb_syndrome_cycle(
    circuit: stim.Circuit,
    error_model: ErrorModel,
    deformation_edges_per_zcheck: Optional[Dict[int, List[int]]] = None,
    deformation_edges_per_xcheck: Optional[Dict[int, List[int]]] = None,
    reset_data: bool = False,
    reset_ancilla: bool = True,
    skip_x_checks: bool = False,
    skip_z_checks: bool = False,
) -> None:
    """
    Append one round of gross-code stabilizer measurement to `circuit`.

    deformation_edges_per_zcheck[z_check_id] = list of edge-qubit indices that
        the Z-check picks up support on (during code deformation).  An extra
        CNOT from each such edge qubit into the Z-ancilla is appended in a
        separate layer at the end of the Z-cycle.

    deformation_edges_per_xcheck: same for X-checks.

    reset_data: if True, reset all 144 data qubits in |0⟩ at the start.
    reset_ancilla: if True (default), reset all ancilla at the start.

    skip_x_checks / skip_z_checks: if True, omit the corresponding
    syndrome-extraction circuit entirely (no CNOTs, no measurements for that
    ancilla block).  Used during LPU rounds where one check-type would
    anticommute with the vertex checks and randomize them.
    """
    p = error_model.p_phys
    pm = error_model.p_meas

    x_anc_all = list(range(X_ANC_BASE, X_ANC_BASE + N_C))
    z_anc_all = list(range(Z_ANC_BASE, Z_ANC_BASE + N_C))

    if reset_data:
        data_all = list(range(N_DATA))
        circuit.append("R", data_all)
        _append_noise(circuit, "X_ERROR", data_all, pm)

    if reset_ancilla:
        if not skip_x_checks:
            circuit.append("R", x_anc_all)
            _append_noise(circuit, "X_ERROR", x_anc_all, pm)
            circuit.append("H", x_anc_all)
            _append_noise(circuit, "DEPOLARIZE1", x_anc_all, p)
        if not skip_z_checks:
            circuit.append("R", z_anc_all)
            _append_noise(circuit, "X_ERROR", z_anc_all, pm)

    # X-check CNOT layers: X-ancilla -> data
    if not skip_x_checks:
        for layer in X_LAYERS:
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)

        # Deformation X-CNOTs: extra CNOTs from X-ancilla -> edge qubits
        if deformation_edges_per_xcheck:
            extra_flat: List[int] = []
            for xc_id, edge_qs in deformation_edges_per_xcheck.items():
                xa = X_ANC_BASE + xc_id
                for eq in edge_qs:
                    extra_flat.extend([xa, eq])
            if extra_flat:
                circuit.append("CX", extra_flat)
                _append_noise(circuit, "DEPOLARIZE2", extra_flat, p)

    if not skip_z_checks:
        # Z-check CNOT layers: data -> Z-ancilla
        for layer in Z_LAYERS:
            flat = [q for pair in layer for q in pair]
            circuit.append("CX", flat)
            _append_noise(circuit, "DEPOLARIZE2", flat, p)

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

    # Measure X-ancilla (Hadamard back, then M)
    if not skip_x_checks:
        circuit.append("H", x_anc_all)
        _append_noise(circuit, "DEPOLARIZE1", x_anc_all, p)
        _append_noise(circuit, "X_ERROR", x_anc_all, pm)
        circuit.append("M", x_anc_all)

    # Measure Z-ancilla
    if not skip_z_checks:
        _append_noise(circuit, "X_ERROR", z_anc_all, pm)
        circuit.append("M", z_anc_all)


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
# Layer 9 — Demo / smoke test
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

    print("="*68)
    print("All checks passed.")
    print("="*68)
