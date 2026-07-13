"""Programmatic derivation of tour-de-gross LPU layouts (arXiv:2506.03094, App. A.1/A.3).

The LPU auxiliary graph G for measuring the logical basis <X1, X7, Z1, Z7> of a BB code is
built from a (p, q, r, s, mu) logical basis (paper eq:pqrs_basis, mu = nu):

    X1 = X(p, q)        Z1 = Z(mu*s^T, mu*r^T)
    X7 = X(r, s)        Z7 = Z(mu*q^T, mu*p^T)

Construction rules (App. A.3, all now encoded here rather than hand-transcribed):
  VERTICES   G_l: one per monomial of pL+qR (the qubits of X1); G_r: one per rL+sR. A vertex
             gamma additionally serves the dual measurement via qubit mu*gamma^T.
  EDGES      (gamma, delta) whenever the two qubits share a Z check (keeps deformed-code check
             degree growth <= 1). Generalization hook: extra "expander" edges may be required
             to reach the target fault distance (gross: none; two-gross: 2 — paper samples
             randomly + certifies with CPLEX; certification is OUT of scope here).
  IDENTIFIED property 3 of the basis guarantees X1 and Z1 overlap on EXACTLY one qubit; the
             G_l vertex of that qubit is identified with the G_r vertex delta satisfying
             mu*delta^T = that qubit (Bell-pair vertex).
  BRIDGES    a Hamiltonian path through V_l \\ {id} in G_l, one through V_r \\ {id} in G_r,
             paired positionally (the "tour"); some rung k must close a triangle through the
             identified vertex (id—top[k] in G_l and id—bot[k] in G_r).
  CYCLES     U_B = the id-triangle + the 10 rung squares (derived from the bridge order).
             U_l/U_r = a minimum cycle basis with paper-chosen eliminations (redundant modulo
             gross-check redundancies, longest eliminated first). The elimination rule is NOT
             re-derived here yet: derived layouts keep a FULL fundamental cycle basis (valid,
             merely more cycle checks than optimal); the gross layout uses the paper's sets.

`certify_layout` checks any layout — including the hand-transcribed gross fixture in
`gross_code_lpu_tdg` — against every structural rule above; `tests/test_lpu_graph.py` pins
that certification. Derived layouts for NEW codes come from `derive_lpu_layout`, which fails
loudly (LPUDerivationError) when the code/basis cannot satisfy the rules — the mini-LPU
existence question is answered by exactly that failure or its absence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from bb_code_sim import BBCodeParams, _poly_matrix

Vertex = Tuple[str, int, int]          # ('L'|'R', i, j) — a data qubit / its graph vertex
Poly = List[Tuple[int, int]]           # monomial exponent list


class LPUDerivationError(RuntimeError):
    """The construction's requirements are violated — the LPU does NOT exist for this input."""


# ------------------------------ polynomial helpers ------------------------------
def _norm(params: BBCodeParams, side: str, i: int, j: int) -> Vertex:
    return (side, i % params.l, j % params.m)


def _transpose(params: BBCodeParams, v: Vertex) -> Vertex:
    side, i, j = v
    return ('R' if side == 'L' else 'L', (-i) % params.l, (-j) % params.m)


def _shift(params: BBCodeParams, v: Vertex, di: int, dj: int) -> Vertex:
    side, i, j = v
    return (side, (i + di) % params.l, (j + dj) % params.m)


def support(params: BBCodeParams, p: Poly, q: Poly) -> List[Vertex]:
    """Qubits of X(p, q) (or Z(p, q)) as vertex tuples: p on the L block, q on the R block."""
    return ([_norm(params, 'L', i, j) for i, j in p]
            + [_norm(params, 'R', i, j) for i, j in q])


def poly_T(params: BBCodeParams, p: Poly) -> Poly:
    return [((-i) % params.l, (-j) % params.m) for i, j in p]


def poly_mul_mono(params: BBCodeParams, p: Poly, mono: Tuple[int, int]) -> Poly:
    return [((i + mono[0]) % params.l, (j + mono[1]) % params.m) for i, j in p]


def poly_mul(params: BBCodeParams, a: Poly, b: Poly) -> Poly:
    """Product in F2[x,y]/(x^l-1, y^m-1) (terms cancel mod 2)."""
    acc: Set[Tuple[int, int]] = set()
    for m1 in a:
        for t in poly_mul_mono(params, b, m1):
            acc ^= {t}
    return sorted(acc)


def qubit_index(params: BBCodeParams, v: Vertex) -> int:
    side, i, j = v
    base = 0 if side == 'L' else params.l * params.m
    return base + (i % params.l) * params.m + (j % params.m)


# ------------------------------ layout container ------------------------------
@dataclass
class LPULayout:
    """Everything the circuit builders need, in the fixture's exact shapes/orders."""
    params: BBCodeParams
    p: Poly
    q: Poly
    r: Poly
    s: Poly
    mu: Tuple[int, int]
    V_l: List[Vertex]
    V_r: List[Vertex]
    identified_l: Vertex
    identified_r: Vertex
    E_l_detail: List[Tuple[Vertex, Vertex, int]]      # (gamma, delta, z_check_index)
    E_r_detail: List[Tuple[Vertex, Vertex, int]]
    bridge_top: List[Vertex]
    bridge_bottom: List[Vertex]
    U_l: List[List[Vertex]]                            # closed vertex walks
    U_r: List[List[Vertex]]
    U_b: List[List[Vertex]] = field(default_factory=list)

    def __post_init__(self):
        if not self.U_b:
            self.U_b = derive_U_b(self)

    @property
    def n_vertices(self) -> int:
        return len(self.V_l) + len(self.V_r) - 1        # identified pair counts once

    @property
    def n_edges(self) -> int:
        return len(self.E_l_detail) + len(self.E_r_detail) + len(self.bridge_top)

    @property
    def cycles(self) -> List[List[Vertex]]:
        return list(self.U_l) + list(self.U_r) + list(self.U_b)


# ------------------------------ derivations ------------------------------
def build_HX_HZ(params: BBCodeParams) -> Tuple[np.ndarray, np.ndarray]:
    A = _poly_matrix(params.l, params.m, params.a_exps)
    B = _poly_matrix(params.l, params.m, params.b_exps)
    return (np.hstack([A, B]).astype(np.uint8),
            np.hstack([B.T % 2, A.T % 2]).astype(np.uint8))


def derive_edges(params: BBCodeParams, vertices: Sequence[Vertex],
                 H_Z: np.ndarray) -> List[Tuple[Vertex, Vertex, int]]:
    """All (gamma, delta, z_check) with both qubits in the check's support — fixture order:
    vertex-list pair order (a < b), then ascending check index."""
    out = []
    for a in range(len(vertices)):
        for b in range(a + 1, len(vertices)):
            qa = qubit_index(params, vertices[a])
            qb = qubit_index(params, vertices[b])
            for zc in np.where((H_Z[:, qa] == 1) & (H_Z[:, qb] == 1))[0]:
                out.append((vertices[a], vertices[b], int(zc)))
    return out


def derive_identified(params: BBCodeParams, p: Poly, q: Poly, r: Poly, s: Poly,
                      mu: Tuple[int, int]) -> Tuple[Vertex, Vertex]:
    """The Bell-pair vertex: the UNIQUE qubit where X1 = X(p,q) and Z1 = Z(mu s^T, mu r^T)
    overlap (basis property 3). Returns (vertex in V_l, vertex in V_r)."""
    x1 = set(support(params, p, q))
    z1 = set(support(params, poly_mul_mono(params, poly_T(params, s), mu),
                     poly_mul_mono(params, poly_T(params, r), mu)))
    overlap = x1 & z1
    if len(overlap) != 1:
        raise LPUDerivationError(
            f"X1 and Z1 overlap on {len(overlap)} qubits (need exactly 1, basis property 3): "
            f"{sorted(overlap)}")
    id_l = overlap.pop()
    # the G_r vertex delta with mu*delta^T = id_l  <=>  delta = (mu^-1 * ... ) — invert directly:
    # mu*delta^T = id_l  =>  delta^T = mu^-1 id_l  =>  delta = (mu^-1 id_l)^T
    side, i, j = id_l
    inv = ((i - mu[0]) % params.l, (j - mu[1]) % params.m)
    id_r = _transpose(params, (side, inv[0], inv[1]))
    v_r = set(support(params, r, s))
    if id_r not in v_r:
        raise LPUDerivationError(
            f"dual of the overlap qubit {id_l} -> {id_r} is not a vertex of G_r "
            "(basis does not satisfy the identification rule)")
    return id_l, id_r


def _adjacency(detail: Sequence[Tuple[Vertex, Vertex, int]]) -> Dict[Vertex, Set[Vertex]]:
    adj: Dict[Vertex, Set[Vertex]] = {}
    for a, b, _ in detail:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def derive_bridges(V: Sequence[Vertex], detail: Sequence[Tuple[Vertex, Vertex, int]],
                   identified: Vertex, start_neighbors_of_id: bool = True
                   ) -> List[List[Vertex]]:
    """ALL Hamiltonian paths through V \\ {identified}, deterministically ordered.

    The tour rule: the path's first vertex must be adjacent to the identified vertex (so the
    closing triangle through id + bridge 0 exists — the fixture's convention). Returns paths
    in lexicographic visiting order; caller pairs one from G_l with one from G_r.
    """
    adj = _adjacency(detail)
    rest = [v for v in V if v != identified]
    starts = sorted(adj.get(identified, set()) & set(rest)) if start_neighbors_of_id else sorted(rest)
    paths: List[List[Vertex]] = []

    def extend(path: List[Vertex], remaining: Set[Vertex]) -> None:
        if not remaining:
            paths.append(list(path))
            return
        for nxt in sorted(adj.get(path[-1], set()) & remaining):
            path.append(nxt)
            remaining.discard(nxt)
            extend(path, remaining)
            remaining.add(nxt)
            path.pop()

    for s0 in starts:
        extend([s0], set(rest) - {s0})
    return paths


def derive_U_b(layout: LPULayout) -> List[List[Vertex]]:
    """The bridge cycle set: the id-triangle (id, top[0], bottom[0]) + one square per
    consecutive rung pair — exactly the fixture's _build_U_B rule."""
    top, bot = layout.bridge_top, layout.bridge_bottom
    tri = [layout.identified_l, top[0], bot[0], layout.identified_l]
    squares = [[top[k], top[k + 1], bot[k + 1], bot[k], top[k]] for k in range(len(top) - 1)]
    return _fixture_ub_order(tri, squares)


def _fixture_ub_order(tri: List[Vertex], squares: List[List[Vertex]]) -> List[List[Vertex]]:
    """Fixture order: the 10 squares first, then the identified-vertex triangle last."""
    return squares + [tri]


def fundamental_cycles(V: Sequence[Vertex], detail: Sequence[Tuple[Vertex, Vertex, int]]
                       ) -> List[List[Vertex]]:
    """Spanning-tree fundamental cycle basis (closed vertex walks), deterministic.

    Used for DERIVED layouts of new codes: a full basis (rank |E|-|V|+components) is always
    a valid cycle-check set — merely larger than the paper's redundancy-eliminated choice.
    """
    adj = _adjacency(detail)
    edges: Set[FrozenSet[Vertex]] = {frozenset((a, b)) for a, b, _ in detail}
    parent: Dict[Vertex, Optional[Vertex]] = {}
    order: List[Vertex] = []
    for root in sorted(adj):
        if root in parent:
            continue
        parent[root] = None
        stack = [root]
        while stack:
            v = stack.pop()
            order.append(v)
            for w in sorted(adj[v], reverse=True):
                if w not in parent:
                    parent[w] = v
                    stack.append(w)
    tree: Set[FrozenSet[Vertex]] = {frozenset((v, p)) for v, p in parent.items() if p is not None}

    def path_to_root(v: Vertex) -> List[Vertex]:
        out = [v]
        while parent[out[-1]] is not None:
            out.append(parent[out[-1]])
        return out

    cycles = []
    for e in sorted(edges - tree, key=lambda fs: sorted(fs)):
        a, b = sorted(e)
        pa, pb = path_to_root(a), path_to_root(b)
        sa, sb = set(pa), set(pb)
        meet = next(v for v in pa if v in sb)
        walk = pa[:pa.index(meet) + 1] + list(reversed(pb[:pb.index(meet)])) + [a]
        cycles.append(walk)
    return cycles


# ------------------------------ certification ------------------------------
def certify_layout(layout: LPULayout, expect_expander_edges: int = 0) -> Dict[str, int]:
    """Check a layout against every structural rule of App. A.3; raise LPUDerivationError
    on any violation. Returns summary counts. This is what pins the hand-transcribed gross
    fixture to the paper's rules (test_lpu_graph.py) and gates any derived layout."""
    P = layout.params
    H_X, H_Z = build_HX_HZ(P)

    # vertices = logical supports
    if sorted(layout.V_l) != sorted(support(P, layout.p, layout.q)):
        raise LPUDerivationError("V_l != supp(X1)")
    if sorted(layout.V_r) != sorted(support(P, layout.r, layout.s)):
        raise LPUDerivationError("V_r != supp(X7)")

    # identified pair from the unique-overlap rule
    id_l, id_r = derive_identified(P, layout.p, layout.q, layout.r, layout.s, layout.mu)
    if (id_l, id_r) != (layout.identified_l, layout.identified_r):
        raise LPUDerivationError(
            f"identified pair {layout.identified_l}/{layout.identified_r} != derived {id_l}/{id_r}")

    # edges = shared-Z-check pairs (allowing declared expander extras)
    for name, V, detail in (("E_l", layout.V_l, layout.E_l_detail),
                            ("E_r", layout.V_r, layout.E_r_detail)):
        derived = derive_edges(P, list(V), H_Z)
        extra = len(detail) - len(derived)
        if list(detail) != list(derived) and extra != expect_expander_edges:
            if sorted(detail) != sorted(derived):
                raise LPUDerivationError(f"{name} != derived shared-Z-check edge set")
            raise LPUDerivationError(f"{name} ordering differs from the fixture pair-scan order")

    # bridges: perfect matching of the two vertex sets minus id, Hamiltonian in each graph,
    # rung 0 closing the id triangle
    top, bot = layout.bridge_top, layout.bridge_bottom
    if sorted(top) != sorted(v for v in layout.V_l if v != layout.identified_l):
        raise LPUDerivationError("bridge_top != V_l \\ {id}")
    if sorted(bot) != sorted(v for v in layout.V_r if v != layout.identified_r):
        raise LPUDerivationError("bridge_bottom != V_r \\ {id}")
    adj_l, adj_r = _adjacency(layout.E_l_detail), _adjacency(layout.E_r_detail)
    for k in range(len(top) - 1):
        if top[k + 1] not in adj_l.get(top[k], set()):
            raise LPUDerivationError(f"bridge_top not a G_l path at rung {k}")
        if bot[k + 1] not in adj_r.get(bot[k], set()):
            raise LPUDerivationError(f"bridge_bottom not a G_r path at rung {k}")
    if top[0] not in adj_l.get(layout.identified_l, set()):
        raise LPUDerivationError("id not adjacent to bridge_top[0] (no closing triangle)")
    if bot[0] not in adj_r.get(layout.identified_r, set()):
        raise LPUDerivationError("id not adjacent to bridge_bottom[0] (no closing triangle)")

    # cycles traverse real edges; U_b matches the ladder rule. Edge membership is checked
    # under canonicalization (identified_r -> identified_l): the closing triangle and any
    # U_r cycle may name the Bell vertex by either of its two labels.
    canon = {layout.identified_r: layout.identified_l}

    def c(v: Vertex) -> Vertex:
        return canon.get(v, v)

    edge_set: Set[FrozenSet[Vertex]] = set()
    for x, y, _ in list(layout.E_l_detail) + list(layout.E_r_detail):
        edge_set.add(frozenset((c(x), c(y))))
    for t, bo in zip(layout.bridge_top, layout.bridge_bottom):
        edge_set.add(frozenset((c(t), c(bo))))

    for cyc in layout.cycles:
        if cyc[0] != cyc[-1]:
            raise LPUDerivationError(f"cycle not closed: {cyc}")
        for a, b in zip(cyc, cyc[1:]):
            if c(a) == c(b):
                continue
            if frozenset((c(a), c(b))) not in edge_set:
                raise LPUDerivationError(f"cycle step {a}->{b} is not an edge")
    if layout.U_b != _fixture_ub_order(
            [layout.identified_l, top[0], bot[0], layout.identified_l],
            [[top[k], top[k + 1], bot[k + 1], bot[k], top[k]] for k in range(len(top) - 1)]):
        raise LPUDerivationError("U_b != ladder rule (10 rung squares + id triangle)")

    return {"n_vertices": layout.n_vertices, "n_edges": layout.n_edges,
            "n_cycles": len(layout.cycles)}
