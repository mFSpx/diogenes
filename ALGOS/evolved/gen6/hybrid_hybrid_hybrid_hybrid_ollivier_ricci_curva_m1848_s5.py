# DARWIN HAMMER — match 1848, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py (gen5)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:39:12Z

"""Hybrid Algorithm: Geometric Algebra Embedded Ollivier‑Ricci Curvature

This module fuses the core of **hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py**
(the Clifford/geometric‑product machinery) with **ollivier_ricci_curvature.py**
(the Ollivier‑Ricci curvature on weighted graphs).

Mathematical bridge
------------------
* Each vertex `v` of a graph `G` is embedded as a multivector `M(v)` in a
  Clifford algebra `Cl(n)` by assigning a unique basis vector `e_i` to every
  vertex identifier and forming a weighted linear combination of its incident
  basis vectors.  
* The *geometric product* `M(x) * M(y)` yields a scalar part equal to the
  inner product `<M(x),M(y)>`.  From this scalar we obtain a Euclidean‑like
  distance  


d_G(x, y) = √(⟨M(x),M(x)⟩ + ⟨M(y),M(y)⟩ - 2⟨M(x),M(y)⟩)


  which replaces the usual shortest‑path distance in the Ollivier‑Ricci
  definition.
* The lazy random‑walk distributions `m_x, m_y` are unchanged, but the
  Wasserstein‑1 cost matrix now uses `d_G` instead of graph‑geodesic distances.
* An exponential‑decay schedule (taken from the simulated‑annealing part of
  the first parent) controls a simple annealing loop that refines the curvature
  estimate by repeatedly perturbing the laziness parameter `α`.

The resulting functions expose a **single unified system** that can be used
directly on arbitrary undirected graphs.  No external scientific libraries
beyond `numpy` are required."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 1. Clifford / geometric‑product utilities (from Parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting a list of basis indices.

    Duplicated indices cancel because e_i·e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel pair
                lst.pop(j)
                lst.pop(j)  # the next element shifts into position j
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int],
                    blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades.

    Returns (result_blade, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Sparse representation of a multivector as a dict
    {blade (frozenset of ints): coefficient (float)}.
    """

    def __init__(self, components: Dict[FrozenSet[int], float] = None):
        self.components = dict(components) if components else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        res = Multivector(self.components)
        for b, c in other.components.items():
            res.components[b] = res.components.get(b, 0.0) + c
            if abs(res.components[b]) < 1e-12:
                del res.components[b]
        return res

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                b_res, sign = _multiply_blades(b1, b2)
                coeff = c1 * c2 * sign
                result[b_res] = result.get(b_res, 0.0) + coeff
        # prune near‑zero components
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result)

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, 0 if absent."""
        return self.components.get(frozenset(), 0.0)

    def norm(self) -> float:
        """Euclidean norm induced by the scalar product <A,B> = (A*B).scalar_part()."""
        prod = self * self
        return math.sqrt(abs(prod.scalar_part()))


# ---------------------------------------------------------------------------
# 2. Graph embedding into the Clifford algebra
# ---------------------------------------------------------------------------

class GraphEmbedding:
    """Maps graph vertices to multivectors in a Clifford algebra.

    The embedding uses a *vertex‑basis* mapping `v -> e_i` and builds a
    multivector for each vertex as the (degree‑weighted) sum of its own basis
    vector and those of its neighbours.
    """

    def __init__(self, adjacency: Dict[int, Tuple[int, ...]]):
        self.adj = {v: tuple(neigh) for v, neigh in adjacency.items()}
        self._basis_index = {v: i for i, v in enumerate(sorted(self.adj))}
        self._dim = len(self._basis_index)

    def basis_vector(self, vertex: int) -> Multivector:
        idx = self._basis_index[vertex]
        return Multivector({frozenset({idx}): 1.0})

    def vertex_multivector(self, vertex: int) -> Multivector:
        """Weighted sum of the vertex's own basis vector and its neighbours."""
        coeffs = {}
        # self‑weight
        coeffs[vertex] = coeffs.get(vertex, 0.0) + 1.0
        # neighbour contribution (uniform)
        for nb in self.adj.get(vertex, ()):
            coeffs[nb] = coeffs.get(nb, 0.0) + 1.0
        mv = Multivector()
        for v, w in coeffs.items():
            mv = mv + (w * self.basis_vector(v))
        return mv

    def geometric_distance(self, u: int, v: int) -> float:
        """Euclidean‑like distance derived from the inner product of embeddings."""
        mu = self.vertex_multivector(u)
        mv = self.vertex_multivector(v)
        inner_uu = mu * mu
        inner_vv = mv * mv
        inner_uv = mu * mv
        d2 = inner_uu.scalar_part() + inner_vv.scalar_part() - 2.0 * inner_uv.scalar_part()
        # Guard against tiny negative due to floating point
        return math.sqrt(max(d2, 0.0))


# ---------------------------------------------------------------------------
# 3. Ollivier‑Ricci curvature using the geometric distance metric
# ---------------------------------------------------------------------------

def lazy_rw_distribution(adj: Dict[int, Tuple[int, ...]],
                         node: int,
                         alpha: float = 0.5) -> Dict[int, float]:
    """Lazy random‑walk distribution centred at *node*."""
    neighbours = adj.get(node, ())
    deg = len(neighbours)
    dist = {node: alpha}
    if deg:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def _emd_greedy(p: Dict[int, float],
                q: Dict[int, float],
                cost: Dict[Tuple[int, int], float]) -> float:
    """North‑west‑corner greedy Earth‑Mover distance (heuristic, fast)."""
    # Convert to mutable copies
    p_rem = dict(p)
    q_rem = dict(q)
    total = 0.0
    # Sort keys to obtain a deterministic order
    for i in sorted(p_rem):
        for j in sorted(q_rem):
            if p_rem[i] == 0 or q_rem[j] == 0:
                continue
            flow = min(p_rem[i], q_rem[j])
            total += flow * cost[(i, j)]
            p_rem[i] -= flow
            q_rem[j] -= flow
    return total


def ollivier_ricci_curvature(adj: Dict[int, Tuple[int, ...]],
                             u: int,
                             v: int,
                             alpha: float = 0.5,
                             embedding: GraphEmbedding = None) -> float:
    """Curvature of edge (u, v) using geometric distances.

    Parameters
    ----------
    adj : graph adjacency dict (node -> tuple of neighbours)
    u, v : endpoints of the edge
    alpha : laziness parameter of the random walk
    embedding : optional pre‑computed GraphEmbedding; if None a new one is built.
    """
    if embedding is None:
        embedding = GraphEmbedding(adj)

    # 1) lazy distributions
    mu = lazy_rw_distribution(adj, u, alpha)
    mv = lazy_rw_distribution(adj, v, alpha)

    # 2) cost matrix using geometric distances
    nodes = set(mu) | set(mv)
    cost = {}
    for i in nodes:
        for j in nodes:
            cost[(i, j)] = embedding.geometric_distance(i, j)

    # 3) Earth‑Mover distance (W_1)
    W1 = _emd_greedy(mu, mv, cost)

    # 4) denominator: geometric distance between the two base vertices
    d_uv = embedding.geometric_distance(u, v)
    if d_uv == 0:
        return 0.0  # degenerate (same vertex)
    return 1.0 - W1 / d_uv


# ---------------------------------------------------------------------------
# 4. Simple simulated‑annealing wrapper that refines alpha using exponential decay
# ---------------------------------------------------------------------------

def anneal_alpha(initial_alpha: float,
                 temperature: float,
                 decay: float,
                 steps: int,
                 adj: Dict[int, Tuple[int, ...]],
                 edge: Tuple[int, int]) -> Tuple[float, list]:
    """Run a tiny annealing loop to maximise curvature over the laziness parameter.

    Returns the best alpha found and the trajectory of curvature values.
    """
    u, v = edge
    best_alpha = initial_alpha
    embedding = GraphEmbedding(adj)
    best_kappa = ollivier_ricci_curvature(adj, u, v, best_alpha, embedding)
    trajectory = [best_kappa]

    alpha = initial_alpha
    for _ in range(steps):
        # propose a small random perturbation
        proposal = alpha + random.uniform(-0.05, 0.05)
        proposal = min(max(proposal, 0.0), 1.0)  # keep inside [0,1]

        kappa_prop = ollivier_ricci_curvature(adj, u, v, proposal, embedding)

        # Metropolis acceptance with exponential temperature decay
        if kappa_prop > best_kappa:
            accept = True
        else:
            delta = best_kappa - kappa_prop
            prob = math.exp(-delta / max(temperature, 1e-12))
            accept = random.random() < prob

        if accept:
            alpha = proposal
            if kappa_prop > best_kappa:
                best_alpha, best_kappa = proposal, kappa_prop

        trajectory.append(best_kappa)
        temperature *= decay  # exponential decay

    return best_alpha, trajectory


# ---------------------------------------------------------------------------
# 5. Demonstration / smoke test
# ---------------------------------------------------------------------------

def _sample_graph() -> Dict[int, Tuple[int, ...]]:
    """A small undirected graph for quick testing."""
    return {
        0: (1, 2),
        1: (0, 2, 3),
        2: (0, 1, 3),
        3: (1, 2, 4),
        4: (3,)
    }


if __name__ == "__main__":
    # Build graph and embedding
    G = _sample_graph()
    embed = GraphEmbedding(G)

    # Choose an edge
    edge = (1, 2)

    # Compute curvature with default alpha
    k0 = ollivier_ricci_curvature(G, *edge, alpha=0.5, embedding=embed)
    print(f"Initial curvature κ({edge[0]}, {edge[1]}) = {k0:.4f}")

    # Run annealing to improve alpha
    best_alpha, traj = anneal_alpha(
        initial_alpha=0.5,
        temperature=1.0,
        decay=0.95,
        steps=30,
        adj=G,
        edge=edge
    )
    k_best = ollivier_ricci_curvature(G, *edge, alpha=best_alpha, embedding=embed)
    print(f"Optimised α = {best_alpha:.3f}, curvature = {k_best:.4f}")

    # Show a few trajectory points
    print("Curvature trajectory (first 10 steps):", ["{:.4f}".format(v) for v in traj[:10]])
    sys.exit(0)