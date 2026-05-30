# DARWIN HAMMER — match 1171, survivor 3
# gen: 3
# parent_a: geometric_product.py (gen0)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# born: 2026-05-29T23:33:12Z

"""Hybrid Geometric–Statistical Algebra (HGSA)

This module fuses:

* **geometric_product.py** – Clifford geometric product over the Euclidean
  algebra Cl(n,0).  Multivectors are built from basis blades; the geometric
  product unifies inner and outer products.

* **hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py** – Gaussian‑based
  Fisher information scoring, minimum‑cost tree evaluation and Bayesian
  marginal updates for graph edges.

**Mathematical bridge**

Both frameworks treat scalar quantities that can be combined linearly.
We embed the scalar Fisher information and Bayesian marginal values as
coefficients of basis blades inside a multivector.  The geometric product
then provides a single algebraic operation that simultaneously:

* propagates directional information of graph edges (grade‑1 blades),
* accumulates uncertainty (scalar coefficients) via inner products,
* respects anti‑commutativity, thus preserving the orientation of edges.

The three core hybrid functions below demonstrate this unified
treatment.  They construct multivectors from graph geometry, weight them
with Gaussian‑derived Fisher scores, and finally update the algebra with
Bayesian marginals.  The resulting scalar (grade‑0) part of the geometric
product yields a single “hybrid cost” that blends deterministic tree cost,
information content and probabilistic belief."""

from __future__ import annotations
import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, Dict, List, Set, FrozenSet

# ---------------------------------------------------------------------------
# Clifford algebra helpers (from geometric_product.py)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort indices by bubble‑sort, tracking sign changes.
    Duplicate indices cancel because e_i*e_i = 1.
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
                # cancel the pair
                del lst[j:j + 2]
                n -= 2
                sign *= 1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int],
                    blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ---------------------------------------------------------------------------
# Multivector class (core of geometric_product.py)
# ---------------------------------------------------------------------------

class Multivector:
    """Clifford multivector stored as a dict {blade_frozenset: coefficient}."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # drop zeroes
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Dimension mismatch in addition")
        comp = self.components.copy()
        for b, v in other.components.items():
            comp[b] = comp.get(b, 0.0) + v
            if abs(comp[b]) < 1e-15:
                del comp[b]
        return Multivector(comp, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -v for b, v in self.components.items()}, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        if self.n != other.n:
            raise ValueError("Dimension mismatch in multiplication")
        result: Dict[FrozenSet[int], float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                bc, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[bc] = result.get(bc, 0.0) + coeff
        return Multivector(result, self.n)

    def inner(self, other: "Multivector") -> float:
        """Scalar (grade‑0) part of the geometric product."""
        prod = self * other
        return prod.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return f"Multivector(0, n={self.n})"
        terms = []
        for blade, coeff in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                indices = "∧".join(f"e{i}" for i in sorted(blade))
                term = f"{coeff:.3g}{indices}"
            terms.append(term)
        return " + ".join(terms)


# ---------------------------------------------------------------------------
# Statistical / graph utilities (from hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py)
# ---------------------------------------------------------------------------

Point = Tuple[float, float]
Edge = Tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def edge_angle(a: Point, b: Point) -> float:
    """Angle of the directed edge a→b in radians."""
    return math.atan2(b[1] - a[1], b[0] - a[0])

def tree_cost(nodes: Dict[str, Point],
              edges: List[Edge],
              root: str,
              path_weight: float = 0.2) -> float:
    """Deterministic material + path cost of a tree."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + length(nodes[cur], nodes[nb])
                stack.append(nb)
    return material + path_weight * sum(dist.values())

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior probability using a simple Bayes rule with false‑positive noise."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * false_positive
    return numerator / denominator if denominator != 0 else 0.0


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def construct_edge_multivector(nodes: Dict[str, Point],
                               edges: List[Edge],
                               n: int) -> Multivector:
    """
    Build a multivector where each edge contributes a grade‑1 blade
    e_i (i = unique edge index) with coefficient equal to the Euclidean
    length of the edge.
    """
    comps: Dict[FrozenSet[int], float] = {}
    for idx, (a, b) in enumerate(edges):
        blade = frozenset({idx})          # grade‑1 blade for this edge
        coeff = length(nodes[a], nodes[b])  # deterministic weight
        comps[blade] = coeff
    return Multivector(comps, n)


def fisher_information_multivector(nodes: Dict[str, Point],
                                   edges: List[Edge],
                                   center: float = 0.0,
                                   width: float = math.pi/4) -> Multivector:
    """
    Encode Fisher information of each edge (computed from its orientation)
    as a scalar attached to the same blade used in ``construct_edge_multivector``.
    """
    comps: Dict[FrozenSet[int], float] = {}
    for idx, (a, b) in enumerate(edges):
        theta = edge_angle(nodes[a], nodes[b])
        info = fisher_score(theta, center, width)
        blade = frozenset({idx})
        comps[blade] = info
    # The algebra dimension must be at least the number of edges.
    return Multivector(comps, n=len(edges))


def hybrid_tree_cost(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     center: float = 0.0,
                     width: float = math.pi/4,
                     false_positive: float = 0.05) -> float:
    """
    Unified cost = deterministic tree cost
                 + inner product of geometric‑product of edge length
                   and Fisher‑information multivectors
                 + Bayesian marginal contribution (scalar part).

    Returns a single scalar that blends geometry, information and probability.
    """
    # 1. Deterministic material/path cost
    deterministic = tree_cost(nodes, edges, root)

    # 2. Geometric product of length‑multivector and Fisher‑info multivector
    length_mv = construct_edge_multivector(nodes, edges, n=len(edges))
    fisher_mv = fisher_information_multivector(nodes, edges, center, width)
    info_coupling = length_mv.inner(fisher_mv)   # scalar grade‑0 part

    # 3. Bayesian marginals per edge, summed as a scalar
    bayes_sum = 0.0
    for (a, b) in edges:
        prior = 0.5                     # placeholder prior
        likelihood = gaussian_beam(edge_angle(nodes[a], nodes[b]),
                                   center, width)
        bayes = bayes_marginal(prior, likelihood, false_positive)
        bayes_sum += bayes

    # Weighted combination (weights chosen heuristically)
    hybrid = deterministic + 0.7 * info_coupling + 0.3 * bayes_sum
    return hybrid


def update_multivector_with_bayes(mv: Multivector,
                                  nodes: Dict[str, Point],
                                  edges: List[Edge],
                                  prior: float = 0.5,
                                  false_positive: float = 0.05,
                                  center: float = 0.0,
                                  width: float = math.pi/4) -> Multivector:
    """
    Produce a new multivector where each blade coefficient is multiplied by
    the Bayesian marginal of the corresponding edge.  This demonstrates how
    probabilistic belief can modulate geometric quantities inside the same
    algebraic object.
    """
    new_comps: Dict[FrozenSet[int], float] = {}
    for idx, (a, b) in enumerate(edges):
        blade = frozenset({idx})
        coeff = mv.components.get(blade, 0.0)
        theta = edge_angle(nodes[a], nodes[b])
        likelihood = gaussian_beam(theta, center, width)
        marginal = bayes_marginal(prior, likelihood, false_positive)
        new_comps[blade] = coeff * marginal
    return Multivector(new_comps, mv.n)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple triangle graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2)
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    print("Deterministic tree cost:", tree_cost(nodes, edges, root))
    print("Hybrid cost:", hybrid_tree_cost(nodes, edges, root))

    # Demonstrate multivector update
    mv_len = construct_edge_multivector(nodes, edges, n=len(edges))
    print("Length multivector:", mv_len)

    mv_updated = update_multivector_with_bayes(mv_len, nodes, edges)
    print("Bayes‑updated multivector:", mv_updated)

    # Verify inner product equals information coupling
    fisher_mv = fisher_information_multivector(nodes, edges)
    coupling = mv_len.inner(fisher_mv)
    print("Info coupling (inner product):", coupling)