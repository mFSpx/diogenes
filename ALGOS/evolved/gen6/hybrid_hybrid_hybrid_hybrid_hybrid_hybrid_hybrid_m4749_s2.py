# DARWIN HAMMER — match 4749, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_s0.py (gen5)
# born: 2026-05-29T23:57:53Z

"""Hybrid Fusion of Multivector Geometry and Bayesian Edge Weights
Parent A: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s2.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2146_s0.py

Mathematical Bridge
-------------------
* The multivector components (grade‑k blades) from Parent A are treated as
  *features* whose importance is modulated by a Bayesian update derived from
  the probabilistic edge model of Parent B.
* For each blade we compute a **Bayesian marginal weight**  

  \\[
  w = \\frac{\\ell \\cdot p}{\\ell \\cdot p + f \\cdot (1-p)}
  \\]

  where `p` is a prior confidence for the blade, `ℓ` a likelihood proportional
  to the absolute coefficient magnitude, and `f` a false‑positive rate taken
  from the edge‑specific Bayes marginal of Parent B.
* The weighted blades are then summed using a similarity matrix that mimics
  the Structural Similarity Index Measure (SSIM) used in Parent A.  The SSIM
  matrix is built from Euclidean distances of the graph nodes (Parent B) and
  provides a *geometric‑probabilistic* decision metric.
* The resulting system lives in the Clifford algebra \\(Cl(n,0)\\) while its
  coefficients encode Bayesian‑adjusted certainty, thus unifying the two
  topologies into a single hybrid algorithm.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict, Any

# ----------------------------------------------------------------------
# Parent A – Multivector core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                del lst[j:j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        # prune near‑zero entries
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        new_comp = self.components.copy()
        for b, c in other.components.items():
            new_comp[b] = new_comp.get(b, 0.0) + c
        return Multivector(new_comp, self.n)

    def __mul__(self, other: Any) -> "Multivector":
        """Geometric product with another multivector or scalar."""
        if isinstance(other, (int, float)):
            return Multivector({b: c * other for b, c in self.components.items()}, self.n)
        if not isinstance(other, Multivector):
            raise TypeError("Unsupported operand type(s) for *")
        result: Dict[frozenset[int], float] = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                b_res, sign = _multiply_blades(b1, b2)
                result[b_res] = result.get(b_res, 0.0) + sign * c1 * c2
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        terms = [f"{c:.3g}*e{sorted(list(b))}" if b else f"{c:.3g}" for b, c in self.components.items()]
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Parent B – Bayesian graph core
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


class HybridAlgorithm:
    """Graph‑based structure carrying Bayesian edge restrictions and node sections."""

    def __init__(self, node_dims: Dict[str, int], edge_list: List[Edge], width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)          # dimensionality per node (unused in fusion)
        self.edges = list(edge_list)
        self.width = width
        self.depth = depth
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[str, np.ndarray] = {}

    def set_restriction(self, edge: Edge, src_map: List[float], dst_map: List[float]) -> None:
        """Associate a pair of linear maps with an edge."""
        u, v = edge
        src = np.array(src_map, dtype=float)
        dst = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src, dst)

    def set_section(self, node: str, value: List[float]) -> None:
        """Assign a vector (section) to a graph node."""
        self._sections[node] = np.array(value, dtype=float)

    @staticmethod
    def length(a: Point, b: Point) -> float:
        """Euclidean distance between two points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
        """
        Compute the Bayesian marginal (posterior) probability.

        posterior = (likelihood * prior) /
                    (likelihood * prior + false_positive * (1 - prior))
        """
        if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
            raise ValueError("All probabilities must be in [0, 1]")
        denominator = likelihood * prior + false_positive * (1.0 - prior)
        if denominator == 0.0:
            return 0.0
        return (likelihood * prior) / denominator

    def edge_false_positive(self, edge: Edge) -> float:
        """Derive a synthetic false‑positive rate from the Euclidean length of the edge."""
        # For demonstration we map length ∈ [0, ∞) to a probability via a sigmoid.
        u, v = edge
        # Dummy coordinates (could be extended to real geometry)
        coord = {
            u: (random.random(), random.random()),
            v: (random.random(), random.random()),
        }
        d = self.length(coord[u], coord[v])
        return 1.0 / (1.0 + math.exp(- (d - 0.5) * 5))  # sigmoid centered at 0.5


# ----------------------------------------------------------------------
# Hybrid Functions – Fusion of the Two Topologies
# ----------------------------------------------------------------------
def bayesian_weighted_multivector(
    mv: Multivector,
    prior_map: Dict[frozenset[int], float],
    hybrid: HybridAlgorithm,
    edge: Edge,
) -> Multivector:
    """
    Apply a Bayesian marginal weight to each blade of ``mv``.
    * ``prior_map`` supplies a prior confidence for each blade (default 0.5).
    * ``likelihood`` is derived from the normalized absolute coefficient magnitude.
    * ``false_positive`` is obtained from the edge geometry via ``HybridAlgorithm.edge_false_positive``.
    Returns a new ``Multivector`` whose coefficients are Bayesian‑adjusted.
    """
    # Normalise magnitudes to obtain a likelihood in [0,1]
    mags = np.array([abs(c) for c in mv.components.values()], dtype=float)
    if mags.size == 0:
        return Multivector({}, mv.n)
    max_mag = mags.max()
    if max_mag == 0:
        likelihoods = np.zeros_like(mags)
    else:
        likelihoods = mags / max_mag

    false_pos = hybrid.edge_false_positive(edge)

    new_components: Dict[frozenset[int], float] = {}
    for (blade, coeff), like in zip(mv.components.items(), likelihoods):
        prior = prior_map.get(blade, 0.5)
        weight = HybridAlgorithm.bayes_marginal(prior, like, false_pos)
        new_components[blade] = coeff * weight
    return Multivector(new_components, mv.n)


def ssim_like_decision_metric(
    mv: Multivector,
    node_positions: Dict[str, Point],
    edge: Edge,
) -> float:
    """
    Compute a decision metric analogous to SSIM by weighting blade coefficients
    with a similarity matrix derived from node positions.
    The similarity between two blades is taken as the Gaussian of the Euclidean
    distance between the edge's endpoint coordinates.
    """
    u, v = edge
    if u not in node_positions or v not in node_positions:
        raise KeyError("Edge endpoints must exist in ``node_positions``")
    d = HybridAlgorithm.length(node_positions[u], node_positions[v])
    similarity = math.exp(- (d ** 2) / (2.0 * (0.3 ** 2)))  # sigma = 0.3

    # Weighted sum of absolute coefficients, scaled by similarity.
    total = sum(abs(c) for c in mv.components.values())
    return similarity * total


def update_hybrid_with_multivector(
    hybrid: HybridAlgorithm,
    mv: Multivector,
    node: str,
) -> None:
    """
    Encode the multivector's scalar part into the section of ``node`` and
    propagate a linear map derived from higher‑grade blades into the edge
    restrictions incident to ``node``.
    """
    # Update the node's section with the scalar part replicated to match its dimension.
    dim = hybrid.node_dims.get(node, 1)
    scalar = mv.scalar_part()
    hybrid.set_section(node, [scalar] * dim)

    # For each incident edge, create a simple restriction map where the
    # source map is the vector of grade‑1 coefficients and the destination map
    # is the vector of grade‑2 coefficients (padded/truncated to ``dim``).
    incident_edges = [e for e in hybrid.edges if node in e]
    grade1 = mv.grade(1).components
    grade2 = mv.grade(2).components

    src_vec = np.array([grade1.get(frozenset([i]), 0.0) for i in range(dim)], dtype=float)
    dst_vec = np.array([grade2.get(frozenset([i, (i + 1) % dim]), 0.0) for i in range(dim)], dtype=float)

    for e in incident_edges:
        if e[0] == node:
            hybrid.set_restriction(e, src_vec.tolist(), dst_vec.tolist())
        else:
            # reverse orientation – swap src/dst
            hybrid.set_restriction(e, dst_vec.tolist(), src_vec.tolist())


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a simple multivector in Cl(3,0)
    comps = {
        frozenset(): 1.0,                     # scalar
        frozenset({0}): 0.6,                  # e1
        frozenset({1}): -0.4,                 # e2
        frozenset({2}): 0.2,                  # e3
        frozenset({0, 1}): 0.1,               # e12
        frozenset({1, 2}): -0.05,             # e23
    }
    mv = Multivector(comps, n=3)

    # Graph definition
    nodes = {"A": 3, "B": 3}
    edges = [("A", "B")]
    hybrid = HybridAlgorithm(node_dims=nodes, edge_list=edges)

    # Prior confidences per blade (arbitrary)
    prior_map = {blade: random.uniform(0.3, 0.9) for blade in comps.keys()}

    # Apply Bayesian weighting
    weighted_mv = bayesian_weighted_multivector(mv, prior_map, hybrid, ("A", "B"))
    print("Weighted Multivector:", weighted_mv)

    # Node positions for SSIM‑like metric
    positions = {"A": (0.0, 0.0), "B": (0.4, 0.3)}
    metric = ssim_like_decision_metric(weighted_mv, positions, ("A", "B"))
    print("Decision metric (SSIM‑like):", metric)

    # Update hybrid structure with the weighted multivector
    update_hybrid_with_multivector(hybrid, weighted_mv, "A")
    print("Section A:", hybrid._sections.get("A"))
    print("Restriction A→B src map:", hybrid._restrictions.get(("A", "B"))[0])
    print("Restriction A→B dst map:", hybrid._restrictions.get(("A", "B"))[1])
    # End of smoke test.