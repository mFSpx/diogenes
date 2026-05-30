# DARWIN HAMMER — match 4141, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s4.py (gen6)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen4)
# born: 2026-05-29T23:53:52Z

"""Hybrid Algorithm integrating Metropolis‑style Bayesian edge updates (Parent A)
with geometric‑algebra blade operations (Parent B).

Mathematical bridge:
- Each graph edge carries a *beta prior* (α,β) describing Bernoulli reliability.
- The posterior mean of that prior is embedded as the coefficient of a
  *grade‑1* basis blade 𝑒_i (i = edge identifier) inside a multivector.
- The geometric (Clifford) product of a sequence of edge‑multivectors yields,
  among other grades, a *scalar* (grade‑0) component.  This scalar is interpreted
  as an “energy change” ΔE for the whole path.
- The scalar ΔE feeds the Metropolis acceptance probability from Parent A,
  while Hoeffding’s bound governs when enough samples have been collected.

Thus the hybrid system fuses Bayesian inference, statistical stopping,
and geometric algebra into a single unified optimisation step.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List, FrozenSet, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (Bayesian + Metropolis + Hoeffding)
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability, never exactly zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    try:
        prob = math.exp(-delta_energy / temperature)
    except OverflowError:
        prob = float('inf')
    return max(min(prob, 1.0), 1e-12)


def hoeffding_decision(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Return True if Hoeffding bound is tighter than *epsilon*."""
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    return bound < epsilon


@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(
    prior: EdgeBetaPrior,
    successes: int,
    failures: int,
) -> Tuple[float, float, EdgeBetaPrior]:
    """Posterior mean, variance and updated prior."""
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    total = new_alpha + new_beta
    posterior_mean = new_alpha / total
    posterior_var = (new_alpha * new_beta) / (total ** 2 * (total + 1))
    return posterior_mean, posterior_var, EdgeBetaPrior(new_alpha, new_beta)


# ----------------------------------------------------------------------
# Parent B components (Geometric Algebra)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Keep only indices with odd multiplicity
    remaining = [i for i, c in counts.items() if c % 2 == 1]

    # Build list preserving original order for odd‑counted indices
    cleaned = []
    temp_counts = counts.copy()
    for i in indices:
        if temp_counts[i] % 2 == 1:
            cleaned.append(i)
            temp_counts[i] = 0

    # Bubble‑sort while tracking sign
    lst = list(cleaned)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Clifford (geometric) product of two basis blades."""
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


def geometric_product(mv_a: Dict[FrozenSet[int], float],
                      mv_b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """
    Full geometric product of two multivectors (limited to scalar and grade‑1 blades).
    The result is a new multivector dictionary mapping basis blades to coefficients.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff
    # Remove near‑zero entries for cleanliness
    return {b: c for b, c in result.items() if abs(c) > 1e-15}


def scalar_part(mv: Dict[FrozenSet[int], float]) -> float:
    """Extract the grade‑0 (scalar) component of a multivector."""
    return mv.get(frozenset(), 0.0)


def vector_to_mv(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Convert a 1‑D array into a multivector containing only grade‑1 blades."""
    mv: Dict[FrozenSet[int], float] = {}
    for i, coeff in enumerate(v):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv


# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------
EdgeId = int
Point = Tuple[float, float]


class HybridGraph:
    """
    Simple undirected graph where each edge stores:
      * a Bayesian beta prior,
      * a multivector whose grade‑1 blade coefficient equals the posterior mean.
    """
    def __init__(self):
        self.edges: Dict[Tuple[EdgeId, EdgeId], EdgeBetaPrior] = {}
        self.mv_cache: Dict[Tuple[EdgeId, EdgeId], Dict[FrozenSet[int], float]] = {}

    def add_edge(self, u: EdgeId, v: EdgeId,
                 prior: Optional[EdgeBetaPrior] = None) -> None:
        key = tuple(sorted((u, v)))
        self.edges[key] = prior if prior is not None else EdgeBetaPrior()
        # Initialise multivector with zero scalar and a blade for this edge
        blade = frozenset({u})  # use first node index as blade identifier
        self.mv_cache[key] = {blade: 0.0}

    def update_edge(self, u: EdgeId, v: EdgeId,
                    successes: int, failures: int) -> None:
        """Bayesian update + embed posterior mean into the edge multivector."""
        key = tuple(sorted((u, v)))
        prior = self.edges[key]
        mean, _, new_prior = bayesian_edge_update(prior, successes, failures)
        self.edges[key] = new_prior
        blade = frozenset({u})
        self.mv_cache[key] = {blade: mean}  # scalar part stays zero

    def edge_mv(self, u: EdgeId, v: EdgeId) -> Dict[FrozenSet[int], float]:
        return self.mv_cache[tuple(sorted((u, v)))]

    def path_energy(self, path: List[Tuple[EdgeId, EdgeId]]) -> float:
        """Geometric product of edge multivectors along *path*; return scalar part."""
        if not path:
            return 0.0
        prod = self.edge_mv(*path[0])
        for edge in path[1:]:
            prod = geometric_product(prod, self.edge_mv(*edge))
        return scalar_part(prod)


def hybrid_acceptance(delta_energy: float, temperature: float) -> bool:
    """Metropolis decision using scalar energy from geometric product."""
    prob = acceptance_probability(delta_energy, temperature)
    return random.random() < prob


def hybrid_assign(points: List[Point],
                  seeds: List[Point],
                  graph: HybridGraph,
                  edge_lookup: Callable[[Point, Point], Tuple[EdgeId, EdgeId]]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest seed, but bias the distance with the
    scalar part of the geometric product of the point vector (as a multivector)
    and the multivector of the edge connecting point and seed.
    """
    assignments: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for pt in points:
        # naive Euclidean nearest seed index
        nearest_idx = min(range(len(seeds)),
                          key=lambda i: math.hypot(pt[0] - seeds[i][0], pt[1] - seeds[i][1]))
        # retrieve edge multivector (if any) between point and that seed
        try:
            edge = edge_lookup(pt, seeds[nearest_idx])
            edge_mv = graph.edge_mv(*edge)
        except Exception:
            edge_mv = {}
        # compute bias = scalar part of geometric product with point vector
        pt_mv = vector_to_mv(np.array(pt))
        bias = scalar_part(geometric_product(pt_mv, edge_mv))
        # modify distance with bias (negative bias makes the seed more attractive)
        biased_dist = math.hypot(pt[0] - seeds[nearest_idx][0],
                                pt[1] - seeds[nearest_idx][1]) - bias
        # if bias flips the order, recompute nearest using biased distance
        # (simple re‑check over all seeds)
        best = nearest_idx
        best_val = biased_dist
        for i, s in enumerate(seeds):
            base = math.hypot(pt[0] - s[0], pt[1] - s[1])
            try:
                e = edge_lookup(pt, s)
                b = scalar_part(geometric_product(pt_mv, graph.edge_mv(*e)))
            except Exception:
                b = 0.0
            val = base - b
            if val < best_val:
                best, best_val = i, val
        assignments[best].append(pt)
    return assignments


def hybrid_hoeffding_stop(sample_counts: List[int],
                          epsilon: float) -> bool:
    """Return True if *all* sample counts satisfy Hoeffding bound."""
    return all(hoeffding_decision(n, epsilon) for n in sample_counts)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny graph with three nodes (0,1,2)
    g = HybridGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)

    # Simulate some observations
    g.update_edge(0, 1, successes=8, failures=2)
    g.update_edge(1, 2, successes=3, failures=7)
    g.update_edge(0, 2, successes=5, failures=5)

    # Define a path and compute its energy
    path = [(0, 1), (1, 2)]
    delta_E = g.path_energy(path)

    # Metropolis acceptance at temperature 1.0
    accepted = hybrid_acceptance(delta_E, temperature=1.0)
    print(f"Path {path} → ΔE={delta_E:.4f}, accepted={accepted}")

    # Simple point assignment demo
    points = [(0.1, 0.2), (0.9, 0.8), (0.5, 0.5)]
    seeds = [(0.0, 0.0), (1.0, 1.0)]

    # Edge lookup function: map a point‑seed pair to the graph edge that shares the seed index
    def lookup(p: Point, s: Point) -> Tuple[EdgeId, EdgeId]:
        # Use seed index as node identifier; points are treated as temporary nodes
        # For the demo we simply connect seed index with a dummy node -1
        seed_idx = seeds.index(s)
        return (seed_idx, -1)  # dummy edge (will raise if not present)

    # Add dummy edges for the lookup to succeed
    g.add_edge(-1, 0)   # dummy edge for seed 0
    g.add_edge(-1, 1)   # dummy edge for seed 1
    # Give them arbitrary posterior means
    g.update_edge(-1, 0, successes=4, failures=1)
    g.update_edge(-1, 1, successes=2, failures=3)

    assignments = hybrid_assign(points, seeds, g, lookup)
    print("Assignments:", assignments)

    # Hoeffding stop test
    sample_counts = [100, 250, 400]
    epsilon = 0.05
    stop = hybrid_hoeffding_stop(sample_counts, epsilon)
    print(f"Hoeffding stop condition met: {stop}")

    sys.exit(0)