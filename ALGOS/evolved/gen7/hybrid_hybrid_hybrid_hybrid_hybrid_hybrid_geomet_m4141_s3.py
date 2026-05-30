# DARWIN HAMMER — match 4141, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1968_s4.py (gen6)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen4)
# born: 2026-05-29T23:53:52Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List, FrozenSet, Callable, Optional
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior for Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0

def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis-style acceptance probability, never exactly zero."""
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

    # Build list preserving original order for odd-counted indices
    cleaned = []
    temp_counts = counts.copy()
    for i in indices:
        if temp_counts[i] % 2 == 1:
            cleaned.append(i)
            temp_counts[i] = 0

    # Bubble-sort while tracking sign
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
    Full geometric product of two multivectors (limited to scalar and grade-1 blades).
    The result is a new multivector dictionary mapping basis blades to coefficients.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff
    # Remove near-zero entries for cleanliness
    return {b: c for b, c in result.items() if abs(c) > 1e-15}

def scalar_part(mv: Dict[FrozenSet[int], float]) -> float:
    """Extract the grade-0 (scalar) component of a multivector."""
    return mv.get(frozenset(), 0.0)

def vector_to_mv(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Convert a 1-D array into a multivector containing only grade-1 blades."""
    mv: Dict[FrozenSet[int], float] = {}
    for i, coeff in enumerate(v):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv

EdgeId = int
Point = Tuple[float, float]

class HybridGraph:
    """
    Simple undirected graph where each edge stores:
      * a Bayesian beta prior,
      * a multivector whose grade-1 blade coefficient equals the posterior mean.
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
    assignments = {}
    for point in points:
        min_dist = float('inf')
        nearest_seed = None
        for seed in seeds:
            edge_id = edge_lookup(point, seed)
            dist = math.sqrt((point[0] - seed[0])**2 + (point[1] - seed[1])**2)
            mv = graph.edge_mv(*edge_id)
            energy = scalar_part(mv)
            biased_dist = dist + energy
            if biased_dist < min_dist:
                min_dist = biased_dist
                nearest_seed = seed
        if nearest_seed not in assignments:
            assignments[nearest_seed] = []
        assignments[nearest_seed].append(point)
    return assignments

def main():
    graph = HybridGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)
    graph.update_edge(0, 1, 10, 5)
    graph.update_edge(1, 2, 8, 6)
    graph.update_edge(2, 0, 12, 4)
    points = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
    seeds = [(0, 0), (1, 1)]
    def edge_lookup(point, seed):
        if point[0] < seed[0]:
            return (0, 1)
        else:
            return (1, 0)
    assignments = hybrid_assign(points, seeds, graph, edge_lookup)
    print(assignments)

if __name__ == "__main__":
    main()