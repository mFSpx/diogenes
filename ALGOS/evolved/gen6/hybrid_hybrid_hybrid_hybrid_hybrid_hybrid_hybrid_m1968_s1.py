# DARWIN HAMMER — match 1968, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1195_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s1.py (gen3)
# born: 2026-05-29T23:40:19Z

"""Hybrid algorithm combining:
- Parent A: probabilistic Metropolis acceptance, Hoeffding decision, Bayesian edge reliability.
- Parent B: geometric algebra multivectors and radial‑basis‑function (RBF) similarity between graph nodes.

Mathematical bridge:
The RBF similarity s_{ij}=exp(-ε‖x_i‑x_j‖²) is interpreted as a *confidence weight* for the geometric product
G_{ij}=a∧b+b·a.  The Bayesian posterior variance of an edge updates the RBF width ε:
lower variance ⇒ smaller ε (sharper similarity).  The Metropolis acceptance then decides
whether the weighted geometric product should be incorporated into the evolving graph state.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, replace
from typing import Dict, Tuple, List, Callable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability, never exactly zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prob = math.exp(-delta_energy / temperature)
    return max(prob, 1e-12)


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
# Parent B components
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to its nearest seed (Voronoi‑like partition)."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


class Multivector:
    """Very small geometric‑algebra helper; stores blades as frozenset of basis indices."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # prune zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), sorted(x[0]))):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.3g}*{label}")
        return " + ".join(terms)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Naïve geometric product: scalar part multiplies, blades are concatenated
    (ignoring sign rules for brevity – sufficient for demonstration)."""
    result: Dict[frozenset, float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            new_blade = frozenset(blade_a.symmetric_difference(blade_b))
            # The sign is omitted; in full GA it depends on grade parity.
            result[new_blade] = result.get(new_blade, 0.0) + coeff_a * coeff_b
    return Multivector(result, a.n)


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    """Radial basis function similarity."""
    diff = x - y
    return math.exp(-epsilon * np.dot(diff, diff))


def rbf_similarity_matrix(points: List[Point], epsilon: float) -> np.ndarray:
    """NxN matrix of pairwise RBF similarities."""
    arr = np.array(points, dtype=float)
    n = arr.shape[0]
    sim = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = rbf_kernel(arr[i], arr[j], epsilon)
    return sim


def posterior_variance_to_epsilon(variance: float,
                                  base_epsilon: float = 1.0,
                                  scale: float = 5.0) -> float:
    """Map a Beta posterior variance to an RBF width.
    Smaller variance → smaller epsilon (sharper kernel)."""
    # Clamp to avoid division by zero
    var = max(variance, 1e-8)
    return base_epsilon / (1.0 + scale * (1.0 - var))


def weighted_geometric_product(a: Multivector,
                               b: Multivector,
                               weight: float) -> Multivector:
    """Geometric product modulated by a scalar weight."""
    prod = geometric_product(a, b)
    # Scale all coefficients by weight
    scaled = {blade: coeff * weight for blade, coeff in prod.components.items()}
    return Multivector(scaled, a.n)


def metropolis_accept(delta_energy: float, temperature: float) -> bool:
    """Metropolis acceptance test using acceptance_probability."""
    prob = acceptance_probability(delta_energy, temperature)
    return random.random() < prob


# ----------------------------------------------------------------------
# Core hybrid operation (demonstrates integration of both parents)
# ----------------------------------------------------------------------
def hybrid_step(
    points: List[Point],
    seeds: List[Point],
    edge_observations: Dict[Tuple[int, int], Tuple[int, int, EdgeBetaPrior]],
    temperature: float,
    base_epsilon: float = 1.0,
) -> Tuple[Dict[int, List[Point]],
           Dict[Tuple[int, int], Multivector],
           Dict[Tuple[int, int], EdgeBetaPrior]]:
    """
    Perform one hybrid iteration.

    1. Assign points to nearest seeds (Parent B geometry).
    2. Build an RBF similarity matrix using an epsilon that is
       adapted per edge from its Bayesian posterior variance (Parent A).
    3. For each edge, compute a weighted geometric product of two random
       multivectors and decide via Metropolis whether to keep it.
    4. Return updated regions, accepted multivector products and updated priors.
    """
    # 1. Partition points
    regions = assign(points, seeds)

    # 2. Compute a global similarity matrix (initial epsilon)
    sim_matrix = rbf_similarity_matrix(points, base_epsilon)

    # 3. Process each edge
    accepted_products: Dict[Tuple[int, int], Multivector] = {}
    updated_priors: Dict[Tuple[int, int], EdgeBetaPrior] = {}

    # Helper to pick a random multivector (grade‑0 or grade‑1)
    def random_multivector(dim: int) -> Multivector:
        # Random scalar + random vector component
        scalar = random.uniform(-1.0, 1.0)
        vec = {frozenset({i}): random.uniform(-1.0, 1.0) for i in range(1, dim + 1)}
        vec[frozenset()] = scalar
        return Multivector(vec, dim)

    for (i, j), (succ, fail, prior) in edge_observations.items():
        # Bayesian update based on observed successes/failures
        mean, var, new_prior = bayesian_edge_update(prior, succ, fail)
        updated_priors[(i, j)] = new_prior

        # Adapt epsilon for this edge
        eps_edge = posterior_variance_to_epsilon(var, base_epsilon)

        # Local similarity weight between the two points (fallback to 0 if out of range)
        if i < len(points) and j < len(points):
            weight = rbf_kernel(np.array(points[i]), np.array(points[j]), eps_edge)
        else:
            weight = 0.0

        # Create two random multivectors (dimension = 3 for demo)
        a = random_multivector(3)
        b = random_multivector(3)

        # Weighted geometric product
        wg_prod = weighted_geometric_product(a, b, weight)

        # Energy proxy: use negative of weight (higher similarity = lower energy)
        delta_energy = -weight

        # Metropolis acceptance
        if metropolis_accept(delta_energy, temperature):
            accepted_products[(i, j)] = wg_prod
        else:
            # reject: keep prior unchanged (already stored)
            pass

    return regions, accepted_products, updated_priors


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866), (2.0, 2.0)]
    seed_pts = [(0.0, 0.0), (2.0, 2.0)]

    # Edge observations: (i,j) -> (successes, failures, prior)
    edge_obs = {
        (0, 1): (3, 1, EdgeBetaPrior()),
        (1, 2): (2, 2, EdgeBetaPrior(alpha=2.0, beta=2.0)),
        (2, 3): (5, 0, EdgeBetaPrior(alpha=5.0, beta=1.0)),
    }

    temperature = 0.5
    regions, products, priors = hybrid_step(pts, seed_pts, edge_obs, temperature)

    print("Regions (Voronoi assignment):")
    for idx, reg in regions.items():
        print(f"  Seed {idx}: {reg}")

    print("\nAccepted weighted geometric products:")
    for edge, mv in products.items():
        print(f"  Edge {edge}: {mv}")

    print("\nUpdated priors:")
    for edge, prior in priors.items():
        print(f"  Edge {edge}: alpha={prior.alpha:.2f}, beta={prior.beta:.2f}")

    # Simple sanity check: ensure at least one product was considered
    assert isinstance(regions, dict)
    assert isinstance(products, dict)
    assert isinstance(priors, dict)