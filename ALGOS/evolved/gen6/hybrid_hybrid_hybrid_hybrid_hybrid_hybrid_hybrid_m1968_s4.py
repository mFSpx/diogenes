# DARWIN HAMMER — match 1968, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1195_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s1.py (gen3)
# born: 2026-05-29T23:40:19Z

import math
import random
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
    # Prevent overflow for very negative delta_energy
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


# ----------------------------------------------------------------------
# Geometric Algebra helpers (full sign handling)
# ----------------------------------------------------------------------
Blade = Tuple[int, ...]  # sorted tuple of basis indices, e.g. (1,3) for e1∧e3


def _blade_sign(blade: List[int]) -> Tuple[Blade, int]:
    """
    Sort a blade (list of basis indices) into canonical order.
    Returns the sorted blade and the parity sign (+1 or -1) incurred by swaps.
    """
    sign = 1
    # In‑place bubble sort counting swaps
    for i in range(len(blade)):
        for j in range(len(blade) - 1, i, -1):
            if blade[j - 1] > blade[j]:
                blade[j - 1], blade[j] = blade[j], blade[j - 1]
                sign = -sign
    # Remove duplicate indices (e_i * e_i = 1)
    i = 0
    while i < len(blade) - 1:
        if blade[i] == blade[i + 1]:
            # e_i*e_i = 1 removes the pair and flips sign if metric is -1;
            # we assume Euclidean (+1) metric, so just drop the pair.
            del blade[i : i + 2]
            # No sign change for Euclidean metric
            i = max(i - 1, 0)
        else:
            i += 1
    return tuple(blade), sign


class Multivector:
    """
    Simple geometric‑algebra multivector.
    Internally stores a mapping from Blade (sorted tuple) to scalar coefficient.
    """

    def __init__(self, components: Dict[Blade, float], dim: int):
        # prune near‑zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.dim = int(dim)

    def scalar_part(self) -> float:
        return self.components.get((), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                  key=lambda x: (len(x[0]), x[0])):
            if blade == ():
                label = "1"
            else:
                label = "e" + "".join(str(i) for i in blade)
            terms.append(f"{coef:+.3g}*{label}")
        return " + ".join(terms)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Full geometric product with sign handling for Euclidean metric.
    Implements (a ∧ b) + (a · b) recursively via blade multiplication.
    """
    result: Dict[Blade, float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            # concatenate basis lists
            combined = list(blade_a) + list(blade_b)
            sorted_blade, sign = _blade_sign(combined)
            result[sorted_blade] = result.get(sorted_blade, 0.0) + sign * coeff_a * coeff_b
    return Multivector(result, a.dim)


def weighted_geometric_product(a: Multivector, b: Multivector, weight: float) -> Multivector:
    """Geometric product scaled by a scalar weight."""
    prod = geometric_product(a, b)
    scaled = {blade: coeff * weight for blade, coeff in prod.components.items()}
    return Multivector(scaled, a.dim)


# ----------------------------------------------------------------------
# RBF utilities
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
    """Radial basis function similarity."""
    diff = x - y
    return math.exp(-epsilon * float(np.dot(diff, diff)))


def rbf_similarity_matrix(points: List[Point], epsilon: float) -> np.ndarray:
    """NxN matrix of pairwise RBF similarities."""
    arr = np.asarray(points, dtype=float)
    n = arr.shape[0]
    sim = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = rbf_kernel(arr[i], arr[j], epsilon)
    return sim


def posterior_variance_to_epsilon(
    variance: float,
    base_epsilon: float = 1.0,
    scale: float = 5.0,
) -> float:
    """
    Map a Beta posterior variance to an RBF width.
    Smaller variance → smaller epsilon (sharper kernel).
    """
    var = max(variance, 1e-8)  # avoid division by zero
    return base_epsilon / (1.0 + scale * (1.0 - var))


# ----------------------------------------------------------------------
# Metropolis utilities
# ----------------------------------------------------------------------
def metropolis_accept(delta_energy: float, temperature: float) -> bool:
    """Metropolis acceptance test."""
    prob = acceptance_probability(delta_energy, temperature)
    return random.random() < prob


# ----------------------------------------------------------------------
# Core hybrid operation (deepened integration)
# ----------------------------------------------------------------------
def hybrid_step(
    points: List[Point],
    seeds: List[Point],
    edge_observations: Dict[Tuple[int, int], Tuple[int, int, EdgeBetaPrior]],
    temperature: float,
    base_epsilon: float = 1.0,
    hoeffding_delta: float = 0.05,
) -> Tuple[
    Dict[int, List[Point]],
    Dict[Tuple[int, int], Multivector],
    Dict[Tuple[int, int], EdgeBetaPrior],
]:
    """
    Perform one hybrid iteration with a tighter coupling between the
    probabilistic (Parent A) and geometric (Parent B) components.

    Steps
    -----
    1. Partition points to seeds (geometric Voronoi).
    2. For each *observed* edge (i, j):
       a. Update its Beta posterior.
       b. Convert posterior variance → edge‑specific ε_ij.
       c. Compute an edge‑specific RBF similarity s_ij.
       d. Generate two random multivectors of dimension = geometric space.
       e. Form a weighted geometric product using s_ij as weight.
       f. Compute a synthetic “energy” ΔE = -log(s_ij) * ||product||_F.
       g. Apply Hoeffding decision to possibly discard low‑confidence edges.
       h. Apply Metropolis acceptance on ΔE.
    3. Return updated regions, accepted products, and updated priors.
    """
    # 1. Voronoi partition (pure geometry)
    regions = assign(points, seeds)

    # 2. Containers for results
    accepted_products: Dict[Tuple[int, int], Multivector] = {}
    updated_priors: Dict[Tuple[int, int], EdgeBetaPrior] = {}

    # Helper: generate a random multivector (scalar + vector part)
    def random_multivector(dim: int) -> Multivector:
        scalar = random.uniform(-1.0, 1.0)
        comps: Dict[Blade, float] = {(): scalar}
        for i in range(1, dim + 1):
            comps[(i,)] = random.uniform(-1.0, 1.0)
        return Multivector(comps, dim)

    # Global dimension inferred from seed count (or fallback to 2)
    dim = max(2, len(seeds))

    # Pre‑compute point coordinates as numpy array for fast RBF queries
    point_array = np.asarray(points, dtype=float)

    for (i, j), (succ, fail, prior) in edge_observations.items():
        # a. Bayesian update
        post_mean, post_var, new_prior = bayesian_edge_update(prior, succ, fail)
        updated_priors[(i, j)] = new_prior

        # b. Edge‑specific epsilon
        eps_ij = posterior_variance_to_epsilon(post_var, base_epsilon)

        # c. Edge‑specific RBF similarity (using the two point coordinates)
        xi = point_array[i]
        xj = point_array[j]
        s_ij = rbf_kernel(xi, xj, eps_ij)

        # d. Random multivectors
        a_mv = random_multivector(dim)
        b_mv = random_multivector(dim)

        # e. Weighted geometric product
        prod_mv = weighted_geometric_product(a_mv, b_mv, weight=s_ij)

        # f. Synthetic energy: larger similarity and larger norm → lower energy
        #    Use Frobenius‑like norm = sqrt(sum coeff^2)
        norm_sq = sum(c * c for c in prod_mv.components.values())
        delta_energy = -math.log(max(s_ij, 1e-12)) * math.sqrt(norm_sq)

        # g. Hoeffding confidence check (use total observations as sample size)
        total_obs = succ + fail
        if not hoeffding_decision(total_obs, epsilon=s_ij, delta=hoeffding_delta):
            # Edge not confident enough – skip Metropolis step
            continue

        # h. Metropolis acceptance
        if metropolis_accept(delta_energy, temperature):
            accepted_products[(i, j)] = prod_mv

    return regions, accepted_products, updated_priors