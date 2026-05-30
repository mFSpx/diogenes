# DARWIN HAMMER — match 5335, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_temporal_moti_m1392_s2.py (gen5)
# born: 2026-05-30T00:01:24Z

"""Hybrid Algorithm Fusion of Geometric Algebra RBF Surrogates and Temporal Gini‑Bayesian Motifs.

Parents:
- hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s3.py (geometric algebra + RBF similarity)
- hybrid_hybrid_hybrid_doomsd_hybrid_temporal_moti_m1392_s2.py (Gini coefficient, Bayesian update, temporal motifs)

Mathematical Bridge:
The blades of a multivector form an orthogonal basis that can be flattened into a feature vector.
Using a radial‑basis‑function (RBF) kernel we obtain a similarity matrix **S** between multivectors.
Each row of **S** is a discrete probability distribution over the other multivectors; the inequality
of this distribution is measured by the Gini coefficient **G**.  **G** is then used as a prior
confidence in a Bayesian update of the similarity weights, yielding a refined posterior matrix
**P** that can drive downstream tasks such as spatial assignment or temporal motif scoring.
Thus the geometric‑algebra representation supplies the feature space, the RBF supplies the
kernel similarity, and the Gini‑Bayesian machinery supplies a principled re‑weighting."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Sequence, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric‑algebra structures (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Multivector:
    """
    Very small multivector implementation.
    Internally stores a mapping from blade (tuple of basis indices) to scalar coefficient.
    Example: {(): 1.0} is the scalar part, {(1,): 2.0} is a vector component, {(1,2): 3.0} a bivector, etc.
    """

    def __init__(self, components: Dict[Tuple[int, ...], float] | None = None):
        self.components: Dict[Tuple[int, ...], float] = dict(components) if components else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components)
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
        return result

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components)
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) - coeff
        return result

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({blade: coeff * scalar for blade, coeff in self.components.items()})

    __rmul__ = __mul__

    def norm(self) -> float:
        """Euclidean norm of the flattened coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def to_vector(self, basis_order: List[Tuple[int, ...]]) -> np.ndarray:
        """Flatten into a fixed‑order numpy vector."""
        return np.array([self.components.get(blade, 0.0) for blade in basis_order], dtype=float)


def random_multivector(max_grade: int = 3, max_basis: int = 4) -> Multivector:
    """Generate a random multivector with blades up to *max_grade* over *max_basis* basis vectors."""
    components: Dict[Tuple[int, ...], float] = {}
    for grade in range(max_grade + 1):
        # number of possible blades of this grade
        if grade == 0:
            blades = [()]
        else:
            blades = [tuple(sorted(combo)) for combo in
                      _combinations(range(max_basis), grade)]
        for blade in blades:
            if random.random() < 0.3:  # sparsity
                components[blade] = random.uniform(-1.0, 1.0)
    return Multivector(components)


def _combinations(iterable, r):
    """Simple combinatorial generator (replacement for itertools.combinations)."""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# ----------------------------------------------------------------------
# RBF similarity (from Parent A) and Gini‑Bayesian re‑weighting (from Parent B)
# ----------------------------------------------------------------------


def rbf_kernel(x: np.ndarray, y: np.ndarray, epsilon: float = 1.0) -> float:
    """Radial Basis Function kernel: exp(-epsilon * ||x‑y||²)."""
    diff = x - y
    return math.exp(-epsilon * np.dot(diff, diff))


def similarity_matrix(
    mvs: List[Multivector],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Compute the pairwise RBF similarity matrix between multivectors.
    Rows are normalized to sum to 1, turning each row into a probability distribution.
    """
    if not mvs:
        raise ValueError("Empty multivector list")
    # Determine a common basis ordering (union of all blades)
    all_blades: Set[Tuple[int, ...]] = set()
    for mv in mvs:
        all_blades.update(mv.components.keys())
    basis_order = sorted(all_blades)

    vectors = np.stack([mv.to_vector(basis_order) for mv in mvs])
    n = len(mvs)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            S[i, j] = rbf_kernel(vectors[i], vectors[j], epsilon)
    # Row‑wise normalization to obtain a stochastic matrix
    row_sums = S.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    return S / row_sums


def gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient for a non‑negative list of numbers.
    Returns 0 for an empty or all‑zero list.
    """
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def gini_vectorized(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Gini coefficient for each row of a probability matrix.
    Returns a 1‑D array of Gini values.
    """
    return np.array([gini_coefficient(row.tolist()) for row in matrix])


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform element‑wise Bayesian update:
        posterior ∝ prior * likelihood
    The result is re‑normalized to remain stochastic (rows sum to 1).
    """
    if prior.shape != likelihood.shape:
        raise ValueError("Shape mismatch between prior and likelihood")
    unnorm = prior * likelihood
    row_sums = unnorm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return unnorm / row_sums


# ----------------------------------------------------------------------
# Hybrid operations demonstrating the fused mathematics
# ----------------------------------------------------------------------


def hybrid_similarity_with_gini(
    mvs: List[Multivector],
    epsilon: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the RBF similarity matrix and the corresponding Gini coefficients per row.
    Returns (S, G) where S is the stochastic similarity matrix and G is a column vector.
    """
    S = similarity_matrix(mvs, epsilon)
    G = gini_vectorized(S)[:, np.newaxis]  # shape (n,1)
    return S, G


def hybrid_posterior_weights(
    mvs: List[Multivector],
    epsilon: float = 1.0,
    gini_weight: float = 0.5,
) -> np.ndarray:
    """
    Produce a posterior similarity matrix that blends the raw RBF similarity (as likelihood)
    with a prior derived from the Gini inequality of each row.
    The prior for row i is a convex combination:
        prior_i = (1‑gini_weight) * uniform + gini_weight * (1‑G_i) * uniform
    where G_i is the Gini coefficient of row i (higher G => lower confidence).
    """
    S, G = hybrid_similarity_with_gini(mvs, epsilon)  # S rows sum to 1
    n = S.shape[0]
    # Uniform prior matrix
    uniform = np.full((n, n), 1.0 / n, dtype=float)
    # Row‑wise confidence factor (higher Gini → lower confidence)
    confidence = (1.0 - G) * gini_weight + (1.0 - gini_weight)
    prior = uniform * confidence  # broadcasting over rows
    posterior = bayesian_update(prior, S)
    return posterior


def assign_points_to_multivectors(
    points: List[Point],
    seeds: List[Point],
    mvs: List[Multivector],
    epsilon: float = 1.0,
) -> Dict[int, List[Point]]:
    """
    Spatial assignment of points to seed locations, weighted by the posterior similarity
    between multivectors attached to each seed.  Each seed is associated with a multivector
    (by index).  The posterior matrix from `hybrid_posterior_weights` modulates the
    Euclidean distance: effective distance = Euclidean distance / similarity.
    The point is assigned to the seed with minimal effective distance.
    """
    if len(seeds) != len(mvs):
        raise ValueError("Number of seeds must match number of multivectors")
    posterior = hybrid_posterior_weights(mvs, epsilon)

    # Pre‑compute Euclidean distances from each point to each seed
    point_array = np.array(points)  # shape (p,2)
    seed_array = np.array(seeds)    # shape (s,2)
    diff = point_array[:, np.newaxis, :] - seed_array[np.newaxis, :, :]  # (p,s,2)
    euclid = np.linalg.norm(diff, axis=2)  # (p,s)

    # Convert posterior similarities to a scaling factor (avoid division by zero)
    sim_factor = posterior.T  # shape (s,p?) actually posterior is (s,s); we need per‑seed similarity
    # For simplicity we use the diagonal (self‑similarity) as the weight for each seed
    seed_weights = np.diag(posterior)  # shape (s,)
    weight_matrix = euclid / (seed_weights + 1e-9)  # broadcast division

    assignments = {i: [] for i in range(len(seeds))}
    nearest_indices = np.argmin(weight_matrix, axis=1)
    for pt, idx in zip(points, nearest_indices):
        assignments[int(idx)].append(pt)
    return assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    random.seed(42)
    np.random.seed(42)

    # Create 5 random seed points and attach a random multivector to each
    seeds = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]
    multivectors = [random_multivector() for _ in range(5)]

    # Generate 100 random data points
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(100)]

    # Run hybrid assignment
    regions = assign_points_to_multivectors(points, seeds, multivectors, epsilon=0.05)

    # Print a concise summary
    for idx, pts in regions.items():
        print(f"Seed {idx} at {seeds[idx]} received {len(pts)} points.")
    # Verify that all points were assigned
    total_assigned = sum(len(v) for v in regions.values())
    assert total_assigned == len(points), "Some points were lost during assignment"
    print("Hybrid algorithm executed successfully.")