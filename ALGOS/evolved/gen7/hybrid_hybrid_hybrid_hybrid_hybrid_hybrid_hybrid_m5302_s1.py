# DARWIN HAMMER — match 5302, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ssim_h_m1837_s2.py (gen6)
# born: 2026-05-30T00:01:14Z

"""
Hybrid algorithm merging:

- Parent A (hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m773_s3.py): Gaussian beam,
  Fisher information score, tree metric construction, Bayesian marginal/update, MinHash.

- Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ssim_h_m1837_s2.py): Pheromone
  probability generation, entropy/expected entropy, and a Structural Similarity Index
  (SSIM) based multivector weighting.

Mathematical bridge:
The Fisher information computed from a Gaussian‑beam‑weighted probability distribution
(equivalent to the pheromone probabilities of Parent B) is used as the likelihood in a
Bayesian update of edge beliefs on a tree (from Parent A).  The resulting posterior edge
probabilities are then modulated by an SSIM similarity score between two attribute
matrices, providing a multivector‑style decision weighting that combines both lineages.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """Build adjacency, edge lengths and distance-from-root for an undirected tree."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        d = length(nodes[a], nodes[b])
        edge_len[(a, b)] = d
        edge_len[(b, a)] = d

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist


def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = np.where(marginal > 0, (likelihood * prior) / marginal, 0.0)
    return posterior


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(limit: int) -> List[float]:
    """Simulate pheromone strengths and normalise them to a probability distribution."""
    pheromones = np.random.rand(limit)
    total = pheromones.sum()
    if total == 0:
        raise ValueError("Pheromone sum cannot be zero")
    return (pheromones / total).tolist()


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = [p / total for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1 - p_hit) * entropy(miss_state)


def simple_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """A lightweight SSIM implementation for two equal‑shaped 2‑D arrays."""
    if x.shape != y.shape:
        raise ValueError("Input arrays must have the same shape")
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def fisher_information_from_pheromones(
    probs: List[float],
    center: float = 0.5,
    width: float = 0.15,
) -> np.ndarray:
    """
    Treat each pheromone probability as a theta value.
    Compute a Fisher‑information‑like score per entry,
    then weight it by a Gaussian beam centred at `center`.
    Returns a vector of the same length as `probs`.
    """
    scores = np.array(
        [fisher_score(p, center, width) for p in probs], dtype=float
    )
    weights = np.array([gaussian_beam(p, center, width) for p in probs], dtype=float)
    return scores * weights


def update_edge_beliefs(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    pheromone_limit: int = 10,
    false_positive: float = 0.01,
) -> Dict[Tuple[str, str], float]:
    """
    1. Build tree metrics.
    2. Generate pheromone probabilities.
    3. Derive a Fisher‑information likelihood per edge (using distance as a proxy for theta).
    4. Perform a Bayesian update of prior edge beliefs.
    Returns a dictionary mapping each directed edge to its posterior probability.
    """
    # 1. Tree structure
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # 2. Pheromone distribution (global, not per edge)
    pher_probs = calculate_pheromone_probabilities(pheromone_limit)

    # 3. Likelihood per edge: map distance -> probability via a simple normalisation
    max_dist = max(dist.values()) if dist else 1.0
    theta_vals = np.array([dist[n] / max_dist for n in nodes])  # normalised distances
    likelihood_vec = fisher_information_from_pheromones(theta_vals.tolist())

    # Align likelihoods with edges (use source node's theta)
    likelihood = np.array(
        [likelihood_vec[list(nodes.keys()).index(src)] for src, _ in edges],
        dtype=float,
    )
    # Prior: uniform
    prior = np.full_like(likelihood, 1.0 / len(likelihood))

    # 4. Bayesian update
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # Return a mapping for both directions (undirected tree)
    posterior_dict: Dict[Tuple[str, str], float] = {}
    for (src, dst), prob in zip(edges, posterior):
        posterior_dict[(src, dst)] = prob
        posterior_dict[(dst, src)] = prob  # symmetry

    return posterior_dict


def ssim_scaled_multivector(
    attr_matrix_a: np.ndarray,
    attr_matrix_b: np.ndarray,
    base_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute SSIM between two attribute matrices and use it to scale a decision multivector.
    The scaling factor is the SSIM value (in [0,1]); the result is a weighted vector.
    """
    ssim_val = simple_ssim(attr_matrix_a, attr_matrix_b)
    return base_vector * ssim_val


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree with 5 nodes
    nodes_demo = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
        "E": (0.5, 0.5),
    }
    edges_demo = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "E")]
    root_demo = "A"

    # Update edge beliefs using the hybrid Bayesian/Fisher pipeline
    posteriors = update_edge_beliefs(nodes_demo, edges_demo, root_demo, pheromone_limit=12)
    print("Posterior edge probabilities:")
    for edge, prob in posteriors.items():
        print(f"  {edge}: {prob:.4f}")

    # Create two synthetic attribute matrices (e.g., image patches)
    rng = np.random.default_rng(42)
    attr_a = rng.integers(0, 256, size=(8, 8)).astype(float)
    attr_b = rng.integers(0, 256, size=(8, 8)).astype(float)

    # Base decision vector (could represent action propensities)
    base_vec = np.array([0.2, 0.5, 0.3])

    weighted_vec = ssim_scaled_multivector(attr_a, attr_b, base_vec)
    print("\nBase vector:", base_vec)
    print("SSIM‑scaled vector:", weighted_vec)
    sys.exit(0)