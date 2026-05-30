# DARWIN HAMMER — match 2285, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_serpentina_se_m872_s2.py (gen5)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m660_s2.py (gen5)
# born: 2026-05-29T23:41:45Z

import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Bayesian utilities
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)P(H) + P(E|¬H)P(¬H)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probability arguments must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Extract a 5‑dimensional count vector from *text*.
    The vector is L2‑normalised to make it comparable across texts of different length.
    """
    patterns = [r'\b\w+\b', r'\d+', r'[a-z]+', r'[A-Z]+', r'\W']
    counts = np.array([len(re.findall(p, text)) for p in patterns], dtype=float)
    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts


# ----------------------------------------------------------------------
# Similarity computation (Mahalanobis distance)
# ----------------------------------------------------------------------
def compute_similarity(v1: np.ndarray, v2: np.ndarray, cov: Optional[np.ndarray] = None) -> float:
    """
    Return a similarity in [0,1] based on a Gaussian kernel.
    If *cov* is not supplied, an identity matrix is used (i.e. Euclidean distance).
    """
    diff = v1 - v2
    if cov is None:
        dist_sq = diff @ diff
    else:
        inv_cov = np.linalg.inv(cov)
        dist_sq = diff @ inv_cov @ diff
    sigma = 1.0
    return math.exp(-dist_sq / (2 * sigma ** 2))


# ----------------------------------------------------------------------
# RBF surrogate model (multi‑center, vectorised)
# ----------------------------------------------------------------------
@dataclass
class RBFSurrogate:
    centers: np.ndarray = field(default_factory=lambda: np.array([[1, 2, 3, 4, 5],
                                                                   [6, 7, 8, 9, 10],
                                                                   [0, 0, 0, 0, 0]], dtype=float))
    weights: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.2], dtype=float))
    sigma: float = 1.0

    def __call__(self, features: np.ndarray) -> float:
        """
        Predict a scalar diffusion coefficient.
        The model is a weighted sum of Gaussian RBFs centred at *centers*.
        """
        if features.shape != self.centers.shape[1:]:
            raise ValueError("Feature dimension mismatch.")
        dists_sq = np.sum((self.centers - features) ** 2, axis=1)
        rbf_vals = np.exp(-dists_sq / (2 * self.sigma ** 2))
        return float(np.dot(self.weights, rbf_vals))


# ----------------------------------------------------------------------
# Morphology‑modulated Fisher information
# ----------------------------------------------------------------------
def fisher_information(morphology: str, features: np.ndarray) -> float:
    """
    Compute a morphology‑aware Fisher information term.
    The morphology string selects a scaling factor and a custom kernel width.
    """
    # Map a few example morphologies to scaling and sigma values
    morph_map: Dict[str, Tuple[float, float]] = {
        "Serpentina": (1.2, 0.8),
        "Linear": (0.9, 1.2),
        "Radial": (1.0, 1.0),
    }
    scale, sigma = morph_map.get(morphology, (1.0, 1.0))

    # Use the variance of the feature vector as a proxy for information content
    var = np.var(features)
    info = scale * (1.0 / (sigma + var + 1e-8))  # avoid division by zero
    return float(info)


# ----------------------------------------------------------------------
# Edge cost computation
# ----------------------------------------------------------------------
def compute_edge_cost(
    edge: Edge,
    morphology: str,
    edge_features: np.ndarray,
    prior: float,
    likelihood: float,
    false_positive: float,
    rbf_model: RBFSurrogate,
) -> float:
    """
    Compute a cost for *edge* by blending Bayesian posterior, Fisher information,
    and a diffusion coefficient predicted by the RBF surrogate.
    """
    # Bayesian part – prior can be adapted per edge if desired
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # Fisher information modulated by morphology
    fisher = fisher_information(morphology, edge_features)

    # Diffusion coefficient (stochastic forcing term)
    diffusion = rbf_model(edge_features)

    # Final cost: higher posterior & Fisher increase cost, diffusion attenuates it
    return posterior * fisher * diffusion


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    edges: List[Edge],
    morphology: str,
    base_text: str,
    *,
    prior: float = 0.5,
    likelihood: float = 0.8,
    false_positive: float = 0.2,
    rbf_model: Optional[RBFSurrogate] = None,
) -> List[float]:
    """
    Perform the hybrid algorithm on *edges*.

    For each edge we:
      1. Build an edge‑specific text by concatenating the node labels with *base_text*.
      2. Extract a normalised feature vector.
      3. Compute similarity with the previous edge (if any) to adapt the Bayesian prior.
      4. Evaluate the edge cost using the adapted prior, morphology‑modulated Fisher
         information and the RBF surrogate diffusion coefficient.
    """
    if rbf_model is None:
        rbf_model = RBFSurrogate()

    # Pre‑compute a global covariance for similarity (optional, identity fallback)
    all_features = np.stack([extract_features(base_text) for _ in edges])
    cov = np.cov(all_features, rowvar=False) if all_features.shape[0] > 1 else None

    edge_costs: List[float] = []
    prev_features: Optional[np.ndarray] = None
    adaptive_prior = prior

    for edge in edges:
        # 1. Edge‑specific text
        edge_text = f"{edge[0]} {edge[1]} {base_text}"

        # 2. Feature extraction
        feats = extract_features(edge_text)

        # 3. Similarity‑driven prior adaptation
        if prev_features is not None:
            sim = compute_similarity(feats, prev_features, cov)
            # Blend the original prior with similarity (more similar → higher confidence)
            adaptive_prior = prior * sim + (1 - sim) * 0.1  # floor at 0.1 to keep it >0
        else:
            adaptive_prior = prior

        # 4. Edge cost
        cost = compute_edge_cost(
            edge,
            morphology,
            feats,
            adaptive_prior,
            likelihood,
            false_positive,
            rbf_model,
        )
        edge_costs.append(cost)
        prev_features = feats

    return edge_costs


# ----------------------------------------------------------------------
# Demo / simple test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_edges = [("A", "B"), ("B", "C"), ("C", "A")]
    demo_morph = "Serpentina"
    demo_text = "The quick brown fox jumps over 13 lazy dogs!"
    costs = hybrid_operation(demo_edges, demo_morph, demo_text)
    print("Edge costs:", costs)