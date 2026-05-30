# DARWIN HAMMER — match 2981, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_semant_m2040_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_gliner_m1234_s1.py (gen4)
# born: 2026-05-29T23:47:11Z

import math
import random
import sys
from pathlib import Path
import numpy as np
from collections.abc import Mapping, Hashable
from typing import Dict, List, Tuple

Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Utility functions from Parent A
# ----------------------------------------------------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    """Normalised pheromone probabilities (Parent A helper)."""
    total = sum(pheromones)
    if total == 0:
        return [0.0 for _ in pheromones]
    return [p / total for p in pheromones]


# ----------------------------------------------------------------------
# Functions from Parent B
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ternary_vector(text: str, labels: List[str]) -> np.ndarray:
    """Produce a ternary (+1/0/‑1) evidence vector for *text* over *labels*."""
    vec = np.zeros(len(labels))
    lowered = text.lower()
    for i, label in enumerate(labels):
        if label in text:
            vec[i] = 1.0
        elif label.lower() in lowered:
            vec[i] = -1.0
    return vec


def gini_coefficient(weights: np.ndarray) -> float:
    """Gini coefficient of a non‑negative weight vector."""
    if weights.size == 0:
        return 0.0
    sorted_w = np.sort(weights)
    n = weights.size
    cumulative = np.cumsum(sorted_w, dtype=float)
    sum_w = cumulative[-1]
    if sum_w == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_w) / n
    return float(gini)


def shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy (base‑2) of a probability distribution."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def ssim_like(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Very light‑weight SSIM‑like similarity for 1‑D vectors.
    Uses luminance, contrast and structure components.
    """
    C1, C2, C3 = 1e-4, 1e-4, 1e-4
    mu1, mu2 = np.mean(vec1), np.mean(vec2)
    sigma1_sq, sigma2_sq = np.var(vec1), np.var(vec2)
    sigma12 = np.mean((vec1 - mu1) * (vec2 - mu2))

    l = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c = (2 * math.sqrt(sigma1_sq) * math.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + C3) / (math.sqrt(sigma1_sq) * math.sqrt(sigma2_sq) + C3) if sigma1_sq * sigma2_sq > 0 else 0.0
    return float(l * c * s)


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------
def combined_similarity(
    feat_i: np.ndarray,
    feat_j: np.ndarray,
    text_i: str,
    text_j: str,
    labels: List[str],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    fisher_center: float,
    fisher_width: float,
) -> float:
    """Compute the hybrid similarity S_ij that merges:

    • Cosine similarity of feature vectors.
    • Fisher‑score weighting of a distance‑derived theta.
    • Entropy and Gini of ternary evidence vectors.
    • SSIM‑like structural similarity of the ternary vectors.

    Parameters
    ----------
    feat_i, feat_j : np.ndarray
        Continuous node embeddings.
    text_i, text_j : str
        Associated textual payloads.
    labels : List[str]
        Vocabulary for ternary vector construction.
    alpha, beta, gamma, delta : float
        Tunable scalars for entropy, Gini, SSIM and the cosine·Fisher term.
    fisher_center, fisher_width : float
        Parameters of the Gaussian beam used by FisherScore.

    Returns
    -------
    float
        Hybrid similarity value for the edge (i, j).
    """
    # 1. Cosine similarity
    cos_sim = _cosine_similarity(feat_i, feat_j)

    # 2. FisherScore based on Euclidean distance
    theta = np.linalg.norm(feat_i - feat_j)
    fisher = fisher_score(theta, fisher_center, fisher_width)

    # 3. Ternary vectors from textual data
    v_i = ternary_vector(text_i, labels)
    v_j = ternary_vector(text_j, labels)

    # 4. Weighted ternary histograms (absolute values multiplied by Fisher)
    w_i = np.abs(v_i) * fisher
    w_j = np.abs(v_j) * fisher

    # 5. Entropy (treat each histogram as a probability distribution)
    prob_i = w_i / (w_i.sum() + 1e-12)
    prob_j = w_j / (w_j.sum() + 1e-12)
    H_i = shannon_entropy(prob_i)
    H_j = shannon_entropy(prob_j)
    H = 0.5 * (H_i + H_j)

    # 6. Gini coefficient of the combined histogram
    G = gini_coefficient(np.concatenate([w_i, w_j]))

    # 7. SSIM‑like similarity between ternary vectors
    S = ssim_like(v_i, v_j)

    # 8. Hybrid blend
    hybrid = alpha * H + beta * G + gamma * S + delta * (cos_sim * fisher)
    return hybrid


def update_weight_matrix(
    W: np.ndarray,
    features: List[np.ndarray],
    texts: List[str],
    labels: List[str],
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
    learning_rate: float = 0.05,
) -> np.ndarray:
    """Perform a single gradient‑descent update of the adjacency matrix *W*.
    The target for each entry W_ij is the hybrid similarity computed by
    ``combined_similarity``; the update follows:
    """
    num_nodes = len(features)
    new_W = W.copy()
    for i in range(num_nodes):
        for j in range(num_nodes):
            feat_i, feat_j = features[i], features[j]
            text_i, text_j = texts[i], texts[j]
            S_ij = combined_similarity(
                feat_i,
                feat_j,
                text_i,
                text_j,
                labels,
                alpha,
                beta,
                gamma,
                delta,
                fisher_center,
                fisher_width,
            )
            new_W[i, j] = new_W[i, j] - learning_rate * (new_W[i, j] - S_ij)
    return new_W


def run_hybrid_process(
    num_nodes: int,
    num_features: int,
    num_labels: int,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
    learning_rate: float = 0.05,
) -> np.ndarray:
    """Run the hybrid process on a random graph."""
    features = [np.random.rand(num_features) for _ in range(num_nodes)]
    texts = [f"node {i}" for i in range(num_nodes)]
    labels = [f"label {i}" for i in range(num_labels)]
    W = np.random.rand(num_nodes, num_nodes)
    new_W = update_weight_matrix(
        W,
        features,
        texts,
        labels,
        alpha,
        beta,
        gamma,
        delta,
        fisher_center,
        fisher_width,
        learning_rate,
    )
    return new_W


if __name__ == "__main__":
    num_nodes = 10
    num_features = 5
    num_labels = 3
    new_W = run_hybrid_process(num_nodes, num_features, num_labels)
    print(new_W)