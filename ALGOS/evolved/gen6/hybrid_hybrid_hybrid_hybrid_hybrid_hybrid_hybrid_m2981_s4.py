# DARWIN HAMMER — match 2981, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_semant_m2040_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_gliner_m1234_s1.py (gen4)
# born: 2026-05-29T23:47:11Z

"""Hybrid Graph-Fisher-Ternary Router (Hybrid A ↔ Hybrid B)

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – graph‑based leader election with adjacency matrix *W* updated by
  gradient descent, cosine similarity between node feature vectors and a Voronoi‑style
  semantic neighbourhood expressed as multivectors.

* **Parent B** – Fisher‑score weighted ternary evidence vectors, Gini‑coefficient
  of the weighted ternary histogram, Shannon entropy and a structural‑similarity
  (SSIM‑like) term.

**Mathematical bridge**

Both parents operate on *weights* derived from similarity measures:

* In **A** the edge weight update uses a cosine similarity  cos(·,·)  between node
  feature vectors *f_i* and *f_j*.
* In **B** a *FisherScore* θ provides a continuous confidence that multiplies a
  ternary evidence vector *v* (derived from textual labels) to obtain a weighted
  histogram *w*.  From *w* the Gini coefficient *G* and Shannon entropy *H* are
  computed and finally combined with a SSIM‑like similarity *S*.

The hybrid therefore builds a **combined similarity** for each node pair *(i,j)*
as  


S_ij = α·H_ij + β·G_ij + γ·S_ij + δ·cos(f_i, f_j)·FisherScore(θ_ij)


where *θ_ij* is a scalar derived from the Euclidean distance of the two feature
vectors.  The resulting scalar drives a gradient‑descent update of the adjacency
matrix *W*.

The implementation below provides three high‑level functions that realise this
fusion:

1. `combined_similarity` – evaluates the hybrid similarity term *S_ij*.
2. `update_weight_matrix` – performs one gradient‑descent step on *W* using the
   hybrid similarity.
3. `run_hybrid_process` – end‑to‑end demo that builds a tiny graph, computes the
   hybrid update and returns the new adjacency matrix.

All code uses only the permitted standard library and NumPy. """

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
    """
    Compute the hybrid similarity S_ij that merges:

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
    """
    Perform a single gradient‑descent update of the adjacency matrix *W*.
    The target for each entry W_ij is the hybrid similarity computed by
    ``combined_similarity``; the update follows:

        W ← W - η·(W - S)

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Current symmetric weight matrix.
    features : List[np.ndarray]
        List of node feature vectors (length N).
    texts : List[str]
        Corresponding textual payloads.
    labels : List[str]
        Vocabulary for ternary vector construction.
    alpha, beta, gamma, delta : float
        Scalars controlling the contribution of each term.
    fisher_center, fisher_width : float
        Parameters for the FisherScore.
    learning_rate : float
        Gradient‑descent step size η.

    Returns
    -------
    np.ndarray
        Updated symmetric weight matrix.
    """
    N = len(features)
    if W.shape != (N, N):
        raise ValueError("Weight matrix shape does not match number of nodes")

    S = np.zeros_like(W, dtype=float)

    # Compute hybrid similarity for each unordered pair (i, j)
    for i in range(N):
        for j in range(i, N):
            sim = combined_similarity(
                features[i],
                features[j],
                texts[i],
                texts[j],
                labels,
                alpha,
                beta,
                gamma,
                delta,
                fisher_center,
                fisher_width,
            )
            S[i, j] = sim
            S[j, i] = sim  # enforce symmetry

    # Gradient step
    W_new = W - learning_rate * (W - S)

    # Optional: keep diagonal zero (no self‑loops)
    np.fill_diagonal(W_new, 0.0)
    return W_new


def run_hybrid_process() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Demonstration driver:

    * Creates a tiny graph with 4 nodes.
    * Random 5‑dimensional feature vectors.
    * Random short texts.
    * Executes one hybrid update of the adjacency matrix.
    * Returns the final matrix and the feature list for inspection.
    """
    random.seed(42)
    np.random.seed(42)

    N = 4
    dim = 5
    labels = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Random features and texts
    features = [np.random.randn(dim) for _ in range(N)]
    sample_texts = [
        "Alpha and beta are present",
        "Gamma appears here",
        "Delta and epsilon are missing",
        "No known label",
    ]

    # Initialise adjacency matrix with small random weights, zero diagonal
    W = np.random.rand(N, N) * 0.1
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    # Perform hybrid update
    W_updated = update_weight_matrix(
        W,
        features,
        sample_texts,
        labels,
        alpha=0.25,
        beta=0.25,
        gamma=0.25,
        delta=0.25,
        fisher_center=0.0,
        fisher_width=1.5,
        learning_rate=0.1,
    )

    # Simple sanity check: matrix must stay symmetric and have zero diagonal
    assert np.allclose(W_updated, W_updated.T, atol=1e-8)
    assert np.allclose(np.diag(W_updated), 0.0, atol=1e-8)

    # Print for visual verification (optional)
    print("Updated adjacency matrix (symmetric, zero diagonal):")
    print(W_updated)

    return W_updated, features


if __name__ == "__main__":
    # Smoke test – run the hybrid process and ensure no exceptions.
    run_hybrid_process()