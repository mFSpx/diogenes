# DARWIN HAMMER — match 602, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_hybrid_model__m99_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s4.py (gen4)
# born: 2026-05-29T23:30:01Z

import numpy as np
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence


# ----------------------------------------------------------------------
# Utility functions – pure, deterministic, and safely handling edge cases
# ----------------------------------------------------------------------
def compute_phash(values: Sequence[float]) -> int:
    """
    Compute a 64‑bit perceptual hash of a numeric sequence.
    The hash is based on whether each element is above or below the mean.
    Empty input yields 0.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:                     # truncate / pad to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def schoolfield_rate(temperature: float) -> float:
    """
    Simple temperature‑performance model (logistic curve).
    Returns a value in (0, 1) that grows with temperature up to a plateau.
    """
    return 1.0 / (1.0 + math.exp(-(temperature - 20.0) / 5.0))


def gini_coefficient(rewards: Sequence[float]) -> float:
    """
    Compute the Gini coefficient of a reward batch.
    Handles zero‑mean and empty inputs gracefully.
    """
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size == 0:
        return 0.0
    mean = rewards_arr.mean()
    if mean == 0.0:
        return 0.0
    # Vectorised Gini: sort, then use the known formula
    sorted_rewards = np.sort(rewards_arr)
    n = rewards_arr.size
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_rewards)) / (n * np.sum(sorted_rewards)) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Graph construction – immutable‑friendly representation
# ----------------------------------------------------------------------
def build_graph(elements: List[List[float]], vram_weights: List[float], max_hamming: int = 4) -> Dict[str, Dict[str, float]]:
    """
    Build an undirected similarity graph.
    Nodes are identified by string indices.
    Edge weight = vram_weights of the *target* node (as in the original code).
    Two nodes are connected if the Hamming distance between their perceptual hashes
    of raw feature vectors is ≤ max_hamming.
    """
    if len(elements) != len(vram_weights):
        raise ValueError("elements and vram_weights must have the same length")

    # Compute a stable hash for each node once – immutable mapping
    node_hashes: Dict[str, int] = {
        str(i): compute_phash(elements[i]) for i in range(len(elements))
    }

    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(len(elements))}

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(node_hashes[str(i)], node_hashes[str(j)]) <= max_hamming:
                # Symmetric edges with weight taken from the *other* node
                graph[str(i)][str(j)] = vram_weights[j]
                graph[str(j)][str(i)] = vram_weights[i]

    return graph


# ----------------------------------------------------------------------
# Feature matrix – deeper integration of topology & performance
# ----------------------------------------------------------------------
def compute_feature_matrix(
    graph: Dict[str, Dict[str, float]],
    temperature: float,
    normalize: bool = True
) -> np.ndarray:
    """
    For each node produce a 3‑dimensional feature vector:
      1. Normalized degree (leader‑election relevance)
      2. Perceptual hash of the sorted neighbor weights (topology fingerprint)
      3. Temperature‑performance factor (Schoolfield)
    The resulting matrix is optionally L2‑normalized row‑wise.
    """
    features = []
    for node, neighbors in graph.items():
        degree = len(neighbors)

        # Fingerprint of neighbor weights – deterministic ordering
        neighbor_weights = [neighbors[nbr] for nbr in sorted(neighbors)]
        hash_val = compute_phash(neighbor_weights)

        # Combine into a float vector; hash is scaled to [0,1] via 64‑bit max
        hash_norm = hash_val / (2 ** 64 - 1)

        temp_factor = schoolfield_rate(temperature)

        features.append([degree, hash_norm, temp_factor])

    feature_matrix = np.array(features, dtype=float)

    if normalize:
        # Row‑wise L2 norm to keep scales comparable across dimensions
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        # Avoid division by zero for isolated nodes
        norms[norms == 0] = 1.0
        feature_matrix = feature_matrix / norms

    return feature_matrix


# ----------------------------------------------------------------------
# Normalized Least‑Mean‑Squares (NLMS) predictor
# ----------------------------------------------------------------------
def nlms_predict(feature_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear prediction using current weight vector."""
    return feature_matrix @ weights


def nlms_update(
    feature_matrix: np.ndarray,
    error: np.ndarray,
    learning_rate: float,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Perform a single NLMS weight update.
    The step size is scaled by the squared norm of each feature vector,
    ensuring stability even when features vary widely.
    """
    # Compute per‑sample step size
    norm_sq = np.sum(feature_matrix ** 2, axis=1, keepdims=True) + epsilon
    step = (learning_rate / norm_sq) * error[:, None] * feature_matrix
    return step.mean(axis=0)  # average contribution over the batch


def hybrid_batch_update(
    rewards: List[float],
    feature_matrix: np.ndarray,
    weights: np.ndarray,
    base_lr: float = 0.1,
    gini_clip: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Update weights using a batch of rewards.
    The Gini coefficient modulates the global learning rate,
    but is clipped to avoid pathological amplification.
    """
    gini = gini_coefficient(rewards)
    gini = np.clip(gini, *gini_clip)          # keep within sensible bounds

    predictions = nlms_predict(feature_matrix, weights)
    error = np.array(rewards, dtype=float) - predictions

    # Adaptive learning rate = base_lr * (1 + gini)
    adaptive_lr = base_lr * (1.0 + gini)

    weight_delta = nlms_update(feature_matrix, error, adaptive_lr)
    return weights + weight_delta


# ----------------------------------------------------------------------
# Example usage – deterministic seed for reproducibility
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic element vectors (could be any high‑dimensional descriptors)
    elements = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.5, 3.5, 4.5]  # extra node to illustrate graph growth
    ]

    # Corresponding VRAM‑related weights (positive scalars)
    vram_weights = [1.0, 2.0, 3.0, 1.5]

    graph = build_graph(elements, vram_weights)
    feature_matrix = compute_feature_matrix(graph, temperature=22.0)

    # Initialise a weight vector matching the feature dimension (3)
    weights = np.random.rand(feature_matrix.shape[1])

    # Simulated reward batch – could come from a bandit arm pull
    rewards = [0.8, 1.2, 0.5, 1.0]

    updated_weights = hybrid_batch_update(rewards, feature_matrix, weights, base_lr=0.05)

    print("Initial weights :", weights)
    print("Updated weights :", updated_weights)