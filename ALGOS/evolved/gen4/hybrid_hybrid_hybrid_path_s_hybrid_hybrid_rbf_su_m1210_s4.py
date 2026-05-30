# DARWIN HAMMER — match 1210, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (gen2)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:34:26Z

"""
Hybrid Module: Path‑Signature + RBF‑Surrogate Fusion

Parents:
- hybrid_hybrid_path_signatur_hybrid_krampus_brain_m25_s3.py (Path‑signature & KAN‑style embedding)
- hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (RBF surrogate similarity on node feature vectors)

Mathematical Bridge:
Both parents operate on high‑dimensional vectors derived from free‑form text.
The bridge is formed by treating the master‑vector extracted from each text as a node
feature in a graph.  An RBF kernel provides a similarity matrix **K** among these
vectors.  The lead‑lag transformation creates a causally‑aware augmented path
𝑋̃ from the ordered vectors; its level‑1 and level‑2 signatures **S** capture the
algebraic geometry of the sequence.  Finally the scalar weight
w = mean(K) modulates the signature, yielding a fused embedding
E = w·S that simultaneously encodes (i) the path‑signature algebra and (ii)
the global similarity structure supplied by the RBF surrogate.

The module implements this unified pipeline with clear, reusable functions.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent B – deterministic master‑vector extractor
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swar"
    ]
    # Pad the key list to a reasonable length if truncated
    while len(keys) < 20:
        keys.append(f"dummy_feature_{len(keys)}")
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Helper: convert dict feature to ordered numpy vector
# ----------------------------------------------------------------------
def dict_to_vector(d: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a deterministic ordered vector."""
    ordered_keys = sorted(d.keys())
    return np.array([d[k] for k in ordered_keys], dtype=np.float64)


# ----------------------------------------------------------------------
# Parent A – lead‑lag transform and level‑1/2 signatures
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag augmentation.
    Input: path of shape (n, d)
    Output: augmented path of shape (2*n-1, 2*d)
    """
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array")
    n, d = path.shape
    aug = np.empty((2 * n - 1, 2 * d), dtype=path.dtype)
    for i in range(n - 1):
        # (lead, lag) = (x_i, x_i)
        aug[2 * i] = np.concatenate([path[i], path[i]])
        # (lead, lag) = (x_i, x_{i+1})
        aug[2 * i + 1] = np.concatenate([path[i], path[i + 1]])
    # final duplicate
    aug[-1] = np.concatenate([path[-1], path[-1]])
    return aug


def signature_level1_and_2(aug_path: np.ndarray) -> np.ndarray:
    """
    Compute flattened level‑1 (total increment) and level‑2 (iterated integral)
    signatures of an augmented path.
    Returns a 1‑D vector of length d + d*d.
    """
    if aug_path.ndim != 2:
        raise ValueError("aug_path must be a 2‑D array")
    d = aug_path.shape[1]
    # level‑1: total increment
    level1 = aug_path[-1] - aug_path[0]          # shape (d,)
    # level‑2: sum of outer products of successive increments
    increments = np.diff(aug_path, axis=0)       # shape (m-1, d)
    level2 = np.einsum('ij,ik->jk', increments, increments)  # shape (d, d)
    # flatten level‑2 (row‑major)
    level2_flat = level2.reshape(-1)
    return np.concatenate([level1, level2_flat])


# ----------------------------------------------------------------------
# RBF surrogate utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel_matrix(features: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute the symmetric RBF kernel matrix K_{ij} = exp(-epsilon^2 * ||x_i - x_j||^2).
    features: (n, d) array.
    """
    n = features.shape[0]
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            r = euclidean(features[i], features[j])
            val = gaussian(r, epsilon)
            K[i, j] = K[j, i] = val
    return K


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_weighted_signature(texts: List[str],
                               epsilon: float = 1.0) -> np.ndarray:
    """
    Full hybrid pipeline:
    1. Extract deterministic master vectors from texts.
    2. Form a lead‑lag augmented path and compute its level‑1/2 signatures.
    3. Build an RBF similarity matrix over the raw vectors.
    4. Modulate the signature by the mean similarity weight.
    Returns the weighted signature embedding.
    """
    # 1. Feature extraction
    raw_vectors = np.stack([dict_to_vector(extract_full_features(t)) for t in texts])
    # 2. Lead‑lag + signature
    aug_path = lead_lag_transform(raw_vectors)
    signature = signature_level1_and_2(aug_path)          # shape (d + d*d,)
    # 3. RBF similarity matrix
    K = rbf_kernel_matrix(raw_vectors, epsilon=epsilon)
    # 4. Global similarity weight (mean off‑diagonal similarity)
    n = K.shape[0]
    if n > 1:
        weight = (K.sum() - n) / (n * (n - 1))  # exclude diagonal 1's
    else:
        weight = 1.0
    # Weighted embedding
    return signature * weight


def rbf_surrogate_predict(train_features: np.ndarray,
                         train_targets: np.ndarray,
                         query_features: np.ndarray,
                         epsilon: float = 1.0) -> np.ndarray:
    """
    Simple RBF surrogate: predict target for each query vector as a
    kernel‑weighted average of training targets.
    """
    K_train = rbf_kernel_matrix(train_features, epsilon=epsilon)  # (m, m)
    # Solve for coefficients α in K_train α = y (ridge regularisation omitted for brevity)
    alpha = np.linalg.solve(K_train + 1e-8 * np.eye(K_train.shape[0]), train_targets)
    # Compute kernel between queries and training points
    m = train_features.shape[0]
    q = query_features.shape[0]
    K_qt = np.empty((q, m), dtype=np.float64)
    for i in range(q):
        for j in range(m):
            K_qt[i, j] = gaussian(euclidean(query_features[i], train_features[j]), epsilon)
    # Prediction
    return K_qt @ alpha


def hybrid_graph_embedding(texts: List[str],
                           epsilon: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a graph‑level embedding by:
    * Computing the weighted signature (as in compute_weighted_signature)
    * Treating each text as a node; using the RBF kernel matrix as adjacency weights.
    Returns (node_embedding, adjacency_matrix).
    """
    # Node features
    node_features = np.stack([dict_to_vector(extract_full_features(t)) for t in texts])
    # Global adjacency via RBF kernel
    adjacency = rbf_kernel_matrix(node_features, epsilon=epsilon)
    # Node embedding via weighted signature (same for all nodes in this simple demo)
    node_emb = compute_weighted_signature(texts, epsilon=epsilon)
    return node_emb, adjacency


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the midst of chaos, there is also opportunity.",
        "All models are wrong, but some are useful."
    ]
    embed = compute_weighted_signature(sample_texts, epsilon=0.5)
    print("Weighted signature shape:", embed.shape)
    node_emb, adj = hybrid_graph_embedding(sample_texts, epsilon=0.5)
    print("Node embedding shape:", node_emb.shape)
    print("Adjacency matrix:\n", adj)