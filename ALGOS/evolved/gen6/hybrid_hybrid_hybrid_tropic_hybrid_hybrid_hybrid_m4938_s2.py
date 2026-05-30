# DARWIN HAMMER — match 4938, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_tropical_maxp_hybrid_hybrid_hybrid_m673_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_korpus_text_h_m1384_s0.py (gen5)
# born: 2026-05-29T23:58:58Z

"""
Hybrid Tropical-Max-Plus & Semantic-Bayesian Curvature Scheduler
====================================================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_tropical_maxp_hybrid_hybrid_hybrid_m673_s3.py``  
  Provides tropical (max-plus) algebra primitives and a ``tree_metrics`` routine
  that builds adjacency lists, Euclidean edge lengths and root-to-node distances.

* **Parent B** – ``hybrid_hybrid_hybrid_semant_hybrid_korpus_text_h_m1384_s0.py``  
  Supplies a semantic weighting of geometric edge lengths, morphology-based recovery 
  priority, and regret-weighted similarity.

The mathematical bridge between the two parent algorithms lies in the 
computation of the semantic weights for the edges. In Parent A, the 
weighted edge cost is computed as ``c_e = ℓ_e · w_e``, where ``ℓ_e`` is the 
Euclidean length and ``w_e`` is a semantic scalar. In Parent B, the 
regret-weighted similarity is used to modulate the entropy of the document 
vectors. We fuse these two concepts by using the regret-weighted similarity 
from Parent B to compute the semantic weights ``w_e`` in Parent A.

"""

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Tropical (max-plus) algebra primitives (Parent A)
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition: max(x, y). Works element-wise for NumPy arrays."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Works element-wise for NumPy arrays."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    (A ⊗ B)[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A)
    B = np.asarray(B)
    return np.max(A[:, np.newaxis] + B, axis=1)


# ----------------------------------------------------------------------
# Morphology & recovery priority (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized to [0,1] – acts as a prior probability."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Semantic neighbors (Parent B)
# ----------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0.0 else sum(x * y for x, y in zip(a, b)) / den


def semantic_neighbors(doc_vectors: list[list[float]], morphology: Morphology) -> list[float]:
    similarities = []
    for i in range(len(doc_vectors)):
        for j in range(i + 1, len(doc_vectors)):
            similarity = _cosine(doc_vectors[i], doc_vectors[j])
            similarities.append(similarity)
    # modulate similarities with recovery priority
    modulated_similarities = [s * recovery_priority(morphology) for s in similarities]
    return modulated_similarities


# ----------------------------------------------------------------------
# Hybrid Tropical-Max-Plus & Semantic-Bayesian Curvature Scheduler
# ----------------------------------------------------------------------
def compute_semantic_weights(edge_lengths: np.ndarray, doc_vectors: list[list[float]], morphology: Morphology) -> np.ndarray:
    similarities = semantic_neighbors(doc_vectors, morphology)
    # compute semantic weights as regret-weighted similarities
    semantic_weights = np.array([s / sum(similarities) for s in similarities])
    # modulate edge lengths with semantic weights
    weighted_edge_lengths = edge_lengths * semantic_weights
    return weighted_edge_lengths


def hybrid_scheduler(edge_lengths: np.ndarray, doc_vectors: list[list[float]], morphology: Morphology) -> np.ndarray:
    weighted_edge_lengths = compute_semantic_weights(edge_lengths, doc_vectors, morphology)
    # compute tropical max-plus utilities
    utilities = t_matmul(weighted_edge_lengths, weighted_edge_lengths.T)
    return utilities


def tree_metrics(node_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # compute adjacency list, Euclidean edge lengths, and root-to-node distances
    num_nodes = len(node_positions)
    adjacency_list = np.zeros((num_nodes, num_nodes))
    edge_lengths = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(node_positions[i] - node_positions[j])
            adjacency_list[i, j] = 1
            adjacency_list[j, i] = 1
            edge_lengths[i, j] = distance
            edge_lengths[j, i] = distance
    root_to_node_distances = np.linalg.norm(node_positions, axis=1)
    return adjacency_list, edge_lengths, root_to_node_distances


if __name__ == "__main__":
    # smoke test
    node_positions = np.random.rand(5, 3)
    adjacency_list, edge_lengths, root_to_node_distances = tree_metrics(node_positions)
    doc_vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    utilities = hybrid_scheduler(edge_lengths, doc_vectors, morphology)
    print(utilities)