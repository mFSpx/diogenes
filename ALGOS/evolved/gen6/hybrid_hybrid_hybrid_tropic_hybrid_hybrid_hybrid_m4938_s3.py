# DARWIN HAMMER — match 4938, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_tropical_maxp_hybrid_hybrid_hybrid_m673_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_semant_hybrid_korpus_text_h_m1384_s0.py (gen5)
# born: 2026-05-29T23:58:58Z

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Tropical (max-plus) algebra primitives 
# ----------------------------------------------------------------------
def t_add(x, y):
    return np.maximum(x, y)


def t_mul(x, y):
    return np.add(x, y)


def t_matmul(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    return np.max(A[:, np.newaxis] + B, axis=1)


# ----------------------------------------------------------------------
# Morphology & recovery priority 
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
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Semantic neighbors 
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
    modulated_similarities = [s * recovery_priority(morphology) for s in similarities]
    return modulated_similarities


# ----------------------------------------------------------------------
# Hybrid Tropical-Max-Plus & Semantic-Bayesian Curvature Scheduler
# ----------------------------------------------------------------------
def compute_semantic_weights(edge_lengths: np.ndarray, doc_vectors: list[list[float]], morphology: Morphology) -> np.ndarray:
    similarities = semantic_neighbors(doc_vectors, morphology)
    semantic_weights = np.array([s / sum(similarities) for s in similarities])
    weighted_edge_lengths = edge_lengths * semantic_weights[:, np.newaxis]
    return weighted_edge_lengths


def hybrid_scheduler(edge_lengths: np.ndarray, doc_vectors: list[list[float]], morphology: Morphology) -> np.ndarray:
    weighted_edge_lengths = compute_semantic_weights(edge_lengths, doc_vectors, morphology)
    utilities = t_matmul(weighted_edge_lengths, weighted_edge_lengths.T)
    return utilities


def tree_metrics(node_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def improved_hybrid_scheduler(edge_lengths: np.ndarray, doc_vectors: list[list[float]], morphology: Morphology) -> np.ndarray:
    num_nodes = edge_lengths.shape[0]
    node_similarities = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity = _cosine(doc_vectors[i], doc_vectors[j])
            node_similarities[i, j] = similarity
            node_similarities[j, i] = similarity
    node_similarities = node_similarities * recovery_priority(morphology)
    weighted_edge_lengths = edge_lengths * node_similarities
    utilities = t_matmul(weighted_edge_lengths, weighted_edge_lengths.T)
    return utilities


if __name__ == "__main__":
    node_positions = np.random.rand(5, 3)
    adjacency_list, edge_lengths, root_to_node_distances = tree_metrics(node_positions)
    doc_vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    utilities = improved_hybrid_scheduler(edge_lengths, doc_vectors, morphology)
    print(utilities)