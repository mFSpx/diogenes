# DARWIN HAMMER — match 3103, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_label__hybrid_hybrid_hybrid_m531_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s5.py (gen6)
# born: 2026-05-29T23:47:56Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
from random import random
from sys import exit
from pathlib import Path
from collections import Counter, defaultdict
from typing import Sequence, Tuple, Hashable, Iterable, Dict, List, Callable

# ----------------------------------------------------------------------
# Basic Types
# ----------------------------------------------------------------------
FeatureVec = Sequence[float]
Point = Tuple[float, float]
Node = Hashable
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# RBF Similarity Utilities 
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_rbf_similarity_matrix(
    features: List[FeatureVec], epsilon: float = 1.0
) -> np.ndarray:
    """Full RBF similarity matrix for a list of feature vectors."""
    n = len(features)
    sim = np.empty((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[i], features[j])
            similarity = gaussian(dist, epsilon)
            sim[i, j] = similarity
            sim[j, i] = similarity
    return sim

# ----------------------------------------------------------------------
# Graph Utilities 
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean length of an edge between two points."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def build_adjacency(
    nodes: Dict[Node, Point], edges: Iterable[Edge]
) -> Dict[Node, List[Tuple[Node, float]]]:
    """Adjacency list with Euclidean edge lengths."""
    adj: Dict[Node, List[Tuple[Node, float]]] = {n: [] for n in nodes}
    for u, v in edges:
        w = length(nodes[u], nodes[v])
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj

def weighted_tree_distances(
    nodes: Dict[Node, Point],
    edges: Iterable[Edge],
    weights: Dict[Edge, float]
) -> Dict[Node, Dict[Node, float]]:
    """Weighted tree distances."""
    dist: Dict[Node, Dict[Node, float]] = {n: {} for n in nodes}
    for u, v in edges:
        w = weights[(u, v)]
        dist[u][v] = w
        dist[v][u] = w
    return dist

# ----------------------------------------------------------------------
# Hybrid Labeling and Graph Algorithm
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float

@dataclass
class LabelError:
    error_probability: float

def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], float]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco

@labeling_function("Hybrid Recovery Priority")
def hybrid_recovery_priority(morphology: Morphology, label_error: LabelError) -> float:
    """Hybrid recovery priority based on morphology and label error."""
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    flatness = flatness_index(morphology.length, morphology.width, morphology.height)
    recovery_priority = (1 - sphericity) * (1 - flatness) + label_error.error_probability
    return recovery_priority

def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity index."""
    return (length * width * height) ** (1/3) / ((length ** 2 + width ** 2 + height ** 2) / 3) ** (1/2)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness index."""
    return min(width, height) / max(width, height)

def stylometric_features(text: str) -> FeatureVec:
    """Stylometric features from raw text."""
    # Simple stylometric features: word count, character count, average word length
    words = text.split()
    return [len(words), len(text), sum(len(word) for word in words) / len(words) if words else 0]

def hybrid_graph_algorithm(nodes: Dict[Node, Point], edges: Iterable[Edge], weights: Dict[Edge, float], 
                           morphologies: Dict[Node, Morphology], label_errors: Dict[Node, LabelError]) -> Dict[Node, Dict[Node, float]]:
    """Hybrid graph algorithm with recovery priority and stylometric features."""
    weighted_dist = weighted_tree_distances(nodes, edges, weights)
    for u, v in edges:
        w = weights[(u, v)]
        recovery_priority_u = hybrid_recovery_priority(morphologies[u], label_errors[u])
        recovery_priority_v = hybrid_recovery_priority(morphologies[v], label_errors[v])
        stylometric_features_u = stylometric_features(f"raw text for node {u}")
        stylometric_features_v = stylometric_features(f"raw text for node {v}")
        similarity = 1 - euclidean(stylometric_features_u, stylometric_features_v) / max(map(max, [stylometric_features_u, stylometric_features_v]))
        weighted_dist[u][v] = w * (recovery_priority_u + recovery_priority_v) / 2 * similarity
        weighted_dist[v][u] = w * (recovery_priority_u + recovery_priority_v) / 2 * similarity
    return weighted_dist

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (3, 4), 'C': (6, 8)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    weights = {('A', 'B'): 2.0, ('B', 'C'): 3.0, ('C', 'A'): 4.0}
    morphologies = {'A': Morphology(1.0, 2.0, 3.0), 'B': Morphology(4.0, 5.0, 6.0), 'C': Morphology(7.0, 8.0, 9.0)}
    label_errors = {'A': LabelError(0.1), 'B': LabelError(0.2), 'C': LabelError(0.3)}
    hybrid_dist = hybrid_graph_algorithm(nodes, edges, weights, morphologies, label_errors)
    for u in nodes:
        print(f"Node {u}:")
        for v, w in hybrid_dist[u].items():
            print(f"  {v}: {w}")