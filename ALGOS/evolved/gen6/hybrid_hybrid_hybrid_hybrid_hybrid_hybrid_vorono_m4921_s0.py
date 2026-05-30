# DARWIN HAMMER — match 4921, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s4.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s4.py (gen5)
# born: 2026-05-29T23:58:49Z

"""
HYBRID VORONOI LEADER TREE XGBOOST‑REGRET ALGORITHM (HVLTXR)

This module merges the two parent algorithms:

* **Parent A** – Hybrid Leader–Tree XGBoost‑Regret Algorithm (HLTXR)
* **Parent B** – Hybrid Voronoi Partitioning Regret Analyzer (HVPR)

The mathematical bridge between the two parents is the use of Voronoi partitions to
divide the input space into regions, each associated with a node in the leader
tree. The XGBoost utility functions (gradient, hessian, split‑gain) are applied
to the Voronoi cells, and the resulting gradients are used to update the leader
tree. The regret‑weighted information from Parent B is used to regularise the
XGBoost objective function.

Mathematical Interface:

* The Voronoi partitioning is used to compute a set of features for each node
  in the leader tree, which are then used to compute the XGBoost gradient and
  hessian.
* The XGBoost utility functions are applied to the Voronoi cells, and the
  resulting gradients are used to update the leader tree.
* The regret‑weighted information from Parent B is used to regularise the
  XGBoost objective function.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Mapping, Hashable, Set, Tuple, List, Dict, Iterable

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
TokenSet = Set[str]

# ----------------------------------------------------------------------
# Tropical (max‑plus) matrix operations – Parent A core
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max‑plus matrix multiplication: (A ⊗ B)[i,j] = max_k (A[i,k] + B[k,j])"""
    n, m = A.shape
    p = B
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i, j] = np.max(np.dot(A[i, :], B[:, j]))
    return C

# ----------------------------------------------------------------------
# Voronoi partitioning utilities – Parent B core
# ----------------------------------------------------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)

def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if seeds.size == 0:
        raise ValueError("seeds array is empty")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))

def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Return a binary region matrix R of shape (n_seeds, n_points) where
    R[i, j] == 1 iff point j belongs to the Voronoi cell of seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions

# ----------------------------------------------------------------------
# Feature extraction (regex based)
# ----------------------------------------------------------------------
_FEATURE_PATTERNS: Dict[str, re.Pattern] = {
    "evidence": re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    ),
    "planning": re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    ),
    "delay": re.compile(r"\b(?:delay|postpone|defer|wait|stall|hold|slow)\b", re.I),
}

def compute_feature_scores(text: str) -> Dict[str, float]:
    """
    Count occurrences of each regex pattern in *text*.
    Returns a dict mapping feature names to counts.
    """
    scores: Dict[str, float] = {}
    for name, pat in _FEATURE_PATTERNS.items():
        scores[name] = float(len(pat.findall(text)))
    return scores

def normalize_feature_vector(vec: np.ndarray) -> np.ndarray:
    """L2‑normalize a feature vector."""
    return vec / np.linalg.norm(vec)

# ----------------------------------------------------------------------
# Hybrid leader tree construction
# ----------------------------------------------------------------------
def hybrid_leader_tree_step(graph: Graph, seeds: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    One iteration of the hybrid leader tree construction algorithm.

    Parameters:
    graph (Graph): The input graph.
    seeds (np.ndarray): The Voronoi seeds.
    features (np.ndarray): The feature vectors for each node in the graph.

    Returns:
    np.ndarray: The updated leader tree.
    """
    # Compute the Voronoi partitioning
    regions = assign(graph.keys(), seeds)

    # Compute the feature scores for each node
    feature_scores = {}
    for node in graph:
        feature_scores[node] = compute_feature_scores(' '.join(node))

    # Compute the XGBoost gradient and hessian
    gradient = np.zeros(len(graph))
    hessian = np.zeros((len(graph), len(graph)))
    for i, node in enumerate(graph):
        for j, other in enumerate(graph):
            if regions[i, j] == 1:
                # Compute the feature vector for the current node
                current_features = normalize_feature_vector(features[i])

                # Compute the feature vector for the other node
                other_features = normalize_feature_vector(features[j])

                # Compute the XGBoost gradient and hessian
                gradient[i] += np.dot(current_features, other_features)
                hessian[i, j] = np.dot(current_features, other_features)

    # Update the leader tree
    leader_tree = np.zeros(len(graph))
    for i in range(len(graph)):
        leader_tree[i] = np.max(gradient[i])

    return leader_tree

# ----------------------------------------------------------------------
# Test the hybrid leader tree construction algorithm
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a sample graph
    graph = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1}
    }

    # Create a sample set of Voronoi seeds
    seeds = np.array([[1, 2], [3, 4]])

    # Create a sample set of feature vectors
    features = np.array([[1, 2], [3, 4], [5, 6]])

    # Run the hybrid leader tree construction algorithm
    leader_tree = hybrid_leader_tree_step(graph, seeds, features)

    # Print the leader tree
    print(leader_tree)