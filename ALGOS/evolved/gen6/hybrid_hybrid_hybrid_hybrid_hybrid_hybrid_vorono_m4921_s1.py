# DARWIN HAMMER — match 4921, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s4.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s4.py (gen5)
# born: 2026-05-29T23:58:49Z

"""
This module fuses the two parent algorithms:

* **Parent A** – Hybrid Leader–Tree XGBoost‑Regret Algorithm (HLTXR)
* **Parent B** – Hybrid Voronoi Partition Hybrid Algorithm

The mathematical bridge is formed by using the XGBoost utilities (gradient, hessian, split‑gain) 
from Parent A to inform the Voronoi partitioning in Parent B. Specifically, the broadcast strength 
vector `b` produced by the tropical max‑plus propagation in Parent A is used to weight the 
Voronoi seeds, influencing the region assignments.
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
    p = B.shape[1]
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            result[i, j] = max(A[i, k] + B[k, j] for k in range(m))
    return result

# ----------------------------------------------------------------------
# Geometry utilities (shared)
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
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def weighted_voronoi(points: np.ndarray, seeds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Perform weighted Voronoi partitioning.

    :param points: Points to be assigned to Voronoi regions.
    :param seeds: Seeds (centroids) of the Voronoi regions.
    :param weights: Weights for the seeds, influencing the region assignments.
    :return: Binary region matrix R of shape (n_seeds, n_points) where
             R[i, j] == 1 iff point j belongs to the Voronoi cell of seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        distances = [euclidean(p, seeds[i]) * weights[i] for i in range(n_seeds)]
        i = int(np.argmin(distances))
        regions[i, j] = 1
    return regions

def hybrid_leader_tree_step(points: np.ndarray, seeds: np.ndarray, graph: Graph) -> np.ndarray:
    """
    Perform one iteration of the hybrid leader tree algorithm.

    :param points: Points to be assigned to Voronoi regions.
    :param seeds: Seeds (centroids) of the Voronoi regions.
    :param graph: Graph representing the relationships between the points.
    :return: Binary region matrix R of shape (n_seeds, n_points) where
             R[i, j] == 1 iff point j belongs to the Voronoi cell of seed i.
    """
    # Compute the broadcast strength vector
    A = np.array([[1 if j in graph[i] else 0 for j in graph] for i in graph])
    b = np.ones((len(graph), 1))
    tropical_b = tropical_matmul(A, b)

    # Normalize the broadcast strength vector to obtain weights
    weights = normalize_feature_vector(tropical_b.flatten())

    # Perform weighted Voronoi partitioning
    regions = weighted_voronoi(points, seeds, weights)

    return regions

if __name__ == "__main__":
    points = np.random.rand(10, 2)
    seeds = np.random.rand(3, 2)
    graph = {i: {j for j in range(10) if j != i} for i in range(10)}
    regions = hybrid_leader_tree_step(points, seeds, graph)
    print(regions)