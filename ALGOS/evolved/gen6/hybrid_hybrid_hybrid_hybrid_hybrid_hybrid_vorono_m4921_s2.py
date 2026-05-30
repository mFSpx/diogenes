# DARWIN HAMMER — match 4921, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s4.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s4.py (gen5)
# born: 2026-05-29T23:58:49Z

"""
Hybrid Voronoi Leader-Tree XGBoost-Regret Algorithm (HVLTXR)

This module fuses two parent algorithms:

* **Parent A** – Hybrid Leader-Tree XGBoost-Regret Algorithm (HLTXR)
  (tropical max-plus broadcast, Hoeffding-bound split test, 
  simulated-annealing acceptance, logistic gradient/hessian, 
  MinHash similarity & Shannon-entropy regulariser).
* **Parent B** – Hybrid Voronoi Partition Algorithm 
  (euclidean distance, Voronoi cell assignment, regex-based feature extraction).

The mathematical bridge between the two parents lies in interpreting 
the Voronoi seeds as leaders in the leader-tree structure. 
The feature scores extracted from text using regex patterns 
are used to compute the similarity between a node's token set 
and a reference set, which is then used to regularize the 
logistic loss in the HLTXR algorithm.

The three public functions below demonstrate the integrated workflow:
* `voronoi_leader_tree_step` – one iteration of the fused algorithm.
* `compute_adjusted_grad_hess` – logistic gradient/hessian with 
  similarity-entropy regularisation and Voronoi-based leader selection.
* `tropical_matmul_with_voronoi` – max-plus matrix multiplication 
  with Voronoi-based leader selection.
"""

import numpy as np
import math
import random
import re
from collections import Counter
from typing import Mapping, Hashable, Set, Tuple, List, Dict

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
TokenSet = Set[str]

# ----------------------------------------------------------------------
# Tropical (max-plus) matrix operations
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max-plus matrix multiplication: (A ⊗ B)[i,j] = max_k (A[i,k] + B[k,j])"""
    n, m = A.shape
    p = B.shape[1]
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            C[i, j] = max(A[i, k] + B[k, j] for k in range(m))
    return C

# ----------------------------------------------------------------------
# Geometry utilities
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
    """L2-normalize a feature vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ----------------------------------------------------------------------
# Hybrid Voronoi Leader-Tree XGBoost-Regret Algorithm
# ----------------------------------------------------------------------
def compute_adjusted_grad_hess(
    leader: np.ndarray, 
    points: np.ndarray, 
    seeds: np.ndarray, 
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adjusted gradient and hessian for logistic loss with 
    similarity-entropy regularisation and Voronoi-based leader selection.
    """
    regions = assign(points, seeds)
    grad = np.zeros_like(leader)
    hess = np.zeros_like(leader)
    for i, region in enumerate(regions):
        pts_in_region = points[region == 1]
        if pts_in_region.size > 0:
            leader_in_region = leader[i]
            feat_scores = compute_feature_scores(" ".join(map(str, pts_in_region)))
            similarity = np.dot(normalize_feature_vector(np.array(list(feat_scores.values()))), 
                                normalize_feature_vector(leader_in_region))
            entropy = -np.sum(np.log2(np.array(list(feat_scores.values())) + 1e-10) * np.array(list(feat_scores.values())))
            adjusted_grad = grad[i] + alpha * similarity * entropy
            adjusted_hess = hess[i] + alpha * similarity * entropy
            grad[i] = adjusted_grad
            hess[i] = adjusted_hess
    return grad, hess

def tropical_matmul_with_voronoi(
    A: np.ndarray, 
    B: np.ndarray, 
    points: np.ndarray, 
    seeds: np.ndarray
) -> np.ndarray:
    """
    Max-plus matrix multiplication with Voronoi-based leader selection.
    """
    leader = np.argmax(A, axis=0)
    regions = assign(points, seeds)
    C = np.zeros_like(A)
    for i, region in enumerate(regions):
        pts_in_region = points[region == 1]
        if pts_in_region.size > 0:
            leader_in_region = leader[i]
            C[leader_in_region] = max(C[leader_in_region], 
                                      np.max(A[:, leader_in_region]) + np.max(B[leader_in_region]))
    return C

def voronoi_leader_tree_step(
    A: np.ndarray, 
    B: np.ndarray, 
    points: np.ndarray, 
    seeds: np.ndarray, 
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One iteration of the hybrid Voronoi leader-tree XGBoost-regret algorithm.
    """
    leader = np.argmax(A, axis=0)
    grad, hess = compute_adjusted_grad_hess(leader, points, seeds, alpha)
    C = tropical_matmul_with_voronoi(A, B, points, seeds)
    return grad, C

if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    points = np.random.rand(100, 10)
    seeds = np.random.rand(10, 10)
    alpha = 0.1
    grad, C = voronoi_leader_tree_step(A, B, points, seeds, alpha)
    print(grad)
    print(C)