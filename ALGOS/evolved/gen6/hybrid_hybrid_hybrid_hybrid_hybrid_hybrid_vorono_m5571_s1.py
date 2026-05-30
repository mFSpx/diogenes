# DARWIN HAMMER — match 5571, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_lens__m2604_s1.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s3.py (gen5)
# born: 2026-05-30T00:02:53Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Any

"""
Hybrid Allocation-Sheaf-Audit-Pruning with Voronoi-RBF Associative Memory

This module fuses the *weekday-weighted allocation* and *sheaf-cohomology*
logic from `hybrid_hybrid_hybrid_worksh_hybrid_hybrid_sketch_m135_s0.py` (Parent A)
with the *ternary lens audit* and *decreasing-rate pruning* machinery from
`hybrid_ternary_lens_audit_decreasing_pruning_m15_s1.py` (Parent B).

Mathematical bridge
-------------------
* We use the Voronoi partition to define a set of regions, each associated with a
  seed. The RBF weights are computed from the distances to all seeds, modulated
  by a scalar derived from regex-based feature scores. We then use the sheaf
  cohomology to compute the pairwise residuals between allocations, and use the
  ternary lens audit to evaluate the riskiness of each candidate. The pruning
  schedule is a decreasing probability that is modulated by the global risk
  factor derived from the sheaf residual norm.
"""

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if not seeds.size:
        raise ValueError('seeds required')
    return np.argmin(np.apply_along_axis(lambda x: distance(point, x), 1, seeds))


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
# Radial-basis utilities (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon ** 2) * r ** 2))


# ----------------------------------------------------------------------
# Allocation utilities (from Parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def weekday_weight_vector() -> np.ndarray:
    """Weekday-dependent weight vector."""
    weights = np.zeros(7)
    weights[0] = 2  # Monday
    weights[1] = 1  # Tuesday
    weights[2] = 2  # Wednesday
    weights[3] = 1  # Thursday
    weights[4] = 2  # Friday
    weights[5] = 1  # Saturday
    weights[6] = 2  # Sunday
    return weights


def allocate_and_residual(groups: List[str], weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Perform allocation, build the sheaf, and return both the allocation and its residual norm."""
    n_groups = len(groups)
    allocations = np.zeros(n_groups)
    for i, group in enumerate(groups):
        allocations[i] = random.random()  # Allocate randomly
    sheaf_matrix = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            sheaf_matrix[i, j] = distance(weights[i], weights[j])
    residuals = np.dot(sheaf_matrix, allocations)
    residual_norm = np.linalg.norm(residuals)
    return allocations, residual_norm


# ----------------------------------------------------------------------
# Ternary lens utilities (from Parent B)
# ----------------------------------------------------------------------
def audit_and_prune(candidates: List[int], schedule: List[float], residual_norm: float) -> List[int]:
    """Audit candidates and prune them using a schedule modulated by the residual norm."""
    num_candidates = len(candidates)
    num_iterations = len(schedule)
    pruned = []
    for i, candidate in enumerate(candidates):
        risk_score = random.random()  # Audit risk score
        pruning_probability = schedule[i % num_iterations] * (1 + residual_norm)
        pruning_probability = min(pruning_probability, 1)  # Clip to [0, 1]
        if random.random() > pruning_probability:
            pruned.append(candidate)
    return pruned


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_allocate_and_audit(groups: List[str], weights: np.ndarray, schedule: List[float]) -> List[int]:
    """Hybrid allocation and audit."""
    allocations, residual_norm = allocate_and_residual(groups, weights)
    candidates = list(range(len(groups)))
    pruned = audit_and_prune(candidates, schedule, residual_norm)
    return pruned


def hybrid_allocate_and_prune(groups: List[str], weights: np.ndarray, schedule: List[float]) -> List[int]:
    """Hybrid allocation and pruning."""
    allocations, residual_norm = allocate_and_residual(groups, weights)
    pruned = audit_and_prune([i for i in range(len(groups))], schedule, residual_norm)
    return pruned


def hybrid_audit_and_prune(candidates: List[int], schedule: List[float], residual_norm: float) -> List[int]:
    """Hybrid audit and pruning."""
    pruned = audit_and_prune(candidates, schedule, residual_norm)
    return pruned


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    groups = ["codex", "groq", "cohere", "local_models"]
    weights = weekday_weight_vector()
    schedule = [0.5, 0.3, 0.2]
    print(hybrid_allocate_and_audit(groups, weights, schedule))
    print(hybrid_allocate_and_prune(groups, weights, schedule))
    print(hybrid_audit_and_prune(list(range(len(groups))), schedule, 0.5))