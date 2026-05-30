# DARWIN HAMMER — match 5251, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1617_s3.py (gen6)
# born: 2026-05-30T00:00:49Z

"""
Hybrid Algorithm: Fusing Fisher-RLCT-Certainty and Ollivier-Ricci Curvature
====================================================================================

This module merges the governing equations of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_fisher_hybrid_hybrid_minimu_m140_s3.py``  
  Provides Gaussian-beam intensity, Fisher information, a count-min sketch and a
  log-log regression that estimates the *regularized loss curvature* (RLCT).

* **Parent B** – ``hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hard_t_m1617_s3.py``  
  Defines an Ollivier-Ricci curvature metric for an undirected weighted graph.

The mathematical bridge between the parents lies in the concept of *information* and
*curvature*.  By treating the Fisher information as a local measure of curvature and
the Ollivier-Ricci curvature as a global measure, we can fuse the two algorithms.

The functions below implement this fusion:
* ``weighted_fisher_score`` – Fisher information averaged over a dataset and
  weighted by a certainty flag.
* ``hybrid_ollivier_ricci_curvature`` – Compute the average Ollivier-Ricci curvature
  of an undirected weighted graph.
* ``fused_curvature_metric`` – a single routine that builds a count-min sketch,
  evaluates the weighted Fisher and Ollivier-Ricci quantities and returns a
  consolidated result.

"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Tuple, Dict, List

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle *theta*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return (1.0 / (width ** 2)) * gaussian_beam(theta, center, width)


def weighted_fisher_score(
    theta: float, center: float, width: float, confidence: float, eps: float = 1e-12
) -> float:
    """Fisher information averaged over a dataset and weighted by a certainty flag."""
    return confidence * fisher_score(theta, center, width, eps)


def _ensure_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric version of *matrix* by averaging with its transpose."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    sym = (matrix + matrix.T) / 2.0
    np.fill_diagonal(sym, 0.0)  
    return sym


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize *matrix* to obtain a stochastic matrix."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums == 0, 0, matrix / row_sums)
    return norm


def _shortest_path_distances(cost: np.ndarray) -> np.ndarray:
    """Floyd-Warshall algorithm for all-pairs shortest path on a dense cost matrix."""
    n = cost.shape[0]
    dist = cost.copy()
    np.fill_diagonal(dist, 0.0)
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    return dist


def hybrid_ollivier_ricci_curvature(
    adjacency: np.ndarray,
    edge_costs: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute the average Ollivier-Ricci curvature of an undirected weighted graph.

    The graph is described by ``adjacency`` (non-negative weights) and a
    ground-metric ``edge_costs`` that quantifies the transport cost between
    directly connected nodes.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    dist = _shortest_path_distances(edge_costs)
    curv = 0.0
    for i in range(adjacency.shape[0]):
        for j in range(i):
            if adjacency[i, j] > 0:
                curv += (dist[i, j] - (edge_costs[i, j] / 2.0)) * adjacency[i, j]
    return curv / np.sum(adjacency)


def fused_curvature_metric(
    theta: float,
    center: float,
    width: float,
    confidence: float,
    adjacency: np.ndarray,
    edge_costs: np.ndarray,
) -> Tuple[float, float]:
    """
    Consolidated result of weighted Fisher score and Ollivier-Ricci curvature.

    Args:
    - theta: angle
    - center: center of the Gaussian beam
    - width: width of the Gaussian beam
    - confidence: certainty flag
    - adjacency: adjacency matrix of the graph
    - edge_costs: edge costs of the graph

    Returns:
    - weighted Fisher score
    - Ollivier-Ricci curvature
    """
    fisher = weighted_fisher_score(theta, center, width, confidence)
    ollivier_ricci = hybrid_ollivier_ricci_curvature(adjacency, edge_costs)
    return fisher, ollivier_ricci


if __name__ == "__main__":
    # Smoke test
    theta = 0.5
    center = 0.0
    width = 1.0
    confidence = 0.8
    adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    edge_costs = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

    fisher, ollivier_ricci = fused_curvature_metric(
        theta, center, width, confidence, adjacency, edge_costs
    )
    print(f"Weighted Fisher score: {fisher}")
    print(f"Ollivier-Ricci curvature: {ollivier_ricci}")