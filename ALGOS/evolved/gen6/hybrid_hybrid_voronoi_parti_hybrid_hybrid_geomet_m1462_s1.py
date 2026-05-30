# DARWIN HAMMER — match 1462, survivor 1
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s2.py (gen5)
# born: 2026-05-29T23:36:29Z

from __future__ import annotations

import math
import random
import sys
import pathlib
import numpy as np

__all__ = ['hybrid_score', 'hybrid_assign', 'hybrid_score_matrix']

"""
Hybrid Geometric-Voronoi-Morphology Optimizer

Parents:
- `hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s2.py` (Euclidean distance metric + Voronoi partition)
- `hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s2.py` (Bilinear learning step + Scalar modulation)

Mathematical bridge:
We combine the Euclidean distance metric from the Voronoi partition with the bilinear learning step from the Hybrid Geometric-Decision-Capybara Optimizer.
The effective learning rate is now a function of both the Euclidean distance and the scalar modulation:
η_w' = η_w * (1 + (1 - d(p, c_e) / D_max)) * (1 + H) * σ(ŝ)
η_r' = η_r * (1 + (1 - d(p, c_e) / D_max)) * (1 + H) * σ(ŝ)
where d(p, c_e) is the Euclidean distance from p to the endpoint's seed coordinate c_e, D_max is the maximal distance among all (point, seed) pairs, H is the Shannon entropy, and σ(ŝ) is the sigmoid of the RBF surrogate output.
"""

# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------

# 1. Voronoi partition utilities (Euclidean distance metric)
# ----------------------------------------------------------------------

def euclidean_distance(p: np.ndarray, c_e: np.ndarray) -> float:
    """Return the Euclidean distance between point p and endpoint's seed coordinate c_e."""
    return np.linalg.norm(p - c_e)


def hybrid_score(endpoint: dict, point: np.ndarray, max_dist: float) -> float:
    """Compute the hybrid score for a single pair (endpoint, point).

    The score is a weighted geometric mean of the Euclidean distance, reliability, recovery priority, sphericity, and flatness.

    Args:
        endpoint (dict): Endpoint dictionary containing the reliability factor, seed coordinate, recovery priority,
            sphericity index, and flatness index.
        point (np.ndarray): Point coordinates.
        max_dist (float): Maximal distance among all (point, seed) pairs.

    Returns:
        float: Hybrid score in the range [0, 1].
    """
    R_e = endpoint['reliability']
    c_e = endpoint['seed']
    P_e = endpoint['recovery_priority']
    σ_e = endpoint['sphericity']
    φ_e = endpoint['flatness']
    d = euclidean_distance(point, c_e)
    D_max = max_dist
    H = shannon_entropy(np.array([1]))  # placeholder for Shannon entropy
    ŝ = np.array([1])  # placeholder for RBF surrogate output
    σ_ŝ = sigmoid(ŝ)
    η_w = 1
    η_r = 1
    η_w_prime = η_w * (1 + (1 - d / D_max)) * (1 + H) * σ_ŝ
    η_r_prime = η_r * (1 + (1 - d / D_max)) * (1 + H) * σ_ŝ
    S = (R_e ** 0.5) * (1 - d / D_max) ** 0.5 * (P_e ** 0.2) * (σ_e ** 0.1) * (1 / φ_e) ** 0.1
    return S


def sigmoid(x: np.ndarray) -> float:
    """Return the sigmoid of the input array."""
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------

def hybrid_score_matrix(endpoints: list, points: np.ndarray) -> np.ndarray:
    """Return a NumPy matrix of hybrid scores for all pairs (endpoint, point).

    Args:
        endpoints (list): List of endpoint dictionaries.
        points (np.ndarray): Array of point coordinates.

    Returns:
        np.ndarray: Matrix of hybrid scores.
    """
    scores = np.zeros((len(endpoints), len(points)))
    for i, endpoint in enumerate(endpoints):
        for j, point in enumerate(points):
            scores[i, j] = hybrid_score(endpoint, point, 1)  # placeholder for max_dist
    return scores


def hybrid_assign(points: np.ndarray, pool: list) -> list:
    """Assign a list of points to the best endpoints.

    Args:
        points (np.ndarray): Array of point coordinates.
        pool (list): List of endpoint dictionaries.

    Returns:
        list: List of assigned endpoints for each point.
    """
    scores = hybrid_score_matrix(pool, points)
    assignments = np.argmax(scores, axis=0)
    return [pool[i] for i in assignments]


# ----------------------------------------------------------------------

if __name__ == "__main__":
    endpoints = [
        {'reliability': 1, 'seed': np.array([0, 0]), 'recovery_priority': 0.5, 'sphericity': 0.5, 'flatness': 0.5},
        {'reliability': 0, 'seed': np.array([1, 1]), 'recovery_priority': 0.8, 'sphericity': 0.8, 'flatness': 0.8},
    ]
    points = np.array([[0, 0], [1, 1]])
    assigned_endpoints = hybrid_assign(points, endpoints)
    print(assigned_endpoints)