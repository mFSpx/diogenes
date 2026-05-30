# DARWIN HAMMER — match 1451, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py (gen4)
# born: 2026-05-29T23:36:35Z

"""
HYBRID KRAMPUS OLLIVIER-SHEAF DOOMSDAY
=====================================

This module fuses the mathematical topologies of two parents:

* `hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s0.py` (Parent A)
* `hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py` (Parent B)

The mathematical bridge between the two structures is the application of Ollivier-Ricci curvature to
neighborhood overlaps between nodes in a graph structure. Specifically, the `hybrid_build_adj`
function from Parent B is used to construct an adjacency matrix, which is then used to compute the
Ollivier-Ricci curvature in the `hybrid_node_curvature` function. This curvature is then used as a
weight in the weighted linear combination used in the Krampus-Ollivier brain map.

The output of this module is a 3-axis space with coordinates augmented by the curvature score.
"""

import datetime as dt
import math
import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Iterable, Dict

def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[dt.date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[(d.weekday() + 1) % 7] += 1
    return counts


def gini_coefficient(values: np.ndarray) -> float:
    """Robust Gini coefficient for a 1‑D non‑negative array."""
    if values.size == 0:
        return 0.0
    values = values.astype(float)
    if np.all(values == 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """Health is a bounded confidence metric."""
    failure_rate = min(failures / failure_threshold, 1.0)
    health = (1 - reconstruction_risk_score * failure_rate) * (1 - recovery_priority)
    return health


def hybrid_build_adj(matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Builds the adjacency structure from a list of master vectors."""
    num_nodes = len(matrix)
    adj_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(matrix[i] - matrix[j])
            if dist < 1.0:
                adj_list.append((i, j))
    return adj_list


def hybrid_node_curvature(adj_list: List[Tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
    """Runs Ollivier-Ricci on the graph and returns per-node average curvature."""
    num_nodes = len(matrix)
    curvature = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        kappa_sum = 0.0
        for j in neighbors:
            dist = np.linalg.norm(matrix[i] - matrix[j])
            kappa = 1 - np.linalg.norm(matrix[i] - matrix[j]) / dist
            kappa_sum += kappa
        curvature[i] = kappa_sum / len(neighbors) if neighbors else 0.0
    return curvature


def krampus_ollivier_sheaf(matrix: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Augments the original brain_xyz with the curvature score to produce the final 3D coordinates."""
    num_nodes = len(matrix)
    brain_xyz = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        weights = [curvature[j] for j in neighbors]
        weights /= sum(weights)
        brain_xyz[i] = np.sum([matrix[j] * weights[j] for j in neighbors], axis=0)
    return brain_xyz


def hybrid_health(matrix: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Computes the health of each node in the graph."""
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1)
        health[i] = compute_health(curvature[i], len(neighbors), num_nodes - 1, 1.0)
    return health


if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    adj_list = hybrid_build_adj(matrix)
    curvature = hybrid_node_curvature(adj_list, matrix)

    # Compute the Krampus-Ollivier brain map with curvature
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature)

    # Compute the health of each node
    health = hybrid_health(matrix, curvature)

    print(brain_xyz)
    print(health)