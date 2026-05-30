# DARWIN HAMMER — match 1451, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py (gen4)
# born: 2026-05-29T23:36:35Z

import datetime as dt
import math
import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Iterable

def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
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
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[(d.weekday() + 1) % 7] += 1
    return counts


def gini_coefficient(values: np.ndarray) -> float:
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
    failure_rate = min(failures / failure_threshold, 1.0)
    health = (1 - reconstruction_risk_score * failure_rate) * (1 - recovery_priority)
    return health


def build_adjacency(matrix: np.ndarray, threshold: float = 1.0) -> List[Tuple[int, int]]:
    num_nodes = len(matrix)
    adj_list = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) 
                 if np.linalg.norm(matrix[i] - matrix[j]) < threshold]
    return adj_list


def node_curvature(adj_list: List[Tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
    num_nodes = len(matrix)
    curvature = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        if neighbors:
            kappa_sum = 0.0
            for j in neighbors:
                dist = np.linalg.norm(matrix[i] - matrix[j])
                if dist > 0:
                    kappa = 1 - np.linalg.norm(matrix[i] - matrix[j]) / dist
                    kappa_sum += kappa
            curvature[i] = kappa_sum / len(neighbors)
    return curvature


def krampus_ollivier_sheaf(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    brain_xyz = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        if neighbors:
            weights = [curvature[j] for j in neighbors]
            weights = np.array(weights) / sum(weights)
            brain_xyz[i] = np.sum([matrix[j] * w for j, w in zip(neighbors, weights)], axis=0)
    return brain_xyz


def hybrid_health(matrix: np.ndarray, curvature: np.ndarray, adj_list: List[Tuple[int, int]]) -> np.ndarray:
    num_nodes = len(matrix)
    health = np.zeros((num_nodes,))
    for i in range(num_nodes):
        neighbors = [j for j in range(num_nodes) if (i, j) in adj_list or (j, i) in adj_list]
        failure_rate = len(neighbors) / (num_nodes - 1) if num_nodes > 1 else 0.0
        health[i] = compute_health(curvature[i], len(neighbors), num_nodes - 1, 1.0)
    return health


if __name__ == "__main__":
    np.random.seed(0)
    matrix = np.random.rand(10, 3)
    adj_list = build_adjacency(matrix)
    curvature = node_curvature(adj_list, matrix)
    brain_xyz = krampus_ollivier_sheaf(matrix, curvature, adj_list)
    health = hybrid_health(matrix, curvature, adj_list)
    print(brain_xyz)
    print(health)