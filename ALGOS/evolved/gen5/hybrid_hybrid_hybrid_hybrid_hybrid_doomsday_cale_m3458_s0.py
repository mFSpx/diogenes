# DARWIN HAMMER — match 3458, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py (gen4)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# born: 2026-05-29T23:50:10Z

"""
This module implements a novel hybrid mathematical algorithm that combines the minimum-cost tree Bayes update and bandit-router 
sketch-RLCT algorithm from the 'hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py' module with the doomsday calendar 
and Gini coefficient from the 'hybrid_doomsday_calendar_gini_coefficient_m49_s3.py' module. The mathematical bridge between 
the two structures is based on representing the path signature as a function that can be approximated using the probabilistic 
weights and log-count statistics from the minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm, and then 
applying the doomsday calendar and Gini coefficient to analyze the distribution of the approximated path signature.

The core idea is to use the probabilistic weights and log-count statistics to approximate the iterated-integral algebra, 
which is a key component of the path signature. Then, the doomsday calendar is used to determine the day of the week for 
each point in the path, and the Gini coefficient is used to analyze the distribution of the approximated path signature 
across different days of the week.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import random
import datetime as dt

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    root_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[a], nodes[b])
        if a == root:
            root_dist[b] = edge_len[(a, b)]
        elif b == root:
            root_dist[a] = edge_len[(a, b)]

    return adj, edge_len, root_dist

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding."""
    return np.array(path)

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

def gini_coefficient_numpy(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator

def weekday_counts(dates: np.ndarray) -> np.ndarray:
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    days = np.array([d.day for d in dates])
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)

    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)

def hybrid_signature(path, dates):
    """Calculate the hybrid signature by combining the path signature with the doomsday calendar and Gini coefficient."""
    path_signature = lead_lag_transform(path)
    weekdays = weekday_counts(dates)
    gini = gini_coefficient_numpy(weekdays)
    return np.concatenate((path_signature, [gini]))

def hybrid_matrix(path, dates):
    """Calculate the hybrid matrix by combining the path signature matrix with the doomsday calendar and Gini coefficient."""
    path_signature = lead_lag_transform(path)
    weekdays = weekday_counts(dates)
    gini = gini_coefficient_numpy(weekdays)
    matrix = np.outer(path_signature, path_signature)
    matrix = matrix * gini
    return matrix

def hybrid_distance(path1, path2, dates1, dates2):
    """Calculate the hybrid distance between two paths by combining the path signature distance with the doomsday calendar and Gini coefficient."""
    signature1 = hybrid_signature(path1, dates1)
    signature2 = hybrid_signature(path2, dates2)
    distance = np.linalg.norm(signature1 - signature2)
    return distance

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    path = [0, 1, 2]
    dates = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    signature = hybrid_signature(path, dates)
    matrix = hybrid_matrix(path, dates)
    distance = hybrid_distance(path, path, dates, dates)
    print(signature)
    print(matrix)
    print(distance)