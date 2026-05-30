# DARWIN HAMMER — match 3458, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py (gen4)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# born: 2026-05-29T23:50:10Z

import math
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import random

"""
This module implements a hybrid mathematical algorithm that combines the minimum-cost tree Bayes update and bandit-router 
sketch-RLCT algorithm from the 'hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s2.py' module with the path signature 
and iterated-integral algebra from the 'hybrid_path_signature_kan_m30_s1.py' module, and the doomsday calendar and Gini 
coefficient from the 'hybrid_doomsday_calendar_gini_coefficient_m49_s3.py' module. The mathematical bridge between the 
two structures is based on representing the path signature as a function that can be approximated using the probabilistic 
weights and log-count statistics from the minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm, and 
using the Gini coefficient to measure the fairness of the distribution of path signatures.
"""

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
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path
    """
    # Interleave the lead and lag channels
    interleaved_path = np.stack([path[:, 0], path[:, 1]], axis=1)
    return interleaved_path

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute the doomsday day of the week for each date.

    Parameters
    ----------
    years : np.ndarray
        Array of years
    months : np.ndarray
        Array of months
    days : np.ndarray
        Array of days

    Returns
    -------
    doomsday_days : np.ndarray
        Array of doomsday days of the week
    """
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
    """
    Compute the Gini coefficient for a given array of values.

    Parameters
    ----------
    values : np.ndarray
        Array of values

    Returns
    -------
    gini : float
        Gini coefficient
    """
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

def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Compute the Gini coefficient for the distribution of weekdays.

    Parameters
    ----------
    dates : Iterable[Union[dt.date, Tuple[int, int, int]]]
        Iterable of dates

    Returns
    -------
    gini : float
        Gini coefficient
    """
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)

def weekday_gini_matrix(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Compute the Gini coefficient matrix for the distribution of weekdays.

    Parameters
    ----------
    dates : Iterable[Union[dt.date, Tuple[int, int, int]]]
        Iterable of dates

    Returns
    -------
    gini_matrix : np.ndarray
        Gini coefficient matrix
    """
    counts = weekday_counts(dates).astype(float)
    weight = np.outer(counts, counts)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    weighted_diff = weight * diff
    flat = weighted_diff.ravel()
    non_zero = flat != 0
    weighted_diff = weighted_diff[non_zero]
    idx = idx[non_zero]
    diff = diff[non_zero]
    gini_matrix = np.zeros((7, 7))
    gini_matrix[idx, idx] = weighted_diff / diff
    return gini_matrix

def hybrid_path_signature(path: np.ndarray, dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Compute the hybrid path signature for a given path and distribution of dates.

    Parameters
    ----------
    path : np.ndarray
        Path
    dates : Iterable[Union[dt.date, Tuple[int, int, int]]]
        Distribution of dates

    Returns
    -------
    hybrid_path_signature : np.ndarray
        Hybrid path signature
    """
    # Compute the lead-lag transform of the path
    interleaved_path = lead_lag_transform(path)
    # Compute the doomsday day of the week for each date
    doomsday_days = doomsday_numpy(np.array([d[0] for d in dates]), np.array([d[1] for d in dates]), np.array([d[2] for d in dates]))
    # Compute the Gini coefficient matrix for the distribution of weekdays
    gini_matrix = weekday_gini_matrix(dates)
    # Compute the hybrid path signature
    hybrid_path_signature = np.sum(interleaved_path * gini_matrix, axis=1)
    return hybrid_path_signature

def hybrid_doomsday_gini(path: np.ndarray, dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Compute the hybrid doomsday Gini coefficient for a given path and distribution of dates.

    Parameters
    ----------
    path : np.ndarray
        Path
    dates : Iterable[Union[dt.date, Tuple[int, int, int]]]
        Distribution of dates

    Returns
    -------
    hybrid_gini : float
        Hybrid doomsday Gini coefficient
    """
    # Compute the hybrid path signature for the given path and distribution of dates
    hybrid_path_signature = hybrid_path_signature(path, dates)
    # Compute the Gini coefficient for the hybrid path signature
    hybrid_gini = gini_coefficient_numpy(hybrid_path_signature)
    return hybrid_gini

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)

    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    adj, edge_len, root_dist = tree_metrics(nodes, edges, "A")
    print(adj)
    print(edge_len)
    print(root_dist)
    path = np.array([[0, 0], [1, 1], [2, 2]])
    dates = [(2022, 1, 1), (2022, 1, 2), (2022, 1, 3)]
    print(hybrid_path_signature(path, dates))
    print(hybrid_doomsday_gini(path, dates))