# DARWIN HAMMER — match 1334, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s1.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s0.py (gen4)
# born: 2026-05-29T23:35:37Z

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def now_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open

def gamma_lanczos(z: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        z_minus_one = z - 1
        a = _LANCZOS_C[0]
        for i in range(1, len(_LANCZOS_C)):
            a += _LANCZOS_C[i] / (z_minus_one + i)
        t = z_minus_one + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * t ** (z_minus_one + 0.5) * math.exp(-t) * a

def caputo_derivative(alpha: float, t: int, series: List[float]) -> float:
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be in (0,1) for the Caputo derivative.")
    if t == 0:
        return 0.0
    integral = 0.0
    for tau in range(t):
        kernel = (t - tau) ** (1 - alpha)
        integral += series[tau] * kernel / gamma_lanczos(2 - alpha)
    return integral / gamma_lanczos(1 - alpha)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    y = nlms_predict(weights, x)
    e = target - y
    norm_x_sq = max(np.dot(x, x), eps)
    step = (mu / (norm_x_sq + eps)) * e * x
    new_weights = weights + step
    return new_weights, e

def bayes_marginal(prior: float, likelihood: float, fp: float) -> float:
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * (1 - likelihood) + fp
    if denominator == 0:
        return 0.0
    return numerator / denominator

def hybrid_edge_weight(
    i: int,
    j: int,
    nodes: Dict[int, Tuple[float, float]],
    nlms_weights: np.ndarray,
    x_i: np.ndarray,
    x_j: np.ndarray,
    certainty: float,
    alpha: float,
    error_series: List[float],
    time_index: int,
    eps: float = 1e-12,
) -> float:
    s_i = nlms_predict(nlms_weights, x_i)
    s_j = nlms_predict(nlms_weights, x_j)

    prior_num = s_i + s_j
    prior_den = max(prior_num + eps, eps)
    prior = prior_num / prior_den

    likelihood = 1.0 - certainty
    fp = certainty * 0.1
    marginal = bayes_marginal(prior, likelihood, fp)

    xi, yi = nodes[i]
    xj, yj = nodes[j]
    d = math.hypot(xi - xj, yi - yj)

    if time_index < len(error_series):
        fractional_factor = abs(caputo_derivative(alpha, time_index, error_series))
    else:
        fractional_factor = 0.0

    weight = d * (1.0 - marginal) * (1.0 + fractional_factor) + eps
    return weight

def kruskal_mst(
    nodes: Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int]],
    alpha: float,
    error_series: List[float],
    nlms_weights: np.ndarray,
    certainty: float,
    eps: float = 1e-12,
) -> List[Tuple[int, int]]:
    edge_weights = {}
    for e in edges:
        i, j = e
        x_i = np.array([i, i**2]) 
        x_j = np.array([j, j**2])
        weight = hybrid_edge_weight(i, j, nodes, nlms_weights, x_i, x_j, certainty, alpha, error_series, len(error_series)-1, eps)
        edge_weights[e] = weight

    mst_edges = []
    parent = {}
    rank = {}

    def make_set(v):
        parent[v] = v
        rank[v] = 0

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(v1, v2):
        root1 = find(v1)
        root2 = find(v2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    for v in nodes:
        make_set(v)

    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])

    breaker = EndpointCircuitBreaker()
    for e, weight in sorted_edges:
        i, j = e
        if find(i) != find(j) and breaker.allow():
            mst_edges.append(e)
            union(i, j)
        else:
            breaker.record_failure()
    return mst_edges