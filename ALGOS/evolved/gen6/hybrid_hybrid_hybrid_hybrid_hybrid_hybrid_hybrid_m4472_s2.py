# DARWIN HAMMER — match 4472, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_h_hybrid_hybrid_path_s_m1310_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_ternar_m1062_s0.py (gen4)
# born: 2026-05-29T23:56:06Z

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple, List, Dict

Point = Tuple[float, float]
Edge = Tuple[str, str]

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T):
        if t == 0:
            out[t] = np.concatenate((path[t], np.zeros(d)))
        elif t == T - 1:
            out[2 * t - 1] = np.concatenate((np.zeros(d), path[t]))
        else:
            out[2 * t - 1] = np.concatenate((path[t - 1], path[t]))
    return out

def bspline_design_matrix(t_vals: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    t_min, t_max = t_vals.min(), t_vals.max()
    n_internal = n_basis - (degree + 1)
    if n_internal < 0:
        raise ValueError("Not enough basis functions for the chosen degree.")
    knots = np.concatenate((
        np.full(degree, t_min),
        np.linspace(t_min, t_max, n_internal + 2),
        np.full(degree, t_max)
    ))
    def basis(i, k, t):
        if k == 0:
            return np.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
        left = (t - knots[i]) / (knots[i + k] - knots[i] + 1e-12) * basis(i, k - 1, t)
        right = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1] + 1e-12) * basis(i + 1, k - 1, t)
        return left + right
    B = np.empty((len(t_vals), n_basis), dtype=float)
    for i in range(n_basis):
        B[:, i] = basis(i, degree, t_vals)
    return B

def path_entropy(path: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(path.ravel(), bins=bins, density=True)
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def tree_cost(nodes: Dict[str, Point], edges: List[Edge]) -> float:
    material = 0.0
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        material += math.hypot(xa - xb, ya - yb)
    return material

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

def nlms_step(
    w: np.ndarray,
    X: np.ndarray,
    y: float,
    mu: float,
    eps: float = 1e-8
) -> Tuple[np.ndarray, float]:
    y_hat = X @ w
    e = y - y_hat
    norm_sq = X @ X + eps
    w_new = w + (mu / norm_sq) * e * X
    return w_new, e

def hybrid_nlms_update(
    path: np.ndarray,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    w: np.ndarray,
    mu0: float = 0.5,
    alpha: float = 0.01,
    beta: float = 0.1,
    eps: float = 1e-8,
    breaker: EndpointCircuitBreaker | None = None
) -> Tuple[np.ndarray, float]:
    L = lead_lag_transform(path)
    t_vals = np.linalg.norm(L, axis=1)
    n_basis = w.shape[0]
    B = bspline_design_matrix(t_vals, n_basis)
    y_target = float(path.sum())
    H = path_entropy(path)
    C = tree_cost(nodes, edges)
    mu = mu0 / (1.0 + alpha * C + beta * H)
    X = B.mean(axis=0)
    w_new, e = nlms_step(w, X, y_target, mu, eps)
    if breaker is not None:
        if abs(e) > 1e-6:
            breaker.record_failure()
        else:
            breaker.record_success()
    return w_new, e

class ImprovedHybridNLMS:
    def __init__(self, n_basis: int, mu0: float = 0.5, alpha: float = 0.01, beta: float = 0.1, eps: float = 1e-8):
        self.n_basis = n_basis
        self.mu0 = mu0
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.w = np.zeros(n_basis)

    def update(self, path: np.ndarray, nodes: Dict[str, Point], edges: List[Edge], root: str) -> Tuple[np.ndarray, float]:
        w_new, e = hybrid_nlms_update(path, nodes, edges, root, self.w, self.mu0, self.alpha, self.beta, self.eps)
        self.w = w_new
        return w_new, e

class ImprovedCircuitBreaker(EndpointCircuitBreaker):
    def __init__(self, failure_threshold: int = 3, recovery_threshold: int = 5):
        super().__init__(failure_threshold)
        self.recovery_threshold = recovery_threshold
        self.recovery_count = 0

    def record_success(self) -> None:
        self.recovery_count += 1
        if self.recovery_count >= self.recovery_threshold:
            self.open = False
        self.failures = 0

    def record_failure(self) -> None:
        self.recovery_count = 0
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

def improved_hybrid_nlms_update(
    path: np.ndarray,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    w: np.ndarray,
    mu0: float = 0.5,
    alpha: float = 0.01,
    beta: float = 0.1,
    eps: float = 1e-8,
    breaker: ImprovedCircuitBreaker | None = None
) -> Tuple[np.ndarray, float]:
    L = lead_lag_transform(path)
    t_vals = np.linalg.norm(L, axis=1)
    n_basis = w.shape[0]
    B = bspline_design_matrix(t_vals, n_basis)
    y_target = float(path.sum())
    H = path_entropy(path)
    C = tree_cost(nodes, edges)
    mu = mu0 / (1.0 + alpha * C + beta * H)
    X = B.mean(axis=0)
    w_new, e = nlms_step(w, X, y_target, mu, eps)
    if breaker is not None:
        if abs(e) > 1e-6:
            breaker.record_failure()
        else:
            breaker.record_success()
    return w_new, e