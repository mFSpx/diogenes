# DARWIN HAMMER — match 3078, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s2.py (gen6)
# born: 2026-05-29T23:47:39Z

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim_n = (2 * mx * my + c1) / (mx ** 2 + my ** 2 + c1)
    ssim_d = (2 * vx * vy + c2) / (vx ** 2 + vy ** 2 + c2)

    return (ssim_n * ssim_d) ** 0.5


class Sheaf:
    def __init__(self, node_dims, edge_list):
        self.node_dims = dict(node_dims)  # type: Dict[int, int]
        self.edges = list(edge_list)  # type: List[Tuple[int, int]]

    def compute_laplacian(self):
        num_nodes = len(self.node_dims)
        L = np.zeros((num_nodes, num_nodes))
        for u, v in self.edges:
            L[u, v] = -1
            L[v, u] = 1
        return L


class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        for name, value in (
            ("length", length),
            ("width", width),
            ("height", height),
            ("mass", mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def __post_init__(self) -> None:
        pass


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def t_add(self, x, y, L):
        return np.maximum(x, y) + np.trace(np.dot(L, np.eye(len(L))))

    def t_mul(self, x, y, L):
        return np.add(x, y) + np.trace(np.dot(L, np.eye(len(L))))

    def t_matmul(self, A, B, L):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :] + np.dot(L, np.eye(len(L))), axis=2)


def hybrid_expand_ssim(v: np.ndarray, prototype: np.ndarray, sheaf: Sheaf) -> float:
    e = np.zeros_like(v)
    for i, x in enumerate(v):
        h = hashlib.sha256((str(x) + str(prototype)).encode()).hexdigest()
        e[i] = int(h, 16) % 1000000
    e_p = np.zeros_like(e)
    for i, x in enumerate(prototype):
        h = hashlib.sha256((str(x) + str(prototype)).encode()).hexdigest()
        e_p[i] = int(h, 16) % 1000000
    ssim = compute_ssim(e, e_p)
    return ssim


def add_laplace_noise(x: float, risk: float, scale: float) -> float:
    noise = random.laplace(loc=0, scale=scale)
    return x + noise * risk


def regret_match_step(
    regrets: List[float],
    utilities: List[float],
    risk: float,
    scale: float,
    sheaf: Sheaf,
    endpoint_circuit_breaker: EndpointCircuitBreaker,
) -> int:
    noisy_utilities = [add_laplace_noise(u, risk, scale) for u in utilities]
    t_max = np.max(np.array(noisy_utilities))
    t_argmax = np.argmax(np.array(noisy_utilities))
    t_matrix = endpoint_circuit_breaker.t_matmul(regrets, noisy_utilities, sheaf.compute_laplacian())
    t_argmax = np.argmax(t_matrix)
    return t_argmax


if __name__ == "__main__":
    # Smoke test
    v = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
    prototype = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
    sheaf = Sheaf({0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    endpoint_circuit_breaker = EndpointCircuitBreaker()
    regrets = [0.1, 0.2, 0.3, 0.4, 0.5]
    utilities = [0.6, 0.7, 0.8, 0.9, 1.0]
    risk = 0.1
    scale = 1.0
    print(hybrid_expand_ssim(v, prototype, sheaf))
    print(add_laplace_noise(1.0, risk, scale))
    print(regret_match_step(regrets, utilities, risk, scale, sheaf, endpoint_circuit_breaker))