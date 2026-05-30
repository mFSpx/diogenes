# DARWIN HAMMER — match 4438, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2421_s0.py (gen5)
# born: 2026-05-29T23:55:50Z

import math
import random
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DIM = 1000
_NODE_DIMS = {i: 5 for i in range(DIM)}  
_EDGE_DIM = 3                  
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

# ----------------------------------------------------------------------
# NLMS core
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Feature extraction & hygiene score
# ----------------------------------------------------------------------
def extract_digit_features(text: str) -> np.ndarray:
    return np.array([text.count(str(i)) for i in range(9)], dtype=np.float64)


def hybrid_hygiene_score(features: np.ndarray) -> float:
    s = np.mean(features)
    H = -np.sum(features * np.log2(features + 1e-9))
    H_max = np.log2(features.size)
    return s * (1.0 + H / H_max)


# ----------------------------------------------------------------------
# Count‑Min Sketch
# ----------------------------------------------------------------------
class CountMinSketch:
    def __init__(self, dim: int = DIM, depth: int = 5, seed: int = 0):
        self.dim = dim
        self.depth = depth
        self.tables = np.zeros((depth, dim), dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.prime = 2 ** 31 - 1
        self.a = rng.integers(1, self.prime, size=depth, dtype=np.int64)
        self.b = rng.integers(0, self.prime, size=depth, dtype=np.int64)

    def _hash(self, item: int, row: int) -> int:
        return ((self.a[row] * item + self.b[row]) % self.prime) % self.dim

    def update(self, item: int, inc: float = 1.0) -> None:
        for r in range(self.depth):
            idx = self._hash(item, r)
            self.tables[r, idx] += inc

    def query(self, item: int) -> float:
        return min(self.tables[r, self._hash(item, r)] for r in range(self.depth))

    def vector(self) -> np.ndarray:
        return self.tables.mean(axis=0)


# ----------------------------------------------------------------------
# Sheaf Laplacian regularizer
# ----------------------------------------------------------------------
def build_weighted_graph(dim: int, node_dims: Dict[int, int], edge_dim: int) -> Dict[int, Dict[int, float]]:
    graph = defaultdict(dict)
    for i in range(dim):
        for j in [i-1, i+1]:
            idx = (j + dim) % dim
            weight = 1 / (1 + abs(node_dims[i] - node_dims[idx]))
            graph[i][idx] = weight
            graph[idx][i] = weight
    return graph


def sheaf_laplacian_regularizer(
    weights: np.ndarray,
    node_dims: Dict[int, int],
    edge_dim: int,
    lam: float = 0.01,
) -> np.ndarray:
    dim = len(weights)
    graph = build_weighted_graph(dim, node_dims, edge_dim)
    grad = np.zeros_like(weights)
    for i in range(dim):
        neighbor_sum = sum(weights[j] * graph[i][j] for j in graph[i])
        degree = sum(graph[i].values())
        laplacian_i = degree * weights[i] - neighbor_sum
        grad[i] = -lam * laplacian_i
    return grad


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_predict(
    weights: np.ndarray,
    sketch: CountMinSketch,
    text: str,
) -> float:
    sketch_vec = sketch.vector()                     
    digit_feat = extract_digit_features(text)       
    x = np.concatenate([sketch_vec, digit_feat])    
    return nlms_predict(weights, x)


def hybrid_update(
    weights: np.ndarray,
    sketch: CountMinSketch,
    text: str,
    target: float,
    base_mu: float = 0.5,
    eps: float = 1e-9,
    reg_lambda: float = 0.01,
) -> Tuple[np.ndarray, float]:
    sketch_vec = sketch.vector()
    digit_feat = extract_digit_features(text)
    x = np.concatenate([sketch_vec, digit_feat])

    H = hybrid_hygiene_score(digit_feat)
    scale = max(1.0, np.mean(digit_feat) + 1e-6)   
    mu = base_mu * (1.0 + H / scale)

    new_w, error = nlms_update(weights, x, target, mu=mu, eps=eps)

    reg_grad = sheaf_laplacian_regularizer(
        new_w[:DIM], _NODE_DIMS, _EDGE_DIM, lam=reg_lambda
    )
    new_w[:DIM] += reg_grad

    return new_w, error


def initialize_hybrid_model(dim: int = DIM) -> Tuple[np.ndarray, CountMinSketch]:
    weights = np.zeros(dim + 9, dtype=np.float64)
    sketch = CountMinSketch(dim=dim, depth=5, seed=42)
    return weights, sketch


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    weights, sketch = initialize_hybrid_model()
    for _ in range(10):
        text = str(random.randint(0, 1000))
        target = random.random()
        weights, _ = hybrid_update(weights, sketch, text, target)
        sketch.update(int(text), 1.0)
    print(hybrid_predict(weights, sketch, "42"))