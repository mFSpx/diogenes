# DARWIN HAMMER — match 1965, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s0.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s2.py (gen2)
# born: 2026-05-29T23:40:10Z

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Sequence, Tuple
import numpy as np

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rbf_kernel_matrix(X: List[Vector], epsilon: float = 1.0) -> np.ndarray:
    n = len(X)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            d = euclidean(X[i], X[j])
            val = gaussian(d, epsilon)
            K[i, j] = K[j, i] = val
    return K

def solve_linear(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    try:
        alpha = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(K) @ y
    return alpha

def nlms_update(
    weights: np.ndarray,
    phi: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    pred = float(weights @ phi)
    error = target - pred
    norm = float(phi @ phi) + eps
    new_weights = weights + mu * error * phi / norm
    return new_weights, error

def prim_mst(cost_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    n = cost_matrix.shape[0]
    if n == 0:
        return []

    visited = [False] * n
    visited[0] = True
    edges = []

    cheap = [(cost_matrix[0, j], 0, j) for j in range(1, n)]

    while len(edges) < n - 1:
        cheap.sort(key=lambda x: x[0])
        cost, u, v = cheap.pop(0)
        if visited[v]:
            continue
        visited[v] = True
        edges.append((u, v, cost))
        for w in range(n):
            if not visited[w]:
                cheap.append((cost_matrix[v, w], v, w))

    return edges

def signature_from_bytes(data: bytes, dim: int = 8) -> List[float]:
    h = hashlib.sha256(data).digest()
    needed = dim * 4
    while len(h) < needed:
        h += hashlib.sha256(h).digest()
    vec = []
    for i in range(dim):
        chunk = int.from_bytes(h[i * 4 : (i + 1) * 4], "big")
        vec.append(chunk / 0xFFFFFFFF)
    return vec

def train_surrogate_with_nlms(
    signatures: List[Vector],
    targets: List[float],
    epsilon: float = 1.0,
    mu: float = 0.4,
    epochs: int = 1,
    learning_rate_schedule: str = 'constant',
    initial_learning_rate: float = 0.4,
) -> np.ndarray:
    K = rbf_kernel_matrix(signatures, epsilon)
    y = np.array(targets, dtype=float)
    alpha = solve_linear(K, y)

    for _ in range(epochs):
        for i, phi in enumerate(K):
            if learning_rate_schedule == 'constant':
                mu_update = mu
            elif learning_rate_schedule == 'linear_decay':
                mu_update = initial_learning_rate * (1 - _ / epochs)
            elif learning_rate_schedule == 'exponential_decay':
                mu_update = initial_learning_rate * np.exp(-_ / epochs)
            else:
                raise ValueError('Invalid learning rate schedule')
            alpha, _ = nlms_update(alpha, phi, y[i], mu=mu_update)
    return alpha

def surrogate_predict(
    signatures: List[Vector],
    alpha: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    K = rbf_kernel_matrix(signatures, epsilon)
    preds = K @ alpha
    pairwise = np.empty_like(K)
    n = K.shape[0]
    for i in range(n):
        pairwise[i, :] = preds[i] * K[i, :]
    pairwise = (pairwise + pairwise.T) / 2.0
    return pairwise

def extract_spans_from_mst(pairwise_costs: np.ndarray) -> List[Tuple[int, int, float]]:
    max_cost = np.max(pairwise_costs)
    cost_matrix = max_cost - pairwise_costs
    np.fill_diagonal(cost_matrix, 0.0)
    edges = prim_mst(cost_matrix)
    return edges

if __name__ == "__main__":
    random.seed(42)
    sigs = [signature_from_bytes(str(i).encode(), dim=8) for i in range(6)]

    base = np.random.rand(6, 6)
    targets = ((base + base.T) / 2.0).tolist()
    for i in range(6):
        targets[i][i] = 0.0

    flat_targets = [targets[i][j] for i in range(6) for j in range(6) if i != j]

    alpha = train_surrogate_with_nlms(sigs, flat_targets, epsilon=1.2, mu=0.3, epochs=10, learning_rate_schedule='exponential_decay')
    pairwise_costs = surrogate_predict(sigs, alpha, epsilon=1.2)
    edges = extract_spans_from_mst(pairwise_costs)
    print(edges)