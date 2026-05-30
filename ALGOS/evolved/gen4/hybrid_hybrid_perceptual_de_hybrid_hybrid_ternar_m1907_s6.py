# DARWIN HAMMER — match 1907, survivor 6
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s2.py (gen3)
# born: 2026-05-29T23:39:49Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Sequence, Any
import numpy as np
from dataclasses import dataclass

Vector = Sequence[float]
Point2D = Tuple[float, float]

def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def compute_combined_hash(values: List[float]) -> int:
    """Merge dhash (high‑order bits) and phash (low‑order bits) into one integer."""
    dh = compute_dhash(values)
    ph = compute_phash(values)
    return (dh << 64) | ph

def hamming_distance(a: int, b: int) -> int:
    """Return Hamming distance between two integers."""
    return bin(a ^ b).count("1")

class RBFSurrogate:
    """Gaussian RBF surrogate trained on (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, gamma: float = 1.0, reg: float = 1e-6):
        """
        Parameters
        ----------
        X : (n_samples, n_features) array
        y : (n_samples,) array
        gamma : kernel width parameter
        reg : Tikhonov regularisation coefficient
        """
        self.gamma = gamma
        self.reg = reg
        self.X = X
        self.y = y
        self._fit()

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gaussian kernel matrix between rows of A and B."""
        sq_norms_A = np.sum(A ** 2, axis=1)[:, None]
        sq_norms_B = np.sum(B ** 2, axis=1)[None, :]
        dists = sq_norms_A + sq_norms_B - 2 * A @ B.T
        return np.exp(-self.gamma * dists)

    def _fit(self):
        K = self._kernel(self.X, self.X)
        n = K.shape[0]
        K_reg = K + self.reg * np.eye(n)
        self.alpha = np.linalg.solve(K_reg, self.y)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict for one or many new points."""
        K_new = self._kernel(X_new, self.X)
        return K_new @ self.alpha

def euclidean(p: Point2D, q: Point2D) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])

class CircuitBreaker:
    """Simple failure counter circuit‑breaker."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.counters: Dict[int, int] = {}

    def record_failure(self, seed_id: int):
        self.counters[seed_id] = self.counters.get(seed_id, 0) + 1

    def is_open(self, seed_id: int) -> bool:
        return self.counters.get(seed_id, 0) >= self.threshold

    def reset(self, seed_id: int):
        self.counters[seed_id] = 0

@dataclass
class Seed:
    """Represents a Voronoi seed linked to a hash cluster."""
    seed_id: int
    position: Point2D
    hash_key: int
    surrogate: RBFSurrogate
    breaker: CircuitBreaker

def fit_surrogates_by_hash(
    data: List[Tuple[List[float], float]],
    gamma: float = 1.0,
    reg: float = 1e-6,
) -> Tuple[Dict[int, RBFSurrogate], Dict[int, Point2D]]:
    clusters: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for feats, target in data:
        h = compute_combined_hash(feats)
        arr = np.asarray(feats, dtype=float)
        clusters.setdefault(h, []).append((arr, target))

    surrogates: Dict[int, RBFSurrogate] = {}
    centroids: Dict[int, Point2D] = {}

    for h, items in clusters.items():
        X = np.stack([it[0] for it in items])
        y = np.array([it[1] for it in items], dtype=float)
        surrogates[h] = RBFSurrogate(X, y, gamma=gamma, reg=reg)

        cx, cy = X[:, 0].mean(), X[:, 1].mean()
        centroids[h] = (float(cx), float(cy))

    return surrogates, centroids

def build_seeds(
    surrogates: Dict[int, RBFSurrogate],
    centroids: Dict[int, Point2D],
    breaker_threshold: int = 3,
) -> List[Seed]:
    seeds: List[Seed] = []
    for idx, (h, surrogate) in enumerate(surrogates.items()):
        pos = centroids[h]
        breaker = CircuitBreaker(threshold=breaker_threshold)
        seeds.append(Seed(seed_id=idx, position=pos, hash_key=h, surrogate=surrogate, breaker=breaker))
    return seeds

def ternary_route(
    query_feat: List[float],
    query_pos: Point2D,
    seeds: List[Seed],
    lam: float = 1.0,
    mu: float = 1.0,
) -> List[int]:
    x = np.asarray(query_feat, dtype=float).reshape(1, -1)

    costs: List[Tuple[float, int]] = []
    for seed in seeds:
        if seed.breaker.is_open(seed.seed_id):
            continue
        dist = euclidean(query_pos, seed.position)
        pred = seed.surrogate.predict(x)[0]
        cost = lam * dist + mu * pred
        costs.append((cost, seed.seed_id))

    costs.sort()
    return [seed_id for _, seed_id in costs[:3]]

def improved_ternary_route(
    query_feat: List[float],
    query_pos: Point2D,
    seeds: List[Seed],
    lam: float = 1.0,
    mu: float = 1.0,
    beta: float = 0.1,
) -> List[int]:
    x = np.asarray(query_feat, dtype=float).reshape(1, -1)

    costs: List[Tuple[float, int]] = []
    for seed in seeds:
        if seed.breaker.is_open(seed.seed_id):
            continue
        dist = euclidean(query_pos, seed.position)
        pred = seed.surrogate.predict(x)[0]
        cost = lam * dist + mu * pred
        penalty = beta * seed.breaker.counters.get(seed.seed_id, 0)
        costs.append((cost + penalty, seed.seed_id))

    costs.sort()
    return [seed_id for _, seed_id in costs[:3]]

def main():
    np.random.seed(0)
    data = [(np.random.rand(10).tolist(), np.random.rand()) for _ in range(100)]
    surrogates, centroids = fit_surrogates_by_hash(data)
    seeds = build_seeds(surrogates, centroids)

    query_feat = np.random.rand(10).tolist()
    query_pos = (np.random.rand(), np.random.rand())

    route = ternary_route(query_feat, query_pos, seeds)
    improved_route = improved_ternary_route(query_feat, query_pos, seeds)

    print(route)
    print(improved_route)

if __name__ == "__main__":
    main()