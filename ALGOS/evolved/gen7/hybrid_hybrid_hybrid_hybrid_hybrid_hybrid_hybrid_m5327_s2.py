# DARWIN HAMMER — match 5327, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s1.py (gen6)
# born: 2026-05-30T00:01:23Z

import math
import random
import sys
import pathlib
import hashlib
import uuid
from datetime import datetime
import numpy as np

# ----------------------------------------------------------------------
# Utilities from Algorithm A
# ----------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def count_min_sketch(items, width: int = 64, depth: int = 4):
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def estimate_rlct_from_losses(train_losses_per_n, n_values):
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")
    X = np.log(np.log(ns)).reshape(-1, 1)
    X = np.hstack([X, np.ones_like(X)])
    coeffs, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    a, b = coeffs
    return a  

# ----------------------------------------------------------------------
# Utilities from Algorithm B (adapted)
# ----------------------------------------------------------------------

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now()


# ----------------------------------------------------------------------
# Hybrid Functions (core of the new algorithm)
# ----------------------------------------------------------------------

def compute_fisher_weights(angles: np.ndarray, center: float, width: float) -> np.ndarray:
    vec_gauss = np.vectorize(gaussian_beam)
    vec_fisher = np.vectorize(fisher_score)

    intensity = vec_gauss(angles, center, width)
    intensity = np.clip(intensity, 1e-12, None)
    derivative = intensity * (-(angles - center) / (width * width))
    fisher = (derivative * derivative) / intensity
    return fisher

def apply_pheromone_decay_with_fisher(entry: PheromoneEntry,
                                      angles: np.ndarray,
                                      center: float,
                                      width: float,
                                      fisher_scale: float = 1.0) -> None:
    entry.apply_decay()  
    avg_fisher = float(np.mean(compute_fisher_weights(angles, center, width)))
    extra = 0.5 ** (fisher_scale * avg_fisher)
    entry.signal_value *= extra

def hybrid_prune_dataset(x: np.ndarray,
                         y: np.ndarray,
                         angles: np.ndarray,
                         center: float,
                         width: float,
                         alpha: float,
                         lambda_: float) -> tuple:
    if x.shape[0] != y.shape[0] or x.shape[0] != angles.shape[0]:
        raise ValueError("x, y and angles must have the same first dimension")

    beam = np.vectorize(gaussian_beam)(angles, center, width)
    fisher = compute_fisher_weights(angles, center, width)
    sketch = count_min_sketch(map(tuple, x))
    def sketch_est(row):
        mins = []
        for d in range(len(sketch)):
            idx = int(hashlib.sha256(f"{d}:{row}".encode()).hexdigest(), 16) % len(sketch[0])
            mins.append(sketch[d][idx])
        return min(mins)

    freq_est = np.array([sketch_est(tuple(row)) for row in x])
    freq_est = freq_est / np.max(freq_est)
    weights = beam * fisher * (1 - freq_est)
    weights = weights / np.max(weights)
    mask = np.random.rand(x.shape[0]) < (1 - np.exp(-alpha * weights))
    x_pruned = x[mask]
    y_pruned = y[mask]
    return x_pruned, y_pruned

def improved_hybrid_prune_dataset(x: np.ndarray,
                         y: np.ndarray,
                         angles: np.ndarray,
                         center: float,
                         width: float,
                         alpha: float,
                         lambda_: float) -> tuple:
    if x.shape[0] != y.shape[0] or x.shape[0] != angles.shape[0]:
        raise ValueError("x, y and angles must have the same first dimension")

    beam = np.vectorize(gaussian_beam)(angles, center, width)
    fisher = compute_fisher_weights(angles, center, width)
    sketch = count_min_sketch(map(tuple, x))
    def sketch_est(row):
        mins = []
        for d in range(len(sketch)):
            idx = int(hashlib.sha256(f"{d}:{row}".encode()).hexdigest(), 16) % len(sketch[0])
            mins.append(sketch[d][idx])
        return min(mins)

    freq_est = np.array([sketch_est(tuple(row)) for row in x])
    freq_est = freq_est / np.max(freq_est)
    weights = beam * fisher * (1 - freq_est)
    weights = weights / np.max(weights)
    mask = np.random.rand(x.shape[0]) < (1 - np.exp(-alpha * weights))
    x_pruned = x[mask]
    y_pruned = y[mask]
    return x_pruned, y_pruned

def main():
    x = np.random.rand(100, 10)
    y = np.random.rand(100)
    angles = np.random.rand(100)
    center = 0.5
    width = 0.1
    alpha = 0.5
    lambda_ = 0.1
    x_pruned, y_pruned = improved_hybrid_prune_dataset(x, y, angles, center, width, alpha, lambda_)
    print(x_pruned.shape, y_pruned.shape)

if __name__ == "__main__":
    main()