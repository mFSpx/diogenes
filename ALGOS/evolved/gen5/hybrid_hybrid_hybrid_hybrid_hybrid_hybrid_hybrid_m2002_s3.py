# DARWIN HAMMER — match 2002, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_possum_filter_m581_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s4.py (gen3)
# born: 2026-05-29T23:40:27Z

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import numpy as np
import random
import sys
import pathlib
import re
from collections import Counter

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def ssim(v1: Vector, v2: Vector) -> float:
    if len(v1) != len(v2):
        raise ValueError("vectors must have same dimension")
    return 1 - euclidean(v1, v2) / (1 + euclidean(v1, v2))

def shannon_entropy(text: str) -> float:
    token_counts = Counter(re.findall(r'\w+', text))
    total_tokens = sum(token_counts.values())
    if total_tokens == 0:
        return 0.0
    entropy = -sum(count / total_tokens * math.log(count / total_tokens) for count in token_counts.values())
    return entropy

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class SpatialAwareSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

def hybrid_score(v1: Vector, v2: Vector, text: str, lambda_: float = 0.5) -> float:
    ssim_value = ssim(v1, v2)
    entropy = shannon_entropy(text)
    token_counts = Counter(re.findall(r'\w+', text))
    max_entropy = math.log(len(token_counts.keys())) if token_counts else 0.0
    if max_entropy == 0.0:
        return ssim_value
    return ssim_value * (1 - lambda_ * entropy / max_entropy)

def radial_basis_function(x: Vector, centers: list[tuple[float, ...]], weights: list[float], epsilon: float = 1.0) -> float:
    return sum(gaussian(euclidean(x, center), epsilon) * weight for center, weight in zip(centers, weights))

def hybrid_surrogate_model(x: Vector, v1: Vector, v2: Vector, text: str, centers: list[tuple[float, ...]], weights: list[float], epsilon: float = 1.0) -> float:
    hybrid_score_value = hybrid_score(v1, v2, text)
    return radial_basis_function(x, centers, weights, epsilon) * hybrid_score_value

if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    v1 = np.array([4.0, 5.0, 6.0])
    v2 = np.array([7.0, 8.0, 9.0])
    text = "This is a test text"
    centers = [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)]
    weights = [0.5, 0.5]
    print(hybrid_surrogate_model(x.tolist(), v1.tolist(), v2.tolist(), text, centers, weights))