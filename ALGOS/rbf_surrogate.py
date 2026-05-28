#!/usr/bin/env python3
"""Tiny OPOSSUM-style radial-basis surrogate model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


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
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))


def fit(points: Iterable[Vector], values: Iterable[float], epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    if not centers or len(centers) != len(y):
        raise ValueError("points and values must be non-empty and same length")
    k = [[gaussian(euclidean(a, b), epsilon) + (ridge if i == j else 0.0) for j, b in enumerate(centers)] for i, a in enumerate(centers)]
    return RBFSurrogate(centers, solve_linear(k, y), epsilon)


def grid_search(model: RBFSurrogate, candidates: Iterable[Vector], minimize: bool = True) -> tuple[tuple[float, ...], float]:
    scored = [(tuple(map(float, c)), model.predict(c)) for c in candidates]
    if not scored:
        raise ValueError("candidates required")
    return min(scored, key=lambda kv: kv[1]) if minimize else max(scored, key=lambda kv: kv[1])
