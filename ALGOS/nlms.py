#!/usr/bin/env python3
"""Normalized least mean squares update."""
from __future__ import annotations
from typing import Iterable


def predict(weights: Iterable[float], x: Iterable[float]) -> float:
    return sum(w * xi for w, xi in zip(weights, x))


def update(weights: list[float], x: list[float], target: float, mu: float = 0.5, eps: float = 1e-9) -> tuple[list[float], float]:
    if len(weights) != len(x):
        raise ValueError('weights and input must have equal length')
    if not 0 < mu < 2:
        raise ValueError('mu must be in the interval (0, 2)')
    y = predict(weights, x)
    error = target - y
    power = sum(xi * xi for xi in x) + eps
    next_weights = [w + mu * error * xi / power for w, xi in zip(weights, x)]
    return next_weights, error
