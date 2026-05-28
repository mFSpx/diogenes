#!/usr/bin/env python3
"""Decreasing-rate pruning schedule."""
from __future__ import annotations
import math, random
from collections.abc import Hashable


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(edges: list[Hashable], t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list[Hashable]:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]
