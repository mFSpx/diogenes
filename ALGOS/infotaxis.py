#!/usr/bin/env python3
"""Gradient-free entropy search helpers."""
from __future__ import annotations
import math


def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)


def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: dict[str, tuple[float, list[float], list[float]]]) -> str:
    if not actions:
        raise ValueError('actions required')
    return min(actions, key=lambda a: (expected_entropy(*actions[a]), a))
