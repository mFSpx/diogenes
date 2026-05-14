#!/usr/bin/env python3
"""Thanatosis / simulated-annealing dormancy primitives."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


@dataclass(frozen=True)
class DormancyDecision:
    accept: bool
    probability: float
    dormant: bool


def decide(delta_e: float, k: int, t0: float = 1.0, alpha: float = 0.95, dormancy_floor: float = 0.05, seed: int | str | None = None) -> DormancyDecision:
    temp = cooling_temperature(k, t0, alpha)
    p = acceptance_probability(delta_e, temp)
    rng = random.Random(seed)
    return DormancyDecision(accept=rng.random() <= p, probability=p, dormant=temp <= dormancy_floor and delta_e >= 0)
