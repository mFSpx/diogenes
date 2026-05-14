#!/usr/bin/env python3
"""Fisher-information scoring for off-axis sensing."""
from __future__ import annotations
import math


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def best_angle(candidates: list[float], center: float, width: float) -> float:
    if not candidates:
        raise ValueError('candidates required')
    return max(candidates, key=lambda t: (fisher_score(t, center, width), -abs(t-center)))
