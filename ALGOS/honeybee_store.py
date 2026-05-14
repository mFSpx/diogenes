#!/usr/bin/env python3
"""Common-store feedback primitive for decentralized resource rate control."""
from __future__ import annotations

def update_store(store: float, inflow: list[float], outflow: list[float], alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    return max(0.0, store + dt * delta), delta


def dance_duration(delta_store: float, base: float = 1.0, gain: float = 1.0, limit: float = 10.0) -> float:
    return max(0.0, min(limit, base + gain * delta_store))
