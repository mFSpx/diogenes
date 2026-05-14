#!/usr/bin/env python3
"""Chelydrid ambush-strike kinematics primitive.

Biology origin: neck-force acceleration opposed by quadratic drag. LUCIDOTA use:
burst/action admission model: short decisive actions accelerate quickly, but
cost/drag dominates if the action is too large or too long.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float


def integrate_strike(force_series: Iterable[float], dt: float, m_head: float, drag_cd: float = 0.3, fluid_density: float = 1000.0, area: float = 0.001, v0: float = 0.0) -> StrikeState:
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)


def pulse_force(peak_force: float, steps: int) -> list[float]:
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]


def burst_admission_score(work_value: float, cost_drag: float, urgency_force: float, steps: int = 12) -> float:
    """Dimensionless score for whether a burst action is worth taking now."""
    state = integrate_strike(pulse_force(max(0.0, urgency_force), steps), dt=0.01, m_head=1.0, drag_cd=max(0.0, cost_drag), fluid_density=1.0, area=1.0)
    return work_value * state.distance
