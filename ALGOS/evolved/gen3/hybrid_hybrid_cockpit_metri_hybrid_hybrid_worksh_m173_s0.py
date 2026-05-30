# DARWIN HAMMER — match 173, survivor 0
# gen: 3
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s2.py (gen1)
# parent_b: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s5.py (gen2)
# born: 2026-05-29T23:25:53Z

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def audit_debt(exports_missing_audit_step, exports_without_audit: int) -> float:
    """Audit debt calculated based on missing audit step, exported and exported without audit"""
    return min(1, exports_missing_audit_step / (exports_missing_audit_step + exports_without_audit))

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """
    Split ``total_units`` into deterministic and LLM residual parts, then
    distribute the residual across ``groups`` using the weekday weight vector.
    Returns a dict mirroring the original schema with added calendar metadata.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    weights = weekday_weight_vector(groups, doomsday(date.year, date.month, date.day))
    return {
        'deterministic_units': deterministic_units,
        'llm_residual_units': llm_units,
        'dow': doomsday(date.year, date.month, date.day),
        'weights': weights,
        'date': {
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'weekday': date.weekday()
        }
    }


def hybrid_workshare(
    groups: Sequence[str],
    total_units: float,
    date: dt.date,
) -> Tuple[float, Dict[str, Any]]:
    """Hybrid workshare allocation."""
    allocation = allocate_hybrid(
        total_units=total_units,
        date=date,
        groups=groups,
    )
    # Introduce cockpit metrics into the hybrid workshare allocation
    # audit_debt as a factor in the deterministic units allocation
    deterministic_units = allocation['deterministic_units'] * (1 - audit_debt(allocation['dow'], allocation['dow']))
    llm_residual_units = allocation['llm_residual_units']
    return deterministic_units, allocation


def hybrid_flow_target(
    x0: np.ndarray,
    x1: np.ndarray,
    h: float,
) -> np.ndarray:
    """Hybrid flow target velocity."""
    return h * (x1 - x0)

def _hybrid_euler_solve(
    y0: np.ndarray,
    t: np.ndarray,
    dt: float,
    h: float,
) -> np.ndarray:
    """Euler integration with adaptive time step."""
    dt *= (1 - audit_debt(t, t))  # Adjust the time step based on audit debt
    y = y0.copy()
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + dt * hybrid_flow_target(y[i], y[i + 1], h)
    return y


def hybrid_euler_solve(
    y0: np.ndarray,
    t: np.ndarray,
    dt: float,
    h: float,
) -> np.ndarray:
    """Hybrid Euler integration."""
    return _hybrid_euler_solve(y0, t, dt, h)


def main() -> None:
    # Smoke test
    date = dt.date.today()
    groups = ('codex', 'groq', 'cohere', 'local_models')
    total_units = 100.0
    deterministic_target_pct = 90.0
    deterministic_units, allocation = hybrid_workshare(groups, total_units, date)
    y0 = np.array([1.0, 2.0])
    t = np.linspace(0, 10, 11)
    dt = 0.5
    h = cockpit_honesty(80, 20)  # cockpit honesty as a factor in the hybrid velocity
    result = hybrid_euler_solve(y0, t, dt, h)
    print("Hybrid workshare allocation:", allocation)
    print("Deterministic units:", deterministic_units)


if __name__ == "__main__":
    main()