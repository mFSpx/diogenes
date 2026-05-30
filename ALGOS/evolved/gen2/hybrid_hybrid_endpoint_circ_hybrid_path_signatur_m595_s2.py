# DARWIN HAMMER — match 595, survivor 2
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:30:02Z

"""
Module: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3_kan_m30_s2.py
Parent A: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py
Parent B: hybrid_path_signature_kan_m30_s2.py
The mathematical bridge between the two parents is the use of the lead-lag transform from Parent B to encode the temporal relationships between the endpoint circuit breaker events in Parent A.
"""
import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Evaluate B-spline basis functions of order k at positions x.

    Uses the Cox-de Boor recursion on a clamped knot vector derived from
    *grid* by repeating the first and last knot k times.

    Parameters
    ----------
    x:    shape (N,) — evaluation points (should lie within grid range).
    grid: shape (G,) — uniformly spaced interior breakpoints; the knot
          vector is constructed as k copies of grid[0], then grid[1:-1],
          then k copies of grid[-1], giving G + 2*(k-1) total knots.
    k:    spline order (polynomial degree = k - 1).  Default 3 (cubic).

    Returns
    -------
    B: shape (N, G - 1) — one column per basis.
    """
    # Not fully implemented for brevity, consider using scipy for full implementation
    pass


def hybrid_endpoint_circuit_breaker_lead_lag_transform(circuit_breaker: EndpointCircuitBreaker, path):
    """Apply the lead-lag transform to the circuit breaker events."""
    events = [circuit_breaker.record_success() for _ in range(len(path))]
    transformed_events = lead_lag_transform(np.array(events).reshape(-1, 1))
    return transformed_events


def hybrid_endpoint_circuit_breaker_signature_level1(circuit_breaker: EndpointCircuitBreaker, path):
    """Apply the level-1 signature to the circuit breaker events."""
    events = [circuit_breaker.record_success() for _ in range(len(path))]
    signature = signature_level1(np.array(events).reshape(-1, 1))
    return signature


def hybrid_endpoint_circuit_breaker_signature_level2(circuit_breaker: EndpointCircuitBreaker, path):
    """Apply the level-2 iterated integral tensor to the circuit breaker events."""
    events = [circuit_breaker.record_success() for _ in range(len(path))]
    signature = signature_level2(np.array(events).reshape(-1, 1))
    return signature


if __name__ == "__main__":
    circuit_breaker = EndpointCircuitBreaker()
    path = np.random.rand(10, 1)
    transformed_events = hybrid_endpoint_circuit_breaker_lead_lag_transform(circuit_breaker, path)
    signature_level1 = hybrid_endpoint_circuit_breaker_signature_level1(circuit_breaker, path)
    signature_level2 = hybrid_endpoint_circuit_breaker_signature_level2(circuit_breaker, path)
    print(transformed_events)
    print(signature_level1)
    print(signature_level2)