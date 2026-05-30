# DARWIN HAMMER — match 595, survivor 4
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:30:02Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives (unchanged, except typing)
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
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


# ----------------------------------------------------------------------
# Parent B – morphology and signature primitives
# ----------------------------------------------------------------------

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


def flatness_index(length: float, width: float, height: float) -> float:
    """Ratio of the smallest to the largest dimension (a simple flatness measure)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform: interleave (lead, lag) channels for causality encoding.

    Parameters
    ----------
    path : (T, d) array‑like
        Original trajectory.

    Returns
    -------
    out : (2T‑1, 2d) ndarray
        Interleaved lead‑lag representation.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 signature: total increment vector (X_T‑1 – X_0)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Level‑2 iterated integral tensor using left‑point Riemann sums."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


# ----------------------------------------------------------------------
# Hybrid functionality
# ----------------------------------------------------------------------

def compute_weighted_signature(
    path: np.ndarray, morphology: Morphology
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute level‑1 and level‑2 signatures of *path* and weight them with morphology
    descriptors.

    The weighting scheme is:
        w1 = sphericity_index
        w2 = flatness_index

    Level‑1 signature is scaled by w1, level‑2 by w2 (applied element‑wise).

    Returns
    -------
    weighted_s1 : (d,) ndarray
        Weighted level‑1 signature.
    weighted_s2 : (d, d) ndarray
        Weighted level‑2 signature.
    """
    s1 = signature_level1(path)
    s2 = signature_level2(path)

    w1 = sphericity_index(morphology.length, morphology.width, morphology.height)
    w2 = flatness_index(morphology.length, morphology.width, morphology.height)

    weighted_s1 = w1 * s1
    weighted_s2 = w2 * s2
    return weighted_s1, weighted_s2


def signature_stress(
    weighted_s1: np.ndarray, weighted_s2: np.ndarray
) -> float:
    """
    Reduce the weighted signatures to a single scalar “stress” measure.

    The stress is defined as the Euclidean norm of the weighted level‑1 vector plus
    the Frobenius norm of the weighted level‑2 tensor.

    Returns
    -------
    stress : float
    """
    norm1 = np.linalg.norm(weighted_s1)          # L2 norm of vector
    norm2 = np.linalg.norm(weighted_s2, ord="fro")  # Frobenius norm of matrix
    return norm1 + norm2


class AdaptiveEndpointCircuitBreaker(EndpointCircuitBreaker):
    """Adaptive circuit breaker with dynamic failure threshold."""

    def __init__(self, initial_failure_threshold: int = 3, 
                 stress_history: int = 10, 
                 learning_rate: float = 0.1):
        super().__init__(initial_failure_threshold)
        self.stress_history = stress_history
        self.stress_values = []
        self.learning_rate = learning_rate

    def update_failure_threshold(self, stress: float) -> None:
        self.stress_values.append(stress)
        if len(self.stress_values) > self.stress_history:
            self.stress_values.pop(0)
        mean_stress = np.mean(self.stress_values)
        std_stress = np.std(self.stress_values)
        self.failure_threshold = int(self.failure_threshold + 
                                     self.learning_rate * (mean_stress + std_stress))

    def monitor_path(
        self, path: np.ndarray, morphology: Morphology, stress_scale: float = 1.0
    ) -> None:
        w1, w2 = compute_weighted_signature(path, morphology)
        stress = signature_stress(w1, w2) * stress_scale
        self.update_failure_threshold(stress)

        if stress > self.failure_threshold:
            self.record_failure()
        else:
            self.record_success()


def hybrid_process(
    paths: List[np.ndarray],
    morphologies: List[Morphology],
    initial_failure_threshold: int = 3,
    stress_history: int = 10, 
    learning_rate: float = 0.1,
    stress_scale: float = 1.0,
) -> List[AdaptiveEndpointCircuitBreaker]:
    """
    Process a batch of trajectories, each paired with a morphology, using a dedicated
    adaptive circuit‑breaker per pair.

    Parameters
    ----------
    paths : list of (T, d) arrays
        Trajectories to analyse.
    morphologies : list of Morphology
        Corresponding geometric descriptors.
    initial_failure_threshold : int
        Initial threshold for each breaker.
    stress_history : int
        Number of past stress values to consider for dynamic threshold.
    learning_rate : float
        Rate at which the failure threshold adapts to stress history.
    stress_scale : float
        Global scaling applied to the stress before threshold comparison.

    Returns
    -------
    breakers : list of AdaptiveEndpointCircuitBreaker
        Final breaker states after processing all paths.
    """
    if len(paths) != len(morphologies):
        raise ValueError("paths and morphologies must have the same length")

    breakers = [AdaptiveEndpointCircuitBreaker(initial_failure_threshold, 
                                              stress_history, 
                                              learning_rate) for _ in paths]

    for i, (p, m) in enumerate(zip(paths, morphologies)):
        breakers[i].monitor_path(p, m, stress_scale)

    return breakers