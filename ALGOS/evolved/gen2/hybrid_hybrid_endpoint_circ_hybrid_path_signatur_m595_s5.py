# DARWIN HAMMER — match 595, survivor 5
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:30:02Z

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.stress_history = []

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
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
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)  
    running = path[:-1] - path[0]               
    return running.T @ increments               


def compute_weighted_signature(
    path: np.ndarray, morphology: Morphology
) -> Tuple[np.ndarray, np.ndarray]:
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
    norm1 = np.linalg.norm(weighted_s1)          
    norm2 = np.linalg.norm(weighted_s2, ord="fro")  
    return norm1 + norm2


def monitor_path(
    path: np.ndarray,
    morphology: Morphology,
    breaker: EndpointCircuitBreaker,
    stress_scale: float = 1.0,
) -> None:
    w1, w2 = compute_weighted_signature(path, morphology)
    stress = signature_stress(w1, w2) * stress_scale

    breaker.stress_history.append(stress)

    if stress > breaker.failure_threshold:
        breaker.record_failure()
    else:
        breaker.record_success()


def hybrid_process(
    paths: List[np.ndarray],
    morphologies: List[Morphology],
    failure_threshold: int = 3,
    stress_scale: float = 1.0,
) -> List[EndpointCircuitBreaker]:
    if len(paths) != len(morphologies):
        raise ValueError("paths and morphologies must have the same length")

    breakers = [EndpointCircuitBreaker(failure_threshold) for _ in paths]

    for i, (p, m) in enumerate(zip(paths, morphologies)):
        monitor_path(p, m, breakers[i], stress_scale)

    return breakers