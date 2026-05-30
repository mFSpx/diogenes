# DARWIN HAMMER — match 2670, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s1.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (gen4)
# born: 2026-05-29T23:43:31Z

# hybrid_hybrid_endpoint_circuit_breaker_morphology_m572_m268_s2.py
# DARWIN HAMMER — match 572 + 268, survivor 2
# gen: 6
# parent_a: hybrid_path_signature_kan_m30_s3.py (gen1)
# parent_b: hybrid_hybrid_endpoint_circuit_breaker_morphology_m18_m26_s5.py (gen5)
# born: 2026-05-30T00:00:00Z

"""
This module implements a hybrid algorithm that combines the Path Signature and Morphology from 
'hybrid_path_signature_kan_m30_s3.py' with the Endpoint Circuit Breaker and Morphology from 
'hybrid_hybrid_endpoint_circuit_breaker_morphology_m18_m26_s5.py'. The mathematical bridge 
between these two structures is the use of the Path Signature to adjust the failure threshold 
of the Endpoint Circuit Breaker, and the Morphology to describe the geometric features of the 
path.

The hybrid algorithm integrates the governing equations of both parents by using the 
signature_level2 function to adjust the failure threshold of the Endpoint Circuit Breaker, and 
the Morphology class to compare the geometry of the paths.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

class Morphology:
    """Geometric description of a physical (or logical) entity."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def as_dict(self) -> dict:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "mass": self.mass
        }

class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

def hybrid_endpoint_circuit_breaker(path, morphology: Morphology):
    signature = signature_level2(path)
    failure_threshold = int(np.mean(signature) * len(path))
    breaker = EndpointCircuitBreaker(failure_threshold)
    breaker.record_success()
    return breaker

def hybrid_morphology(path):
    morphology = Morphology(0.0, 0.0, 0.0, 0.0)
    for i in range(len(path)):
        morphology.length += path[i][0]
        morphology.width += path[i][1]
        morphology.height += path[i][2]
        morphology.mass += 1.0
    return morphology

def hybrid_ssim(path1, path2):
    morphology1 = hybrid_morphology(path1)
    morphology2 = hybrid_morphology(path2)
    ssim1 = np.mean([morphology1.length, morphology1.width, morphology1.height, morphology1.mass]) / 4
    ssim2 = np.mean([morphology2.length, morphology2.width, morphology2.height, morphology2.mass]) / 4
    return np.mean([ssim1, ssim2])

if __name__ == "__main__":
    path1 = np.random.rand(10, 3)
    path2 = np.random.rand(10, 3)
    morphology1 = hybrid_morphology(path1)
    morphology2 = hybrid_morphology(path2)
    ssim = hybrid_ssim(path1, path2)
    print(ssim)
    breaker = hybrid_endpoint_circuit_breaker(path1, morphology1)
    print(breaker.allow())