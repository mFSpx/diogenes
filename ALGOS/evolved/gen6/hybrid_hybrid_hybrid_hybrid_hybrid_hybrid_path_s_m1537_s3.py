# DARWIN HAMMER — match 1537, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (gen4)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:37:23Z

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 0.0  

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


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


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    if subset_size < 0 or subset_size >= feature_count:
        raise ValueError("Invalid subset size.")
    numerator = math.comb(feature_count, subset_size) * math.comb(
        feature_count - subset_size - 1, feature_count - subset_size - 1
    )
    denominator = math.comb(2 * feature_count - 1, feature_count)
    return numerator / denominator


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


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    B = np.zeros((len(x), len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((len(x), len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros_like(x)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros_like(x)
            B_new[:, i] = term_l + term_r
        B = B_new
    return B


def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    if not actions:
        raise ValueError("Action list cannot be empty.")
    ev = np.array([a.expected_value for a in actions], dtype=float)
    risk = np.array([a.risk for a in actions], dtype=float)
    raw = ev * np.exp(-risk)
    total = raw.sum()
    if total == 0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / total


def morphology_path_signature(morphologies: List[Morphology]) -> Tuple[np.ndarray, np.ndarray]:
    if len(morphologies) < 2:
        raise ValueError("At least two morphology snapshots are required.")
    path = np.array(
        [[m.length, m.width, m.height] for m in morphologies],
        dtype=float,
    )
    lvl1 = signature_level1(path)
    lvl2 = signature_level2(path)
    return lvl1, lvl2


def sphericity_math_action(morphology: Morphology, circuit_breaker: EndpointCircuitBreaker) -> MathAction:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    risk = 1.0 if not circuit_breaker.allow() else 0.0
    return MathAction(
        id="sphericity",
        expected_value=sphericity,
        risk=risk,
    )


def fused_morphology_analysis(morphologies: List[Morphology], circuit_breaker: EndpointCircuitBreaker) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lvl1, lvl2 = morphology_path_signature(morphologies)
    actions = [
        sphericity_math_action(m, circuit_breaker) 
        for m in morphologies
    ]
    probabilities = calculate_regret_weighted_probabilities(actions)
    return lvl1, lvl2, probabilities


def main():
    # Example usage
    morphologies = [
        Morphology(1.0, 2.0, 3.0),
        Morphology(4.0, 5.0, 6.0),
        Morphology(7.0, 8.0, 9.0),
    ]
    circuit_breaker = EndpointCircuitBreaker()
    lvl1, lvl2, probabilities = fused_morphology_analysis(morphologies, circuit_breaker)
    print("Level 1 Signature:", lvl1)
    print("Level 2 Signature:", lvl2)
    print("Probabilities:", probabilities)

if __name__ == "__main__":
    main()