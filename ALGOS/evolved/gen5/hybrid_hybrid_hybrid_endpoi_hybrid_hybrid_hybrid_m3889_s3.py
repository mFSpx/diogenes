# DARWIN HAMMER — match 3889, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_gliner_m1234_s0.py (gen4)
# born: 2026-05-29T23:52:25Z

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
        }

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def health_score(cb: EndpointCircuitBreaker, morphology_factor: float) -> float:
    failure_rate = min(cb.failures / max(cb.failure_threshold, 1), 1.0)
    reliability = 1.0 - failure_rate
    morphology = max(0.0, min(morphology_factor, 1.0))
    return reliability * (1.0 - morphology)

def weighted_voronoi_assign(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    healths: List[float],
) -> List[int]:
    if len(seeds) != len(healths):
        raise ValueError("seeds and healths must have the same length")
    assignments = []
    for pt in points:
        weighted_distances = [
            euclidean(pt, seed) / (h if h > 0 else 1e-9) for seed, h in zip(seeds, healths)
        ]
        assignments.append(int(np.argmin(weighted_distances)))
    return assignments

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = (theta - center) / (width * width) * intensity
    return derivative * derivative / intensity

def ternary_vector(value: float, thresholds: Tuple[float, float] = (-0.33, 0.33)) -> Tuple[int, int, int]:
    low, high = thresholds
    if value < low:
        return (-1, 0, 0)
    elif value > high:
        return (1, 0, 0)
    else:
        return (0, 1, 0)

def gini_coefficient(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    sorted_w = np.sort(weights)
    n = weights.size
    cumulative = np.cumsum(sorted_w, dtype=float)
    sum_w = cumulative[-1]
    if sum_w == 0:
        return 0.0
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum((n - np.arange(1, n + 1) + 1) * sorted_w) / sum_w)
    return gini

def entropy(weights: np.ndarray, eps: float = 1e-12) -> float:
    total = np.sum(weights)
    if total == 0:
        return 0.0
    p = weights / total
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p))

def simple_ssim(a: str, b: str) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    intersect = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersect / union

def compute_region_weights(
    assignments: List[int],
    healths: List[float],
    num_seeds: int,
) -> np.ndarray:
    w = np.zeros(num_seeds, dtype=float)
    for idx, seed_idx in enumerate(assignments):
        w[seed_idx] += healths[idx]
    return w / np.sum(w)  # Normalize to ensure weights sum to 1

def hybrid_fisher_weights(
    theta: float,
    seed_coords: List[Tuple[float, float]],
    width: float = 1.0,
) -> np.ndarray:
    scores = []
    for x, y in seed_coords:
        angle = math.atan2(y, x)  
        scores.append(fisher_score(theta, angle, width))
    return np.array(scores, dtype=float)

def hybrid_decision(
    region_weights: np.ndarray,
    fisher_weights: np.ndarray,
    packet: str,
    reference: str,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    weighted_histogram = region_weights * fisher_weights
    gini = gini_coefficient(weighted_histogram)
    ent = entropy(weighted_histogram)
    ssim = simple_ssim(packet, reference)
    decision = alpha * gini + beta * ssim + gamma * ent
    return decision

def improved_hybrid_decision(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    healths: List[float],
    theta: float,
    packet: str,
    reference: str,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    assignments = weighted_voronoi_assign(points, seeds, healths)
    region_weights = compute_region_weights(assignments, healths, len(seeds))
    fisher_weights = hybrid_fisher_weights(theta, seeds)
    return hybrid_decision(region_weights, fisher_weights, packet, reference, alpha, beta, gamma)