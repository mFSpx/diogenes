# DARWIN HAMMER — match 3545, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1697_s1.py (gen5)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
# born: 2026-05-29T23:50:38Z

import math
import random
import sys
from pathlib import Path
from typing import Any, Tuple, Dict, List
import numpy as np

class DarwinHammer:
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
        }

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

Point = Tuple[float, float]

def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x.any():
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def text_to_signal(text: str) -> np.ndarray:
    """Convert a Unicode string to a numeric signal (code‑point float array)."""
    return np.array([float(ord(ch)) for ch in text])

def voronoi_partition(points: np.ndarray) -> np.ndarray:
    """Voronoi partition of a set of points."""
    vor = Voronoi(points)
    return vor.points

def hybrid_metric(theta: float, center: float, width: float,
                  packet_text: str, reference_text: str) -> float:
    """Combined quality metric H = Fisher(θ) × SSIM(text, reference)."""
    f = fisher_score(theta, center, width)
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))
    return f * s

def hybrid_angle(points: np.ndarray, center: float, width: float,
                  packet_text: str, reference_text: str) -> float:
    """Select the angle that maximises the hybrid metric."""
    voronoi_points = voronoi_partition(points)
    return best_hybrid_angle(voronoi_points, center, width, packet_text, reference_text)

def best_hybrid_angle(candidates: np.ndarray, center: float, width: float,
                      packet_text: str, reference_text: str) -> float:
    """Select the angle that maximises the hybrid metric.

    Tie‑breaker: choose the angle closest to the centre when metrics are equal.
    """
    hybrid_metrics = [hybrid_metric(theta, center, width, packet_text, reference_text) for theta in candidates]
    best_index = np.argmax(hybrid_metrics)
    return candidates[best_index]

class HybridVoronoiPartition:
    def __init__(self, points: np.ndarray):
        self.points = points

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": self.points.tolist(),
        }

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

if __name__ == "__main__":
    points = np.random.rand(10, 2)
    voronoi = HybridVoronoiPartition(points)
    hybrid_angle = hybrid_angle(points, 0.5, 0.1, "Hello", "World")
    print(hybrid_angle)