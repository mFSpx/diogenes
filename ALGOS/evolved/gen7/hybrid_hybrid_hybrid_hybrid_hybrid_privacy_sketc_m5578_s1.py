# DARWIN HAMMER — match 5578, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_indy_l_m1306_s1.py (gen6)
# parent_b: hybrid_privacy_sketches_m15_s2.py (gen1)
# born: 2026-05-30T00:03:02Z

import math
import random
import sys
import pathlib
import numpy as np
from typing import Any, Dict, Tuple, List
import hashlib

# ----------------------------------------------------------------------
# Gaussian / Fisher utilities
# ----------------------------------------------------------------------

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ) with centre `center` and standard-deviation `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    For a Gaussian N(center, width²) the Fisher information w.r.t. the mean is
    I = 1 / width².  The implementation follows the original code and returns
    (∂G/∂θ)² / G, which is algebraically equivalent.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width ** 2))
    return (derivative ** 2) / intensity


# ----------------------------------------------------------------------
# Count-Min Sketch utilities
# ----------------------------------------------------------------------

def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Build a Count-Min Sketch matrix as a NumPy int array."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms


def _estimate_cardinality_from_cms(cms: np.ndarray, precision: float) -> int:
    """
    Weighted cardinality estimator: count distinct (row, col) cells
    that received at least one increment, weighted by precision.
    """
    nonzero = np.count_nonzero(cms)
    depth = cms.shape[0]
    return max(1, nonzero // (depth * precision))


def hybrid_cardinality_estimate(items: List[str], width: int = 64, depth: int = 4, 
                                center: float = 0.0, sigma: float = 1.0) -> int:
    """Estimate cardinality using weighted Count-Min Sketch."""
    cms = count_min_sketch(items, width, depth)
    fisher_info = fisher_score(center, center, sigma)
    precision = 1 / (1 + fisher_info)
    return _estimate_cardinality_from_cms(cms, precision)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------

def hybrid_reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Ratio-based risk, clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, 1.0 - unique_quasi_identifiers / total_records)


def hybrid_privacy_aggregation(cms: np.ndarray, laplace_noise: float) -> np.ndarray:
    """Add Laplace noise to CMS cells before reconstructing the sum."""
    noise = np.random.laplace(loc=0, scale=laplace_noise, size=cms.shape)
    return cms + noise


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    items = ["item1", "item2", "item3"]
    cms = count_min_sketch(items)
    cardinality_estimate = hybrid_cardinality_estimate(items)
    print(f"Cardinality estimate: {cardinality_estimate}")
    reconstructed_risk_score = hybrid_reconstruction_risk_score(cardinality_estimate, 100)
    print(f"Reconstructed risk score: {reconstructed_risk_score}")