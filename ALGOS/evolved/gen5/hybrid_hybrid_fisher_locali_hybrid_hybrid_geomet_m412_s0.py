# DARWIN HAMMER — match 412, survivor 0
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen4)
# born: 2026-05-29T23:28:44Z

"""
This module fuses the core topologies of hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py and 
hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py. It combines the Fisher information metric 
with the geometric product from Clifford algebra to create a unified hybrid system.

The mathematical bridge between the two parents lies in the fact that both deal with 
multivariate data and involve operations that can be represented as matrix transformations. 
The Fisher information metric provides a measure of the sensitivity of a probability distribution 
to changes in its parameters, while the geometric product from Clifford algebra provides a way 
to combine and manipulate multivectors.

The hybrid system integrates the Fisher information metric with the geometric product 
by using the metric to weight the multivectors in the geometric product. This allows for 
a more nuanced and flexible way of combining and manipulating multivariate data.
"""

import numpy as np
from typing import Dict, FrozenSet, Tuple
import math

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


def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    remaining = [i for i, c in counts.items() if c % 2 == 1]
    cleaned = []
    for i in indices:
        if counts[i] % 2 == 1:
            cleaned.append(i)
            counts[i] = 0  
    lst = list(cleaned)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Clifford (geometric) product of two basis blades.
    Returns (resulting_blade, sign).
    """
    combined = tuple(list(blade_a) + list(blade_b))
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


def vector_to_mv(v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """Convert a 1‑D array into a multivector containing only grade‑1 blades."""
    mv: Dict[FrozenSet[int], float] = {}
    for i, coeff in enumerate(v):
        if coeff != 0.0:
            mv[frozenset({i})] = float(coeff)
    return mv


def hybrid_metric(theta: float, center: float, width: float, 
                  packet_vector: np.ndarray, reference_vector: np.ndarray) -> float:
    """Combined quality metric H = Fisher(θ) × Geometric Product."""
    f = fisher_score(theta, center, width)
    packet_mv = vector_to_mv(packet_vector)
    reference_mv = vector_to_mv(reference_vector)
    
    # Compute geometric product
    geometric_product = 0.0
    for blade_a, coeff_a in packet_mv.items():
        for blade_b, coeff_b in reference_mv.items():
            resulting_blade, sign = _multiply_blades(blade_a, blade_b)
            geometric_product += sign * coeff_a * coeff_b
    
    return f * geometric_product


def best_hybrid_angle(candidates: np.ndarray, center: float, width: float, 
                      packet_vector: np.ndarray, reference_vector: np.ndarray) -> float:
    """Select the angle that maximises the hybrid metric.

    Tie‑breaker: choose the angle closest to the centre when metrics are equal.
    """
    max_metric = -np.inf
    best_angle = None
    for theta in candidates:
        metric = hybrid_metric(theta, center, width, packet_vector, reference_vector)
        if metric > max_metric:
            max_metric = metric
            best_angle = theta
        elif metric == max_metric:
            if abs(theta - center) < abs(best_angle - center):
                best_angle = theta
    return best_angle


if __name__ == "__main__":
    # Smoke test
    theta = 0.5
    center = 0.0
    width = 1.0
    packet_vector = np.array([1.0, 2.0, 3.0])
    reference_vector = np.array([4.0, 5.0, 6.0])
    print(hybrid_metric(theta, center, width, packet_vector, reference_vector))
    candidates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(best_hybrid_angle(candidates, center, width, packet_vector, reference_vector))