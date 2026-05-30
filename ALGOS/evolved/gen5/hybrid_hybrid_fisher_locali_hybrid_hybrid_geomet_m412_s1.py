# DARWIN HAMMER — match 412, survivor 1
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen4)
# born: 2026-05-29T23:28:44Z

"""
Module hybrid_fusion.py: Fusing hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py and hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py.
The mathematical bridge between the two parent algorithms lies in the combination of Fisher information and geometric product.
The Fisher information from parent A is used to weight the geometric product from parent B, effectively fusing their core topologies.

Parents:
- hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen 2)
- hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s3.py (gen 4)
"""

import numpy as np
import math
from typing import Dict, FrozenSet, Tuple

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
    # Count occurrences
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Cancel even occurrences
    remaining = [i for i, c in counts.items() if c % 2 == 1]
    # The remaining list may contain duplicates (odd count >1) – keep one copy per odd count
    # Build the list respecting original multiplicities (odd only)
    cleaned = []
    for i in indices:
        if counts[i] % 2 == 1:
            cleaned.append(i)
            counts[i] = 0  # ensure we keep only one copy
    # Sort while tracking sign via bubble‑sort swaps
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


def hybrid_fusion(theta: float, center: float, width: float, v: np.ndarray) -> Dict[FrozenSet[int], float]:
    """
    Fused function combining Fisher information and geometric product.

    Args:
    - theta: angle
    - center: center of the Gaussian beam
    - width: width of the Gaussian beam
    - v: 1-D array to convert to a multivector

    Returns:
    - A multivector with Fisher information as weights
    """
    fisher_info = fisher_score(theta, center, width)
    mv = vector_to_mv(v)
    weighted_mv = {}
    for blade, coeff in mv.items():
        weighted_mv[blade] = fisher_info * coeff
    return weighted_mv


def geometric_product(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], float]:
    """
    Clifford (geometric) product of two basis blades.

    Args:
    - blade_a: first blade
    - blade_b: second blade

    Returns:
    - resulting blade and its weight
    """
    resulting_blade, sign = _multiply_blades(blade_a, blade_b)
    return resulting_blade, sign


def test_hybrid_fusion():
    theta = 0.5
    center = 0.0
    width = 1.0
    v = np.array([1.0, 2.0, 3.0])
    mv = hybrid_fusion(theta, center, width, v)
    print(mv)


if __name__ == "__main__":
    test_hybrid_fusion()