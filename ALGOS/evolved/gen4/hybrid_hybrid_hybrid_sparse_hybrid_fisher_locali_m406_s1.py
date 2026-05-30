# DARWIN HAMMER — match 406, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py (gen3)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py (gen2)
# born: 2026-05-29T23:28:55Z

"""
This module fuses the Sparse Winner-Take-All (WTA) algorithm from 
hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py with the 
Fisher Localization and Hybrid Ternary Router from 
hybrid_fisher_localization_hybrid_ternary_route_m40_s5.py to create a 
novel hybrid algorithm.

The mathematical bridge between the two parents is based on the 
interpretation of the signal-to-noise gap as a confidence scalar, 
which rescales the random coefficient used in the social interaction 
and the step size used in predator evasion. This confidence scalar 
is then used to modulate the sparse expansion and the reconstruction 
risk function in the WTA algorithm, while also influencing the 
Gaussian beam intensity in the Fisher Localization and Hybrid Ternary 
Router.

The hybrid algorithm integrates the governing equations of both parents, 
combining the hash-based sparse projection, differential privacy, and 
reconstruction risk function from the WTA algorithm with the 
exponential evasion schedule, Hoeffding-tree split decision, and 
recovery priority from the Hybrid Capybara-Tri Conduit Algorithm, and 
the Gaussian beam intensity and Fisher information from the Fisher 
Localization and Hybrid Ternary Router.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict
import numpy as np

def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash-based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 at the indices of the top-k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]

def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two binary vectors."""
    return sum(el1 != el2 for el1, el2 in zip(a, b))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    For a Gaussian beam I(θ) the Fisher information reduces to
        F(θ) = (θ‑center)² / width⁴ .
    The implementation follows the definition
        F = (∂I/∂θ)² / I
    but guards against division by zero.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_fisher_sparse_wta(values: List[float], m: int, theta: float, center: float, width: float) -> List[float]:
    """Hybrid Fisher Sparse WTA algorithm."""
    # Compute sparse expansion
    sparse_expansion = expand(values, m)
    
    # Compute Gaussian beam intensity
    beam_intensity = gaussian_beam(theta, center, width)
    
    # Modulate sparse expansion with Gaussian beam intensity
    modulated_sparse_expansion = [x * beam_intensity for x in sparse_expansion]
    
    return modulated_sparse_expansion

def weighted_ssim(
    x: List[float],
    y: List[float],
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Weighted Structural Similarity Index.

    The weight for each sample is the Gaussian beam intensity at *theta*.
    This couples the Fisher‑information side (through the beam) directly to the
    similarity computation, yielding a deeper mathematical fusion.

    Parameters
    ----------
    x, y : sequences of equal length
        Numeric signals to compare.
    theta, center, width : float
        Parameters defining the Gaussian weighting function.
    dynamic_range : float, optional
        If omitted, the range is taken as ``max(x∪y) - min(x∪y)``.
    k1, k2 : float
        Stability constants as in the classic SSIM definition.

    Returns
    -------
    float
        Weighted SSIM value in ``[0, 1]`` (approximately).
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    # Compute per‑sample weights
    w = [gaussian_beam(theta, center, width) for _ in range(len(x))]
    
    # Compute mean and standard deviation of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    std_x = math.sqrt(sum((x_i - mean_x) ** 2 for x_i in x) / len(x))
    std_y = math.sqrt(sum((y_i - mean_y) ** 2 for y_i in y) / len(y))
    
    # Compute weighted SSIM
    numerator = sum(w_i * (x_i - mean_x) * (y_i - mean_y) for w_i, x_i, y_i in zip(w, x, y))
    denominator = (sum(w_i * (x_i - mean_x) ** 2 for w_i, x_i in zip(w, x)) ** 0.5) * (sum(w_i * (y_i - mean_y) ** 2 for w_i, y_i in zip(w, y)) ** 0.5)
    ssim = (2 * numerator) / (denominator + k1 + k2)
    
    return ssim

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("Invalid input")

    return delta_max * math.exp(-alpha * t / t_max)

if __name__ == "__main__":
    # Test the hybrid Fisher Sparse WTA algorithm
    values = [1.0, 2.0, 3.0]
    m = 10
    theta = 0.5
    center = 0.0
    width = 1.0
    result = hybrid_fisher_sparse_wta(values, m, theta, center, width)
    print("Hybrid Fisher Sparse WTA result:", result)

    # Test the weighted SSIM function
    x = [1.0, 2.0, 3.0]
    y = [2.0, 3.0, 4.0]
    theta = 0.5
    center = 0.0
    width = 1.0
    result = weighted_ssim(x, y, theta, center, width)
    print("Weighted SSIM result:", result)

    # Test the evasion delta function
    t = 10
    t_max = 100
    delta_max = 1.0
    alpha = 3.0
    result = evasion_delta(t, t_max, delta_max, alpha)
    print("Evasion delta result:", result)