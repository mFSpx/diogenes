# DARWIN HAMMER — match 4466, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s7.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_korpus_m1956_s2.py (gen6)
# born: 2026-05-29T23:55:56Z

"""
Hybrid algorithm combining:
- Parent A: Structural Similarity Index Measure (SSIM) with morphological scaling,
           Gaussian beam, Fisher score, and Euclidean distance.
- Parent B: Temperature-dependent developmental rate (Schoolfield model) used to
           modulate the SSIM calculation.

Mathematical bridge:
The developmental rate ρ(T) from Parent B is employed as a scaling factor for the
dynamic range in the SSIM calculation.  This allows the temperature-dependent
physiological model to directly influence the perceptual similarity surface of
the SSIM metric, creating a unified system that adapts its sensitivity to the
current temperature.

Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s7.py
Parent B: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_korpus_m1956_s2.py
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np

# ---------- Parent A components ----------

Point = Tuple[float, float]
Edge = Tuple[str, str]

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, sphericity: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width, sphericity), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03, morphology: Morphology = None) -> float:
    if x.shape != y.shape:
        raise ValueError("Input images must have the same dimensions")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    if morphology:
        scale = sphericity_index(morphology.length, morphology.width, morphology.height)
        C1 *= scale
        C2 *= scale

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

# ---------- Parent B components ----------

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15   # reference temperature (Kelvin)

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C (K25)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15        # lower temperature bound (K)
    t_high: float = 307.15       # upper temperature bound (K)
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate ρ(T).

    Args:
        temp_k: Temperature in Kelvin (must be > 0).
        params: Parameter set for the model.

    Returns:
        Scaled developmental rate.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")
    # Helper exponent term
    def exp_term(delta_h: float, t: float) -> float:
        return math.exp(-delta_h / (params.r_cal * t))
    # Full formulation
    rho = params.rho_25 * exp_term(params.delta_h_activation, temp_k) * \
           ((exp_term(params.delta_h_low, temp_k) + exp_term(params.delta_h_high, temp_k)) ** -1)
    return rho

# ---------- Hybrid components ----------

def hybrid_ssim(x: np.ndarray, y: np.ndarray, temp_c: float, morphology: Morphology = None) -> float:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    dynamic_range = 255.0 * rho
    return ssim(x, y, dynamic_range, morphology=morphology)

def hybrid_fisher_score(theta: float, center: float, width: float, sphericity: float, temp_c: float) -> float:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    width_scaled = width * rho
    return fisher_score(theta, center, width_scaled, sphericity)

def hybrid_gaussian_beam(theta: float, center: float, width: float, sphericity: float, temp_c: float) -> float:
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    width_scaled = width * rho
    return gaussian_beam(theta, center, width_scaled, sphericity)

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(256, 256)
    y = np.random.rand(256, 256)
    morphology = Morphology(10.0, 5.0, 2.0, 1.0)
    temp_c = 25.0

    hybrid_ssim_value = hybrid_ssim(x, y, temp_c, morphology)
    hybrid_fisher_score_value = hybrid_fisher_score(0.5, 0.0, 1.0, 0.8, temp_c)
    hybrid_gaussian_beam_value = hybrid_gaussian_beam(0.5, 0.0, 1.0, 0.8, temp_c)

    print("Hybrid SSIM:", hybrid_ssim_value)
    print("Hybrid Fisher Score:", hybrid_fisher_score_value)
    print("Hybrid Gaussian Beam:", hybrid_gaussian_beam_value)