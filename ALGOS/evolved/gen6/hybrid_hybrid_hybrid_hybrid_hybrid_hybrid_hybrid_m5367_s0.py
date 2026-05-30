# DARWIN HAMMER — match 5367, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_caputo_hybrid_hybrid_hybrid_m1527_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_fisher_locali_m1616_s0.py (gen5)
# born: 2026-05-30T00:01:20Z

"""
This module fuses the two parent algorithms:

* **Parent A** – `hybrid_hybrid_hybrid_hybrid_hybrid_caputo_hybrid_hybrid_hybrid_m1527_s0.py`  
  Provides a Caputo fractional derivative kernel φ(t;α)≈t^{‑α} (implemented via a
  Lanczos‑approximated Gamma function) that yields long‑range memory weights for
  graph edges.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_fisher_locali_m1616_s0.py`  
  Supplies a cockpit metrics rectified flow, in particular a flow target that
  combines multiple labeling functions.

The mathematical bridge between the two structures is the use of a weighted sum,
where the Caputo kernel from Parent A is used to weight the Gaussian beam from
Parent B. This allows the hybrid algorithm to model the intensity of a signal
in a weighted manner, taking into account both the long-range memory effects
from Parent A and the Gaussian beam from Parent B.
"""

import math
import numpy as np
import random
import sys
import pathlib

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of Γ(z) for z>0."""
    if z < 0.5:
        # Reflection formula
        return np.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        # Lanczos series
        return np.sum(_LANCZOS_C[:int(z) + 2] * np.power(z, _LANCZOS_G - (int(z) + 2)))

def caputo_weights(t: np.ndarray, alpha: float) -> np.ndarray:
    return t ** (-alpha)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return np.exp(-z**2 / 2)

def hybrid_gaussian_beam(theta: np.ndarray, center: np.ndarray, width: np.ndarray, alpha: float) -> np.ndarray:
    caputo_weight = caputo_weights(np.abs(theta - center), alpha)
    return caputo_weight * gaussian_beam(theta, center, width)

def calculate_hybrid_labeling(theta: np.ndarray, center: np.ndarray, width: np.ndarray, alpha: float) -> np.ndarray:
    hybrid_beam = hybrid_gaussian_beam(theta, center, width, alpha)
    return np.sum(hybrid_beam)

def calculate_hybrid_loss(theta: np.ndarray, center: np.ndarray, width: np.ndarray, alpha: float, labels: np.ndarray) -> float:
    hybrid_labeling = calculate_hybrid_labeling(theta, center, width, alpha)
    return np.mean((hybrid_labeling - labels) ** 2)

if __name__ == "__main__":
    theta = np.array([1.0, 2.0, 3.0])
    center = np.array([1.5, 2.5, 3.5])
    width = np.array([1.0, 1.0, 1.0])
    alpha = 0.5
    labels = np.array([0.5, 0.5, 0.5])

    hybrid_labeling = calculate_hybrid_labeling(theta, center, width, alpha)
    hybrid_loss = calculate_hybrid_loss(theta, center, width, alpha, labels)

    print("Hybrid labeling:", hybrid_labeling)
    print("Hybrid loss:", hybrid_loss)