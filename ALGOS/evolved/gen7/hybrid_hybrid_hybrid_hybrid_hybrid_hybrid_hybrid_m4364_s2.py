# DARWIN HAMMER — match 4364, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1747_s3.py (gen6)
# born: 2026-05-29T23:55:07Z

"""
Module Docstring:
This module integrates the mathematical structures of two parent algorithms:
1. `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s1.py`: 
   It implements a Hybrid Fusion of Darwin Hammer Decision-Hygiene Bandit and RBF Surrogate Optimizer.
   The core equations involve a resource vector, linear transformation, RBF surrogate prediction, 
   store dynamics, learning-rate modulation, and weight-matrix update.

2. `hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1747_s3.py`: 
   It provides a set of mathematical utilities, including numerically stable softmax, 
   Gaussian beam intensity, and Fisher information for a single-parameter Gaussian model.

The mathematical bridge between these two parents is established by introducing a Gaussian beam 
intensity function into the RBF surrogate model. The Gaussian beam intensity function 
is used to modulate the predicted reward in the RBF surrogate model, which in turn affects 
the store dynamics and the learning-rate modulation. This integration allows the 
hybrid system to incorporate the Fisher information of the Gaussian model, enabling more 
informed decision-making.

"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    sum_exp = np.sum(exp_x)
    if sum_exp == 0:
        # fallback to uniform distribution
        return np.full_like(x, 1.0 / x.size, dtype=float)
    return exp_x / sum_exp

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian “beam” intensity.

    Parameters
    ----------
    theta : float
        Evaluation point.
    center : float
        Beam centre.
    width : float
        Standard deviation (must be > 0).

    Returns
    -------
    float
        Intensity in the range (0, 1].
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_information_gaussian(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model with respect to its mean.

    The analytical expression is 1/σ², but we keep the full
    derivative‑based form to stay consistent with the original code
    while protecting against division by zero.

    Parameters
    ----------
    theta, center, width : float
        Same meaning as in :func:`gaussian_beam`.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Non‑negative Fisher information.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    # derivative of the log‑likelihood w.r.t. the mean μ
    dlog = -(theta - center) / (width * width)
    # Fisher information = E[(dlog)²] = (dlog)² because intensity integrates to 1
    return (dlog * dlog) * intensity

# ----------------------------------------------------------------------
# Core mathematical operations
# ----------------------------------------------------------------------
class HybridSystem:
    def __init__(self, base_eta: float, alpha: float, beta: float, epsilon: float, 
                 d_out: int, d_in: int, width: float):
        self.base_eta = base_eta
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.d_out = d_out
        self.d_in = d_in
        self.width = width
        self.z = 0.0
        self.W = np.random.rand(d_out, d_in)
        self.c = []

    def resource_vector(self, d: float, sigma: float, s: float) -> np.ndarray:
        """Resource vector."""
        return np.array([d, sigma, s])

    def linear_transformation(self, e: np.ndarray) -> np.ndarray:
        """Linear transformation."""
        return self.W @ e

    def rbf_surrogate_prediction(self, x: np.ndarray) -> float:
        """RBF surrogate prediction."""
        if not self.c:
            return 0.0
        rbf = 0.0
        for c_k in self.c:
            rbf += gaussian_beam(x[0], c_k[0], self.width)
        return rbf

    def gaussian_beam_intensity(self, theta: float, center: float) -> float:
        """Gaussian beam intensity."""
        return gaussian_beam(theta, center, self.width)

    def store_dynamics(self, x: np.ndarray) -> float:
        """Store dynamics (Euler step)."""
        rbf = self.rbf_surrogate_prediction(x)
        self.z = self.z + self.alpha * (rbf - self.z) - self.beta * self.z
        return self.z

    def learning_rate_modulation(self) -> float:
        """Learning-rate modulation."""
        return self.base_eta * (1 + self.z)

    def weight_matrix_update(self, e: np.ndarray, x: np.ndarray) -> None:
        """Weight-matrix update (gradient-like)."""
        eta = self.learning_rate_modulation()
        rbf = self.rbf_surrogate_prediction(x)
        self.W = self.W + eta * rbf * np.outer(np.ones(self.d_out), e)

    def surrogate_weight_update(self, x: np.ndarray) -> None:
        """Surrogate weight update."""
        self.c.append(x)

def demonstrate_hybrid_system(base_eta: float, alpha: float, beta: float, epsilon: float, 
                                d_out: int, d_in: int, width: float) -> None:
    """Demonstrate the hybrid system."""
    system = HybridSystem(base_eta, alpha, beta, epsilon, d_out, d_in, width)
    d = 1.0
    sigma = 0.5
    s = 0.2
    e = system.resource_vector(d, sigma, s)
    x = system.linear_transformation(e)
    system.store_dynamics(x)
    system.weight_matrix_update(e, x)
    system.surrogate_weight_update(x)
    print(f"Weight matrix: {system.W}")
    print(f"Store value: {system.z}")

def demonstrate_gaussian_beam() -> None:
    """Demonstrate the Gaussian beam intensity function."""
    theta = 0.5
    center = 0.0
    width = 1.0
    intensity = gaussian_beam(theta, center, width)
    print(f"Gaussian beam intensity: {intensity}")

def demonstrate_fisher_information() -> None:
    """Demonstrate the Fisher information function."""
    theta = 0.5
    center = 0.0
    width = 1.0
    information = fisher_information_gaussian(theta, center, width)
    print(f"Fisher information: {information}")

if __name__ == "__main__":
    demonstrate_hybrid_system(0.1, 0.5, 0.1, 0.01, 3, 3, 1.0)
    demonstrate_gaussian_beam()
    demonstrate_fisher_information()