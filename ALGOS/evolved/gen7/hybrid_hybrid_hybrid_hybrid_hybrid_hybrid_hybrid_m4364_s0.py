# DARWIN HAMMER — match 4364, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1747_s3.py (gen6)
# born: 2026-05-29T23:55:07Z

"""
Module for the hybrid algorithm combining the core topologies of 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s1.py and 
hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m1747_s3.py.

The mathematical bridge between the two algorithms lies in the use of 
the RBF surrogate model from the first parent and the Gaussian beam 
intensity and Fisher information functions from the second parent. 
The RBF surrogate model is used to predict a reward, which is then 
used to drive the store dynamics and update the weight matrix. 
The Gaussian beam intensity and Fisher information functions are 
used to introduce uncertainty and information gain into the 
reward prediction and weight update processes.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# Define types
Vector = list[float]

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


def hybrid_fusion(
    d: float, 
    beta: float, 
    sigma: float, 
    s: float, 
    W: np.ndarray, 
    z: float, 
    base_eta: float, 
    alpha: float, 
    eps: float, 
    width: float, 
    center: float
) -> tuple[float, np.ndarray, float]:
    """
    Hybrid fusion of the two parent algorithms.

    Parameters
    ----------
    d, beta, sigma, s : float
        Parameters for the resource vector.
    W : np.ndarray
        Weight matrix.
    z : float
        Store dynamics variable.
    base_eta : float
        Base learning rate.
    alpha : float
        Store dynamics coefficient.
    eps : float
        Small constant to avoid division by zero.
    width : float
        Standard deviation for the Gaussian beam.
    center : float
        Center for the Gaussian beam.

    Returns
    -------
    tuple[float, np.ndarray, float]
        Predicted reward, updated weight matrix, and updated store dynamics variable.
    """
    # Resource vector
    e = np.array([d, beta * sigma, s])

    # Linear transformation
    x = W @ e

    # RBF surrogate prediction
    r_hat = gaussian_beam(x[0], center, width)

    # Store dynamics
    z_new = z + alpha * (r_hat - z)

    # Learning rate modulation
    eta = base_eta * (1 + z_new)

    # Weight matrix update
    W_new = W + eta * r_hat * (e[:, None] * np.ones(W.shape[1]))

    # Fisher information
    fisher_info = fisher_information_gaussian(x[0], center, width)

    return r_hat, W_new, z_new


def demo_fusion(
    d: float, 
    beta: float, 
    sigma: float, 
    s: float, 
    W: np.ndarray, 
    z: float, 
    base_eta: float, 
    alpha: float, 
    eps: float, 
    width: float, 
    center: float
) -> None:
    """
    Demonstration of the hybrid fusion algorithm.

    Parameters
    ----------
    d, beta, sigma, s : float
        Parameters for the resource vector.
    W : np.ndarray
        Weight matrix.
    z : float
        Store dynamics variable.
    base_eta : float
        Base learning rate.
    alpha : float
        Store dynamics coefficient.
    eps : float
        Small constant to avoid division by zero.
    width : float
        Standard deviation for the Gaussian beam.
    center : float
        Center for the Gaussian beam.
    """
    r_hat, W_new, z_new = hybrid_fusion(d, beta, sigma, s, W, z, base_eta, alpha, eps, width, center)
    print(f"Predicted reward: {r_hat}")
    print(f"Updated weight matrix:\n{W_new}")
    print(f"Updated store dynamics variable: {z_new}")


def main() -> None:
    # Example usage
    d = 1.0
    beta = 0.5
    sigma = 0.2
    s = 0.1
    W = np.random.rand(3, 3)
    z = 0.5
    base_eta = 0.1
    alpha = 0.2
    eps = 1e-12
    width = 1.0
    center = 0.5

    demo_fusion(d, beta, sigma, s, W, z, base_eta, alpha, eps, width, center)


if __name__ == "__main__":
    main()