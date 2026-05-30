# DARWIN HAMMER — match 2744, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s2.py (gen4)
# born: 2026-05-29T23:45:38Z

"""Hybrid RBF‑Fisher‑SSIM Model

This module fuses the two parent algorithms:

* **Parent A** – an RBF surrogate model updated with a Normalised Least‑Mean‑Squares (NLMS)
  rule.
* **Parent B** – a Fisher‑information weighting scheme for Gaussian beams together with a
  Structural Similarity Index (SSIM) that modulates associative‑memory updates.

**Mathematical bridge**

For every RBF centre *cᵢ* we compute a Fisher‑information score
`Fᵢ = fisher_score(θ = ‖x‑cᵢ‖, centre = 0, width = w)`.  
`Fᵢ` acts as a per‑centre adaptive learning‑rate factor inside the NLMS weight update.
A global similarity factor `S = ssim(x, μ_c)` – where `μ_c` is the mean of all centres –
scales the whole update, thus marrying the continuous Fisher weighting (Parent B) with
the discrete NLMS adaptation of the RBF surrogate (Parent A).  The resulting hybrid
update rule is


Δw = (μ · S · F) ⊙ (e·x / (‖x‖²+ε))


where `μ` is the base NLMS step size, `e` the prediction error, `⊙` element‑wise
multiplication and `F` the vector of Fisher scores.  The prediction itself remains the
standard RBF sum, and an energy metric weighted by Fisher scores provides a unified
objective for training.

The three core functions below demonstrate prediction, hybrid NLMS update and a
Fisher‑weighted energy evaluation.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

Vector = np.ndarray


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial‑basis Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a - b))


def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a linear system Ax = b using Gaussian elimination (no external libs)."""
    n = len(b)
    m = np.hstack((a.astype(float), b[:, None].astype(float)))
    for col in range(n):
        pivot = np.argmax(np.abs(m[col:, col])) + col
        if abs(m[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[[col, pivot]] = m[[pivot, col]]
        m[col] = m[col] / m[col, col]
        for row in range(n):
            if row == col:
                continue
            factor = m[row, col]
            m[row] -= factor * m[col]
    return m[:, -1]


@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate with fixed centres and adaptable weights."""
    centers: np.ndarray          # shape (n_centres, dim)
    weights: np.ndarray          # shape (n_centres,)
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Weighted sum of Gaussian RBFs evaluated at x."""
        kernels = np.array([gaussian(euclidean(x, c), self.epsilon) for c in self.centers])
        return float(np.dot(self.weights, kernels))


def gaussian_beam(theta: float, centre: float, width: float) -> float:
    """Gaussian beam intensity (Parent B)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - centre) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, centre: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam (Parent B)."""
    intensity = max(gaussian_beam(theta, centre, width), eps)
    derivative = intensity * (-(theta - centre) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """One‑dimensional Structural Similarity Index (Parent B)."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


def compute_fisher_weights(surrogate: RBFSurrogate,
                           x: Vector,
                           width: float = 1.0) -> np.ndarray:
    """
    Fisher information for each centre based on the distance from the input vector.
    Returns an array of shape (n_centres,).
    """
    distances = np.linalg.norm(surrogate.centers - x, axis=1)
    return np.array([fisher_score(theta=d, centre=0.0, width=width) for d in distances])


def hybrid_nlms_update(surrogate: RBFSurrogate,
                       x: Vector,
                       target: float,
                       mu: float = 0.5,
                       eps: float = 1e-9,
                       width: float = 1.0) -> RBFSurrogate:
    """
    Perform a single NLMS weight update on the RBF surrogate,
    modulated by per‑centre Fisher scores (continuous weighting) and a global
    SSIM similarity factor (discrete topology).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Prediction and error
    y = surrogate.predict(x)
    error = target - y

    # Power normalisation
    power = np.dot(x, x) + eps

    # Fisher‑based per‑centre scaling
    fisher_vec = compute_fisher_weights(surrogate, x, width)

    # Global SSIM similarity (reference = mean of centres)
    reference = np.mean(surrogate.centers, axis=0)
    similarity = ssim(x, reference)

    # NLMS delta (element‑wise multiplication with Fisher vector)
    delta = (mu * similarity * fisher_vec) * (error * x / power)

    new_weights = surrogate.weights + delta
    return RBFSurrogate(surrogate.centers, new_weights, surrogate.epsilon)


def hybrid_energy(surrogate: RBFSurrogate,
                  X: np.ndarray,
                  Y: np.ndarray,
                  width: float = 1.0) -> float:
    """
    Fisher‑weighted mean‑squared error over a dataset.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples")
    total = 0.0
    for x, y in zip(X, Y):
        pred = surrogate.predict(x)
        err = y - pred
        fisher_vec = compute_fisher_weights(surrogate, x, width)
        total += np.sum(fisher_vec) * (err ** 2)
    return total / X.shape[0]


def train_hybrid_surrogate(centers: np.ndarray,
                           X: np.ndarray,
                           Y: np.ndarray,
                           epochs: int = 10,
                           mu: float = 0.5,
                           width: float = 1.0) -> RBFSurrogate:
    """
    Initialise an RBFSurrogate and train it with the hybrid NLMS rule.
    Returns the trained surrogate.
    """
    n_centres = centers.shape[0]
    # Initialise weights randomly
    init_weights = np.random.randn(n_centres)
    surrogate = RBFSurrogate(centers, init_weights)

    for _ in range(epochs):
        # Shuffle data each epoch
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        for idx in indices:
            surrogate = hybrid_nlms_update(surrogate, X[idx], Y[idx],
                                           mu=mu, width=width)
    return surrogate


if __name__ == "__main__":
    # Simple smoke test
    dim = 5
    n_centres = 8
    n_samples = 50

    # Random centres and synthetic data
    rng = np.random.default_rng(42)
    centres = rng.normal(size=(n_centres, dim))
    X = rng.normal(size=(n_samples, dim))
    # Underlying true function: linear combination of a hidden weight vector
    true_w = rng.normal(size=n_centres)
    surrogate_true = RBFSurrogate(centres, true_w)
    Y = np.array([surrogate_true.predict(x) for x in X]) + 0.05 * rng.normal(size=n_samples)

    # Train hybrid surrogate
    trained = train_hybrid_surrogate(centres, X, Y, epochs=5, mu=0.7, width=1.2)

    # Evaluate energy before and after training
    energy_before = hybrid_energy(RBFSurrogate(centres, np.random.randn(n_centres)), X, Y)
    energy_after = hybrid_energy(trained, X, Y)

    print(f"Energy before training: {energy_before:.6f}")
    print(f"Energy after  training: {energy_after:.6f}")
    print("Hybrid training completed without errors.")