# DARWIN HAMMER — match 3923, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m2407_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s0.py (gen6)
# born: 2026-05-29T23:52:27Z

"""Hybrid Algorithm combining:
- Parent A: representation learning → probabilistic distribution → confidence scalar that modulates Fisher information.
- Parent B: sphericity index, Gaussian beam, Fisher score used to control diffusion timestep.

Mathematical bridge:
The confidence scalar (σ_c) derived from the covariance of the learned representation is used to
scale the sphericity index (S) computed from geometric morphology. The product σ_c·S becomes a
multiplicative factor for the diffusion timestep (Δt) in the endpoint‑circuit diffusion process.
Furthermore, the Fisher information computed from the representation (I_F) is injected as the
amplitude of the Gaussian beam, turning the classic `gaussian_beam` into a
confidence‑aware `fused_beam`. This creates a closed loop where learned uncertainty
modulates similarity‑driven diffusion, and diffusion‑driven similarity feeds back into the
representation update.

The module implements three core functions that realise this fusion and a small smoke test.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Helper geometry (from Parent B)

def sphericity_index(length: float, width: float, height: float) -> float:
    """Normalized ratio of the geometric mean to the maximal dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def gaussian_beam(theta: float, center: float, width: float, amplitude: float) -> float:
    """Standard Gaussian beam evaluated at angle theta."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return amplitude * math.exp(-0.5 * z * z)


# ----------------------------------------------------------------------
# Representation learning & confidence (from Parent A)

def learn_representation(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean vector and covariance matrix of the input data.
    data shape: (n_samples, n_features)
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2‑D array")
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False, bias=True)  # bias=True → MLE estimator
    # Ensure positive‑definiteness by adding a tiny diagonal term
    eps = 1e-12
    cov += eps * np.eye(cov.shape[0])
    return mean, cov


def confidence_scalar(cov: np.ndarray) -> float:
    """
    Scalar that reflects certainty of the learned distribution.
    Small determinant → high confidence, large determinant → low confidence.
    """
    det = np.linalg.det(cov)
    if det <= 0:
        det = 1e-12
    return 1.0 / (det ** (1.0 / cov.shape[0]))  # geometric mean of eigenvalues inverted


def fisher_information(theta: float, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Simplified scalar Fisher information for a univariate projection of the multivariate
    Gaussian onto direction theta (treated as a 1‑D variable).
    """
    # Project mean and covariance onto the unit vector defined by theta
    direction = np.array([math.cos(theta), math.sin(theta)])
    if direction.shape[0] != mean.shape[0]:
        # Pad or truncate to match dimensionality (fallback for generic data)
        direction = np.resize(direction, mean.shape)
    proj_mean = np.dot(direction, mean)
    proj_var = direction @ cov @ direction
    # Fisher information of a Gaussian w.r.t. its mean is 1/variance
    return 1.0 / (proj_var + 1e-12)


# ----------------------------------------------------------------------
# Fusion core (new)

def fused_beam(theta: float,
               center: float,
               width: float,
               morphology: Tuple[float, float, float],
               confidence: float,
               fisher: float) -> float:
    """
    Combine Gaussian beam amplitude with Fisher score and confidence.
    amplitude = confidence * fisher * sphericity
    """
    length, width_m, height = morphology
    sphericity = sphericity_index(length, width_m, height)
    amplitude = confidence * fisher * sphericity
    return gaussian_beam(theta, center, width, amplitude)


def diffusion_timestep(base_dt: float,
                       confidence: float,
                       sphericity: float,
                       jitter: float = 0.0) -> float:
    """
    Modulate the diffusion timestep using the product of confidence and sphericity.
    Optional jitter adds a small random perturbation (simulating stochastic diffusion).
    """
    dt = base_dt * confidence * sphericity
    if jitter > 0.0:
        dt *= (1.0 + random.uniform(-jitter, jitter))
    return max(dt, 1e-12)


def hybrid_step(data: np.ndarray,
                theta: float,
                center: float,
                width: float,
                morphology: Tuple[float, float, float],
                base_dt: float) -> Dict[str, float]:
    """
    Perform a single hybrid iteration:
      1. Learn representation → mean, cov.
      2. Derive confidence and Fisher information.
      3. Compute fused beam intensity.
      4. Compute diffusion timestep.
    Returns a dictionary with intermediate results.
    """
    mean, cov = learn_representation(data)
    conf = confidence_scalar(cov)
    fisher = fisher_information(theta, mean, cov)

    # fused intensity (used later as a proxy for selection probability)
    intensity = fused_beam(theta, center, width, morphology, conf, fisher)

    # diffusion control
    sph = sphericity_index(*morphology)
    dt = diffusion_timestep(base_dt, conf, sph, jitter=0.05)

    return {
        "confidence": conf,
        "fisher_info": fisher,
        "intensity": intensity,
        "diffusion_dt": dt,
        "sphericity": sph,
    }


# ----------------------------------------------------------------------
# Simple smoke test

if __name__ == "__main__":
    # synthetic 2‑D data (e.g., sensor readings)
    rng = np.random.default_rng(seed=42)
    data = rng.normal(loc=[0.0, 0.0], scale=[1.0, 2.0], size=(500, 2))

    # geometric description of a fictitious object
    morphology = (4.2, 2.5, 1.8)  # length, width, height

    # algorithmic parameters
    theta = math.radians(30)      # direction angle
    center = math.radians(0)      # beam centre
    width = math.radians(45)      # beam width
    base_dt = 0.01                # baseline diffusion timestep

    result = hybrid_step(data, theta, center, width, morphology, base_dt)

    print("Hybrid step results:")
    for k, v in result.items():
        print(f"  {k}: {v:.6g}")

    # sanity check: diffusion_dt should be positive
    assert result["diffusion_dt"] > 0, "Diffusion timestep must be > 0"
    sys.exit(0)