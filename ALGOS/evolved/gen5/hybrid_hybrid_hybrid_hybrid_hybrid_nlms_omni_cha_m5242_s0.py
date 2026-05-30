# DARWIN HAMMER — match 5242, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1353_s4.py (gen4)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s5.py (gen1)
# born: 2026-05-30T00:00:46Z

"""
This module implements a novel hybrid algorithm that mathematically fuses the core topologies of 
hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1353_s4.py and hybrid_nlms_omni_chaotic_sprint_m59_s5.py.
The mathematical bridge between the two structures lies in the use of geometric mean and dot-product 
operations, which are combined to create a unified system that leverages the strengths of both parents.
"""

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class Morphology:
    """Physical shape descriptors of a document."""
    length: float
    width: float
    height: float
    mass: float

def _check_positive(*values: float) -> None:
    """Utility to ensure all supplied values are strictly positive."""
    for v in values:
        if v <= 0:
            raise ValueError("All geometric parameters must be > 0")

def sphericity_index(length: float, width: float, height: float) -> float:
    """Compactness ratio: geometric mean of dimensions over the longest side."""
    _check_positive(length, width, height)
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio: larger when height is small relative to the base."""
    _check_positive(length, width, height)
    return (length + width) / (2.0 * height)

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Approximate time for a body to right itself after being tipped.
    A simple physics‑inspired surrogate.
    """
    _check_positive(m.mass, neck_lever)
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass * (m.length ** b)) / (k * neck_lever * fi)

def compute_recovery_priority(m: Morphology) -> float:
    """
    Smooth priority in [0, 1] that grows with “good” morphology.
    Combines sphericity, flatness and righting time via a weighted geometric mean.
    """
    sph = sphericity_index(m.length, m.width, m.height)          # ∈ (0,1]
    flt = flatness_index(m.length, m.width, m.height)           # >0
    rti = righting_time_index(m)                                # >0

    # Normalise flatness to (0,1) using a soft‑clipping exponential.
    flt = 1 / (1 + exp(-flt))

    # Weighted geometric mean
    weights = np.array([0.4, 0.3, 0.3])
    values = np.array([sph, flt, rti])
    return np.prod(values ** weights) ** (1.0 / sum(weights))

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)

def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Predictions and errors
    preds = X @ weights
    errors = targets - preds

    # Normalized step for each sample
    powers = np.sum(X * X, axis=1) + eps  # shape (N,)
    steps = (mu * errors / powers)[:, None] * X   # shape (N, d)

    # Aggregate the per‑sample steps
    delta_w = steps.sum(axis=0)
    new_weights = weights + delta_w
    return new_weights, errors

def hybrid_predict(m: Morphology, weights: np.ndarray) -> float:
    """
    Hybrid prediction function that combines the geometry-based recovery priority with the NLMS prediction.
    """
    priority = compute_recovery_priority(m)
    nlms_pred = nlms_predict(weights, np.array([m.length, m.width, m.height]))
    return priority * nlms_pred

def hybrid_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    morphologies: List[Morphology],
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid update function that combines the NLMS update with the geometry-based recovery priority.
    """
    new_weights, errors = nlms_batch_update(weights, X, targets, mu, eps)
    priorities = [compute_recovery_priority(m) for m in morphologies]
    weighted_errors = errors * np.array(priorities)
    return new_weights, weighted_errors

def generate_synthetic_data(num_samples: int, num_features: int) -> Tuple[np.ndarray, np.ndarray, List[Morphology]]:
    """
    Generate synthetic data for testing the hybrid algorithm.
    """
    X = np.random.rand(num_samples, num_features)
    targets = np.random.rand(num_samples)
    morphologies = [Morphology(length=np.random.rand(), width=np.random.rand(), height=np.random.rand(), mass=np.random.rand()) for _ in range(num_samples)]
    return X, targets, morphologies

if __name__ == "__main__":
    num_samples = 10
    num_features = 3
    X, targets, morphologies = generate_synthetic_data(num_samples, num_features)
    weights = np.random.rand(num_features)
    new_weights, errors = hybrid_update(weights, X, targets, morphologies)
    print("Updated weights:", new_weights)
    print("Prediction errors:", errors)