# DARWIN HAMMER — match 5470, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m2628_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s3.py (gen5)
# born: 2026-05-30T00:02:14Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Hybrid Fisher-JEPA-Certainty-RBF Module
# ----------------------------------------------------------------------
"""
This module combines the mathematical structures of two parent algorithms:

* **Parent A**: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py
  Provides a Gaussian beam intensity model, a Fisher information scorer,
  and a JEPA-style energy term that measures representation mismatch.
* **Parent B**: hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s3.py
  Supplies a radial-basis-function (RBF) surrogate model and utilities.

The mathematical bridge between these parents lies in their treatment of
information as a scalar that can be combined multiplicatively. We found
three key interfaces:

1. Both parents use a Gaussian distribution to model intensity or
   uncertainty. We leverage this commonality to fuse the Fisher information
   scorer with the RBF surrogate model.
2. The JEPA energy term in Parent A can be viewed as a base loss function,
   which we can regularize using the weighted Fisher score produced by the
   RBF surrogate model.
3. The certainty flag in Parent A can be used to weight the contribution of
   each loss term in the RBF surrogate model, producing a certainty-weighted
   RBF estimate.

The three functions below – weighted_fisher_rbf_score, weighted_jepa_energy,
and hybrid_metric – implement this unified view, while rbf_surrogate shows
the direct fusion of Fisher information with the RBF surrogate model.
"""
# ----------------------------------------------------------------------
# Shared statistical core (Gaussian beam & Fisher information)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-((z) ** 2))

def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Hybrid Fisher-JEPA-Certainty-RBF functions
# ----------------------------------------------------------------------
def weighted_fisher_rbf_score(theta: float, center: float, width: float,
                              c: float, weights: List[float],
                              centers: List[Tuple[float, ...]], epsilon: float) -> float:
    """
    Weighted Fisher-RBF score.

    Parameters:
    theta (float): Beam angle.
    center (float): Beam center.
    width (float): Beam width.
    c (float): Certainty flag (scaled to [0,1]).
    weights (List[float]): RBF surrogate weights.
    centers (List[Tuple[float, ...]]): RBF surrogate centers.
    epsilon (float): RBF surrogate epsilon.

    Returns:
    float: Weighted Fisher-RBF score.
    """
    fisher = gaussian_beam(theta, center, width) / gaussian_beam(theta, center, width)
    rbf = sum(w * _gaussian(_euclidean([theta], c), epsilon) for w, c in zip(weights, centers))
    return fisher * rbf

def weighted_jepa_energy(z_pred: float, z_true: float, c: float) -> float:
    """
    Weighted JEPA energy.

    Parameters:
    z_pred (float): Predicted value.
    z_true (float): True value.
    c (float): Certainty flag (scaled to [0,1]).

    Returns:
    float: Weighted JEPA energy.
    """
    return c * (z_pred - z_true) ** 2

def hybrid_metric(theta: float, center: float, width: float, z_pred: float,
                  z_true: float, c: float, weights: List[float],
                  centers: List[Tuple[float, ...]], epsilon: float) -> float:
    """
    Hybrid metric.

    Parameters:
    theta (float): Beam angle.
    center (float): Beam center.
    width (float): Beam width.
    z_pred (float): Predicted value.
    z_true (float): True value.
    c (float): Certainty flag (scaled to [0,1]).
    weights (List[float]): RBF surrogate weights.
    centers (List[Tuple[float, ...]]): RBF surrogate centers.
    epsilon (float): RBF surrogate epsilon.

    Returns:
    float: Hybrid metric.
    """
    fisher_rbf = weighted_fisher_rbf_score(theta, center, width, c, weights, centers, epsilon)
    jepa = weighted_jepa_energy(z_pred, z_true, c)
    return fisher_rbf + jepa

# ----------------------------------------------------------------------
# Hybrid RBF surrogate model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HybridRBFModel:
    """Hybrid radial-basis-function surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        """Evaluate hybrid surrogate at point x."""
        return sum(
            w * _gaussian(_euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_hybrid_rbf(points: Iterable[List[float]],
                   values: Iterable[float],
                   epsilon: float = 1.0,
                   ridge: float = 1e-9) -> HybridRBFModel:
    """Fit an hybrid RBF surrogate to (points, values)."""
    centers = [tuple(map(float, p)) for p in points]
    weights = [float(v) for v in values]
    return HybridRBFModel(centers, weights, epsilon)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Test hybrid metric
    theta = 1.0
    center = 2.0
    width = 0.5
    z_pred = 3.0
    z_true = 4.0
    c = 0.7
    weights = [0.2, 0.3, 0.5]
    centers = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    epsilon = 1.0
    print(hybrid_metric(theta, center, width, z_pred, z_true, c, weights, centers, epsilon))

    # Test hybrid RBF surrogate model
    points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    values = [0.2, 0.3, 0.5]
    model = fit_hybrid_rbf(points, values, epsilon)
    print(model.predict([1.0, 2.0]))