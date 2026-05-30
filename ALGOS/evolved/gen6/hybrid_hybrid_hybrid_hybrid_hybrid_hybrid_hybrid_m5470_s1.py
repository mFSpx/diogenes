# DARWIN HAMMER — match 5470, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m2628_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s3.py (gen5)
# born: 2026-05-30T00:02:14Z

"""
Hybrid Fisher-RBF surrogate Module
================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m2628_s1.py``  
  Provides a *Gaussian beam* intensity model, a *Fisher information* scorer and a
  JEPA-style energy term that measures representation mismatch.

* **Parent B** – ``hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s3.py``  
  Supplies a radial basis function (RBF) surrogate model.

The mathematical bridge between these two parents is found in the ability to use the
RBF surrogate model to approximate the Fisher information and JEPA-style energy term.
By doing so, we can leverage the strength of the RBF model in approximating complex
functions and the Fisher information in measuring the uncertainty of the model.

This module implements this unified view through three main functions: 
``gaussian_beam``, ``weighted_fisher_score``, and ``hybrid_rbf_surrogate``.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, List, Dict
import numpy as np

# ----------------------------------------------------------------------
# Shared statistical core (Gaussian beam & Fisher information)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian intensity I(θ) of a beam centred at *center* with *width*.
    
    Args:
    theta (float): The point at which to evaluate the Gaussian beam.
    center (float): The center of the Gaussian beam.
    width (float): The width of the Gaussian beam.
    
    Returns:
    float: The intensity of the Gaussian beam at theta.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-((z) ** 2))

# ----------------------------------------------------------------------
# Parent B – RBF surrogate utilities (re‑implemented)
# ----------------------------------------------------------------------
Vector = List[float]

def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """
    Gaussian radial basis function.
    
    Args:
    r (float): The distance from the center of the RBF.
    epsilon (float): The width of the RBF.
    
    Returns:
    float: The value of the Gaussian RBF.
    """
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: Vector, b: Vector) -> float:
    """
    Euclidean distance between two vectors.
    
    Args:
    a (Vector): The first vector.
    b (Vector): The second vector.
    
    Returns:
    float: The Euclidean distance between the two vectors.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFModel:
    """
    Radial-basis-function surrogate.
    
    Attributes:
    centers (List[Tuple[float, ...]]): The centers of the RBFs.
    weights (List[float]): The weights of the RBFs.
    epsilon (float): The width of the RBFs.
    """
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """
        Evaluate surrogate at point x.
        
        Args:
        x (Vector): The point at which to evaluate the surrogate.
        
        Returns:
        float: The value of the surrogate at x.
        """
        return sum(
            w * _gaussian(_euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector],
            values: Iterable[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFModel:
    """
    Fit an RBF surrogate to (points, values).
    
    Args:
    points (Iterable[Vector]): The points at which to fit the surrogate.
    values (Iterable[float]): The values of the function at the points.
    epsilon (float): The width of the RBFs.
    ridge (float): The regularization parameter.
    
    Returns:
    RBFModel: The fitted RBF surrogate.
    """
    centers = [tuple(map(float, p)) for p in points]
    y = [float(v) for v in values]
    A = [[_gaussian(_euclidean(p1, p2), epsilon) for p1 in points] for p2 in points]
    weights = np.linalg.solve(np.array(A) + ridge * np.eye(len(points)), np.array(y))
    return RBFModel(centers, weights.tolist(), epsilon)

# ----------------------------------------------------------------------
# Hybrid Fisher-RBF surrogate
# ----------------------------------------------------------------------
def weighted_fisher_score(points: Iterable[Vector], values: Iterable[float], epsilon: float = 1.0) -> float:
    """
    Calculate the weighted Fisher score using the RBF surrogate.
    
    Args:
    points (Iterable[Vector]): The points at which to calculate the Fisher score.
    values (Iterable[float]): The values of the function at the points.
    epsilon (float): The width of the RBFs.
    
    Returns:
    float: The weighted Fisher score.
    """
    rbf_model = fit_rbf(points, values, epsilon)
    fisher_score = 0.0
    for point in points:
        fisher_score += rbf_model.predict(point)
    return fisher_score

def hybrid_rbf_surrogate(points: Iterable[Vector], values: Iterable[float], epsilon: float = 1.0) -> RBFModel:
    """
    Create a hybrid RBF surrogate that incorporates the Fisher information.
    
    Args:
    points (Iterable[Vector]): The points at which to fit the surrogate.
    values (Iterable[float]): The values of the function at the points.
    epsilon (float): The width of the RBFs.
    
    Returns:
    RBFModel: The hybrid RBF surrogate.
    """
    weights = [gaussian_beam(point[0], 0.0, 1.0) for point in points]
    weighted_values = [weight * value for weight, value in zip(weights, values)]
    return fit_rbf(points, weighted_values, epsilon)

if __name__ == "__main__":
    points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    values = [10.0, 20.0, 30.0]
    epsilon = 1.0
    fisher_score = weighted_fisher_score(points, values, epsilon)
    hybrid_model = hybrid_rbf_surrogate(points, values, epsilon)
    print(f"Weighted Fisher score: {fisher_score}")
    print(f"Hybrid RBF surrogate: {hybrid_model}")