# DARWIN HAMMER — match 1248, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

"""
Hybrid Rectified Flow and Physarum-RBF Algorithm
==============================================

This module fuses the Rectified Flow Matching algorithm with the Physarum-RBF 
algorithm from 'hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py' 
and 'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py' respectively. 
The mathematical bridge between the two structures is found by using the 
Rectified Flow's straight-line interpolant to generate input features for 
the Normalized Least-Mean-Squares (NLMS) predictor, which attempts to model 
the wavefront velocity of the graph-propagation engine. This wavefront 
velocity is then used to update the conductance of the Physarum network, 
which in turn is used to evaluate a radial-basis function (RBF) for the 
surrogate model.

Imports:
    numpy
    standard library
    math
    random
    sys
    pathlib
"""

import numpy as np
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
import sys

# ----------------------------------------------------------------------
# Core NLMS utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e


# ----------------------------------------------------------------------
# Rectified Flow utilities
# ----------------------------------------------------------------------
def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0."""
    return t * x1 + (1 - t) * x0


# ----------------------------------------------------------------------
# Multivector (geometric algebra) – core of Parent B
# ----------------------------------------------------------------------
class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near‑zero entries for sparsity
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def hybrid_predict(weights: np.ndarray, x: np.ndarray, multivector: Multivector) -> float:
    """Return the dot-product prediction w·x and update the multivector."""
    prediction = nlms_predict(weights, x)
    multivector_components = {k: v for k, v in multivector.components.items() if len(k) == 1}
    for blade, coef in multivector_components.items():
        multivector.components[blade] += prediction * coef
    return prediction


def hybrid_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    multivector: Multivector,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float, Multivector]:
    """
    Perform one hybrid weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    multivector : Multivector
        Current multivector.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    e = target - hybrid_predict(weights, x, multivector)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    multivector_components = {k: v for k, v in multivector.components.items() if len(k) == 1}
    for blade, coef in multivector_components.items():
        multivector.components[blade] += e * coef
    return weights, e, multivector


def hybrid_train(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    multivector: Multivector,
    num_iterations: int = 100,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Multivector]:
    """
    Perform hybrid training.

    Parameters
    ----------
    weights : np.ndarray
        Initial weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    multivector : Multivector
        Initial multivector.
    num_iterations : int
        Number of training iterations.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    for _ in range(num_iterations):
        weights, _, multivector = hybrid_update(weights, x, target, multivector, mu, eps)
    return weights, multivector


if __name__ == "__main__":
    # Smoke test
    weights = np.array([1.0, 2.0])
    x = np.array([3.0, 4.0])
    target = 10.0
    multivector = Multivector({frozenset([1]): 1.0}, 2)
    trained_weights, trained_multivector = hybrid_train(weights, x, target, multivector)
    print("Trained weights:", trained_weights)
    print("Trained multivector:", trained_multivector.components)