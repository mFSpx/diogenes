# DARWIN HAMMER — match 1248, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

"""
Hybrid Rectified Flow and Physarum-RBF Algorithm
=====================================================

This module fuses the Rectified Flow Matching algorithm with the Physarum-RBF 
surrogate model. The mathematical bridge between the two structures is found 
by using the Rectified Flow's straight-line interpolant to generate input 
features for the Physarum-RBF model, which attempts to model the wavefront 
velocity of the graph-propagation engine. The Physarum-RBF model's gradient 
is then used to adapt the Rectified Flow's weight vector.

The Rectified Flow's straight-line interpolant is used to generate input 
vectors for the Physarum-RBF model, which are then used to predict the 
wavefront velocity of the graph-propagation engine. The Physarum-RBF model's 
error is then used to adapt the global weight vector.

Parent A: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py
"""

import numpy as np
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
import sys

class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near-zero entries for sparsity
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Return the grade-k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade-0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0."""
    return t * x1 + (1 - t) * x0

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
        Current weight vector (1-D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e

def physarum_rbf_predict(multivector: Multivector, x: np.ndarray) -> float:
    """Return the Physarum-RBF prediction."""
    # calculate the Physarum-RBF surrogate model's prediction
    # using the multivector's scalar part as the input feature
    scalar_part = multivector.scalar_part()
    return np.exp(-np.linalg.norm(x - scalar_part) ** 2)

def physarum_rbf_update(
    multivector: Multivector,
    x: np.ndarray,
    target: float,
    eta: float = 0.1,
    lambda_: float = 0.5,
) -> Tuple[Multivector, float]:
    """
    Perform one Physarum-RBF update.

    Parameters
    ----------
    multivector : Multivector
        Current multivector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    eta : float
        Learning rate.
    lambda_ : float
        Coupling constant.
    """
    # calculate the Physarum-RBF surrogate model's gradient
    # using the multivector's scalar part as the input feature
    scalar_part = multivector.scalar_part()
    gradient = -2 * (target - np.exp(-np.linalg.norm(x - scalar_part) ** 2)) * (x - scalar_part)
    # update the multivector using the Physarum-RBF gradient
    multivector.components[frozenset()] += eta * (gradient - lambda_ * multivector.scalar_part())
    return multivector, gradient

def hybrid_predict(weights: np.ndarray, multivector: Multivector, x: np.ndarray) -> float:
    """Return the hybrid prediction."""
    # generate input features for the Physarum-RBF model using the Rectified Flow's straight-line interpolant
    interpolant_x = interpolant(x, weights, 0.5)
    # calculate the Physarum-RBF prediction using the input features
    physarum_rbf_pred = physarum_rbf_predict(multivector, interpolant_x)
    # calculate the NLMS prediction using the input features
    nlms_pred = nlms_predict(weights, x)
    # return the hybrid prediction
    return physarum_rbf_pred + nlms_pred

def hybrid_update(
    weights: np.ndarray,
    multivector: Multivector,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eta: float = 0.1,
    lambda_: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Multivector, float]:
    """
    Perform one hybrid update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1-D).
    multivector : Multivector
        Current multivector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eta : float
        Learning rate.
    lambda_ : float
        Coupling constant.
    eps : float
        Small constant to avoid division by zero.
    """
    # generate input features for the Physarum-RBF model using the Rectified Flow's straight-line interpolant
    interpolant_x = interpolant(x, weights, 0.5)
    # calculate the Physarum-RBF prediction using the input features
    physarum_rbf_pred = physarum_rbf_predict(multivector, interpolant_x)
    # calculate the NLMS prediction using the input features
    nlms_pred = nlms_predict(weights, x)
    # calculate the hybrid prediction
    hybrid_pred = physarum_rbf_pred + nlms_pred
    # calculate the hybrid error
    hybrid_error = target - hybrid_pred
    # update the multivector using the Physarum-RBF gradient
    multivector, physarum_rbf_grad = physarum_rbf_update(multivector, interpolant_x, target, eta, lambda_)
    # update the weights using the NLMS update
    weights, nlms_error = nlms_update(weights, x, target, mu, eps)
    return weights, multivector, hybrid_error

if __name__ == "__main__":
    # initialize the weights and multivector
    weights = np.array([1.0, 2.0, 3.0])
    multivector = Multivector({frozenset(): 1.0}, 3)
    # initialize the input feature vector and target
    x = np.array([4.0, 5.0, 6.0])
    target = 10.0
    # perform a hybrid update
    weights, multivector, hybrid_error = hybrid_update(weights, multivector, x, target)
    print("Hybrid error:", hybrid_error)