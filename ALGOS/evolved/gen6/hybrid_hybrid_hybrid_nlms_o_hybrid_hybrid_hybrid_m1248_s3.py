# DARWIN HAMMER — match 1248, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

"""
Hybrid Rectified Flow - Physarum - NLMS Algorithm
====================================================

This module fuses the Rectified Flow Matching algorithm (Parent A: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py) 
with the Physarum‑RBF Algorithm (Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py) 
and the Normalized Least-Mean-Squares (NLMS) adaptive filter.

The mathematical bridge between the Rectified Flow and Physarum-RBF structures 
is found by using the Rectified Flow's straight-line interpolant to generate 
input features for the NLMS predictor, which attempts to model the wavefront 
velocity of the Physarum network. The Physarum network state is encoded as a 
multivector **C** = Σ g_i e_i, where g_i are edge conductances and e_i are 
orthogonal basis vectors of a Clifford algebra. The surrogate model of 
Parent B provides a scalar functional 𝔈(**C**) ≈ free‑energy of the network 
by evaluating a radial-basis function (RBF) on the conductance vector (the 
scalar part of **C**).

The hybrid system integrates the governing equations of both parents: 
the NLMS update rule, the Rectified Flow interpolant, and the Physarum 
update rule with RBF-based free-energy approximation.

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

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:.2f}{label}")
        return " + ".join(terms)

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
# RBF-based free-energy approximation
# ----------------------------------------------------------------------

def rbf_free_energy(conductances: np.ndarray) -> float:
    """
    Approximate free-energy of the Physarum network using a radial-basis function.

    Parameters
    ----------
    conductances : np.ndarray
        Vector of edge conductances.
    """
    # Simple RBF implementation (e.g., Gaussian)
    sigma = 1.0
    return np.sum(np.exp(-conductances ** 2 / (2 * sigma ** 2)))

# ----------------------------------------------------------------------
# Hybrid Physarum - NLMS - Rectified Flow update
# ----------------------------------------------------------------------

def hybrid_update(
    multivector: Multivector,
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eta: float = 0.1,
    lambda_: float = 0.5,
) -> Tuple[Multivector, np.ndarray, float]:
    """
    Perform one hybrid update.

    Parameters
    ----------
    multivector : Multivector
        Current Physarum network state.
    weights : np.ndarray
        Current NLMS weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        NLMS learning rate in (0, 2).
    eta : float
        Physarum learning rate.
    lambda_ : float
        Coupling constant.
    """
    # Compute RBF-based free-energy approximation
    conductances = np.array([multivector.components.get(frozenset({i}), 0.0) for i in range(multivector.n)])
    free_energy = rbf_free_energy(conductances)

    # Compute gradient of free-energy w.r.t. conductances
    grad_free_energy = -2 * conductances / (2 * 1.0 ** 2)

    # Update Physarum network state
    new_multivector_components = multivector.components.copy()
    for i in range(multivector.n):
        new_multivector_components[frozenset({i})] = multivector.components.get(frozenset({i}), 0.0) + eta * (multivector.components.get(frozenset({i}), 0.0) - lambda_ * grad_free_energy[i])
    new_multivector = Multivector(new_multivector_components, multivector.n)

    # Update NLMS weights
    new_weights, e = nlms_update(weights, x, target, mu)

    return new_multivector, new_weights, e

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize Physarum network state and NLMS weights
    multivector = Multivector({frozenset({0}): 1.0}, 2)
    weights = np.array([0.5, 0.5])

    # Example input feature vector and target output
    x = np.array([1.0, 2.0])
    target = 3.0

    # Perform hybrid update
    new_multivector, new_weights, e = hybrid_update(multivector, weights, x, target)

    print("New Physarum network state:", new_multivector)
    print("New NLMS weights:", new_weights)
    print("NLMS error:", e)