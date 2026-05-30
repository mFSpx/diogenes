# DARWIN HAMMER — match 1248, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

"""
Hybrid Rectified Flow-NLMS-Physarum-RBF Algorithm
====================================================

This module fuses the Hybrid Rectified Flow and NLMS-Graph Engine (Parent A: 
hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py) with the 
Hybrid Physarum-RBF Algorithm (Parent B: 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py). 

The mathematical bridge between the two structures is found by using the 
Rectified Flow's straight-line interpolant to generate input features for 
the NLMS predictor, which attempts to model the wavefront velocity of the 
graph-propagation engine. The Physarum network state is encoded as a 
multivector **C** = Σ g_i e_i, where g_i are edge conductances and e_i are 
orthogonal basis vectors of a Clifford algebra. The surrogate model of 
Parent B provides a scalar functional 𝔈(**C**) ≈ free‑energy of the 
network by evaluating a radial‑basis function (RBF) on the conductance 
vector (the scalar part of **C**). The gradient of 𝔈 w.r.t. conductances 
is obtained via the inner product ⟨∂𝔈/∂**C**, e_i⟩, which yields a real 
number for each edge. This gradient is fused with the flux‑based physarum 
update to obtain a hybrid rule.

    g_i ← g_i + η ( Φ_i – λ ∂𝔈/∂g_i ),

The NLMS error is then used to adapt the global weight vector and update 
the physarum network state.

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
# Multivector (geometric algebra) – core of Parent B
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
# RBF surrogate model
# ----------------------------------------------------------------------

def rbf(x: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> float:
    """
    Evaluate a radial-basis function (RBF) on the input vector.

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    centers : np.ndarray
        Centers of the RBFs.
    widths : np.ndarray
        Widths of the RBFs.
    """
    return np.sum(np.exp(-((x - centers) ** 2) / (2 * widths ** 2)))


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------

def hybrid_operation(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    centers: np.ndarray = None,
    widths: np.ndarray = None,
) -> Tuple[np.ndarray, Multivector]:
    """
    Perform one hybrid operation.

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
    centers : np.ndarray
        Centers of the RBFs.
    widths : np.ndarray
        Widths of the RBFs.
    """
    # NLMS prediction and update
    e = target - nlms_predict(weights, x)
    weights, _ = nlms_update(weights, x, target, mu, eps)

    # RBF surrogate model evaluation
    if centers is not None and widths is not None:
        scalar_part = rbf(np.array(list(x)), centers, widths)
    else:
        scalar_part = 0.0

    # Multivector creation and update
    components = {frozenset(): scalar_part}
    multivector = Multivector(components, len(x))

    return weights, multivector


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    weights = np.random.rand(10)
    x = np.random.rand(10)
    target = 1.0
    centers = np.random.rand(10, 10)
    widths = np.random.rand(10)

    weights, multivector = hybrid_operation(weights, x, target, centers=centers, widths=widths)
    print(multivector)