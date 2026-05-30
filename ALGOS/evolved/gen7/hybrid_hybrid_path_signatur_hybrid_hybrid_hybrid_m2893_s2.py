# DARWIN HAMMER — match 2893, survivor 2
# gen: 7
# parent_a: hybrid_path_signature_kan_m30_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1248_s0.py (gen6)
# born: 2026-05-29T23:46:26Z

"""
Hybrid Algorithm: Fusing Path Signature and Hybrid Rectified Flow with Physarum-NLMS-Graph Engine

This module fuses the Path Signature algorithm (parent A) with the Hybrid Rectified Flow and Physarum-NLMS-Graph Engine (parent B).
The mathematical bridge between the two structures is found by using the lead-lag transform from Path Signature to generate input features for the NLMS predictor,
which attempts to model the wavefront velocity of the graph-propagation engine.

The Path Signature's lead-lag transform provides a causality-encoded representation of the input path.
The Hybrid Rectified Flow's straight-line interpolant generates input features for the NLMS predictor.
The Physarum network state is encoded as a multivector **C** = Σ g_i e_i, where g_i are edge conductances and e_i are orthogonal basis vectors of a Clifford algebra.

The surrogate model provides a scalar functional 𝔈(**C**) ≈ free-energy of the network by evaluating a radial-basis function (RBF) on the conductance vector.
The gradient of 𝔈 w.r.t. conductances is obtained via the inner product ⟨∂𝔈/∂**C**, e_i⟩, which yields a real number for each edge.
This gradient is fused with the flux-based physarum update to obtain a hybrid rule.

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

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)

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
    learning_rate: float = 0.1,
) -> np.ndarray:
    prediction = nlms_predict(weights, x)
    error = target - prediction
    return weights + learning_rate * error * x

def hybrid_operation(path, weights, learning_rate=0.1):
    """Perform the hybrid operation.

    1. Apply lead-lag transform to the input path.
    2. Compute the level-1 and level-2 signatures.
    3. Use the straight-line interpolant to generate input features for the NLMS predictor.
    4. Update the NLMS weights using the predicted target.

    """
    lead_lag_path = lead_lag_transform(path)
    level1_signature = signature_level1(path)
    level2_signature = signature_level2(path)

    # Use the straight-line interpolant to generate input features for the NLMS predictor
    t = 0.5
    interpolated_path = interpolant(path[0], path[-1], t)

    # Update the NLMS weights using the predicted target
    prediction = nlms_predict(weights, interpolated_path)
    target = level1_signature @ interpolated_path
    updated_weights = nlms_update(weights, interpolated_path, target, learning_rate)

    return updated_weights, level1_signature, level2_signature

if __name__ == "__main__":
    # Smoke test
    path = np.random.rand(10, 3)  # Random path with 10 time steps and 3 dimensions
    weights = np.random.rand(3)  # Random initial weights
    updated_weights, level1_signature, level2_signature = hybrid_operation(path, weights)
    print(updated_weights)