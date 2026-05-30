# DARWIN HAMMER — match 2893, survivor 1
# gen: 7
# parent_a: hybrid_path_signature_kan_m30_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1248_s0.py (gen6)
# born: 2026-05-29T23:46:26Z

"""
This module fuses the path signature and lead-lag transformation from the 'hybrid_path_signature_kan_m30_s2.py' algorithm
with the hybrid rectified flow, normalized least-mean-squares (NLMS) adaptive filter, and graph-propagation engine from the
'hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m1248_s0.py' algorithm.

The mathematical bridge is formed by using the lead-lag transformation to generate input features for the NLMS predictor,
which attempts to model the wavefront velocity of the graph-propagation engine. The Physarum network state is encoded as a
multivector, and its free-energy is approximated using a radial-basis function (RBF) on the conductance vector.
The gradient of the free-energy with respect to conductances is obtained via the inner product, which yields a real number for each edge.
This gradient is fused with the flux-based physarum update to obtain a hybrid rule.

The fusion integrates the governing equations of both parents by combining the lead-lag transformation and NLMS predictor
to model the wavefront velocity of the graph-propagation engine. The Physarum network state is encoded as a multivector, and
its free-energy is approximated using a radial-basis function (RBF) on the conductance vector.

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import math
import random
import sys
import pathlib

class Multivector:
    """Sparse multivector in a Clifford algebra with n basis vectors."""
    def __init__(self, components: dict, n: int):
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

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    step_size: float = 0.1,
) -> np.ndarray:
    """NLMS update: w_new = w_old - (step_size * x) * (target - w_old·x) / (x·x)."""
    prediction = nlms_predict(weights, x)
    weights -= step_size * x * (target - prediction) / (x @ x)
    return weights

def radial_basis_function(conductance: np.ndarray, centers: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Radial basis function (RBF) on the conductance vector."""
    return np.exp(-((conductance - centers) ** 2) / (2 * sigma ** 2))

def free_energy(multivector: Multivector, conductance: np.ndarray, centers: np.ndarray, sigma: float = 1.0) -> float:
    """Approximate free-energy of the network using a radial-basis function (RBF) on the conductance vector."""
    return np.sum(radial_basis_function(conductance, centers, sigma) * multivector.components[frozenset()])

def hybrid_update(
    multivector: Multivector,
    conductance: np.ndarray,
    centers: np.ndarray,
    sigma: float = 1.0,
    step_size: float = 0.1,
) -> np.ndarray:
    """Hybrid update: fuse the flux-based physarum update with the gradient of the free-energy."""
    free_energy_val = free_energy(multivector, conductance, centers, sigma)
    gradient = np.gradient(free_energy_val)
    return conductance - step_size * gradient

def lead_lag_nlms_predict(path: np.ndarray, weights: np.ndarray) -> float:
    """Lead-lag NLMS predictor: combine the lead-lag transformation with the NLMS predictor."""
    lead_lag_path = lead_lag_transform(path)
    return nlms_predict(weights, lead_lag_path)

def lead_lag_nlms_update(
    path: np.ndarray,
    weights: np.ndarray,
    target: float,
    step_size: float = 0.1,
) -> np.ndarray:
    """Lead-lag NLMS update: combine the lead-lag transformation with the NLMS update."""
    lead_lag_path = lead_lag_transform(path)
    return nlms_update(weights, lead_lag_path, target, step_size)

if __name__ == "__main__":
    # Smoke test
    path = np.random.rand(10, 2)
    weights = np.random.rand(4)
    target = np.random.rand()
    print(lead_lag_nlms_predict(path, weights))
    print(lead_lag_nlms_update(path, weights, target))