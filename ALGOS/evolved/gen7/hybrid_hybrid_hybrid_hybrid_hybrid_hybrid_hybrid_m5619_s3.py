# DARWIN HAMMER — match 5619, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py (gen6)
# born: 2026-05-30T00:03:29Z

"""
This module fuses the core topologies of two parent algorithms: 
hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py and 
hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py.

The mathematical bridge between their structures lies in the integration of the 
NLMS weight update from the first parent and the flux-based conductance update 
from the Physarum network in the second parent. The interface is established 
through the concept of a shared feature vector that influences both the NLMS 
weight update and the conductance update.

The hybrid system uses the NLMS predictor to forecast the flow velocity, 
which is then used to update the conductance in the Physarum network. 
The conductance update, in turn, affects the propensity in the Hybrid Bandit model, 
which modulates the NLMS weight update.

The governing equations of both parents are integrated through the following 
interface:

- The interpolated flow state `Z_t` from the first parent is used as a 
  feature vector in the NLMS predictor.
- The predicted flow velocity `v̂` from the NLMS predictor is used to 
  update the conductance in the Physarum network.
- The conductance update affects the propensity in the Hybrid Bandit model, 
  which modulates the NLMS weight update.

This integration enables a unified system that leverages the strengths of both 
parent algorithms.
"""

import numpy as np
import math
import random
import sys
import pathlib

# Define constants
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.

# Define NLMS parameters
NLMS_MU = 0.5
NLMS_EPS = 1e-9

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot-product prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = NLMS_MU,
    eps: float = NLMS_EPS,
) -> Tuple[np.ndarray, float]:
    """
    One Normalised LMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 1).
    eps : float
        Small positive value for regularization.

    Returns
    -------
    weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error.
    """
    prediction = nlms_predict(weights, x)
    error = target - prediction
    weights_update = weights + mu * error * x / (eps + np.dot(x, x))
    return weights_update, error

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = DT, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_step(
    weights: np.ndarray,
    conductance: float,
    Z_t: np.ndarray,
    context_vector: np.ndarray,
    target: float,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """
    One step of the hybrid algorithm.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    conductance : float
        Current conductance value.
    Z_t : np.ndarray
        Interpolated flow state.
    context_vector : np.ndarray
        Context vector for the Gaussian RBF kernel.
    target : float
        Desired scalar output.
    sigma : float
        Standard deviation for the Gaussian RBF kernel.

    Returns
    -------
    weights : np.ndarray
        Updated weight vector.
    conductance : float
        Updated conductance value.
    error : float
        Prediction error.
    """
    # Compute Gaussian RBF kernel
    kernel = np.exp(-np.linalg.norm(context_vector - Z_t) ** 2 / (2 * sigma ** 2))

    # Form NLMS input vector
    phi = np.concatenate((Z_t, [kernel]))

    # NLMS prediction and update
    prediction = nlms_predict(weights, phi)
    weights_update, error = nlms_update(weights, phi, target)

    # Update conductance
    conductance_update = update_conductance(conductance, prediction)

    return weights_update, conductance_update, error

if __name__ == "__main__":
    # Smoke test
    weights = np.random.rand(10)
    conductance = 1.0
    Z_t = np.random.rand(5)
    context_vector = np.random.rand(5)
    target = 1.0

    weights_update, conductance_update, error = hybrid_step(
        weights, conductance, Z_t, context_vector, target
    )

    print("Weights update:", weights_update)
    print("Conductance update:", conductance_update)
    print("Error:", error)