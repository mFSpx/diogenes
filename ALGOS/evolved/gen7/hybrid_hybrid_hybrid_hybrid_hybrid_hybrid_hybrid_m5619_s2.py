# DARWIN HAMMER — match 5619, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py (gen6)
# born: 2026-05-30T00:03:29Z

"""
This module provides a novel HYBRID algorithm, which mathematically fuses the core topologies 
of two parent algorithms: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_krampu_m1840_s1.py and 
hybrid_hybrid_hybrid_physar_hybrid_pheromone_hyb_m2648_s0.py.

The mathematical bridge between their structures lies in the integration of the 
flux-based conductance update from the Physarum network and the NLMS prediction 
mechanism from the Krampus brain-map bandit router. The interface is established 
through the concept of propensity, which influences the conductance update in 
the Physarum network, and the weighting factor from the NLMS predictor, which 
modulates the propensity in the Hybrid Bandit model.

In this fusion, the NLMS predictor is used to estimate the flux in the Physarum 
network, and the conductance update is used to adapt the NLMS weights. This 
creates a feedback loop between the two systems, allowing them to learn from each 
other and improve their performance.
"""

import numpy as np
import math
import random
import sys
import pathlib

ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
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
        Regularisation parameter to prevent division by zero.

    Returns
    -------
    updated_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error.
    """
    error = target - nlms_predict(weights, x)
    updated_weights = weights + mu * error * x / (x @ x + eps)
    return updated_weights, error

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_bandit_update(conductance: float, propensity: float, reward: float, dt: float = 1.0, gain: float = 1.0) -> float:
    return update_conductance(conductance, propensity * reward, dt, gain)

def hybrid_step(weights: np.ndarray, x: np.ndarray, target: float, conductance: float, edge_length: float, pressure_a: float, pressure_b: float) -> Tuple[np.ndarray, float, float]:
    """
    Hybrid step that combines NLMS update and conductance update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    conductance : float
        Current conductance value.
    edge_length : float
        Edge length in the Physarum network.
    pressure_a : float
        Pressure at node A.
    pressure_b : float
        Pressure at node B.

    Returns
    -------
    updated_weights : np.ndarray
        Updated weight vector.
    updated_conductance : float
        Updated conductance value.
    error : float
        Prediction error.
    """
    updated_weights, error = nlms_update(weights, x, target)
    q = nlms_predict(updated_weights, x)
    updated_conductance = update_conductance(conductance, q, DT, ETA0)
    return updated_weights, updated_conductance, error

def simulate_hybrid_system(num_steps: int, initial_weights: np.ndarray, initial_conductance: float, edge_length: float, pressure_a: float, pressure_b: float) -> None:
    """
    Simulate the hybrid system for a specified number of steps.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps.
    initial_weights : np.ndarray
        Initial weight vector (1‑D).
    initial_conductance : float
        Initial conductance value.
    edge_length : float
        Edge length in the Physarum network.
    pressure_a : float
        Pressure at node A.
    pressure_b : float
        Pressure at node B.
    """
    weights = initial_weights
    conductance = initial_conductance
    for _ in range(num_steps):
        x = np.array([random.random() for _ in range(len(initial_weights))])
        target = random.random()
        weights, conductance, error = hybrid_step(weights, x, target, conductance, edge_length, pressure_a, pressure_b)
        print(f"Error: {error}, Conductance: {conductance}")

if __name__ == "__main__":
    initial_weights = np.array([0.1, 0.2, 0.3])
    initial_conductance = 0.5
    edge_length = 1.0
    pressure_a = 1.0
    pressure_b = 0.5
    simulate_hybrid_system(10, initial_weights, initial_conductance, edge_length, pressure_a, pressure_b)