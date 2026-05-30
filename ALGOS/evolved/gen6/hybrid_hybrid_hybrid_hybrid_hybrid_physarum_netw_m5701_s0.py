# DARWIN HAMMER — match 5701, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1666_s3.py (gen5)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s6.py (gen3)
# born: 2026-05-30T00:04:15Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

"""
Hybrid Pheromone-Physarum-Bandit Algorithm
=====================================

This module fuses the Hybrid Pheromone-Tree-Bayesian-Entropy Algorithm (Parent A)
with the Hybrid Physarum-Bandit Model (Parent B).  The mathematical bridge is
the identification of **conductance** with the **edge prior** (e.g. a physical-distance-based cost)
and **flux** with the **edge likelihood** calculated from the normalized Hamming similarity
between two node hashes.

*   Hybrid Pheromone-Tree-Bayesian-Entropy Algorithm:
    `edge prior π` is the physical-distance-based cost of an edge.
    `edge likelihood L` is the normalized Hamming similarity between two node hashes.
    `Bayesian marginal π' = bayes_marginal(π, L, ε)` updates the edge prior with the edge likelihood.
*   Hybrid Physarum-Bandit Model:
    `conductance` is the propensity (inflow rate) of a bandit action.
    `flux` is the reward signal that drives the bandit’s learning update.

By feeding the Bayesian marginal `π'` into the conductance update rule we obtain a new
conductance, which is then used by the bandit’s linear weight update.  The VRAM-store variable
provides a global modulation of the learning rate, closing the feedback loop.

The implementation provides three core functions that demonstrate the hybrid operation and a
lightweight `HybridPhysarumBandit` class that encapsulates the state.
"""

# ----------------------------------------------------------------------
# Perceptual hashing utilities (Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64-bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    bits <<= max(0, 64 - len(values))  # pad with zeros if needed
    return bits & ((1 << 64) - 1)


# ----------------------------------------------------------------------
# Hybrid Physarum-Bandit Model (Parent B)
# ----------------------------------------------------------------------
def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute the flux on a single edge.

    Parameters
    ----------
    conductance : float
        Current conductance (capacity) of the edge.
    edge_length : float
        Physical length of the edge (must be > 0).
    pressure_a, pressure_b : float
        Node pressures at the two ends of the edge.
    eps : float, optional
        Small number to avoid division by zero.

    Returns
    -------
    float
        Flux value `q = G/L * (p_a - p_b)`.
    """
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    """
    Conductance update rule derived from Physarum.

    Parameters
    ----------
    conductance : float
        Current conductance (capacity) of the edge.
    q : float
        Flux value on the edge.
    dt : float, optional
        Time step (default: 1.0).
    gain : float, optional
        Gain factor (default: 1.0).
    decay : float, optional
        Decay rate (default: 0.05).

    Returns
    -------
    float
        Updated conductance value.
    """
    return conductance * (dt * gain * q - decay)


def bayes_marginal(
    prior: float,
    likelihood: float,
    epsilon: float,
) -> float:
    """
    Compute the Bayesian marginal of the prior and likelihood.

    Parameters
    ----------
    prior : float
        Prior probability of an event.
    likelihood : float
        Likelihood of observing an event given the prior.
    epsilon : float
        False positive rate.

    Returns
    -------
    float
        Bayesian marginal `π' = bayes_marginal(π, L, ε)`.
    """
    return prior * likelihood / (prior + likelihood * epsilon)


# ----------------------------------------------------------------------
# Hybrid Pheromone-Physarum-Bandit Algorithm
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    prior: float,
    likelihood: float,
    epsilon: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute the hybrid edge weight.

    Parameters
    ----------
    conductance : float
        Current conductance (capacity) of the edge.
    edge_length : float
        Physical length of the edge (must be > 0).
    pressure_a, pressure_b : float
        Node pressures at the two ends of the edge.
    prior : float
        Prior probability of an event.
    likelihood : float
        Likelihood of observing an event given the prior.
    epsilon : float
        False positive rate.
    dt : float, optional
        Time step (default: 1.0).
    gain : float, optional
        Gain factor (default: 1.0).
    decay : float, optional
        Decay rate (default: 0.05).
    alpha : float, optional
        Tunable parameter (default: 1.0).
    beta : float, optional
        Tunable parameter (default: 1.0).

    Returns
    -------
    float
        Hybrid edge weight `w = σ·π'`.
    """
    flux_value = flux(
        conductance,
        edge_length,
        pressure_a,
        pressure_b,
        eps=1e-12,
    )
    updated_conductance = update_conductance(
        conductance,
        flux_value,
        dt=dt,
        gain=gain,
        decay=decay,
    )
    bayesian_marginal = bayes_marginal(
        prior,
        likelihood,
        epsilon,
    )
    scaling_factor = 1 + alpha * updated_conductance + beta * flux_value
    return scaling_factor * bayesian_marginal


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    values = np.random.rand(1000)
    phash = compute_phash(values.tolist())
    print(f"Pheromone hash: {phash}")

    conductance = 1.0
    edge_length = 1.0
    pressure_a = 1.0
    pressure_b = 2.0
    prior = 0.5
    likelihood = 0.3
    epsilon = 0.1
    dt = 1.0
    gain = 1.0
    decay = 0.05
    alpha = 1.0
    beta = 1.0
    hybrid_weight = hybrid_edge_weight(
        conductance,
        edge_length,
        pressure_a,
        pressure_b,
        prior,
        likelihood,
        epsilon,
        dt=dt,
        gain=gain,
        decay=decay,
        alpha=alpha,
        beta=beta,
    )
    print(f"Hybrid edge weight: {hybrid_weight}")