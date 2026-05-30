# DARWIN HAMMER — match 1695, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1182_s2.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s1.py (gen4)
# born: 2026-05-29T23:38:12Z

"""
Module for the hybrid algorithm that mathematically fuses the core topologies of the Physarum network update from 
hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py and the pheromone decision-making process from 
hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1. The mathematical bridge between these two algorithms is found 
in the concept of propensity and pheromone signals. The hybrid algorithm combines these two concepts by using the 
vector representation from hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s1 as the input to the pheromone 
decision-making process in hybrid_krampus_brainmap_hybrid_pheromone_inf_m37_s1.

The mathematical interface between these two algorithms lies in the use of propensity as a weighting factor for the 
pheromone signals. This allows the algorithm to adaptively select actions based on the expected reward and 
confidence bound of the bandit problem, while also considering the geometric context provided by the Voronoi 
partitions.

This fusion of algorithms represents a novel approach to solving complex decision-making problems in high-dimensional 
spaces, and demonstrates the power of combining different mathematical structures to create a more robust and 
generalizable algorithm.
"""

import numpy as np
import random
import math
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Physarum core
# ----------------------------------------------------------------------
def flux(conductance, edge_length, pressure_a, pressure_b, eps=1e-12):
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05):
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ----------------------------------------------------------------------
# Pheromone core
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Return the multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc())

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_update_conductance(conductance, propensity, pheromone_signal, dt=1.0, gain=1.0, decay=0.05):
    q = propensity * pheromone_signal
    return update_conductance(conductance, q, dt, gain, decay)

def hybrid_flux(conductance, edge_length, pressure_a, pressure_b, pheromone_signal, eps=1e-12):
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return flux(conductance, edge_length, pressure_a, pressure_b, eps) * pheromone_signal

def hybrid_action_selection(propensity, pheromone_signal, confidence_bound):
    return propensity * pheromone_signal / (propensity + confidence_bound)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize conductance and pheromone signals
    conductance = 1.0
    pheromone_signal = 0.5

    # Update conductance using hybrid update rule
    updated_conductance = hybrid_update_conductance(conductance, 0.8, pheromone_signal)
    print(updated_conductance)

    # Compute flux using hybrid flux function
    flux_value = hybrid_flux(updated_conductance, 1.0, 1.0, 0.0, pheromone_signal)
    print(flux_value)

    # Select action using hybrid action selection rule
    action = hybrid_action_selection(0.8, pheromone_signal, 0.2)
    print(action)