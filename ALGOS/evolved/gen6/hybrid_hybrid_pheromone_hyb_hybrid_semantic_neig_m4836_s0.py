# DARWIN HAMMER — match 4836, survivor 0
# gen: 6
# parent_a: hybrid_pheromone_hybrid_hybrid_hybrid_m1143_s0.py (gen5)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s1.py (gen2)
# born: 2026-05-29T23:58:12Z

"""
Hybrid Algorithm: Fusing Darwinian Surface Pheromone with Semantic Neighborhood Recovery Priority

This module integrates the Darwinian surface pheromone algorithm with the semantic neighborhood recovery priority algorithm.
The mathematical bridge between the two parents lies in the application of the surface pheromone to modulate the recovery priority calculation.
The governing equations of the parent algorithms are fused as follows:

- The store equation from the Darwinian surface pheromone algorithm is used to update the surface pheromone based on the recovery priority.
- The righting time index and recovery priority from the semantic neighborhood recovery priority algorithm are used to inform the evasion-driven position perturbation.

The resulting hybrid algorithm couples resource-allocation dynamics with continuous optimisation dynamics and decision hygiene evaluation.
"""

import numpy as np
import math
import random
from pathlib import Path
from dataclasses import dataclass

# Shared constants
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule

@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def update_pheromone(pheromone: float, recovery_priority: float, alpha: float, beta: float, dt: float) -> float:
    return pheromone + alpha * recovery_priority * dt - beta * pheromone * dt

def perturb_position(position: float, recovery_priority: float, delta_max: float, alpha_evasion: float) -> float:
    return position + random.uniform(-1, 1) * delta_max * recovery_priority * math.exp(-alpha_evasion * recovery_priority)

def hybrid_operation(morphology: Morphology, pheromone: float, position: float) -> (float, float):
    recovery_prior = recovery_priority(morphology)
    new_pheromone = update_pheromone(pheromone, recovery_prior, ALPHA, BETA, DT)
    new_position = perturb_position(position, recovery_prior, DELTA_MAX, ALPHA_EVASION)
    return new_pheromone, new_position

if __name__ == "__main__":
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    pheromone = 1.0
    position = 0.0
    new_pheromone, new_position = hybrid_operation(morphology, pheromone, position)
    print(f"New pheromone: {new_pheromone}, New position: {new_position}")