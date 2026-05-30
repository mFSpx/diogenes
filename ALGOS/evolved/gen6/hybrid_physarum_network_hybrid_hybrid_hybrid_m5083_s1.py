# DARWIN HAMMER — match 5083, survivor 1
# gen: 6
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s1.py (gen5)
# born: 2026-05-29T23:59:41Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: physarum_network.py and hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s1.py. 
The mathematical bridge between the two parents lies in the interface between the 
flux-based conductance update primitive and the state space duality. Specifically, 
the flux can be viewed as a mechanism for updating the state transition matrix A 
in the state space duality, while the pheromone infotaxis can be used to update the 
conductance in the flux-based conductance update primitive. This fusion enables the 
hybrid algorithm to leverage the strengths of both parents, combining the adaptability 
of the pheromone infotaxis with the efficiency of the flux-based conductance update.
"""

import numpy as np
from collections import defaultdict
import math
import random

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def pheromone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

class HybridModel:
    def __init__(self):
        self.state_transition_matrix = np.zeros((2, 2))
        self.conductance = 1.0
        self.pressure_a = 1.0
        self.pressure_b = 0.0
        self.edge_length = 1.0
        self.pheromone = 1.0
        self.log_count_ratio = 1.0

    def update_state_transition_matrix(self):
        q = flux(self.conductance, self.edge_length, self.pressure_a, self.pressure_b)
        self.state_transition_matrix[0, 0] = pheromone_infotaxis(self.pheromone, self.log_count_ratio)
        self.state_transition_matrix[0, 1] = q
        self.state_transition_matrix[1, 0] = q
        self.state_transition_matrix[1, 1] = 1.0 - pheromone_infotaxis(self.pheromone, self.log_count_ratio)

    def update_conductance_and_pheromone(self):
        self.conductance = update_conductance(self.conductance, flux(self.conductance, self.edge_length, self.pressure_a, self.pressure_b))
        self.pheromone = pheromone_infotaxis(self.pheromone, self.log_count_ratio)

    def run_hybrid_model(self):
        self.update_state_transition_matrix()
        self.update_conductance_and_pheromone()
        return self.state_transition_matrix, self.conductance, self.pheromone

if __name__ == "__main__":
    model = HybridModel()
    state_transition_matrix, conductance, pheromone = model.run_hybrid_model()
    print(state_transition_matrix)
    print(conductance)
    print(pheromone)