# DARWIN HAMMER — match 5272, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1776_s3.py (gen5)
# born: 2026-05-30T00:00:56Z

"""
This module represents a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py` 
and `hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s1.py`. The mathematical 
bridge between these two structures lies in the integration of the Hodgkin-Huxley 
cable model from the first parent with the tropical (max-plus) network from the 
second parent. This is achieved by representing the dendritic tree's electrical 
activity as a vector of aggregated pheromone signals, which is then fed into the 
tropical network as input. The network's output (gain candidates) is compared 
against a Hoeffding bound that decides when the Hodgkin-Huxley dynamics should be 
updated.
"""

import numpy as np
import random
import sys
import pathlib
import math

__all__ = [
    "HybridDendriticCompartment",
    "tropical_hybrid_update_rule",
    "tropical_hybrid_energy",
    "tropical_hybrid_retrieve",
    "sodium_current",
    "potassium_current",
    "leak_current",
    "phremone_decay",
]

class HybridDendriticCompartment:
    """
    A hybrid compartment that integrates Hodgkin-Huxley cable model with tropical network.
    """

    def __init__(self, node_dims: dict, edges: list, pheromone_decay_rate: float):
        self.node_dims = node_dims
        self.edges = edges
        self.pheromone_decay_rate = pheromone_decay_rate
        self._restrictions = {}
        self._sections = {}
        self._phremone_entries = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map), np.asarray(dst_map))

    def add_pheromone_entry(self, surface_key: str, signal_kind: str, signal_value: float):
        """Add a decaying pheromone signal to the compartment."""
        self._phremone_entries[surface_key] = PheromoneEntry(surface_key, signal_kind, signal_value, self.pheromone_decay_rate)

    def update_phremone(self):
        """Update the pheromone signals based on the decay rate."""
        for entry in self._phremone_entries.values():
            entry.signal_value *= math.exp(-entry.half_life_seconds * self.pheromone_decay_rate)
            entry.last_decay = datetime.now(timezone.utc)

def phremone_decay(pheromone_entry: PheromoneEntry):
    """Compute the decayed pheromone signal."""
    return pheromone_entry.signal_value * math.exp(-pheromone_entry.half_life_seconds * pheromone_entry.half_life_seconds)

def sodium_current(v_m: float, v_na: float, g_na: float):
    """Compute the sodium current."""
    return g_na * (v_na - v_m)

def potassium_current(v_m: float, v_k: float, g_k: float):
    """Compute the potassium current."""
    return g_k * (v_k - v_m)

def leak_current(v_m: float, g_l: float):
    """Compute the leak current."""
    return g_l * v_m

def tropical_hybrid_energy(compartment: HybridDendriticCompartment):
    """Compute the hybrid energy of the compartment."""
    pheromone_signals = [phremone_decay(entry) for entry in compartment._phremone_entries.values()]
    return np.sum(pheromone_signals)

def tropical_hybrid_update_rule(compartment: HybridDendriticCompartment):
    """Update the Hodgkin-Huxley dynamics based on the tropical network output."""
    pheromone_signals = [phremone_decay(entry) for entry in compartment._phremone_entries.values()]
    v_m = np.sum(pheromone_signals)
    i_na = sodium_current(v_m, 120, 35)
    i_k = potassium_current(v_m, -80, 9)
    i_l = leak_current(v_m, 0.3)
    return v_m + i_na + i_k + i_l

def tropical_hybrid_retrieve(compartment: HybridDendriticCompartment):
    """Retrieve the pheromone signals from the compartment."""
    return [entry.signal_value for entry in compartment._phremone_entries.values()]

if __name__ == "__main__":
    node_dims = {"u": 3, "v": 4}
    edges = [("u", "v")]
    compartment = HybridDendriticCompartment(node_dims, edges, 0.1)
    compartment.add_pheromone_entry("surface_key", "signal_kind", 10.0)
    print(tropical_hybrid_energy(compartment))
    print(tropical_hybrid_update_rule(compartment))
    print(tropical_hybrid_retrieve(compartment))