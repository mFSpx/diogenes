# DARWIN HAMMER — match 1651, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s2.py (gen4)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:38:08Z

"""
Unified Algorithm: Flux-Based Morphology Hybrid
Fuses the principles of Flux-Based Conductance Update (Parent Algorithm A: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s2.py)
and a Self-Righting Morphology Primitive (Parent Algorithm B: serpentina_self_righting.py).

The mathematical bridge between the two parents lies in the integration of the 
store differential equation in the UnifiedBanditTTT class (Parent A) with the 
morphology-based righting time index in the Morphology class (Parent B). Specifically, 
the update_conductance function from Parent A can be used to influence the 
recovery priority of the morphology in Parent B, based on the flux computed 
from the conductance and morphology dimensions.

By fusing these two components, we develop a unified algorithm that leverages the 
strengths of both parents to compute recovery priorities based on a flux-based 
conductance update mechanism and morphology dimensions.
"""

import math
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


@dataclass(frozen=True)
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


# ----------------------------------------------------------------------
# Unified Flux-Based Morphology Hybrid
# ----------------------------------------------------------------------
class UnifiedFluxMorphology:
    def __init__(self, morphology: Morphology, conductance: float, pressure_a: float, pressure_b: float):
        self.morphology = morphology
        self.conductance = conductance
        self.pressure_a = pressure_a
        self.pressure_b = pressure_b

    def compute_recovery_priority(self, max_index: float = 10.0) -> float:
        edge_length = self.morphology.length
        flux_value = flux(self.conductance, edge_length, self.pressure_a, self.pressure_b)
        updated_conductance = update_conductance(self.conductance, flux_value)
        righting_time = righting_time_index(self.morphology)
        return max(0.0, min(1.0, righting_time / max_index * updated_conductance))

    def compute_flux_based_sphericity(self) -> float:
        edge_length = self.morphology.length
        flux_value = flux(self.conductance, edge_length, self.pressure_a, self.pressure_b)
        return sphericity_index(self.morphology.length, self.morphology.width, self.morphology.height) * flux_value

    def compute_morphology_based_conductance(self) -> float:
        righting_time = righting_time_index(self.morphology)
        return update_conductance(self.conductance, righting_time)


if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 3.0, 20.0)
    unified_system = UnifiedFluxMorphology(morphology, 1.0, 10.0, 5.0)
    print(unified_system.compute_recovery_priority())
    print(unified_system.compute_flux_based_sphericity())
    print(unified_system.compute_morphology_based_conductance())