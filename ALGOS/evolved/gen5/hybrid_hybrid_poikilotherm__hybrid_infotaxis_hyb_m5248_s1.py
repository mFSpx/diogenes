# DARWIN HAMMER — match 5248, survivor 1
# gen: 5
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hybrid_hybrid_m1080_s0.py (gen4)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s0.py (gen3)
# born: 2026-05-30T00:00:57Z

"""
This module represents a hybrid algorithm, combining the core topologies of 
Hybrid Poikilotherm Schoolfield (parent algorithm A) and Hybrid Infotaxis-Morphology (parent algorithm B).
The mathematical bridge between the two structures is established by modulating the 
temperature-dependent rate calculation of the poikilotherm model using the recovery priority 
from the hybrid infotaxis-morphology algorithm. This allows the pheromone signals to influence 
the exploration intensity and confidence bounds used by the rate calculation, 
while the morphology of the system adjusts the pheromone signal value.

The hybrid algorithm fuses the governing equations of both parents by using the 
recovery priority as a multiplicative scaling factor for the pheromone signal value, 
which in turn affects the temperature-dependent rate calculation.

Author: [Your Name]
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict
import random
import sys
from pathlib import Path

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

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

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def calculate_pheromone_signal(surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: float, 
                              morphology: Morphology) -> float:
    recovery_p = recovery_priority(morphology)
    modulated_signal_value = signal_value * recovery_p
    return modulated_signal_value

def temperature_dependent_rate(schoolfield_params: SchoolfieldParams, temperature: float, 
                               pheromone_signal: float) -> float:
    rho = schoolfield_params.rho_25 * math.exp((schoolfield_params.delta_h_activation / schoolfield_params.r_cal) * 
                                              (1.0 / K25 - 1.0 / temperature))
    modulated_rho = rho * pheromone_signal
    return modulated_rho

def hybrid_operation(morphology: Morphology, schoolfield_params: SchoolfieldParams, temperature: float, 
                    signal_kind: str, signal_value: float, half_life_seconds: float) -> float:
    pheromone_signal = calculate_pheromone_signal('surface_key', signal_kind, signal_value, half_life_seconds, morphology)
    rate = temperature_dependent_rate(schoolfield_params, temperature, pheromone_signal)
    return rate

if __name__ == "__main__":
    morphology = Morphology(1.0, 2.0, 3.0, 10.0)
    schoolfield_params = SchoolfieldParams()
    temperature = 300.0
    signal_kind = 'test_signal'
    signal_value = 1.0
    half_life_seconds = 10.0

    rate = hybrid_operation(morphology, schoolfield_params, temperature, signal_kind, signal_value, half_life_seconds)
    print(f'Hybrid rate: {rate}')