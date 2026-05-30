# DARWIN HAMMER — match 5248, survivor 0
# gen: 5
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hybrid_hybrid_m1080_s0.py (gen4)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s0.py (gen3)
# born: 2026-05-30T00:00:57Z

import math
import numpy as np
from dataclasses import dataclass
import random
import sys
from pathlib import Path

"""
Hybrid Algorithm: fusion of hybrid_poikilotherm_schoolf_hybrid_hybrid_hybrid_m1080_s0 and hybrid_infotaxis_hybrid_semantic_neig_m739_s0
The mathematical bridge is established by using the temperature-dependent rate calculation to modulate the expected entropy.
This allows the topology of the information space to be influenced by the physical-temperature space.
"""

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


def c_to_k(celsius: float) -> float:
    return celsius + 273.15

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

def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

def hybrid_expected_entropy(p_hit: float, hit_state: list[float], miss_state: list[float], m: Morphology, t: float) -> float:
    """Temperature-dependent expected entropy."""
    temperature_factor = 1 + (t - K25) / 50  # arbitrary scaling
    return recovery_priority(m) * expected_entropy(p_hit, hit_state, miss_state) * temperature_factor

def schoolfield_rate(m: Morphology, t: float, rho_25: float, delta_h_activation: float, delta_h_low: float, delta_h_high: float, r_cal: float) -> float:
    """Temperature-dependent rate calculation from Schoolfield model."""
    t_k = c_to_k(t)
    if t_k < 283.15:
        return delta_h_low * r_cal * rho_25
    elif t_k > 307.15:
        return delta_h_high * r_cal * rho_25
    else:
        return delta_h_activation * r_cal * rho_25

def hybrid_poikilotherm_infotaxis(m: Morphology, t: float) -> float:
    """Hybrid calculation of expected entropy and rate."""
    hit_state = [0.5, 0.3, 0.2]
    miss_state = [0.7, 0.2, 0.1]
    p_hit = 0.8
    return hybrid_expected_entropy(p_hit, hit_state, miss_state, m, t) * schoolfield_rate(m, t, 1.0, 12000.0, -45000.0, 65000.0, 1.987)

if __name__ == "__main__":
    m = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    t = 300.0
    print(hybrid_poikilotherm_infotaxis(m, t))