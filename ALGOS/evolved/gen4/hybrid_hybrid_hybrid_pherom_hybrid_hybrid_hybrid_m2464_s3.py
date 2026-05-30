# DARWIN HAMMER — match 2464, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s1.py (gen3)
# born: 2026-05-29T23:42:22Z

"""
This module integrates the mathematical structures of 'hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2.py' 
and 'hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s1.py' to create a novel hybrid algorithm. 
The mathematical bridge between the two algorithms is formed by applying the temperature-dependent 
developmental rate from the poikilotherm model in 'hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s1.py' 
to the burst admission model in 'hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s2.py', which in turn 
informs the pheromone algorithm's signal recording process. This fusion enables the creation of a more 
dynamic and adaptive clustering of the graph, where leaders are chosen from clusters of similar nodes 
with high burst action scores, and the state transition and output projection are adapted based on the 
current temperature or state of the system.
"""

import numpy as np
import random
import math
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime, timezone

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def pulse_force(peak_force: float, steps: int) -> list[float]:
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, rho_25: float = 1.0, delta_h_activation: float = 12_000.0, 
                        t_low: float = 283.15, t_high: float = 307.15, delta_h_low: float = -45_000.0, 
                        delta_h_high: float = 65_000.0, r_cal: float = 1.987) -> float:
    if temp_k <= 0 or rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = rho_25 * (temp_k / 298.15) * math.exp((delta_h_activation / r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((delta_h_low / r_cal) * ((1.0 / t_low) - (1.0 / temp_k)))
    high = math.exp((delta_h_high / r_cal) * ((1.0 / t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_dependent_burst_admission(peak_force: float, steps: int, temp_k: float, 
                                          rho_25: float = 1.0, delta_h_activation: float = 12_000.0, 
                                          t_low: float = 283.15, t_high: float = 307.15, delta_h_low: float = -45_000.0, 
                                          delta_h_high: float = 65_000.0, r_cal: float = 1.987) -> list[float]:
    rate = developmental_rate(temp_k, rho_25, delta_h_activation, t_low, t_high, delta_h_low, delta_h_high, r_cal)
    return [rate * force for force in pulse_force(peak_force, steps)]

def hybrid_pheromone_signal_record(values: list[float], temp_k: float, 
                                    rho_25: float = 1.0, delta_h_activation: float = 12_000.0, 
                                    t_low: float = 283.15, t_high: float = 307.15, delta_h_low: float = -45_000.0, 
                                    delta_h_high: float = 65_000.0, r_cal: float = 1.987) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    rate = developmental_rate(temp_k, rho_25, delta_h_activation, t_low, t_high, delta_h_low, delta_h_high, r_cal)
    return int(bits * rate)

def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    rho_25: float = 1.0,
    delta_h_activation: float = 12_000.0,
    t_low: float = 283.15,
    t_high: float = 307.15,
    delta_h_low: float = -45_000.0,
    delta_h_high: float = 65_000.0,
    r_cal: float = 1.987,
) -> tuple[np.ndarray, np.ndarray]:
    rate = developmental_rate(temp_k, rho_25, delta_h_activation, t_low, t_high, delta_h_low, delta_h_high, r_cal)
    A_rate = rate * A
    h_next = np.dot(A_rate, h) + np.dot(B, x)
    x_out = np.dot(C, h_next)
    return h_next, x_out

if __name__ == "__main__":
    temp_k = c_to_k(25)
    peak_force = 10.0
    steps = 10
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    h = np.array([1.0, 2.0])
    x = np.array([3.0, 4.0])
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[0.0, 0.0], [0.0, 0.0]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])

    print(temperature_dependent_burst_admission(peak_force, steps, temp_k))
    print(hybrid_pheromone_signal_record(values, temp_k))
    print(ssm_step(h, x, A, B, C, temp_k))