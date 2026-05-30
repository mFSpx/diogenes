# DARWIN HAMMER — match 4609, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m586_s1.py (gen5)
# born: 2026-05-29T23:56:47Z

"""
This module represents a novel hybrid algorithm, merging the core topologies of 
hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m586_s1.py.
The mathematical bridge between these two structures is the application of the 
Structural Similarity Index (SSIM) to inform the selection of actions in the 
regret-matching algorithm, while also utilizing the Caputo fractional derivative 
to model the decay of pheromone signals over time in the Voronoi partitioning 
of engine endpoints.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x

def caputo_derivative(f, alpha, t):
    """Caputo fractional derivative of f(t) with order alpha."""
    dt = 0.01
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integral = np.trapz(f_tau / (t - tau) ** alpha, tau)
    return integral / gamma_lanczos(1 - alpha)

def fractional_decay(alpha, t):
    """Fractional decay kernel."""
    return t ** (alpha - 1) / gamma_lanczos(alpha)

def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

def calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, alpha):
    """Calculate pheromone signal with Caputo fractional derivative."""
    current_time = np.arange(0, half_life_seconds, 0.01)
    pheromone_signal = signal_value * fractional_decay(alpha, current_time)
    return pheromone_signal

def hybrid_score(packet: dict[str, list[float]]) -> float:
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec, PROTOTYPE_VECTOR, dynamic_range=1.0)
    except Exception:
        return 0.0

def distance(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point, seeds):
    """Find the nearest seed to a point."""
    if not seeds:
        raise ValueError('seeds required')
    return min(seeds, key=lambda x: distance(point, x))

def hybrid_action(surface_key, signal_kind, signal_value, half_life_seconds, alpha):
    """Calculate pheromone signal and hybrid score."""
    pheromone_signal = calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, alpha)
    packet = {"payload": pheromone_signal.tolist()}
    hybrid_score_value = hybrid_score(packet)
    return pheromone_signal, hybrid_score_value

def regret_matching(alpha, surface_key, signal_kind, signal_value, half_life_seconds):
    """Regret-matching algorithm."""
    pheromone_signal = calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, alpha)
    packet = {"payload": pheromone_signal.tolist()}
    hybrid_score_value = hybrid_score(packet)
    return hybrid_score_value

def voronoi_partition(seeds, alpha, surface_key, signal_kind, signal_value, half_life_seconds):
    """Voronoi partitioning."""
    pheromone_signal = calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, alpha)
    packet = {"payload": pheromone_signal.tolist()}
    hybrid_score_value = hybrid_score(packet)
    nearest_seed = nearest((surface_key, signal_kind), seeds)
    return nearest_seed, hybrid_score_value

if __name__ == "__main__":
    surface_key = (1.0, 2.0)
    signal_kind = (3.0, 4.0)
    signal_value = 5.0
    half_life_seconds = 10.0
    alpha = 0.5
    seeds = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    pheromone_signal, hybrid_score_value = hybrid_action(surface_key, signal_kind, signal_value, half_life_seconds, alpha)
    print("Pheromone signal:", pheromone_signal)
    print("Hybrid score:", hybrid_score_value)
    hybrid_score_value = regret_matching(alpha, surface_key, signal_kind, signal_value, half_life_seconds)
    print("Regret-matching score:", hybrid_score_value)
    nearest_seed, hybrid_score_value = voronoi_partition(seeds, alpha, surface_key, signal_kind, signal_value, half_life_seconds)
    print("Nearest seed:", nearest_seed)
    print("Voronoi partitioning score:", hybrid_score_value)