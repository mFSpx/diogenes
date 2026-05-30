# DARWIN HAMMER — match 5227, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1811_s4.py (gen5)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-30T00:00:41Z

"""
Hybrid Darwin Hammer and Capybara Tri Conduit Algorithm.

This module fuses the *Hybrid Decision Hygiene and Shannon Entropy* algorithm 
(hybrid_hybrid_hybrid_decisi_hybrid_hdc_hybrid_hy_m996_s2.py) with the 
*Hybrid Capybara‑Tri Conduit Algorithm* (hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py) 
using a novel mathematical bridge based on the intersection of their decision hygiene metrics 
and temperature-dependent modulation factors.

The bridge integrates the bipolar vector operations from the *Hybrid Decision Hygiene* 
algorithm with the temperature-dependent modulation factor from the *Hybrid Capybara* 
algorithm, and uses the developmental rate from the SchoolfieldParams to modulate 
the signal and noise scores in the LearningVector.

The mathematical bridge is established by combining the signal‑to‑noise gap `Δ = signal - noise` 
from the *Hybrid Capybara* algorithm with the temperature-dependent modulation factor from 
the *Hybrid Decision Hygiene* algorithm, resulting in a hybrid decision hygiene metric 
that incorporates both the confidence scalar and the temperature-dependent activity curve.

This fusion enables the evaluation of decision-making cues under different temperature conditions 
while maintaining the diversity of decision-making cues, as measured by the vectorized representation 
of decision hygiene metrics.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clamp each component of a vector to [lower, upper]."""
    return np.clip(x, lower, upper)

# ----------------------------------------------------------------------
# Minimal re‑implementation of Parent B primitives
# ----------------------------------------------------------------------
def _shannon_entropy(data: np.ndarray) -> float:
    """Return Shannon entropy in bits for a sequence of byte values."""
    if not data.size:
        return 0.0
    counts = np.bincount(data.astype(np.uint8), minlength=256)
    probabilities = counts / data.size
    return -np.sum(probabilities * np.log2(probabilities))

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def hybrid_decision_hygiene(metrics: np.ndarray, temperature: float, params: SchoolfieldParams) -> np.ndarray:
    """Hybrid decision hygiene metric."""
    signal = np.sum(metrics * _POSITIVE_WEIGHTS)
    noise = np.sum(metrics * _NEGATIVE_WEIGHTS)
    delta = signal - noise
    modulation_factor = np.exp(-params.rho_25 * (temperature - params.t_low) / (params.t_high - params.t_low))
    return delta * modulation_factor

def hybrid_evasion_magnitude(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0, epsilon: float = 0.1) -> float:
    """Hybrid evasion magnitude."""
    delta = evasion_delta(t, t_max, delta_max, alpha)
    return delta * (1 + epsilon)

def hybrid_attract_magnitude(gain_gap: float, global_best: float, attraction_probability: float) -> float:
    """Hybrid attraction magnitude."""
    return gain_gap * global_best * attraction_probability

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    params = SchoolfieldParams()
    metrics = np.random.rand(9)
    temperature = 298.15
    t = 100
    t_max = 1000
    delta_max = 1.0
    alpha = 3.0
    epsilon = 0.1
    gain_gap = 0.5
    global_best = 1.0
    attraction_probability = 0.7

    hybrid_metrics = hybrid_decision_hygiene(metrics, temperature, params)
    hybrid_evasion = hybrid_evasion_magnitude(t, t_max, delta_max, alpha, epsilon)
    hybrid_attraction = hybrid_attract_magnitude(gain_gap, global_best, attraction_probability)

    print("Hybrid decision hygiene metric:", hybrid_metrics)
    print("Hybrid evasion magnitude:", hybrid_evasion)
    print("Hybrid attraction magnitude:", hybrid_attraction)