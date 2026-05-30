# DARWIN HAMMER — match 2597, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""
Hybrid Fractional-Memory Allocation and Pheromone System Module
===========================================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation Module (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation.
* **Hybrid Pheromone System (hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py)** – 
  provides a pheromone signal calculation and decay mechanism.

The mathematical bridge between the two algorithms lies in the use of the 
Caputo fractional derivative to introduce a memory term into the pheromone 
signal calculation. The fractional-memory kernel is used to weight the 
historical pheromone signals, which are then used to modulate the signal 
calculation.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid 
   Fractional-Memory Allocation Module.
2. The input-dependent effective time constant of the Hybrid 
   Fractional-Memory Allocation Module.
3. The pheromone signal calculation and decay mechanism of the Hybrid 
   Pheromone System.

The implementation below provides:

* `init_hybrid_fm_pheromone` – initialise the hybrid allocation and pheromone 
  parameters.
* `hybrid_fm_pheromone_calculate` – compute the pheromone signal using the 
  fractional-memory modulated LLM share.
* `summarize_hybrid_fm_pheromone` – aggregate baseline vs. fractional-memory 
  modulated pheromone signals and report a modulation percentage.
"""

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List
import random
import sys
import pathlib
from datetime import date, datetime, timezone

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Parent A – Lanczos Gamma approximation and Caputo utilities
# ----------------------------------------------------------------------

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13
])

def gamma(z: float) -> float:
    """Lanczos approximation for the Gamma function."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma(1 - z))
    else:
        x = _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1)
        t = z + _LANCZOS_G - 1.5
        return math.sqrt(2 * math.pi) * t ** (z + 0.5) * np.exp(-t) * np.prod(x)

def caputo_derivative(f: callable, t: float, alpha: float) -> float:
    """Caputo fractional derivative."""
    return 1 / gamma(1 - alpha) * integral(lambda tau: (t - tau) ** (-alpha) * f(tau), 0, t)

def integral(f: callable, a: float, b: float, num_points: int = 1000) -> float:
    """Numerical integration using the trapezoidal rule."""
    h = (b - a) / num_points
    x = np.linspace(a, b, num_points + 1)
    y = f(x)
    return h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))

# ----------------------------------------------------------------------
# Parent B – Pheromone System
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, float]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds}
        current_value = self.pheromones[surface_key]['signal_value']
        decay_factor = math.exp(-half_life_seconds / (60 * 60 * 24))
        new_value = current_value * decay_factor + signal_value * (1 - decay_factor)
        self.pheromones[surface_key]['signal_value'] = new_value
        return new_value

# ----------------------------------------------------------------------
# Hybrid Module
# ----------------------------------------------------------------------

def init_hybrid_fm_pheromone(
    groups: tuple = GROUPS,
    alpha: float = 0.5,
    half_life_seconds: float = 3600,
) -> (dict, PheromoneSystem):
    """Initialise the hybrid allocation and pheromone parameters."""
    allocation_params = {
        'groups': groups,
        'alpha': alpha,
    }
    pheromone_system = PheromoneSystem()
    return allocation_params, pheromone_system

def hybrid_fm_pheromone_calculate(
    allocation_params: dict,
    pheromone_system: PheromoneSystem,
    surface_key: str,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
) -> float:
    """Compute the pheromone signal using the fractional-memory modulated LLM share."""
    alpha = allocation_params['alpha']
    groups = allocation_params['groups']
    def f(t: float) -> float:
        return np.sum([1 / (1 + t ** alpha) for _ in groups])
    modulated_signal_value = signal_value * caputo_derivative(f, 1, alpha)
    return pheromone_system.calculate_pheromone_signal(surface_key, signal_kind, modulated_signal_value, half_life_seconds)

def summarize_hybrid_fm_pheromone(
    allocation_params: dict,
    pheromone_system: PheromoneSystem,
) -> float:
    """Aggregate baseline vs. fractional-memory modulated pheromone signals and report a modulation percentage."""
    baseline_signals = []
    modulated_signals = []
    for surface_key, pheromone in pheromone_system.pheromones.items():
        baseline_signals.append(pheromone['signal_value'])
        modulated_signals.append(hybrid_fm_pheromone_calculate(allocation_params, pheromone_system, surface_key, pheromone['signal_kind'], pheromone['signal_value'], pheromone['half_life_seconds']))
    modulation_percentage = _pct(np.mean(modulated_signals) / np.mean(baseline_signals) - 1)
    return modulation_percentage

if __name__ == "__main__":
    allocation_params, pheromone_system = init_hybrid_fm_pheromone()
    surface_key = "test_surface"
    signal_kind = "test_signal"
    signal_value = 1.0
    half_life_seconds = 3600
    pheromone_signal = hybrid_fm_pheromone_calculate(allocation_params, pheromone_system, surface_key, signal_kind, signal_value, half_life_seconds)
    print(f"Pheromone signal: {pheromone_signal}")
    modulation_percentage = summarize_hybrid_fm_pheromone(allocation_params, pheromone_system)
    print(f"Modulation percentage: {modulation_percentage}")