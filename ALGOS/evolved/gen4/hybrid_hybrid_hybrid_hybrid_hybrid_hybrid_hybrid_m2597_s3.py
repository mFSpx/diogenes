# DARWIN HAMMER — match 2597, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""
Hybrid Fractional-Memory Pheromone Allocation Module

This module fuses two parent algorithms:
- `hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py` provides a deterministic/LLM split 
  and group-wise division with an input-dependent effective time constant that modulates the LLM allocation.
- `hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py` provides a pheromone system 
  that simulates the release and decay of pheromone signals.

The mathematical bridge between the two algorithms lies in the use of the Caputo 
fractional derivative to introduce a memory term into the allocation process, while 
incorporating the pheromone system to modulate the allocation based on the pheromone signals.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Allocation-LTC Module.
2. The input-dependent effective time constant of the Hybrid Allocation-LTC Module.
3. The pheromone system of the Pheromone Module.

The implementation below provides:
- `init_hybrid_fm_allocation` – initialise the hybrid allocation parameters.
- `hybrid_fm_allocate_by_dates` – compute per-day, per-group allocations using 
  the fractional-memory modulated LLM share and pheromone signals.
- `summarize_hybrid_fm_savings` – aggregate baseline vs. fractional-memory modulated 
  allocations and report a savings percentage.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")
LANCZOS_G = 7
LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
])

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def lanczos_gamma(z):
    z = complex(z)
    if z.real < 0.5:
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    z = z - 1
    x = LANCZOS_C[0]
    for i in range(1, LANCZOS_G + 2):
        x += LANCZOS_C[i] / (z + i)
    x *= math.sqrt(2 * math.pi) * (z + LANCZOS_G + 0.5) ** z * math.exp(-(z + LANCZOS_G + 0.5))
    return x.real

def caputo_derivative(f, t, alpha):
    integral = 0
    for i in range(1, len(f)):
        integral += (f[i] - f[i - 1]) * (t[i] - t[i - 1]) ** alpha
    return integral / math.gamma(1 - alpha)

class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: dict = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = date.today()
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds}
        current_value = self.pheromones[surface_key]['signal_value']
        decay_factor = math.exp(-half_life_seconds / (60 * 60 * 24))
        new_value = current_value * decay_factor + signal_value * (1 - decay_factor)
        self.pheromones[surface_key]['signal_value'] = new_value
        return new_value

class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0):
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m, max_index: float = 10.0):
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def init_hybrid_fm_allocation():
    # Initialize pheromone system and morphology
    pheromone_system = PheromoneSystem()
    morphology = Morphology(1.0, 1.0, 1.0, 1.0)
    return pheromone_system, morphology

def hybrid_fm_allocate_by_dates(dates, amounts, pheromone_system, morphology):
    # Compute Caputo derivative
    caputo_derivatives = []
    for i in range(1, len(amounts)):
        caputo_derivative_value = caputo_derivative(amounts[:i+1], dates[:i+1], 0.5)
        caputo_derivatives.append(caputo_derivative_value)
    
    # Compute pheromone signals
    pheromone_signals = []
    for i in range(len(dates)):
        signal_value = recovery_priority(morphology)
        pheromone_signal = pheromone_system.calculate_pheromone_signal(f"date_{i}", "recovery", signal_value, 3600)
        pheromone_signals.append(pheromone_signal)
    
    # Compute allocations
    allocations = []
    for i in range(len(dates)):
        allocation = amounts[i] * (1 + pheromone_signals[i]) * (1 + caputo_derivatives[i] if i > 0 else 1)
        allocations.append(allocation)
    return allocations

def summarize_hybrid_fm_savings(baseline_allocations, hybrid_allocations):
    # Compute savings percentage
    savings = 0
    for i in range(len(baseline_allocations)):
        savings += baseline_allocations[i] - hybrid_allocations[i]
    savings_percentage = (savings / sum(baseline_allocations)) * 100
    return savings_percentage

if __name__ == "__main__":
    pheromone_system, morphology = init_hybrid_fm_allocation()
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    amounts = [100.0, 200.0, 300.0]
    allocations = hybrid_fm_allocate_by_dates(dates, amounts, pheromone_system, morphology)
    baseline_allocations = amounts
    savings_percentage = summarize_hybrid_fm_savings(baseline_allocations, allocations)
    print(f"Savings percentage: {savings_percentage}%")