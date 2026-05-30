# DARWIN HAMMER — match 2597, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""
Hybrid Fractional-Memory Allocation and Pheromone System Module
===========================================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation Module (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py)**
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, using a Caputo 
  fractional derivative to introduce a memory term into the allocation process.
* **Hybrid Pheromone System (hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py)**
  provides a pheromone signal calculation and decay mechanism.

The mathematical bridge between the two algorithms lies in the use of the 
exponential decay factor from the Pheromone System to modulate the 
fractional-memory kernel in the Hybrid Fractional-Memory Allocation Module.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Allocation Module.
2. The fractional-memory tree cost of the Hybrid fractional-memory tree cost module.
3. The pheromone signal calculation and decay mechanism of the Hybrid Pheromone System.

The implementation below provides:

* `init_hybrid_fm_pheromone_allocation` – initialise the hybrid allocation parameters.
* `hybrid_fm_pheromone_allocate_by_dates` – compute per-day, per-group allocations using 
  the fractional-memory modulated LLM share and pheromone signal.
* `summarize_hybrid_fm_pheromone_savings` – aggregate baseline vs. fractional-memory modulated 
  allocations and report a savings percentage.
"""

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List
import random
import sys
import pathlib
from datetime import date, datetime, timezone

# Constants & Helpers
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

# Lanczos Gamma approximation and Caputo utilities
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13
])

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

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

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

def caputo_fractional_derivative(x: float, alpha: float) -> float:
    return (1 / math.gamma(1 - alpha)) * (x ** (-alpha))

def hybrid_fm_pheromone_allocation(
    group: str, 
    date: date, 
    signal_value: float, 
    half_life_seconds: float, 
    alpha: float
) -> float:
    pheromone_system = PheromoneSystem()
    pheromone_signal = pheromone_system.calculate_pheromone_signal(
        group, 'signal', signal_value, half_life_seconds
    )
    fractional_derivative = caputo_fractional_derivative(pheromone_signal, alpha)
    return fractional_derivative

def init_hybrid_fm_pheromone_allocation(
    groups: List[str], 
    dates: List[date], 
    signal_values: List[float], 
    half_life_seconds: float, 
    alpha: float
) -> Dict[str, Dict[date, float]]:
    allocations = {}
    for group in groups:
        allocations[group] = {}
        for i, date in enumerate(dates):
            allocation = hybrid_fm_pheromone_allocation(
                group, date, signal_values[i], half_life_seconds, alpha
            )
            allocations[group][date] = allocation
    return allocations

def summarize_hybrid_fm_pheromone_savings(
    allocations: Dict[str, Dict[date, float]], 
    baseline_allocations: Dict[str, Dict[date, float]]
) -> float:
    total_savings = 0
    for group, group_allocations in allocations.items():
        baseline_group_allocations = baseline_allocations[group]
        for date, allocation in group_allocations.items():
            baseline_allocation = baseline_group_allocations[date]
            savings = (baseline_allocation - allocation) / baseline_allocation * 100
            total_savings += savings
    return total_savings / len(allocations)

if __name__ == "__main__":
    groups = list(GROUPS)
    dates = [date(2022, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
    signal_values = [1.0] * len(dates)
    half_life_seconds = 86400  # 1 day
    alpha = 0.5

    allocations = init_hybrid_fm_pheromone_allocation(
        groups, dates, signal_values, half_life_seconds, alpha
    )

    baseline_allocations = {
        group: {date: 1.0 for date in dates} for group in groups
    }

    savings = summarize_hybrid_fm_pheromone_savings(
        allocations, baseline_allocations
    )
    print(f"Savings: {_pct(savings)}%")