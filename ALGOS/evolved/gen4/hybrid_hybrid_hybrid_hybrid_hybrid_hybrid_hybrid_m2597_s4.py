# DARWIN HAMMER — match 2597, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""
Hybrid Fractional-Memory Pheromone Module
=====================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation Module (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, and a power-law memory kernel.
* **Hybrid Endpoint Pheromone Module (hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py)** – 
  provides a pheromone system and morphological indices for engine endpoints.

The mathematical bridge between the two algorithms lies in the use of the 
power-law memory kernel to modulate the pheromone signals, and the application of 
morphological indices to the group-wise division of the Hybrid Fractional-Memory 
Allocation Module. The fractional-memory kernel is used to weight the historical 
allocations, which are then used to modulate the pheromone signals.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Allocation Module.
2. The power-law memory kernel of the Hybrid Fractional-Memory Allocation Module.
3. The pheromone system and morphological indices of the Hybrid Endpoint Pheromone Module.
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
    -1259.139216000391,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
])

def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    z = complex(z)
    if z.real < 0.5:
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    z -= 1
    x = _LANCZOS_P / (z + _LANCZOS_G)
    for i in range(_LANCZOS_G + 1):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

_LANCZOS_P = np.sqrt(2 * math.pi)

# ----------------------------------------------------------------------
# Parent B – Pheromone System and Morphology
# ----------------------------------------------------------------------

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
        self.pheromones: dict = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                'signal_kind': signal_kind, 
                'signal_value': signal_value, 
                'half_life_seconds': half_life_seconds
            }
        current_value = self.pheromones[surface_key]['signal_value']
        decay_factor = math.exp(-half_life_seconds / (60 * 60 * 24))
        new_value = current_value * decay_factor + signal_value * (1 - decay_factor)
        self.pheromones[surface_key]['signal_value'] = new_value
        return new_value

# ----------------------------------------------------------------------
# Hybrid Module
# ----------------------------------------------------------------------

def init_hybrid_fm_allocation(
    groups: list, 
    time_constant: float, 
    memory_kernel: float, 
    pheromone_system: PheromoneSystem
) -> dict:
    """
    Initialize the hybrid allocation parameters.

    Args:
    groups (list): List of groups.
    time_constant (float): Time constant for the LLM allocation.
    memory_kernel (float): Memory kernel for the fractional-memory allocation.
    pheromone_system (PheromoneSystem): Pheromone system for the hybrid allocation.

    Returns:
    dict: Hybrid allocation parameters.
    """
    allocation_params = {
        'groups': groups,
        'time_constant': time_constant,
        'memory_kernel': memory_kernel,
        'pheromone_system': pheromone_system
    }
    return allocation_params

def hybrid_fm_allocate_by_dates(
    allocation_params: dict, 
    dates: list, 
    group_allocations: dict
) -> dict:
    """
    Compute per-day, per-group allocations using the fractional-memory modulated LLM share.

    Args:
    allocation_params (dict): Hybrid allocation parameters.
    dates (list): List of dates.
    group_allocations (dict): Group allocations.

    Returns:
    dict: Per-day, per-group allocations.
    """
    groups = allocation_params['groups']
    time_constant = allocation_params['time_constant']
    memory_kernel = allocation_params['memory_kernel']
    pheromone_system = allocation_params['pheromone_system']

    allocations = {}
    for date in dates:
        allocations[date] = {}
        for group in groups:
            signal_value = group_allocations.get(group, 0.0)
            pheromone_signal = pheromone_system.calculate_pheromone_signal(
                group, 'allocation', signal_value, time_constant
            )
            allocations[date][group] = pheromone_signal * memory_kernel
    return allocations

def summarize_hybrid_fm_savings(
    allocation_params: dict, 
    dates: list, 
    group_allocations: dict
) -> float:
    """
    Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage.

    Args:
    allocation_params (dict): Hybrid allocation parameters.
    dates (list): List of dates.
    group_allocations (dict): Group allocations.

    Returns:
    float: Savings percentage.
    """
    groups = allocation_params['groups']
    time_constant = allocation_params['time_constant']
    memory_kernel = allocation_params['memory_kernel']
    pheromone_system = allocation_params['pheromone_system']

    baseline_allocations = 0.0
    hybrid_allocations = 0.0
    for date in dates:
        for group in groups:
            signal_value = group_allocations.get(group, 0.0)
            baseline_allocations += signal_value
            pheromone_signal = pheromone_system.calculate_pheromone_signal(
                group, 'allocation', signal_value, time_constant
            )
            hybrid_allocations += pheromone_signal * memory_kernel
    savings = (baseline_allocations - hybrid_allocations) / baseline_allocations
    return savings

if __name__ == "__main__":
    pheromone_system = PheromoneSystem()
    allocation_params = init_hybrid_fm_allocation(
        GROUPS, 
        time_constant=1.0, 
        memory_kernel=0.5, 
        pheromone_system=pheromone_system
    )
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    group_allocations = {
        'codex': 0.2,
        'groq': 0.3,
        'cohere': 0.1,
        'local_models': 0.4
    }
    allocations = hybrid_fm_allocate_by_dates(
        allocation_params, 
        dates, 
        group_allocations
    )
    savings = summarize_hybrid_fm_savings(
        allocation_params, 
        dates, 
        group_allocations
    )
    print("Allocations:", allocations)
    print("Savings:", savings)