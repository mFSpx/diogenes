# DARWIN HAMMER — match 46, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:26:37Z

"""
Hybrid Fractional-Memory Allocation Module
=======================================

This module fuses two parent algorithms:

* **Hybrid Allocation-LTC Module (hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation.
* **Hybrid fractional-memory tree cost module (hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py)** – 
  provides a power-law memory kernel and a fractional-memory tree cost.

The mathematical bridge between the two algorithms lies in the use of the Caputo 
fractional derivative to introduce a memory term into the allocation process. 
The fractional-memory kernel is used to weight the historical allocations, 
which are then used to modulate the LLM allocation.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Allocation-LTC Module.
2. The input-dependent effective time constant of the Hybrid Allocation-LTC Module.
3. The fractional-memory tree cost of the Hybrid fractional-memory tree cost module.

The implementation below provides:

* `init_hybrid_fm_allocation` – initialise the hybrid allocation parameters.
* `hybrid_fm_allocate_by_dates` – compute per-day, per-group allocations using 
  the fractional-memory modulated LLM share.
* `summarize_hybrid_fm_savings` – aggregate baseline vs. fractional-memory modulated 
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
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1 - z))
    t = 1 / (z * z)
    t += _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1)
    return np.sqrt(2 * np.pi) * np.power(z, z + 0.5) * np.exp(-z) * np.sqrt(t)

def caputo_weights(alpha: float, T: int) -> np.ndarray:
    """Compute normalized Caputo kernel weights for a history."""
    w = np.zeros(T)
    for k in range(T):
        w[k] = np.power(k + 1, alpha - 1) / gamma_lanczos(alpha)
    return w / np.sum(w)

# ----------------------------------------------------------------------
# Hybrid Fractional-Memory Allocation
# ----------------------------------------------------------------------

class HybridFMAllocation:
    def __init__(self, 
                 total_units: float, 
                 llm_base: float, 
                 tau_max: float, 
                 alpha: float):
        self.total_units = total_units
        self.llm_base = llm_base
        self.tau_max = tau_max
        self.alpha = alpha

    def hybrid_fm_allocate_by_dates(self, 
                                    dates: list[date], 
                                    deterministic_split: dict) -> dict:
        T = len(dates)
        w = caputo_weights(self.alpha, T)
        allocations = {}
        for group in GROUPS:
            group_allocations = []
            for i, date in enumerate(dates):
                day_of_week = date.weekday() / 7
                tau_sys = 1 / (1 + np.sum([w[j] * (date - dates[j]).days for j in range(i+1)]))
                llm_units = self.llm_base * (tau_sys / self.tau_max)
                group_allocation = deterministic_split[group] + llm_units
                group_allocations.append(group_allocation)
            allocations[group] = group_allocations
        return allocations

    def summarize_hybrid_fm_savings(self, 
                                     allocations: dict, 
                                     baseline_allocations: dict) -> float:
        total_savings = 0
        for group in GROUPS:
            group_allocations = allocations[group]
            baseline_group_allocations = baseline_allocations[group]
            total_savings += np.sum(baseline_group_allocations) - np.sum(group_allocations)
        return _pct(total_savings / np.sum([allocations[group] for group in GROUPS]))

def init_hybrid_fm_allocation(total_units: float, 
                              llm_base: float, 
                              tau_max: float, 
                              alpha: float) -> HybridFMAllocation:
    deterministic_split = {group: total_units * 0.5 for group in GROUPS}
    return HybridFMAllocation(total_units, llm_base, tau_max, alpha)

if __name__ == "__main__":
    dates = [date(2022, 1, 1) + i for i in range(30)]
    total_units = 100
    llm_base = 20
    tau_max = 1
    alpha = 0.5

    hybrid_fm_allocation = init_hybrid_fm_allocation(total_units, llm_base, tau_max, alpha)
    allocations = hybrid_fm_allocation.hybrid_fm_allocate_by_dates(dates, {group: total_units * 0.5 for group in GROUPS})
    baseline_allocations = {group: [total_units * 0.5 for _ in range(30)] for group in GROUPS}

    savings = hybrid_fm_allocation.summarize_hybrid_fm_savings(allocations, baseline_allocations)
    print(f"Savings: {savings}%")