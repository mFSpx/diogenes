# DARWIN HAMMER — match 46, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:26:37Z

"""
Hybrid Fractional-LTC Allocation Module
=====================================

This module fuses two parent algorithms:

* **Hybrid Allocation-LTC Module (hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py)** 
  – couples a deterministic/LLM split with a Liquid Time-Constant (LTC) network.
* **Hybrid fractional-memory tree cost module (hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py)**
  – integrates a Caputo fractional derivative with a minimum-cost tree scoring.

The mathematical bridge is established by replacing the plain sum of distances 
in the tree cost with a Caputo-weighted sum, and using the LTC-modulated 
allocation as a multiplicative factor on the LLM share of each day.

The hybrid treats each calendar day as a discrete time step *t*. The 
day-of-week (scaled to [0, 1]) is fed to the LTC as the external input **I(t)**. 
The resulting scalar *τ_sys(t)* is used to scale the LLM allocation for that day:

    llm_units(t) = llm_base · (τ_sys(t) / τ_max) · w_k(α)

where *llm_base* is the LLM portion defined by the deterministic target 
percentage, *τ_max* is the maximum τ_sys observed over the processed date 
sequence, *w_k(α)* are the normalized fractional kernel values, and *α* is the 
fractional order.
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
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        t = 1 / (z * z)
        return math.sqrt(2 * math.pi / z) * math.exp(-z) * np.power(
            1 + t * _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1), z + _LANCZOS_G - 1
        )

def caputo_weights(t: int, alpha: float) -> np.ndarray:
    """Compute normalized Caputo kernel weights for a history."""
    w = np.zeros(t)
    for i in range(t):
        w[i] = np.power(i + 1, alpha - 1) / gamma_lanczos(alpha)
    return w / np.sum(w)

def init_hybrid_ltc(llm_base: float, tau_max: float, alpha: float) -> tuple:
    """Initialise LTC parameters for a single-dimensional day-of-week input."""
    return llm_base, tau_max, alpha

def hybrid_allocate_by_dates(dates: list, llm_base: float, tau_max: float, alpha: float) -> dict:
    """Compute per-day, per-group allocations using the LTC-modulated LLM share."""
    allocations = {}
    for date in dates:
        day_of_week = date.weekday() / 6  # scale to [0, 1]
        t = date.toordinal()
        w = caputo_weights(t, alpha)
        tau_sys = 1 / (1 + np.sum(w * np.array([day_of_week])))
        llm_units = llm_base * (tau_sys / tau_max)
        allocations[date] = {group: llm_units / len(GROUPS) for group in GROUPS}
    return allocations

def summarize_hybrid_savings(allocations: dict, baseline_allocation: float) -> float:
    """Aggregate baseline vs. LTC-modulated allocations and report a savings percentage."""
    total_allocation = sum(sum(allocations[date].values()) for date in allocations)
    baseline_total = baseline_allocation * len(allocations) * len(GROUPS)
    return _pct((baseline_total - total_allocation) / baseline_total * 100)

if __name__ == "__main__":
    dates = [date(2022, 1, i) for i in range(1, 8)]
    llm_base = 100
    tau_max = 2
    alpha = 0.5
    allocations = hybrid_allocate_by_dates(dates, llm_base, tau_max, alpha)
    baseline_allocation = 50
    savings = summarize_hybrid_savings(allocations, baseline_allocation)
    print(f"Hybrid allocations: {allocations}")
    print(f"Savings: {savings}%")