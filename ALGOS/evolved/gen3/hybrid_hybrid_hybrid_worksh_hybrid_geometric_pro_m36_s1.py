# DARWIN HAMMER — match 36, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py (gen2)
# born: 2026-05-29T23:25:25Z

"""
Hybrid Allocation-LTC Geometric Product Module
=============================================

Parents:
- **Hybrid Allocation-LTC Module** (PARENT ALGORITHM A)
- **Fusion of geometric_product.py and hybrid_model_vram_scheduler_ttt_linear_m11_s1.py** (PARENT ALGORITHM B)

Mathematical Bridge
-------------------
The hybrid integrates the governing equation of Algorithm A with the Clifford geometric product from Algorithm B. The mathematically coupled system treats each calendar day as a discrete time step *t*.  The day-of-week (scaled to [0, 1]) is fed to the LTC as the external input **I(t)**.  The resulting scalar *τ_sys(t)* is used to scale a portion of the VRAM allocation for that day, which is in turn determined by the geometric product-based update rule. This creates a novel hybrid algorithm that adapts to changing memory requirements while reshaping resource allocation schedules.

The module therefore fuses:
1. The deterministic/LLM split and group-wise division of Algorithm A.
2. The input-dependent effective time constant of Algorithm B as a multiplicative factor on the LLM share of each day.
3. The Clifford geometric product for optimizing the update rule of the TTT-Linear model.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
import math

# ---------------------------------------------------------------------------
# Constants & Helpers (from Parent A)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self):
        """Return the scalar (grade-0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __mul__(self, other):
        result = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade, sign = _multiply_blades(blade_a, blade_b)
                result[blade] = (result.get(blade, 0.0) + sign * coef_a * coef_b)
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def init_hybrid_ltc(total_units, groups):
    """Initialise LTC parameters for a single-dimensional day-of-week input."""
    deterministic_target_percentage = 0.5
    llm_base = total_units - (total_units * deterministic_target_percentage)
    max_tau = 1.0  # arbitrary initialisation
    return deterministic_target_percentage, llm_base, max_tau

def hybrid_allocate_by_dates(dates, groups, deterministic_target_percentage, llm_base, max_tau):
    """Compute per-day, per-group allocations using the LTC-modulated LLM share."""
    tau_sys = np.zeros(len(dates))
    for i, date_ in enumerate(dates):
        day_of_week = (date_.weekday() + 1) / 7  # scaled to [0, 1]
        tau_sys[i] = 1 / (1 + 1 * day_of_week)
    llm_allocations = (llm_base * tau_sys / max_tau) * np.ones((len(groups), len(tau_sys)))
    deterministic_allocations = (deterministic_target_percentage * total_units) * np.ones((len(groups), len(tau_sys)))
    return llm_allocations, deterministic_allocations

def summarize_hybrid_savings(llm_allocations, deterministic_allocations):
    """Aggregate baseline vs. LTC-modulated allocations and report a savings percentage."""
    baseline_usage = sum(sum(deterministic_allocations))
    hybrid_usage = sum(sum(llm_allocations))
    savings = (baseline_usage - hybrid_usage) / baseline_usage * 100
    return _pct(savings)

if __name__ == "__main__":
    total_units = 100  # example value
    groups = GROUPS  # example value
    dates = [date(2024, 3, 1) + date.timedelta(days=i) for i in range(7)]
    deterministic_target_percentage, llm_base, max_tau = init_hybrid_ltc(total_units, groups)
    llm_allocations, deterministic_allocations = hybrid_allocate_by_dates(dates, groups, deterministic_target_percentage, llm_base, max_tau)
    savings = summarize_hybrid_savings(llm_allocations, deterministic_allocations)
    print("Savings:", savings)