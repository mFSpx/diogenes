# DARWIN HAMMER — match 36, survivor 6
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_geometric_product_hybrid_model_vram_sc_m22_s1.py (gen2)
# born: 2026-05-29T23:25:25Z

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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

def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade-0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        result = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_c, sign = _multiply_blades(blade_a, blade_b)
                result[blade_c] = result.get(blade_c, 0.0) + sign * coef_a * coef_b
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __repr__(self) -> str:
        return f"Multivector(components={self.components}, n={self.n})"

def init_hybrid_ltc_gp(llm_base: float, tau: float, day_of_week: float) -> Tuple[Multivector, float]:
    """Initialise LTC and geometric product parameters."""
    multivector = Multivector({frozenset(): 1.0}, 1)
    tau_sys = tau / (1 + tau * day_of_week)
    llm_units = llm_base * (tau_sys / tau)
    return multivector, llm_units

def hybrid_allocate_by_dates(dates: List[str], llm_base: float, tau: float, groups: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    """Compute per-day, per-group allocations using the LTC-modulated LLM share and the geometric product."""
    allocations = {}
    for date_str in dates:
        date_obj = date.fromisoformat(date_str)
        day_of_week = date_obj.weekday() / 6  # scale to [0, 1]
        multivector, llm_units = init_hybrid_ltc_gp(llm_base, tau, day_of_week)
        group_allocations = {}
        for group in groups:
            group_allocation = llm_units * multivector.scalar_part() / len(groups)
            group_allocations[group] = group_allocation
        allocations[date_str] = group_allocations
    return allocations

def summarize_hybrid_savings(allocations: Dict[str, Dict[str, float]]) -> float:
    """Aggregate baseline vs. LTC-modulated allocations and report a savings percentage."""
    total_baseline = sum(sum(allocation.values()) for allocation in allocations.values())
    # Calculate baseline assuming equal allocation
    baseline_allocation = sum([llm_base / len(GROUPS) for _ in range(len(allocations))])
    savings = (baseline_allocation - total_baseline) / baseline_allocation * 100
    return savings

if __name__ == "__main__":
    llm_base = 100.0
    tau = 10.0
    groups = GROUPS
    dates = [date.today().isoformat() for _ in range(7)]
    allocations = hybrid_allocate_by_dates(dates, llm_base, tau, groups)
    savings = summarize_hybrid_savings(allocations)
    print(f"Savings: {savings:.2f}%")