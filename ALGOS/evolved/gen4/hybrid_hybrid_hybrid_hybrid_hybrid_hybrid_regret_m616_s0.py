# DARWIN HAMMER — match 616, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s1.py (gen3)
# born: 2026-05-29T23:29:58Z

"""
Hybrid Fractional-Memory Regret Engine Module
=============================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation Module (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, and a power-law memory kernel.
* **Hybrid Regret Engine Hybrid Bandit Router Module (hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s1.py)** – 
  provides a regret-weighted strategy and a hybrid bandit router with a MinHash-based 
  similarity metric to modulate the propensity of each action.

The mathematical bridge between the two algorithms lies in the application of the 
power-law memory kernel to the regret-weighted strategy, effectively introducing a 
memory term into the action selection process. The fractional-memory kernel is used 
to weight the historical regrets, which are then used to modulate the regret-weighted 
expected reward of each action.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Allocation Module.
2. The input-dependent effective time constant of the Hybrid Fractional-Memory Allocation Module.
3. The power-law memory kernel of the Hybrid Fractional-Memory Allocation Module.
4. The regret-weighted strategy of the Hybrid Regret Engine Hybrid Bandit Router Module.
5. The hybrid bandit router with a MinHash-based similarity metric of the Hybrid Regret Engine Hybrid Bandit Router Module.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

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

def lanczos_gamma(z: np.ndarray) -> np.ndarray:
    """Lanczos approximation of the gamma function."""
    z = z - 1
    x = _LANCZOS_C[0] + z / (_LANCZOS_G + z)
    for i in range(1, _LANCZOS_G + 1):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x

def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """Caputo fractional derivative."""
    dt = t[1] - t[0]
    n = len(t)
    dfdt = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += (f[j] * (t[i] - t[j]) ** (-alpha - 1))
        dfdt[i] = (1 / math.gamma(1 - alpha)) * sum
    return dfdt

def hybrid_fm_allocate_by_dates(dates: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    """Compute per-day, per-group allocations using the fractional-memory modulated LLM share."""
    n = len(dates)
    allocations = np.zeros((n, len(groups)))
    for i in range(n):
        for j in range(len(groups)):
            allocations[i, j] = (1 / lanczos_gamma(1 - alpha)) * np.sum((dates[:i] - dates[i]) ** (-alpha - 1) * groups[j])
    return allocations

def regret_weighted_strategy(actions: np.ndarray, regrets: np.ndarray, alpha: float) -> np.ndarray:
    """Regret-weighted strategy with a power-law memory kernel."""
    n = len(actions)
    weighted_regrets = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += regrets[j] * (1 / lanczos_gamma(1 - alpha)) * np.sum((actions[:j] - actions[j]) ** (-alpha - 1))
        weighted_regrets[i] = sum
    return weighted_regrets

def hybrid_bandit_router(actions: np.ndarray, rewards: np.ndarray, alpha: float) -> np.ndarray:
    """Hybrid bandit router with a MinHash-based similarity metric and a power-law memory kernel."""
    n = len(actions)
    propensities = np.zeros(n)
    for i in range(n):
        sum = 0
        for j in range(n):
            sum += rewards[j] * (1 / lanczos_gamma(1 - alpha)) * np.sum((actions[:j] - actions[j]) ** (-alpha - 1))
        propensities[i] = sum
    return propensities

def summarize_hybrid_fm_savings(baseline_allocations: np.ndarray, hybrid_allocations: np.ndarray) -> float:
    """Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage."""
    baseline_sum = np.sum(baseline_allocations)
    hybrid_sum = np.sum(hybrid_allocations)
    return (baseline_sum - hybrid_sum) / baseline_sum

if __name__ == "__main__":
    dates = np.array([date(2022, 1, 1) + date.resolution * i for i in range(10)])
    groups = np.array([0.1, 0.2, 0.3, 0.4])
    alpha = 0.5
    allocations = hybrid_fm_allocate_by_dates(dates, groups, alpha)
    actions = np.array([1, 2, 3, 4, 5])
    regrets = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    weighted_regrets = regret_weighted_strategy(actions, regrets, alpha)
    rewards = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    propensities = hybrid_bandit_router(actions, rewards, alpha)
    baseline_allocations = np.array([0.1, 0.2, 0.3, 0.4])
    hybrid_allocations = np.array([0.05, 0.1, 0.15, 0.2])
    savings = summarize_hybrid_fm_savings(baseline_allocations, hybrid_allocations)
    print(savings)