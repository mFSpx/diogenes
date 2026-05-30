# DARWIN HAMMER — match 4669, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2484_s3.py (gen5)
# born: 2026-05-29T23:57:19Z

"""
This module fuses the hybrid allocator from `hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py` 
and the regret-weighted strategy with Fisher localization from `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2484_s3.py`. 
The mathematical bridge between these two structures lies in the application of 
the Fisher information scoring to modulate the propensity scores in the regret-weighted 
strategy, allowing the strategy to consider the uncertainty of the diffusion 
schedule when selecting actions, while integrating the weekday-based weight vector 
from the allocator with the VRAM-aware GPU selection logic from the scheduler.

The governing equations of the allocator are used to optimize the propensity scores 
in the regret-weighted strategy, while the Fisher information scoring is used to 
modulate the weekday-based weight vector.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple
from datetime import date as dt

# Constants
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (dt(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def vram_aware_gpu_selection(gpus: List[Dict[str, int]], budget_mb: int, reserve_mb: int) -> List[Dict[str, int]]:
    """
    Select GPUs that have sufficient VRAM to meet the budget and reserve requirements.
    """
    selected_gpus = []
    for gpu in gpus:
        if gpu['memory.free'] >= budget_mb + reserve_mb:
            selected_gpus.append(gpu)
    return selected_gpus

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_fisher_regret(
    groups: List[str], 
    gpus: List[Dict[str, int]], 
    budget_mb: int, 
    reserve_mb: int, 
    theta: float, 
    center: float, 
    width: float
) -> Dict[str, float]:
    """
    Integrate Fisher information scoring with regret-weighted strategy and weekday-based weight vector.
    """
    dow = doomsday(dt.today().year, dt.today().month, dt.today().day)
    weight_vec = weekday_weight_vector(groups, dow)
    selected_gpus = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    regret = {}
    for group in groups:
        regret[group] = fisher_score(theta, center, width) * weight_vec[groups.index(group)]
    return regret

def hybrid_predict(
    groups: List[str], 
    gpus: List[Dict[str, int]], 
    budget_mb: int, 
    reserve_mb: int, 
    theta: float, 
    center: float, 
    width: float
) -> Dict[str, float]:
    """
    Prediction using the scaled schedule, Fisher information scoring, and regret-weighted strategy.
    """
    regret = hybrid_fisher_regret(groups, gpus, budget_mb, reserve_mb, theta, center, width)
    predict = {}
    for group in groups:
        predict[group] = regret[group] * (1 + math.exp(-regret[group]))
    return predict

def hybrid_allocation(
    *,
    total_units: float,
    date: dt,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
    theta: float = 0.0,
    center: float = 0.0,
    width: float = 1.0
) -> Dict[str, float]:
    """
    Hybrid allocation using the scaled schedule, Fisher information scoring, and regret-weighted strategy.
    """
    gpus = [{'memory.free': 1024}, {'memory.free': 2048}, {'memory.free': 4096}]
    predict = hybrid_predict(list(groups), gpus, budget_mb, reserve_mb, theta, center, width)
    allocation = {}
    for group in groups:
        allocation[group] = predict[group] * total_units
    return allocation

if __name__ == "__main__":
    groups = list(GROUPS)
    gpus = [{'memory.free': 1024}, {'memory.free': 2048}, {'memory.free': 4096}]
    budget_mb = DEFAULT_BUDGET_MB
    reserve_mb = DEFAULT_RESERVE_MB
    theta = 0.0
    center = 0.0
    width = 1.0
    total_units = 100.0
    date = dt.today()
    deterministic_target_pct = 90.0
    allocation = hybrid_allocation(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
        budget_mb=budget_mb,
        reserve_mb=reserve_mb,
        theta=theta,
        center=center,
        width=width
    )
    print(allocation)