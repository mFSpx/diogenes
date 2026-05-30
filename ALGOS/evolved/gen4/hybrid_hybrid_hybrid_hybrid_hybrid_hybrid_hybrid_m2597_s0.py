# DARWIN HAMMER — match 2597, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_pherom_m227_s2.py (gen3)
# born: 2026-05-29T23:43:02Z

"""
Hybrid Fractional-Memory Phantasmagoria Module
=============================================

This module fuses two parent algorithms:

* **Hybrid Allocation-LTC Module (hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation.
* **Hybrid Pheromone-Infused Endpoint Duality Module (hybrid_hybrid_endpoint_circ_state_space_duality_m1_s1.py)** – 
  provides a pheromone-infused system that computes recovery priorities based on shape morphologies.

The mathematical bridge between the two algorithms lies in the use of the Caputo 
fractional derivative to introduce a memory term into the allocation process, 
which is then used to modulate the pheromone-infused recovery priorities.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Allocation-LTC Module.
2. The input-dependent effective time constant of the Hybrid Allocation-LTC Module.
3. The pheromone-infused recovery priorities of the Hybrid Pheromone-Infused Endpoint Duality Module.

The implementation below provides:

* `init_hybrid_fm_phantasmagoria` – initialise the hybrid allocation parameters.
* `hybrid_fm_allocate_by_dates` – compute per-day, per-group allocations using 
  the fractional-memory modulated LLM share and pheromone-infused recovery priorities.
* `summarize_hybrid_fm_savings` – aggregate baseline vs. fractional-memory modulated 
  allocations and report a savings percentage.
"""

import math
import numpy as np
from datetime import date
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Constants & Helpers
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Hybrid Fractional-Memory Phantasmagoria Module
# ----------------------------------------------------------------------

def _lanczos_gamma_approximation(x: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
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
    return (_LANCZOS_G * x ** (_LANCZOS_G - 1) * np.exp(-x)) * np.sum(_LANCZOS_C * x ** np.arange(_LANCZOS_G))

def caputo_fractional_derivative(f: np.ndarray, alpha: float, t: np.ndarray) -> np.ndarray:
    """Compute the Caputo fractional derivative of order alpha."""
    gamma = _lanczos_gamma_approximation(1 + alpha)
    return gamma * np.diff(f, alpha) / np.diff(t ** alpha)

def hybrid_fm_phantasmagoria(
    morphologies: Dict[str, Dict[str, float]],
    pheromone_system: PheromoneSystem,
    allocations: np.ndarray,
    time_constants: np.ndarray,
) -> np.ndarray:
    """Compute the fractional-memory modulated LLM share and pheromone-infused recovery priorities."""
    # Compute the fractional-memory modulated LLM share
    fm_modulated_llm_share = np.zeros_like(allocations)
    for i in range(allocations.shape[0]):
        fm_modulated_llm_share[i] = np.sum(
            caputo_fractional_derivative(allocations[i], 0.5, time_constants[i]) * time_constants[i]
        )
    
    # Compute the pheromone-infused recovery priorities
    recovery_priorities = np.zeros_like(allocations)
    for i in range(allocations.shape[0]):
        surface_key = f"{morphologies[i]['length']}-{morphologies[i]['width']}-{morphologies[i]['height']}"
        signal_value = pheromone_system.calculate_pheromone_signal(
            surface_key,
            "recovery_priority",
            recovery_priority(morphologies[i]),
            24 * 60 * 60,
        )
        recovery_priorities[i] = signal_value
    
    # Combine the fractional-memory modulated LLM share and pheromone-infused recovery priorities
    hybrid_fm_phantasmagoria_share = fm_modulated_llm_share + recovery_priorities
    
    return hybrid_fm_phantasmagoria_share

def init_hybrid_fm_phantasmagoria(
    morphologies: Dict[str, Dict[str, float]],
    pheromone_system: PheromoneSystem,
    allocations: np.ndarray,
    time_constants: np.ndarray,
) -> Dict[str, float]:
    """Initialise the hybrid allocation parameters."""
    return {
        "morphologies": morphologies,
        "pheromone_system": pheromone_system,
        "allocations": allocations,
        "time_constants": time_constants,
    }

def hybrid_fm_allocate_by_dates(
    hybrid_fm_phantasmagoria_params: Dict[str, float],
    dates: List[date],
) -> np.ndarray:
    """Compute per-day, per-group allocations using the fractional-memory modulated LLM share and pheromone-infused recovery priorities."""
    hybrid_fm_phantasmagoria_share = hybrid_fm_phantasmagoria(
        hybrid_fm_phantasmagoria_params["morphologies"],
        hybrid_fm_phantasmagoria_params["pheromone_system"],
        hybrid_fm_phantasmagoria_params["allocations"],
        hybrid_fm_phantasmagoria_params["time_constants"],
    )
    
    per_day_allocations = np.zeros((len(dates), len(GROUPS)))
    for i in range(len(dates)):
        for j in range(len(GROUPS)):
            per_day_allocations[i, j] = hybrid_fm_phantasmagoria_share[i] * 0.5
    
    return per_day_allocations

def summarize_hybrid_fm_savings(
    hybrid_fm_phantasmagoria_params: Dict[str, float],
    allocations: np.ndarray,
) -> float:
    """Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage."""
    hybrid_fm_phantasmagoria_share = hybrid_fm_phantasmagoria(
        hybrid_fm_phantasmagoria_params["morphologies"],
        hybrid_fm_phantasmagoria_params["pheromone_system"],
        hybrid_fm_phantasmagoria_params["allocations"],
        hybrid_fm_phantasmagoria_params["time_constants"],
    )
    
    savings_percentage = np.mean(hybrid_fm_phantasmagoria_share) / np.mean(allocations) * 100
    
    return savings_percentage

if __name__ == "__main__":
    # Smoke test
    morphologies = {
        "morphology1": {"length": 10.0, "width": 5.0, "height": 2.0},
        "morphology2": {"length": 5.0, "width": 10.0, "height": 3.0},
    }
    pheromone_system = PheromoneSystem()
    allocations = np.random.rand(2, 4)
    time_constants = np.random.rand(2)
    
    hybrid_fm_phantasmagoria_params = init_hybrid_fm_phantasmagoria(
        morphologies,
        pheromone_system,
        allocations,
        time_constants,
    )
    
    per_day_allocations = hybrid_fm_allocate_by_dates(hybrid_fm_phantasmagoria_params, [date.today()] * 10)
    print(per_day_allocations)
    
    savings_percentage = summarize_hybrid_fm_savings(hybrid_fm_phantasmagoria_params, allocations)
    print(f"Savings percentage: {savings_percentage:.6f}%")