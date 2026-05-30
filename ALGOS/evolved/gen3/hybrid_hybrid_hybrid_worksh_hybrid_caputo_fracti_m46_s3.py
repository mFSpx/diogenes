# DARWIN HAMMER — match 46, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:26:38Z

import math
import random
import sys
from pathlib import Path
import numpy as np
from datetime import date

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

def gamma(z):
    """Lanczos approximation for the gamma function."""
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma(1 - z))
    else:
        x = _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1)
        t = z + _LANCZOS_G + 0.5
        return np.sqrt(2 * np.pi) * np.power(t, z + 0.5) * np.exp(-t) * np.sum(x)

def caputo_weights(alpha, t):
    """Compute normalized Caputo kernel weights for a history."""
    return np.power(t, alpha - 1) / gamma(alpha)

# ----------------------------------------------------------------------
# Hybrid LTC module
# ----------------------------------------------------------------------
def init_hybrid_ltc():
    """Initialise LTC parameters for a single-dimensional day-of-week input."""
    tau_max = 1.0  # maximum effective time constant
    llm_base = 0.5  # baseline LLM share
    return tau_max, llm_base

def hybrid_allocate_by_dates(tau_sys, llm_base, tau_max, dates):
    """Compute per-day, per-group allocations using the LTC-modulated LLM share."""
    allocations = []
    for date in dates:
        day_of_week = (date.weekday() + 1) / 7  # scale to [0, 1]
        llm_units = llm_base * (tau_sys / tau_max) * (1 + 0.1 * day_of_week)  # introduce day-of-week dependency
        allocation = {group: llm_units for group in GROUPS}
        allocations.append(allocation)
    return allocations

# ----------------------------------------------------------------------
# Parent B – Caputo utilities and fractional-memory tree cost
# ----------------------------------------------------------------------
def fractional_weighted_sum(weights, history):
    """Apply the weights to an arbitrary numeric history."""
    return np.sum(weights * history)

def length(edge):
    """Euclidean edge length (from the tree module)."""
    return np.linalg.norm(edge)

def incremental_fractional_tree_cost(alpha, material, path_weight, edges, distances):
    """Build the tree edge-by-edge, updates distances, and evaluates the hybrid cost."""
    t = np.arange(1, len(distances) + 1)
    caputo_weights = caputo_weights(alpha, t)
    caputo_weights = caputo_weights / np.sum(caputo_weights)  # normalize weights
    history = [length(edge) for edge in edges] + distances
    weighted_sum = fractional_weighted_sum(caputo_weights, history)
    return material + path_weight * weighted_sum

def fractional_ssm_step(alpha, x, u):
    """Generic state-space update that also uses the same Caputo weighting."""
    t = np.arange(1, len(x) + 1)
    weights = caputo_weights(alpha, t)
    weighted_sum = fractional_weighted_sum(weights, x)
    return weighted_sum + u

# ----------------------------------------------------------------------
# Hybrid system
# ----------------------------------------------------------------------
def hybrid_system(tau_sys, alpha, material, path_weight, edges, distances, dates):
    """Combine the LTC module and fractional-memory tree cost module."""
    tau_max, llm_base = init_hybrid_ltc()
    allocations = hybrid_allocate_by_dates(tau_sys, llm_base, tau_max, dates)
    fractional_cost = incremental_fractional_tree_cost(alpha, material, path_weight, edges, distances)
    return allocations, fractional_cost

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # test parameters
    alpha = 0.5  # fractional order
    material = 1.0  # material length
    path_weight = 0.5  # path weight
    edges = [[1, 2], [2, 3], [3, 4]]  # tree edges
    distances = [1, 2, 3]  # tree distances
    dates = [date(2022, 7, 1), date(2022, 7, 2), date(2022, 7, 3)]  # dates

    # run the hybrid system
    tau_sys = 1.0  # effective time constant
    allocations, fractional_cost = hybrid_system(tau_sys, alpha, material, path_weight, edges, distances, dates)

    # print the results
    print("Allocations:")
    for allocation in allocations:
        print(allocation)
    print("Fractional cost:", fractional_cost)