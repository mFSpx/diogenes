# DARWIN HAMMER — match 46, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:26:37Z

"""
Hybrid Allocation-LTC & Fractional-Memory Tree Cost Module

Parents:
- **Hybrid Allocation-LTC** (hybrid_hybrid_workshare_all_liquid_time_constant_m67_s2.py)
- **Hybrid Fractional-Memory Tree Cost** (hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py)

Mathematical Bridge
-------------------
The hybrid fuses the temporal dynamics of the Liquid Time-Constant (LTC) module
with the summation-over-history of the Minimum-Cost Tree scoring.  The key
interface is the *effective time constant* τ_sys(t) that modulates the LLM
allocation in the LTC module, which is analogous to the Caputo weights used in
the fractional-memory tree cost.  We leverage this analogy to introduce a
further fractional weighting into the tree cost calculation.

The module therefore fuses:
1. The temporal dynamics of the LTC module as a multiplicative factor on the
   LLM share of each day.
2. The summation-over-history of the Minimum-Cost Tree scoring, replaced by a
   Caputo-weighted sum.

The resulting hybrid system has the following structure:

- The LTC module computes the effective time constant τ_sys(t) based on the
  day-of-week input and the learned gating function f.
- The fractional-memory tree cost module computes the tree cost C_frac using the
  Caputo weights, material length, and path weight.
- The hybrid system combines the two modules, using the effective time constant
  as a multiplicative factor on the LLM share of each day, and introducing a
  Caputo-weighted sum into the tree cost calculation.

The three public functions demonstrate the hybrid operation:
- `init_hybrid_ltc` – initialise LTC parameters for a single-dimensional
  day-of-week input.
- `hybrid_allocate_by_dates` – compute per-day, per-group allocations using the
  LTC-modulated LLM share.
- `incremental_fractional_tree_cost` – builds the tree edge-by-edge, updates
  distances, and evaluates the hybrid cost using the fractional memory term.
- `fractional_ssm_step` – a generic state-space update that also uses the same
  Caputo weighting, illustrating the deeper algebraic connection.
"""

import math
import random
import sys
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
        llm_units = llm_base * (tau_sys / tau_max)
        allocation = {group: llm_units for group in GROUPS}
        allocations.append(allocation)
    return allocations

# ----------------------------------------------------------------------
# Parent B – Caputo utilities and fractional-memory tree cost
# ----------------------------------------------------------------------
def caputo_weights(alpha, t):
    """Compute normalized Caputo kernel weights for a history."""
    return t**(alpha - 1) / np.math.gamma(alpha)

def fractional_weighted_sum(weights, history):
    """Apply the weights to an arbitrary numeric history."""
    return np.sum(weights * history)

def length(edge):
    """Euclidean edge length (from the tree module)."""
    return np.linalg.norm(edge)

def incremental_fractional_tree_cost(alpha, material, path_weight, edges, distances):
    """Build the tree edge-by-edge, updates distances, and evaluates the hybrid cost."""
    caputo_weights = [caputo_weights(alpha, t) for t in range(len(distances))]
    caputo_weights = [weight / np.sum(caputo_weights) for weight in caputo_weights]
    history = [length(edge) for edge in edges] + [dist for dist in distances]
    weighted_sum = fractional_weighted_sum(caputo_weights, history)
    return material + path_weight * weighted_sum

def fractional_ssm_step(alpha, x, u):
    """Generic state-space update that also uses the same Caputo weighting."""
    weights = caputo_weights(alpha, np.arange(len(x)))
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