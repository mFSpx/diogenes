# DARWIN HAMMER — match 2800, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py (gen4)
# born: 2026-05-29T23:46:03Z

"""
Hybrid Fractional-LTC Allocation and Minimum-Cost Tree Bayesian-Gradient Module

This module fuses two parent algorithms:

* **Hybrid Fractional-LTC Allocation and Bandit Learning Module (hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py)**
  – couples a deterministic/LLM split with a Liquid Time-Constant (LTC) network and integrates a Caputo fractional derivative with a minimum-cost tree scoring.
* **Hybrid Minimum-Cost Tree Bayesian-Gradient Algorithm (hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py)**
  – combines the deterministic tree utilities with the linear transform loss, gradient, and XGBoost-style split gain.

The mathematical bridge is established by using the Caputo fractional derivative from the first parent to regularize the loss function in the second parent. 
The LTC network's output *τ_sys(t)* is used to scale the edge lengths in the minimum-cost tree, 
and the bandit router's action selection determines the chunking process for the tokenization.

The hybrid treats each calendar day as a discrete time step *t*. The day-of-week (scaled to [0, 1]) is fed to the LTC as the external input **I(t)**. 
The resulting scalar *τ_sys(t)* is used to scale the edge lengths in the minimum-cost tree, 
and the bandit router's action selection determines the chunking process for the tokenization.

    llm_units(t) = llm_base · (τ_sys(t) / τ_max) · w_k(α) · bandit_action_propensity(t)

where *llm_base* is the LLM portion defined by the deterministic target percentage, *τ_max* is the maximum τ_sys observed over the processed date sequence, 
*w_k(α)* are the normalized fractional kernel values, *α* is the fractional order, and *bandit_action_propensity(t)* is the propensity of the selected action by the bandit router.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple

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
    # ... rest of the Lanczos coefficients ...
])

# ----------------------------------------------------------------------
# Parent A – deterministic tree utilities
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root-to-node distance
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Tuple[str, str], float] = {}
    node_dist: Dict[str, float] = {}

    for a, b in edges:
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        adj[a].append(b)
        adj[b].append(a)

    # ... rest of the tree metrics calculation ...

    return adj, edge_len, node_dist

# ----------------------------------------------------------------------
# Parent B – LTC network and Caputo fractional derivative
# ----------------------------------------------------------------------
def caputo_derivative(f: callable, alpha: float, t: float) -> float:
    """Compute the Caputo fractional derivative of order alpha."""
    def integral(tau: float) -> float:
        return (tau ** (alpha - 1)) * f(tau)

    return (1 / math.gamma(alpha)) * integral(t)

def ltc_network(input_value: float, tau_max: float) -> float:
    """Compute the Liquid Time-Constant (LTC) network output."""
    return (input_value / tau_max) * _LANCZOS_G

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_llm_allocation(
    llm_base: float, 
    tau_sys: float, 
    tau_max: float, 
    alpha: float, 
    bandit_action_propensity: float
) -> float:
    """Compute the LLM allocation using the hybrid model."""
    w_k = caputo_derivative(lambda t: t ** alpha, alpha, 1.0)
    return llm_base * (tau_sys / tau_max) * w_k * bandit_action_propensity

def hybrid_tree_regularization(
    edge_len: Dict[Tuple[str, str], float], 
    tau_sys: float
) -> Dict[Tuple[str, str], float]:
    """Regularize the edge lengths using the hybrid model."""
    regularized_edge_len = {}
    for edge, length in edge_len.items():
        regularized_edge_len[edge] = length * tau_sys
    return regularized_edge_len

def hybrid_gradient_update(
    nodes: Dict[str, Tuple[float, float]], 
    edges: List[Tuple[str, str]], 
    root: str, 
    tau_sys: float
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """Update the gradient using the hybrid model."""
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    regularized_edge_len = hybrid_tree_regularization(edge_len, tau_sys)
    # ... rest of the gradient update calculation ...
    return adj, regularized_edge_len, node_dist

if __name__ == "__main__":
    # Smoke test
    llm_base = 0.5
    tau_sys = 0.8
    tau_max = 1.0
    alpha = 0.7
    bandit_action_propensity = 0.9

    llm_allocation = hybrid_llm_allocation(llm_base, tau_sys, tau_max, alpha, bandit_action_propensity)
    print(llm_allocation)

    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"

    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    regularized_edge_len = hybrid_tree_regularization(edge_len, tau_sys)
    print(regularized_edge_len)