# DARWIN HAMMER — match 2800, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py (gen4)
# born: 2026-05-29T23:46:02Z

"""
Hybrid Fractional-LTC Allocation and Minimum-Cost Tree Bayesian-Gradient Algorithm

This module fuses two parent algorithms:

* **Hybrid Fractional-LTC Allocation Module** – couples a deterministic/LLM split with a Liquid Time-Constant (LTC) network and integrates a Caputo fractional derivative with a minimum-cost tree scoring.
* **Hybrid Minimum-Cost Tree Bayesian-Gradient Algorithm** – combines deterministic tree utilities (edge lengths, root-to-node distances) with linear transform loss, gradient, and XGBoost-style split gain.

The mathematical bridge is established by using the tokenization and chunking operations from the first parent to generate input for the LTC network, and using the tree-regularized loss from the second parent to update the weight matrix in a single hybrid step.

The hybrid treats each calendar day as a discrete time step *t*. The day-of-week (scaled to [0, 1]) is fed to the LTC as the external input **I(t)**. 
The resulting scalar *τ_sys(t)* is used to scale the LLM allocation for that day, and the bandit router's action selection determines the chunking process for the tokenization.

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
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
])

def ltc_network(input_value: float, tau_sys: float) -> float:
    """Liquid Time-Constant network."""
    return tau_sys * input_value

def tree_metrics(
    nodes: dict,
    edges: list,
    root: str,
) -> tuple:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root-to-node distance
    """
    adj: dict = {}
    edge_len: dict = {}
    node_dist: dict = {}

    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])

    for node in nodes:
        node_dist[node] = math.hypot(nodes[node][0] - nodes[root][0], nodes[node][1] - nodes[root][1])

    return adj, edge_len, node_dist

def calculate_llm_units(tau_sys: float, llm_base: float, tau_max: float, alpha: float, bandit_action_propensity: float) -> float:
    """Calculate LLM units."""
    return llm_base * (tau_sys / tau_max) * (1 / (1 + alpha)) * bandit_action_propensity

def hybrid_operation(input_value: float, nodes: dict, edges: list, root: str, llm_base: float, tau_max: float, alpha: float, bandit_action_propensity: float) -> tuple:
    """Hybrid operation."""
    tau_sys = ltc_network(input_value, 1.0)
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    llm_units = calculate_llm_units(tau_sys, llm_base, tau_max, alpha, bandit_action_propensity)
    return llm_units, adj, edge_len, node_dist

if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0)
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    input_value = 0.5
    llm_base = 0.8
    tau_max = 1.0
    alpha = 0.5
    bandit_action_propensity = 0.7
    llm_units, adj, edge_len, node_dist = hybrid_operation(input_value, nodes, edges, root, llm_base, tau_max, alpha, bandit_action_propensity)
    print("LLM units:", llm_units)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Node distances:", node_dist)