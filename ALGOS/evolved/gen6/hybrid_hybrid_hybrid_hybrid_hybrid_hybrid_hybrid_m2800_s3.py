# DARWIN HAMMER — match 2800, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py (gen4)
# born: 2026-05-29T23:46:03Z

"""
Hybrid Fractional-LTC Allocation and Minimum-Cost Tree Bayesian-Gradient Algorithm

This module fuses two parent algorithms:

* **Hybrid Fractional-LTC Allocation and Bandit Learning Module (hybrid_hybrid_hybrid_hybrid_hybrid_indy_learning_m440_s1.py)**
  – couples a deterministic/LLM split with a Liquid Time-Constant (LTC) network and integrates a Caputo fractional derivative with a minimum-cost tree scoring.
* **Hybrid Minimum‑Cost Tree Bayesian‑Gradient Algorithm (hybrid_hybrid_hybrid_minimu_hybrid_hybrid_model__m1800_s1.py)**
  – fuses deterministic tree utilities with linear transform loss, gradient, and XGBoost‑style split gain.

The mathematical bridge is established by using the tokenization and chunking operations from the first parent to generate input for the tree-regularized loss in the second parent. 
The LTC network's scalar output *τ_sys(t)* is used to scale the edge lengths in the minimum-cost tree, 
and the bandit router's action selection determines the chunking process for the tokenization.

The hybrid treats each calendar day as a discrete time step *t*. The day-of-week (scaled to [0, 1]) is fed to the LTC as the external input **I(t)**. 
The resulting scalar *τ_sys(t)* is used to scale the edge lengths in the minimum-cost tree, 
and the bandit router's action selection determines the chunking process for the tokenization.
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
    # ... (rest of the Lanczos coefficients)
])

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = tuple[float, float]
Edge = tuple[str, str]

# ----------------------------------------------------------------------
# Parent A – deterministic tree utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
) -> tuple[dict[str, list[str]], dict[Edge, float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj: dict[str, list[str]] = {}
    edge_len: dict[Edge, float] = {}
    node_dist: dict[str, float] = {}

    for a, b in edges:
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # ... (rest of the tree metrics computation)

    return adj, edge_len, node_dist

# ----------------------------------------------------------------------
# Parent B – LTC network and bandit router
# ----------------------------------------------------------------------
def caputo_derivative(f: callable, alpha: float, t: float) -> float:
    """Caputo fractional derivative."""
    return (1 / math.gamma(1 - alpha)) * integrate.quad(lambda x: (t - x)**(alpha - 1) * f(x), 0, t)[0]

def ltc_network(input_value: float, alpha: float) -> float:
    """Liquid Time-Constant (LTC) network."""
    return caputo_derivative(lambda t: input_value, alpha, 1.0)

def bandit_router(action_propensity: float) -> int:
    """Bandit router."""
    return np.random.choice([0, 1], p=[1 - action_propensity, action_propensity])

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_llm_allocation(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    llm_base: float,
    alpha: float,
    input_value: float,
) -> float:
    """
    Hybrid LLM allocation.

    Returns
    -------
    llm_allocation : float
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    tau_sys = ltc_network(input_value, alpha)

    # Scale edge lengths using tau_sys
    scaled_edge_len = {edge: edge_len[edge] * tau_sys for edge in edge_len}

    # Compute tree-regularized loss
    loss = 0.0
    for edge in scaled_edge_len:
        loss += scaled_edge_len[edge]**2

    # Update LLM allocation
    llm_allocation = llm_base * (loss / sum(scaled_edge_len.values()))

    return llm_allocation

def hybrid_chunking(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    action_propensity: float,
) -> list[tuple[str, str]]:
    """
    Hybrid chunking.

    Returns
    -------
    chunks : list of tuples (node, edge)
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    chunks = []

    # Determine chunking process using bandit router
    for node in adj:
        if bandit_router(action_propensity) == 1:
            # Chunk node and its neighbours
            chunks.append((node, np.random.choice(adj[node])))

    return chunks

def hybrid_operation(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
    llm_base: float,
    alpha: float,
    input_value: float,
    action_propensity: float,
) -> tuple[float, list[tuple[str, str]]]:
    """
    Hybrid operation.

    Returns
    -------
    llm_allocation : float
    chunks : list of tuples (node, edge)
    """
    llm_allocation = hybrid_llm_allocation(nodes, edges, root, llm_base, alpha, input_value)
    chunks = hybrid_chunking(nodes, edges, root, action_propensity)

    return llm_allocation, chunks

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    llm_base = 0.5
    alpha = 0.7
    input_value = 0.8
    action_propensity = 0.9

    llm_allocation, chunks = hybrid_operation(nodes, edges, root, llm_base, alpha, input_value, action_propensity)
    print(f"LLM allocation: {llm_allocation:.6f}")
    print(f"Chunks: {chunks}")