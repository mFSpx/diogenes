# DARWIN HAMMER — match 3466, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m616_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py (gen3)
# born: 2026-05-29T23:50:16Z

"""
Hybrid Fractional-Memory Regret-Weighted Module with Minimum-Cost Tree Bayesian Update
====================================================================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Regret-Weighted Module (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m616_s1.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, using a Caputo fractional derivative 
  to introduce a memory term into the allocation process.
* **Hybrid Minimum-Cost Tree Bayesian Update (hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py)** – 
  provides a decision hygiene scoring system applied to the allocation of VRAM resources, 
  and the expected cost of the minimum-cost tree computed using Bayesian update.

The mathematical bridge between these two structures lies in the application of the fractional-memory 
kernel to the Bayesian update, effectively modulating the probability of successful VRAM allocation 
based on its similarity to a set of reference allocations, and the use of the Caputo fractional derivative 
to introduce a memory term into the Bayesian update.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Regret-Weighted Module.
2. The input-dependent effective time constant of the Hybrid Fractional-Memory Regret-Weighted Module.
3. The fractional-memory tree cost of the Hybrid fractional-memory tree cost module.
4. The regret-weighted strategy of the Hybrid Fractional-Memory Regret-Weighted Module.
5. The decision hygiene scoring system of the Hybrid Minimum-Cost Tree Bayesian Update.
6. The Bayesian update of the Hybrid Minimum-Cost Tree Bayesian Update.

The implementation below provides:

* `init_hybrid_fm_regret_mct` – initialise the hybrid regret parameters and minimum-cost tree.
* `hybrid_fm_regret_mct_allocate` – compute per-action, per-group allocations using 
  the fractional-memory modulated regret-weighted strategy and minimum-cost tree Bayesian update.
* `summarize_hybrid_fm_regret_mct_savings` – aggregate baseline vs. fractional-memory modulated 
  regret-weighted strategy and report a savings percentage.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field

GROUPS = ("codex", "groq", "cohere", "local_mod")

@dataclass
class Node:
    id: str
    x: float
    y: float

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Node],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    dist: Dict[str, float] = {root: 0}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length((nodes[u].x, nodes[u].y), (nodes[v].x, nodes[v].y))
        edge_len[(v, u)] = edge_len[(u, v)]

    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        for neighbour in adj[node]:
            if neighbour not in dist:
                dist[neighbour] = d + edge_len[(node, neighbour)]
                stack.append((neighbour, d + edge_len[(node, neighbour)]))

    return adj, edge_len, dist

def caputo_derivative(f: callable, t: float, alpha: float) -> float:
    """Caputo fractional derivative."""
    return (1 / math.gamma(1 - alpha)) * integral(lambda tau: (f(tau) - f(0)) / (t - tau)**alpha, 0, t)

def integral(f: callable, a: float, b: float) -> float:
    """Numerical integration using the trapezoidal rule."""
    n = 1000
    h = (b - a) / n
    return (h / 2) * (f(a) + f(b) + 2 * sum(f(a + i * h) for i in range(1, n)))

def init_hybrid_fm_regret_mct(
    nodes: Dict[str, Node],
    edges: List[Tuple[str, str]],
    root: str,
    alpha: float,
    beta: float,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], float]:
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    regret = {node: 0 for node in nodes}
    return adj, edge_len, dist, regret

def hybrid_fm_regret_mct_allocate(
    adj: Dict[str, List[str]],
    edge_len: Dict[Tuple[str, str], float],
    dist: Dict[str, float],
    regret: Dict[str, float],
    alpha: float,
    beta: float,
) -> Dict[str, float]:
    for node in regret:
        regret[node] = caputo_derivative(lambda t: dist[node]**t, 1, alpha) * beta
    return regret

def summarize_hybrid_fm_regret_mct_savings(
    regret: Dict[str, float],
    baseline: Dict[str, float],
) -> float:
    savings = 0
    for node in regret:
        savings += (baseline[node] - regret[node]) / baseline[node]
    return (savings / len(regret)) * 100

if __name__ == "__main__":
    nodes = {
        "A": Node("A", 0, 0),
        "B": Node("B", 1, 0),
        "C": Node("C", 1, 1),
        "D": Node("D", 0, 1),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"
    alpha = 0.5
    beta = 0.1

    adj, edge_len, dist, regret = init_hybrid_fm_regret_mct(nodes, edges, root, alpha, beta)
    regret = hybrid_fm_regret_mct_allocate(adj, edge_len, dist, regret, alpha, beta)
    baseline = {node: 1 for node in nodes}
    savings = summarize_hybrid_fm_regret_mct_savings(regret, baseline)
    print(f"Savings: {savings}%")