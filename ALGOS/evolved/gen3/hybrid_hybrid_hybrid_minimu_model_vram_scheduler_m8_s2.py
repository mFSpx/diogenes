# DARWIN HAMMER — match 8, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:25:05Z

"""
This module integrates the hybrid_minimum_cost_tree_bayes_update_m6_s2 and model_vram_scheduler algorithms into a single hybrid system.
The bridge between the two structures is the concept of decision hygiene scoring system applied to 
the allocation of VRAM resources, and the expected cost of the minimum-cost tree computed using Bayesian update.
The mathematical interface is formed by the idea of applying the Bayesian update to the probability of successful 
VRAM allocation, given the likelihood of a specific combination of resident DeepSeek/Qwen synthesis model, 
transient embedding lane, and selected LoRA adapter cartridges.

The governing equations for the hybrid system are formed by the following components:
- The decision hygiene scoring system from the hybrid_minimum_cost_tree_bayes_update_m6_s2 algorithm, 
  which is used to evaluate the probability of successful VRAM allocation.
- The Bayesian update from the hybrid_minimum_cost_tree_bayes_update_m6_s2 algorithm, 
  which is used to update the probability of successful VRAM allocation based on the likelihood of a specific combination.
- The VRAM allocation planning from the model_vram_scheduler algorithm, 
  which is used to evaluate the feasibility of a specific combination of resident DeepSeek/Qwen synthesis model, 
  transient embedding lane, and selected LoRA adapter cartridges.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
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
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P
    """
    return likelihood * prior / marginal

def vram_allocation_planning(available_vram: int, model_size: int, embedding_size: int, adapter_size: int) -> bool:
    """
    Evaluate the feasibility of a specific combination of resident DeepSeek/Qwen synthesis model, 
    transient embedding lane, and selected LoRA adapter cartridges.
    """
    total_size = model_size + embedding_size + adapter_size
    return total_size <= available_vram

def hybrid_vram_allocation_planning(
    available_vram: int, 
    model_size: int, 
    embedding_size: int, 
    adapter_size: int, 
    prior_probability: np.ndarray, 
    likelihood: np.ndarray, 
    false_positive: float
) -> Tuple[bool, np.ndarray]:
    """
    Apply the Bayesian update to the probability of successful VRAM allocation, 
    given the likelihood of a specific combination of resident DeepSeek/Qwen synthesis model, 
    transient embedding lane, and selected LoRA adapter cartridges.
    """
    marginal_probability = bayes_marginal(prior_probability, likelihood, false_positive)
    posterior_probability = bayes_update(prior_probability, likelihood, marginal_probability)
    allocation_feasibility = vram_allocation_planning(available_vram, model_size, embedding_size, adapter_size)
    return allocation_feasibility, posterior_probability

def calculate_expected_cost(
    adj: Dict[str, List[str]], 
    edge_len: Dict[Tuple[str, str], float], 
    dist: Dict[str, float], 
    root: str, 
    available_vram: int, 
    model_size: int, 
    embedding_size: int, 
    adapter_size: int
) -> float:
    """
    Calculate the expected cost of the minimum-cost tree, given the adjacency, edge lengths, 
    and distances from the root, as well as the availability of VRAM and the sizes of the model, 
    embedding, and adapter.
    """
    total_cost = 0.0
    for node in adj:
        if node != root:
            path_cost = dist[node]
            allocation_feasibility, _ = hybrid_vram_allocation_planning(
                available_vram, 
                model_size, 
                embedding_size, 
                adapter_size, 
                np.array([0.5]), 
                np.array([0.8]), 
                0.1
            )
            if allocation_feasibility:
                total_cost += path_cost
    return total_cost

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (3.0, 4.0), "C": (6.0, 8.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    available_vram = 4096
    model_size = 1800
    embedding_size = 384
    adapter_size = 128

    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    expected_cost = calculate_expected_cost(adj, edge_len, dist, root, available_vram, model_size, embedding_size, adapter_size)
    print(f"Expected cost: {expected_cost}")