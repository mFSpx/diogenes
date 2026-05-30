# DARWIN HAMMER — match 8, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:25:05Z

"""
This module integrates the hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0 and 
model_vram_scheduler algorithms into a single hybrid system. The bridge between the two 
structures is the concept of information entropy applied to the decision hygiene scoring 
system and the expected cost of the minimum-cost tree computed using Bayesian update, 
which is then used to inform the advisory VRAM preemption planner. The governing equations 
of both parents are integrated through the use of Bayesian update to inform the planning 
of VRAM allocation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

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
    return (likelihood * prior) / marginal

def vram_planning(bayes_posterior: np.ndarray, vram_budget: int, reserve_mb: int) -> VramSlotPlan:
    """
    Use Bayesian update to inform the planning of VRAM allocation.
    """
    estimated_mb = int(vram_budget * bayes_posterior[0])
    reason = "Bayesian update"
    detail = {"bayes_posterior": bayes_posterior.tolist(), "vram_budget": vram_budget}
    return VramSlotPlan("vram_slot", "vram", "allocate", estimated_mb, reason, detail)

def calculate_expected_cost(tree_metrics_result: Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]], vram_slot_plan: VramSlotPlan) -> float:
    """
    Calculate the expected cost of the minimum-cost tree using the VRAM allocation plan.
    """
    adj, edge_len, dist = tree_metrics_result
    expected_cost = 0.0
    for node in adj:
        expected_cost += dist[node] * vram_slot_plan.estimated_mb
    return expected_cost

def hybrid_operation(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, prior: np.ndarray, likelihood: np.ndarray, false_positive: float, vram_budget: int, reserve_mb: int) -> Tuple[float, VramSlotPlan]:
    """
    Perform the hybrid operation, integrating the governing equations of both parents.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    vram_slot_plan = vram_planning(posterior, vram_budget, reserve_mb)
    expected_cost = calculate_expected_cost((adj, edge_len, dist), vram_slot_plan)
    return expected_cost, vram_slot_plan

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 1.0), "C": (2.0, 2.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.7, 0.3])
    false_positive = 0.1
    vram_budget = 4096
    reserve_mb = 768
    expected_cost, vram_slot_plan = hybrid_operation(nodes, edges, root, prior, likelihood, false_positive, vram_budget, reserve_mb)
    print("Expected cost:", expected_cost)
    print("VRAM slot plan:", vram_slot_plan.as_dict())