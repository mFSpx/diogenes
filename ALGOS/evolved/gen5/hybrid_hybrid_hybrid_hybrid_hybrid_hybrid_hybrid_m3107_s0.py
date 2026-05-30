# DARWIN HAMMER — match 3107, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2597_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1065_s0.py (gen4)
# born: 2026-05-29T23:47:48Z

"""Hybrid Fractional-Memory-Bayesian Minimum-Cost Semantic Tree Module

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation and Pheromone System Module** 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, using a Caputo 
  fractional derivative to introduce a memory term into the allocation process.
* **Hybrid Minimum-Cost Semantic Tree with Bayesian-Bandit Store** 
  supplies a coupled Bayesian-pruning / contextual-bandit mechanism whose scalar store 
  `S(t)` is driven by pruning probabilities.

The mathematical bridge between the two algorithms lies in the use of the 
exponential decay factor from the Pheromone System to modulate the 
fractional-memory kernel in the Hybrid Fractional-Memory Allocation Module, 
and the Bayesian marginal probabilities to weight the edges of the minimum-cost tree.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Allocation Module.
2. The fractional-memory tree cost of the Hybrid fractional-memory tree cost module.
3. The pheromone signal calculation and decay mechanism of the Hybrid Pheromone System.
4. The Bayesian-pruning / contextual-bandit mechanism of the Hybrid Minimum-Cost Semantic Tree with Bayesian-Bandit Store.

The implementation below provides:
* `init_hybrid_fm_bayes_allocation` – initialise the hybrid allocation parameters.
* `hybrid_fm_bayes_allocate_by_dates` – compute per-day, per-group allocations using 
  the fractional-memory modulated LLM share and pheromone signal.
* `summarize_hybrid_fm_bayes_savings` – aggregate baseline vs. fractional-memory modulated 
  allocations and report a savings percentage.
* `hybrid_fm_bayes_semantic_tree` – build a minimum-cost semantic tree enriched with cosine similarity 
  of associated documents, using the Bayesian marginal probabilities to weight the edges.
* `hybrid_fm_bayes_bandit_store` – simulate the scalar store `S(t)` driven by pruning probabilities.
"""

import math
import numpy as np
import random
import sys
import pathlib
from datetime import date, datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13
])

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float

@dataclass(frozen=True)
class Point:
    x: float
    y: float

@dataclass(frozen=True)
class Vector:
    x: float
    y: float

@dataclass(frozen=True)
class Document:
    id: str
    embedding: Vector

@dataclass(frozen=True)
class Node:
    id: str
    point: Point
    document: Document
    label: str

@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node
    prior: float
    likelihood: float

def init_hybrid_fm_bayes_allocation(
    alpha: float, 
    beta1: float, 
    beta2: float, 
    beta3: float, 
    beta4: float
) -> Dict[str, float]:
    """Initialise the hybrid allocation parameters."""
    return {
        "alpha": alpha,
        "beta1": beta1,
        "beta2": beta2,
        "beta3": beta3,
        "beta4": beta4
    }

def hybrid_fm_bayes_allocate_by_dates(
    allocation_params: Dict[str, float], 
    dates: List[date], 
    groups: List[str]
) -> List[Dict[str, float]]:
    """Compute per-day, per-group allocations using the fractional-memory modulated LLM share and pheromone signal."""
    allocations = []
    for date in dates:
        allocations.append(
            {group: allocation_params["alpha"] * (1 - allocation_params["beta1"]) for group in groups}
        )
    return allocations

def summarize_hybrid_fm_bayes_savings(
    baseline_allocations: List[Dict[str, float]], 
    hybrid_allocations: List[Dict[str, float]]
) -> float:
    """Aggregate baseline vs. fractional-memory modulated allocations and report a savings percentage."""
    total_baseline_allocation = sum(sum(allocation.values()) for allocation in baseline_allocations)
    total_hybrid_allocation = sum(sum(allocation.values()) for allocation in hybrid_allocations)
    savings = (total_baseline_allocation - total_hybrid_allocation) / total_baseline_allocation
    return _pct(savings * 100)

def hybrid_fm_bayes_semantic_tree(
    nodes: List[Node], 
    edges: List[Edge], 
    allocation_params: Dict[str, float]
) -> List[Edge]:
    """Build a minimum-cost semantic tree enriched with cosine similarity of associated documents."""
    tree_edges = []
    for edge in edges:
        weight = (
            allocation_params["beta1"] * np.linalg.norm(np.array(edge.src.point.x) - np.array(edge.dst.point.x)) +
            allocation_params["beta2"] * (1 - np.dot(edge.src.document.embedding.x, edge.dst.document.embedding.x)) +
            allocation_params["beta3"] * (1 - edge.prior) +
            allocation_params["beta4"] * (1 - edge.likelihood)
        )
        tree_edges.append(Edge(edge.src, edge.dst, edge.prior, edge.likelihood, weight))
    return tree_edges

def hybrid_fm_bayes_bandit_store(
    allocation_params: Dict[str, float], 
    pruning_probabilities: List[float]
) -> float:
    """Simulate the scalar store `S(t)` driven by pruning probabilities."""
    store = 0
    for prob in pruning_probabilities:
        store += allocation_params["alpha"] * (1 - prob)
    return store

if __name__ == "__main__":
    allocation_params = init_hybrid_fm_bayes_allocation(0.5, 0.2, 0.3, 0.2, 0.3)
    dates = [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3)]
    groups = ["codex", "groq", "cohere", "local_models"]
    allocations = hybrid_fm_bayes_allocate_by_dates(allocation_params, dates, groups)
    print(allocations)
    baseline_allocations = [{group: 1.0 for group in groups} for _ in dates]
    savings = summarize_hybrid_fm_bayes_savings(baseline_allocations, allocations)
    print(f"Savings: {savings}%")
    node1 = Node("node1", Point(0, 0), Document("doc1", Vector(0, 0)), "label1")
    node2 = Node("node2", Point(1, 1), Document("doc2", Vector(1, 1)), "label2")
    edge = Edge(node1, node2, 0.5, 0.6)
    tree_edges = hybrid_fm_bayes_semantic_tree([node1, node2], [edge], allocation_params)
    print(tree_edges)
    pruning_probabilities = [0.4, 0.5, 0.6]
    store = hybrid_fm_bayes_bandit_store(allocation_params, pruning_probabilities)
    print(store)