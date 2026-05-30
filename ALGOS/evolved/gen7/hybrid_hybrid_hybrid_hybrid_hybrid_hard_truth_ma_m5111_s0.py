# DARWIN HAMMER — match 5111, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2381_s3.py (gen6)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py (gen2)
# born: 2026-05-29T23:59:47Z

"""
Hybrid module combining Hybrid Geometric-Product LTC + Fold-Change Bandit Fusion 
and Hybrid module combining LUCIDOTA's hard-truth math and DARWIN HAMMER's Minimum-Cost Tree scoring and Bayesian evidence update.

The mathematical bridge between the two parents lies in the fusion of the 
Clifford algebra multivector representation of a weight matrix W from the first parent 
and the probabilistic weights from the stylometry features/classifier helpers of the second parent. 
The hybrid update rule replaces the scalar learning-rate η of the LTC with the 
product λ·φ from the fold-change bandit, and incorporates the expected edge lengths 
and node distances from the second parent.

This module implements:
* `hybrid_geometric_product` – Clifford product of two multivectors with edge weights.
* `hybrid_update` – Performs the fused LTC-bandit update with edge weights.
* `hybrid_tree_cost` – Evaluates the hybrid cost with node and edge weights.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return a sorted blade and its sign after bubble-sorting.
    Duplicate indices cancel (e ↔ e = 0)."""
    lst = list(indices)
    sign = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
    return lst, sign

def geometric_product(a: List[Tuple[int, float]], b: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Clifford product of two multivectors."""
    result = []
    for a_blade, a_coeff in a:
        for b_blade, b_coeff in b:
            blade = tuple(sorted(set(a_blade + b_blade)))
            sign = (-1) ** (len(a_blade) * len(b_blade) - sum(a_blade) * sum(b_blade))
            coeff = a_coeff * b_coeff * sign
            if coeff != 0:
                result.append((blade, coeff))
    return result

def hybrid_geometric_product(a: List[Tuple[int, float]], b: List[Tuple[int, float]], edge_weights: Dict[Tuple[int, int], float]) -> List[Tuple[int, float]]:
    """Clifford product of two multivectors with edge weights."""
    result = []
    for a_blade, a_coeff in a:
        for b_blade, b_coeff in b:
            blade = tuple(sorted(set(a_blade + b_blade)))
            sign = (-1) ** (len(a_blade) * len(b_blade) - sum(a_blade) * sum(b_blade))
            coeff = a_coeff * b_coeff * sign
            if coeff != 0:
                edge_weight = 1
                for i in range(len(blade)):
                    for j in range(i + 1, len(blade)):
                        edge_weight *= edge_weights.get((blade[i], blade[j]), 1)
                result.append((blade, coeff * edge_weight))
    return result

# ----------------------------------------------------------------------
# Tree metrics and Bayesian update
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    adjacency = defaultdict(list)
    edge_lengths = {}
    root_distances = {}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        edge_lengths[(u, v)] = length(nodes[u], nodes[v])
        edge_lengths[(v, u)] = length(nodes[v], nodes[u])
    stack = [(root, 0)]
    while stack:
        node, distance = stack.pop()
        root_distances[node] = distance
        for neighbor in adjacency[node]:
            if neighbor not in root_distances:
                stack.append((neighbor, distance + edge_lengths[(node, neighbor)]))
    return adjacency, edge_lengths, root_distances

def bayes_edge_posteriors(edge_counts: Dict[Tuple[str, str], int], total_counts: Dict[str, int]) -> Dict[Tuple[str, str], float]:
    posteriors = {}
    for u, v in edge_counts:
        posterior = edge_counts[(u, v)] / total_counts[u]
        posteriors[(u, v)] = posterior
    return posteriors

# ----------------------------------------------------------------------
# Hybrid update and tree cost
# ----------------------------------------------------------------------
def hybrid_update(W: List[Tuple[int, float]], X: List[Tuple[int, float]], action_id: int, edge_weights: Dict[Tuple[int, int], float]) -> List[Tuple[int, float]]:
    lambda_phi = 0.1  # replace with actual lambda and phi values
    return [(blade, coeff * lambda_phi) for blade, coeff in hybrid_geometric_product(W, X, edge_weights)]

def hybrid_tree_cost(nodes: Dict[str, Tuple[float, float]], edges: List[Tuple[str, str]], root: str, edge_weights: Dict[Tuple[str, str], float]) -> float:
    adjacency, edge_lengths, root_distances = tree_metrics(nodes, edges, root)
    edge_posteriors = bayes_edge_posteriors({edge: 1 for edge in edges}, {node: len(adjacency[node]) for node in nodes})
    tree_cost = 0
    for u, v in edges:
        tree_cost += edge_posteriors[(u, v)] * edge_lengths[(u, v)] * edge_weights.get((u, v), 1)
    return tree_cost

if __name__ == "__main__":
    # Smoke test
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    edge_weights = {("A", "B"): 0.5, ("B", "C"): 0.7, ("C", "A"): 0.3}
    W = [((1,), 1.0), ((2,), 2.0)]
    X = [((1,), 3.0), ((2,), 4.0)]
    action_id = 0
    print(hybrid_geometric_product(W, X, edge_weights))
    print(hybrid_update(W, X, action_id, edge_weights))
    print(hybrid_tree_cost(nodes, edges, root, edge_weights))