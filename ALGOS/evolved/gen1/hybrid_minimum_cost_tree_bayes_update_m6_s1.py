# DARWIN HAMMER — match 6, survivor 1
# gen: 1
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: bayes_update.py (gen0)
# born: 2026-05-29T23:15:26Z

#!/usr/bin/env python3
"""Hybrid algorithm combining minimum-cost tree scoring from minimum_cost_tree.py and Bayesian evidence update from bayes_update.py.
The bridge between the two structures is the notion of uncertainty in the tree edges and nodes. By assigning prior probabilities to the edges and nodes, we can update these probabilities based on new evidence using the Bayesian update rule.
The tree cost function is then modified to take into account the updated probabilities, allowing for a more informed decision-making process in the presence of uncertainty."""

import math
import random
import sys
import pathlib
from typing import Tuple, Dict, List

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal

def hybrid_tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float], path_weight: float = 0.2) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b]) * edge_priors[(a, b)]
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    updated_material = 0.0
    for a, b in edges:
        likelihood = 0.8  # example likelihood
        marginal = bayes_marginal(edge_priors[(a, b)], likelihood, 0.1)
        updated_material += length(nodes[a], nodes[b]) * bayes_update(edge_priors[(a, b)], likelihood, marginal)
    return updated_material + path_weight * sum(dist.values())

def hybrid_tree_edge_uncertainty(nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float]) -> Dict[Edge, float]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for a, b in edges:
        likelihood = 0.8  # example likelihood
        marginal = bayes_marginal(edge_priors[(a, b)], likelihood, 0.1)
        edge_priors[(a, b)] = bayes_update(edge_priors[(a, b)], likelihood, marginal)
    return edge_priors

def hybrid_tree_node_uncertainty(nodes: Dict[str, Point], edges: List[Edge], root: str, node_priors: Dict[str, float]) -> Dict[str, float]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for node in nodes:
        likelihood = 0.8  # example likelihood
        marginal = bayes_marginal(node_priors[node], likelihood, 0.1)
        node_priors[node] = bayes_update(node_priors[node], likelihood, marginal)
    return node_priors

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    edge_priors = {("A", "B"): 0.5, ("B", "C"): 0.7, ("C", "A"): 0.3}
    print(hybrid_tree_cost(nodes, edges, root, edge_priors))
    print(hybrid_tree_edge_uncertainty(nodes, edges, root, edge_priors))
    print(hybrid_tree_node_uncertainty(nodes, edges, root, {"A": 0.5, "B": 0.7, "C": 0.3}))