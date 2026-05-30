# DARWIN HAMMER — match 392, survivor 0
# gen: 3
# parent_a: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:28:28Z

"""
Module for hybrid algorithm combining sparse winner-take-all tags and hybrid privacy model pool management with minimum-cost tree scoring and Bayesian evidence update.
The mathematical bridge between the two structures is the assignment of prior probabilities to the edges and nodes in the minimum-cost tree, 
which can be informed by the winner-take-all (WTA) mechanism in the model pool management.
The WTA mechanism is used to select the model with the highest score, and the reconstruction risk score is used to inform the WTA mechanism.
The prior probabilities in the minimum-cost tree are then updated based on new evidence using the Bayesian update rule.
"""

from __future__ import annotations
from typing import Any, Iterable, Dict, Tuple, List
import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}

    def is_loaded(self, name: str) -> bool: 
        return name in self.loaded

    def _used(self) -> int: 
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): 
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: 
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name]=model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

def expand(values: list[float], m: int, salt: str = '') -> list[float]:
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]

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
    return material + path_weight * sum(dist.values())

def hybrid_model_selection(model_pool: ModelPool, nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float]) -> ModelTier:
    model_scores = [model.ram_mb for model in model_pool.loaded.values()]
    wta_mask = top_k_mask(model_scores, 1)
    selected_model = [model for i, model in enumerate(model_pool.loaded.values()) if wta_mask[i] == 1][0]
    tree_cost_with_model = hybrid_tree_cost(nodes, edges, root, edge_priors)
    return selected_model

def hybrid_model_update(model_pool: ModelPool, nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float], new_model: ModelTier) -> None:
    model_pool.load_with_eviction(new_model)
    updated_tree_cost = hybrid_tree_cost(nodes, edges, root, edge_priors)
    new_edge_priors = {edge: bayes_update(edge_priors[edge], 0.5, updated_tree_cost) for edge in edge_priors}
    return new_edge_priors

def hybrid_model_inference(model_pool: ModelPool, nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float]) -> float:
    selected_model = hybrid_model_selection(model_pool, nodes, edges, root, edge_priors)
    updated_tree_cost = hybrid_tree_cost(nodes, edges, root, edge_priors)
    return updated_tree_cost

if __name__ == "__main__":
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (0.0, 1.0)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    root = 'A'
    edge_priors = {('A', 'B'): 0.5, ('B', 'C'): 0.6, ('C', 'A'): 0.7}
    model_pool = ModelPool()
    new_model = ModelTier('new_model', 1000, 'T1')
    model_pool.load(new_model)
    updated_tree_cost = hybrid_model_inference(model_pool, nodes, edges, root, edge_priors)
    print(updated_tree_cost)