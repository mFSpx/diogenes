# DARWIN HAMMER — match 743, survivor 0
# gen: 4
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s4.py (gen2)
# parent_b: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s3.py (gen3)
# born: 2026-05-29T23:30:39Z

"""
Darwin Hammer — match 182, survivor 4
gen: 4
parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s4.py (gen2)
parent_b: hybrid_jepa_energy_hybrid_sparse_wta_hy_m181_s3.py (gen3)
born: 2026-05-29T23:31:14Z

The mathematical bridge between the two parents is the variational free energy (VFE) surrogate from the ModelPool class in the second parent, which can be used to manage the nodes and edges in the first parent's tree structure. This allows us to integrate the edge priors and likelihoods from the first parent with the variational free energy management from the second parent.

In this hybrid algorithm, we will use the VFE surrogate to manage the nodes and edges in the tree, and update the edge priors and likelihoods using the same Bayesian update rules as in the first parent.
"""

import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


Point = Tuple[float, float]
Edge = Tuple[str, str]


def _euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ModelTier:
    """Immutable descriptor of a model tier."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
    """
    Manages a pool of loaded models under a RAM ceiling.
    Uses a variational free‑energy (VFE) surrogate to decide loading/eviction.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb: int = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._vfe: float = 0.0  # variational free energy (lower is better)

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _vfe_penalty(self, delta: float) -> None:
        self._vfe += delta

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def free_energy(self) -> float:
        return self._vfe

    def add_model(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._vfe_penalty(1e10)  # heavy penalty
        if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
            self._vfe_penalty(1e10)  # heavy penalty


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> float:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + _euclidean_length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    model_pool = ModelPool()
    for a, b in edges:
        model_pool.add_model(ModelTier(f"{a}_{b}", 10, "T1"))
    updated_material = 0.0
    for a, b in edges:
        prior = edge_priors[(a, b)]
        marginal = likelihood * prior + false_positive * (1.0 - prior)
        posterior = prior * likelihood / marginal if marginal > 0 else 0.0
        updated_material += _euclidean_length(nodes[a], nodes[b]) * posterior

    return updated_material + path_weight * sum(dist.values())


def bayes_update_edge_priors(
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, float],
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> Dict[Edge, float]:
    updated = {}
    model_pool = ModelPool()
    for edge, prior in edge_priors.items():
        if not model_pool.is_loaded(edge):
            model_pool.add_model(ModelTier(edge, 10, "T1"))
        eff_likelihood = likelihood * evidence.get(edge, 1.0)
        marginal = eff_likelihood * prior + false_positive * (1.0 - prior)
        posterior = prior * eff_likelihood / marginal if marginal > 0 else 0.0
        updated[edge] = posterior
    return updated


def hybrid_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    evidence: Dict[Edge, float],
    path_weight: float = 0.2,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> float:
    updated_material = hybrid_tree_cost(nodes, edges, root, edge_priors, path_weight, likelihood, false_positive)
    updated_priors = bayes_update_edge_priors(edge_priors, evidence, likelihood, false_positive)
    return updated_material + path_weight * sum([_euclidean_length(nodes[a], nodes[b]) for a, b in updated_priors])


def hybrid_jepa_energy(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_priors: Dict[Edge, float],
    path_weight: float = 0.2,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
) -> float:
    updated_material = 0.0
    for a, b in edges:
        prior = edge_priors[(a, b)]
        marginal = likelihood * prior + false_positive * (1.0 - prior)
        posterior = prior * likelihood / marginal if marginal > 0 else 0.0
        updated_material += _euclidean_length(nodes[a], nodes[b]) * posterior
    return updated_material + path_weight * sum(_euclidean_length(nodes[a], nodes[b]) for a, b in edge_priors)


if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    edge_priors = {("A", "B"): 0.5, ("B", "C"): 0.5, ("C", "A"): 0.5}
    evidence = {("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "A"): 1.0}
    print(hybrid_update(nodes, edges, root, edge_priors, evidence))
    print(hybrid_jepa_energy(nodes, edges, root, edge_priors))