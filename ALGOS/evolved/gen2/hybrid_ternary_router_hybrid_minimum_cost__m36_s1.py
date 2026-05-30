# DARWIN HAMMER — match 36, survivor 1
# gen: 2
# parent_a: ternary_router.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s1.py (gen1)
# born: 2026-05-29T23:23:31Z

"""
Hybrid algorithm combining the FairyFuse ternary router from ternary_router.py 
and the minimum-cost tree scoring with Bayesian evidence update from hybrid_minimum_cost_tree_bayes_update_m6_s1.py.
The mathematical bridge between the two structures is the notion of uncertainty 
in the tree edges and nodes, which can be updated using the Bayesian update rule 
and integrated into the routing decisions in the FairyFuse ternary router.
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

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

def route_packet(packet: Dict[str, Any], nodes: Dict[str, Point], edges: List[Edge], root: str, edge_priors: Dict[Edge, float]) -> Dict[str, Any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    # Integrate the hybrid tree cost into the routing decision
    cost = hybrid_tree_cost(nodes, edges, root, edge_priors)
    route = {
        "cost": cost,
        "text": text,
        "intent": intent,
        "context": context,
    }
    return route

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
    packet = {
        "text_surface": "example text",
        "normalized_intent": "example intent",
        "context": {
            "source": "example source",
            "source_ref": "example source ref",
            "ontology_terms": ["term1", "term2"],
            "epistemic_flag": True,
            "payload": {"key": "value"},
        },
    }
    route = route_packet(packet, nodes, edges, root, edge_priors)
    print(route)
    edge_uncertainty = hybrid_tree_edge_uncertainty(nodes, edges, root, edge_priors)
    print(edge_uncertainty)
    node_uncertainty = hybrid_tree_node_uncertainty(nodes, edges, root, {"A": 0.5, "B": 0.7, "C": 0.3})
    print(node_uncertainty)