# DARWIN HAMMER — match 2805, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py (gen3)
# born: 2026-05-29T23:46:10Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable
import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist

def leaf_nodes(adj: Dict[str, List[str]], root: str) -> List[str]:
    leaves = []
    for node, neighbours in adj.items():
        if node == root:
            continue
        if len(neighbours) == 1:
            leaves.append(node)
    return leaves

def expected_cost(dist: Dict[str, float], leaves: Iterable[str]) -> float:
    leaf_dists = [dist[l] for l in leaves]
    if not leaf_dists:
        return 0.0
    return sum(leaf_dists) / len(leaf_dists)

def shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log(p) for p in probs)

@dataclass
class Endpoint:
    id: int
    failure_threshold: int = 3
    failures: int = 0
    open: bool = False
    last_event_at: str = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

    @property
    def failure_rate(self) -> float:
        return min(self.failures / self.failure_threshold, 1.0)

class Multivector:
    def __init__(self, components: Dict[str, float], n: int):
        self.components: Dict[str, float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-12
        }
        self.n = int(n)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scale(self, factor: float) -> "Multivector":
        return Multivector({b: c * factor for b, c in self.components.items()}, self.n)

    def copy(self) -> "Multivector":
        return Multivector(dict(self.components), self.n)

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    result: Dict[str, float] = {}

    for blade, ca in a.components.items():
        if blade in b.components:
            result[blade] = result.get(blade, 0.0) + ca * b.components[blade]

    for ba, ca in a.components.items():
        for bb, cb in b.components.items():
            if ba == bb:
                continue
            new_blade = "".join(sorted(set(ba + bb)))
            result[new_blade] = result.get(new_blade, 0.0) + ca * cb

    return Multivector(result, a.n)

def compute_endpoint_weights(
    dist: Dict[str, float],
    leaves: List[str],
    endpoints: Dict[str, Endpoint],
    hygiene_counts: Dict[str, int],
    base_multivector: Multivector,
) -> Dict[str, Multivector]:
    entropy = shannon_entropy(hygiene_counts)
    entropy_factor = math.exp(-entropy)

    weights: Dict[str, Multivector] = {}
    for leaf in leaves:
        endpoint = endpoints.get(leaf)
        if endpoint is None or not endpoint.allow():
            weights[leaf] = Multivector({}, base_multivector.n)
            continue
        cost_factor = math.exp(-dist[leaf])
        health_factor = 1.0 - endpoint.failure_rate
        weight = cost_factor * health_factor * entropy_factor
        weights[leaf] = base_multivector.scale(weight)

    return weights

def main():
    nodes = {
        'A': (0.0, 0.0),
        'B': (3.0, 0.0),
        'C': (3.0, 4.0),
        'D': (0.0, 4.0),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    leaves = leaf_nodes(adj, root)

    endpoints = {
        'A': Endpoint(1),
        'B': Endpoint(2),
        'C': Endpoint(3),
        'D': Endpoint(4),
    }
    hygiene_counts = {'A': 10, 'B': 20, 'C': 30, 'D': 40}

    base_multivector = Multivector({'e1': 1.0, 'e2': 2.0}, 2)
    weights = compute_endpoint_weights(dist, leaves, endpoints, hygiene_counts, base_multivector)
    for leaf, weight in weights.items():
        print(f"Leaf: {leaf}, Weight: {weight}")

if __name__ == "__main__":
    main()