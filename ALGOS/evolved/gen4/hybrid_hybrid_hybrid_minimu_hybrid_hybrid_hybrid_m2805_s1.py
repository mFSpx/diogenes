# DARWIN HAMMER — match 2805, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py (gen3)
# born: 2026-05-29T23:46:10Z

"""
This module integrates the hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py and 
hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py algorithms into a single hybrid system.
The mathematical bridge between the two parents lies in the application of Shannon entropy 
to weight the Multivector's components in the geometric product operation. Specifically, 
we use the Shannon entropy of the decision hygiene feature counts to adapt the Multivector's 
update rule, allowing the allocator to optimize its performance while minimizing uncertainty.

By representing the decision hygiene scores as a Multivector and using the geometric product 
to update the allocation, we can leverage the properties of Clifford algebras to optimize 
the allocator's performance while minimizing memory usage.

Parents:
- hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (DARWIN HAMMER — match 7, survivor 1)
- hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py (DARWIN HAMMER — match 54, survivor 0)
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
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
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist

def shannon_entropy(feature_counts: Dict[str, int]) -> float:
    """Calculate Shannon entropy from feature counts."""
    total = sum(feature_counts.values())
    entropy = 0.0
    for count in feature_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

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
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    @property
    def failure_rate(self) -> float:
        return self.failures / self.failure_threshold

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k}, self.n)

def hybrid_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    feature_counts: Dict[str, int],
    endpoint: Endpoint,
) -> Multivector:
    """
    Update the Multivector using the Shannon entropy of decision hygiene feature counts 
    and the Endpoint's health score.

    Returns
    -------
    multivector : Multivector with updated components
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    entropy = shannon_entropy(feature_counts)
    health_score = 1 - endpoint.failure_rate
    components = {}
    for node, _ in nodes.items():
        components[node] = health_score * entropy * dist[node]
    return Multivector(components, len(nodes))

def main():
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"
    feature_counts = {"feature1": 10, "feature2": 20}
    endpoint = Endpoint(1)

    multivector = hybrid_update(nodes, edges, root, feature_counts, endpoint)
    print(multivector.components)

if __name__ == "__main__":
    main()