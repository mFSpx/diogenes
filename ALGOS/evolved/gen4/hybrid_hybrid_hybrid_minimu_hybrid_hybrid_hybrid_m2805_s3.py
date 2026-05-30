# DARWIN HAMMER — match 2805, survivor 3
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

# ----------------------------------------------------------------------
# Types shared by both parents
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – tree utilities
# ----------------------------------------------------------------------
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
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]   # undirected convenience

    # BFS to compute distances from root
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
    """Return nodes that have degree 1 except the root (i.e. tree leaves)."""
    leaves = []
    for node, neighbours in adj.items():
        if node == root:
            continue
        if len(neighbours) == 1:
            leaves.append(node)
    return leaves


def expected_cost(dist: Dict[str, float], leaves: Iterable[str]) -> float:
    """Simple average of root‑to‑leaf distances (uniform leaf probability)."""
    leaf_dists = [dist[l] for l in leaves]
    if not leaf_dists:
        return 0.0
    return sum(leaf_dists) / len(leaf_dists)


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (base e) of a discrete count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log(p) for p in probs)


# ----------------------------------------------------------------------
# Algorithm B – endpoint & multivector utilities
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    @property
    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        return min(self.failures / self.failure_threshold, 1.0)


class Multivector:
    """Element of a Clifford algebra Cl(n,0) represented as a dict of blade→coefficient."""

    def __init__(self, components: Dict[str, float], n: int):
        self.components: Dict[str, float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-12
        }
        self.n = int(n)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"

    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector containing only blades of grade *k* (length of blade string)."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scale(self, factor: float) -> "Multivector":
        """Scalar multiplication."""
        return Multivector({b: c * factor for b, c in self.components.items()}, self.n)

    def copy(self) -> "Multivector":
        return Multivector(dict(self.components), self.n)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Very simplified geometric product:
    - For each matching blade we multiply the coefficients.
    - For non‑matching blades we keep them separate (outer product style).
    This is *not* a full Clifford product but suffices for the hybrid demo.
    """
    result: Dict[str, float] = {}

    # matching blades → scalar multiplication
    for blade, ca in a.components.items():
        if blade in b.components:
            result[blade] = result.get(blade, 0.0) + ca * b.components[blade]

    # outer‑like combination of distinct blades (concatenation, order ignored)
    for ba, ca in a.components.items():
        for bb, cb in b.components.items():
            if ba == bb:
                continue
            new_blade = "".join(sorted(set(ba + bb)))  # naive symmetrisation
            result[new_blade] = result.get(new_blade, 0.0) + ca * cb

    return Multivector(result, a.n)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_endpoint_weights(
    dist: Dict[str, float],
    leaves: List[str],
    endpoints: Dict[str, Endpoint],
    hygiene_counts: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute a scalar weight for each leaf endpoint.

    Weight = exp(-cost) * (1‑failure_rate) * exp(-entropy)

    - *cost*  : root‑to‑leaf Euclidean distance.
    - *failure_rate* : endpoint health metric.
    - *entropy* : Shannon entropy of the hygiene feature count vector.
    """
    entropy = shannon_entropy(hygiene_counts)
    weights: Dict[str, float] = {}
    for leaf in leaves:
        endpoint = endpoints.get(leaf)
        if endpoint is None or not endpoint.allow():
            # Closed circuit or missing endpoint → zero contribution
            weights[leaf] = 0.0
            continue
        cost_factor = math.exp(-dist[leaf])
        health_factor = 1.0 - endpoint.failure_rate
        weights[leaf] = cost_factor * health_factor * math.exp(-entropy)
    return weights


def allocate_multivector(
    weights: Dict[str, float],
    base_multivector: Multivector,
) -> Multivector:
    """
    Scale a multivector by the weights.

    Returns
    -------
    A new multivector with scaled components.
    """
    scaled_components = {}
    for blade, coef in base_multivector.components.items():
        scaled_coef = coef
        for leaf, weight in weights.items():
            scaled_coef *= weight
        scaled_components[blade] = scaled_coef
    return Multivector(scaled_components, base_multivector.n)


def hybrid_decision_tree_endpoint_allocator(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    endpoints: Dict[str, Endpoint],
    hygiene_counts: Dict[str, int],
    base_multivector: Multivector,
) -> Multivector:
    """
    Hybrid allocator.

    Returns
    -------
    A multivector representing the resource allocation.
    """
    adj, _, dist = tree_metrics(nodes, edges, root)
    leaves = leaf_nodes(adj, root)
    weights = compute_endpoint_weights(dist, leaves, endpoints, hygiene_counts)
    return allocate_multivector(weights, base_multivector)


# Example usage
if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"
    endpoints = {"B": Endpoint(1), "C": Endpoint(2), "D": Endpoint(3)}
    hygiene_counts = {"h1": 10, "h2": 20}
    base_multivector = Multivector({"1": 1.0, "12": 2.0}, 2)

    result = hybrid_decision_tree_endpoint_allocator(
        nodes, edges, root, endpoints, hygiene_counts, base_multivector
    )
    print(result)