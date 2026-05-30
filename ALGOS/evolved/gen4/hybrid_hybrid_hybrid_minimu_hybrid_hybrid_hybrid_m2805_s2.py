# DARWIN HAMMER — match 2805, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py (gen3)
# born: 2026-05-29T23:46:10Z

"""Hybrid Decision‑Tree Endpoint Allocator

Parents:
- hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py
- hybrid_hybrid_hybrid_endpoi_hybrid_geometric_pro_m54_s0.py

Mathematical bridge
-------------------
Algorithm A provides *expected traversal cost* for each leaf of a decision
tree (root‑to‑node Euclidean distance).  Algorithm B supplies a *multivector*
representation of a resource allocation together with an *endpoint health
score* (failure rate).  The hybrid system uses the expected cost as a
multiplicative weight for the multivector components and also incorporates
the Shannon entropy of a decision‑hygiene feature count vector.  The final
allocation for an endpoint *e* is

    w_e = exp(‑cost(e)) · (1‑failure_rate(e)) · H

where *H* = exp(‑S) with *S* the Shannon entropy of the hygiene counts.
The weighted scalar *w_e* scales every blade of the base multivector,
producing a cost‑aware, health‑aware allocation.

The module implements the combined pipeline and provides three core
functions demonstrating the hybrid operation.
"""

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
    entropy_factor = math.exp(-entropy)

    weights: Dict[str, float] = {}
    for leaf in leaves:
        endpoint = endpoints.get(leaf)
        if endpoint is None or not endpoint.allow():
            # Closed circuit or missing endpoint → zero contribution
            weights[leaf] = 0.0
            continue
        cost_factor = math.exp(-dist[leaf])
        health_factor = 1.0 - endpoint.failure_rate
        weights[leaf] = cost_factor * health_factor * entropy_factor
    return weights


def allocate_multivector_to_endpoints(
    base_mv: Multivector,
    endpoint_weights: Dict[str, float],
) -> Dict[str, Multivector]:
    """
    Produce a weighted multivector for each leaf endpoint by scaling the base
    multivector with the previously computed scalar weight and then applying
    a self‑geometric product to embed the weight into the algebraic structure.
    """
    allocation: Dict[str, Multivector] = {}
    for leaf, w in endpoint_weights.items():
        if w == 0.0:
            allocation[leaf] = Multivector({}, base_mv.n)  # zero multivector
            continue
        # Scale the base multivector
        scaled = base_mv.scale(w)
        # Embed the weight via a geometric product with a scalar‑blade multivector
        scalar_mv = Multivector({"": w}, base_mv.n)  # empty string denotes scalar blade
        allocation[leaf] = geometric_product(scaled, scalar_mv)
    return allocation


def hybrid_allocation_pipeline(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    endpoints: Dict[str, Endpoint],
    base_mv: Multivector,
    hygiene_counts: Dict[str, int],
) -> Tuple[Dict[str, Multivector], float]:
    """
    End‑to‑end hybrid process:

    1. Build tree metrics.
    2. Identify leaf nodes (candidate endpoints).
    3. Compute expected cost (for diagnostic purposes).
    4. Derive per‑endpoint scalar weights using cost, health and hygiene entropy.
    5. Allocate a weighted multivector to each leaf.

    Returns a mapping leaf→allocated multivector and the global expected cost.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    leaves = leaf_nodes(adj, root)
    exp_cost = expected_cost(dist, leaves)
    weights = compute_endpoint_weights(dist, leaves, endpoints, hygiene_counts)
    allocation = allocate_multivector_to_endpoints(base_mv, weights)
    return allocation, exp_cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: a root with three children, each child is an endpoint.
    nodes = {
        "R": (0.0, 0.0),
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (-1.0, 0.0),
    }
    edges = [("R", "A"), ("R", "B"), ("R", "C")]

    # Endpoints attached to leaf identifiers
    endpoints = {
        "A": Endpoint(id=1, failures=0),
        "B": Endpoint(id=2, failures=2),  # partially unhealthy
        "C": Endpoint(id=3, failures=3),  # will be opened (failure_threshold=3)
    }

    # Base multivector (grade‑1 blades for a 3‑dim space)
    base_mv = Multivector(
        {
            "e1": 1.0,
            "e2": 2.0,
            "e3": 3.0,
        },
        n=3,
    )

    # Decision‑hygiene feature counts (arbitrary example)
    hygiene_counts = {
        "clean": 8,
        "dirty": 2,
        "unknown": 0,
    }

    allocation, exp_cost = hybrid_allocation_pipeline(
        nodes,
        edges,
        root="R",
        endpoints=endpoints,
        base_mv=base_mv,
        hygiene_counts=hygiene_counts,
    )

    print(f"Expected tree cost: {exp_cost:.4f}")
    for leaf, mv in allocation.items():
        print(f"Leaf {leaf} → Allocated {mv}")
    # Verify that closed endpoint (C) receives a zero multivector
    assert allocation["C"].components == {}, "Closed endpoint should have zero allocation"
    print("Smoke test completed successfully.")