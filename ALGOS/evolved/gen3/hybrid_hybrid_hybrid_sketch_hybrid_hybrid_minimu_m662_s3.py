# DARWIN HAMMER — match 662, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s1.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s1.py (gen2)
# born: 2026-05-29T23:30:19Z

"""Hybrid Sketch-Tree Epistemic Algorithm
=======================================

This module fuses **Parent Algorithm A** (count‑min sketch, MinHash LSH and the
sheaf‑cohomology inspired log‑log regression) with **Parent Algorithm B**
(minimum‑cost spanning tree weighted by Bayesian epistemic certainty).

Mathematical bridge
------------------
* The sketch utilities from *A* compress each node’s high‑dimensional item
  multiset into a low‑dimensional count‑min matrix and a MinHash signature.
* The similarity of two nodes is obtained from the Jaccard estimate derived
  from their MinHash signatures – a classic dimensionality‑reduction metric.
* In *B* the edge weight is a Euclidean distance multiplied by a factor that
  reflects epistemic certainty (via a Bayesian update).
* The hybrid algorithm therefore defines a **hybrid edge cost**


d   = Euclidean distance between node positions
s   = Jaccard similarity from MinHash signatures
p   = confidence (0..1) extracted from an epistemic flag
L   = likelihood = s
M   = bayes_marginal(p, L, fp)            # fp = false‑positive rate
p'  = bayes_update(p, L, M)               # posterior certainty
cost = d * (1 - s) * (1 - p')


Thus the topological information (similarity) and epistemic information
both modulate the geometric cost, yielding a tree that simultaneously
preserves spatial proximity, feature similarity and belief strength.

The module provides three public hybrid operations:
1. `hybrid_node_representation` – builds a sketch + signature for a node.
2. `hybrid_edge_cost` – computes the cost of an edge using the formula above.
3. `build_hybrid_minimum_spanning_tree` – constructs a minimum‑cost spanning
   tree over the graph using the hybrid costs.

All code is pure Python 3 and relies only on the standard library and NumPy.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (count‑min sketch, MinHash)
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4):
    """Return a depth×width count‑min sketch matrix for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            table[d][h % width] += 1
    return np.array(table, dtype=np.int32)


def minhash_signature(items, num_perm: int = 64):
    """Compute a MinHash signature (list of integers) for *items*."""
    signature = []
    for i in range(num_perm):
        min_hash = None
        for item in items:
            h = int(hashlib.sha1(f"{i}:{item}".encode()).hexdigest(), 16)
            if min_hash is None or h < min_hash:
                min_hash = h
        # Use a large sentinel when the set is empty
        signature.append(min_hash if min_hash is not None else 2 ** 64 - 1)
    return np.array(signature, dtype=np.uint64)


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig_a == sig_b))


# ----------------------------------------------------------------------
# Parent B utilities (geometry, Bayesian epistemic certainty)
# ----------------------------------------------------------------------
Point = tuple[float, float]
Edge = tuple[str, str]

EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = L·P + fp·(1‑P)"""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)"""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def certainty_flag(label: str, *, confidence_bps: int, authority_class: str,
                  rationale: str, evidence_refs: tuple[str, ...] = ()):
    """Create an epistemic certainty dictionary."""
    if label not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {label!r}")
    if not (0 <= confidence_bps <= 10000):
        raise ValueError("confidence_bps must be in 0..10000")
    return {
        "label": label,
        "confidence_bps": int(confidence_bps),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_node_representation(node_id: str, items: list[str]) -> dict:
    """
    Build the hybrid representation for a graph node.

    Returns a dict with:
        - 'sketch'   : count‑min matrix (numpy array)
        - 'signature': MinHash signature (numpy array)
    """
    sketch = count_min_sketch(items)
    signature = minhash_signature(items)
    return {
        "id": node_id,
        "sketch": sketch,
        "signature": signature,
    }


def hybrid_edge_cost(
    a_point: Point,
    b_point: Point,
    rep_a: dict,
    rep_b: dict,
    flag: dict,
    false_positive_rate: float = 0.01,
) -> float:
    """
    Compute the hybrid cost for an edge (a,b).

    Parameters
    ----------
    a_point, b_point : Point
        Euclidean coordinates of the two endpoints.
    rep_a, rep_b : dict
        Hybrid node representations from ``hybrid_node_representation``.
    flag : dict
        Epistemic certainty dictionary for this edge.
    false_positive_rate : float, optional
        Fixed false‑positive probability used in the Bayesian marginal.

    Returns
    -------
    float
        The hybrid cost, lower values indicate a more desirable edge.
    """
    # 1. Geometric component
    d = length(a_point, b_point)

    # 2. Feature‑similarity component (Jaccard estimate from MinHash)
    s = jaccard_estimate(rep_a["signature"], rep_b["signature"])

    # 3. Epistemic component
    prior = flag["confidence_bps"] / 10000.0          # map bps → [0,1]
    likelihood = s                                   # treat similarity as likelihood
    marginal = bayes_marginal(prior, likelihood, false_positive_rate)
    posterior = bayes_update(prior, likelihood, marginal)

    # 4. Hybrid combination (geometric * dissimilarity * epistemic penalty)
    cost = d * (1.0 - s) * (1.0 - posterior)
    return float(cost)


def build_hybrid_minimum_spanning_tree(
    nodes: dict[str, Point],
    node_items: dict[str, list[str]],
    edges: list[Edge],
    edge_flags: dict[Edge, dict],
) -> list[Edge]:
    """
    Construct a minimum‑cost spanning tree using the hybrid edge costs.

    Parameters
    ----------
    nodes : dict
        Mapping node identifier → Point.
    node_items : dict
        Mapping node identifier → list of items (raw data for sketching).
    edges : list[Edge]
        Candidate edges (undirected, each as a tuple of node ids).
    edge_flags : dict
        Mapping Edge → epistemic certainty dictionary.

    Returns
    -------
    list[Edge]
        Edges of the resulting spanning tree (size = |V|‑1).
    """
    # Pre‑compute hybrid representations for all nodes
    reps = {nid: hybrid_node_representation(nid, node_items[nid]) for nid in nodes}

    # Compute hybrid costs for each edge
    edge_costs = {}
    for u, v in edges:
        flag = edge_flags.get((u, v)) or edge_flags.get((v, u))
        if flag is None:
            raise KeyError(f"Missing epistemic flag for edge {(u, v)}")
        cost = hybrid_edge_cost(nodes[u], nodes[v], reps[u], reps[v], flag)
        edge_costs[(u, v)] = cost

    # Prim's algorithm (simple O(E log V) implementation using a heap)
    import heapq

    start = next(iter(nodes))  # arbitrary start node
    visited = {start}
    heap = []
    for (u, v), c in edge_costs.items():
        if u == start or v == start:
            heapq.heappush(heap, (c, u, v))

    tree_edges = []
    while heap and len(visited) < len(nodes):
        cost, u, v = heapq.heappop(heap)
        if u in visited and v in visited:
            continue
        new_node = v if u in visited else u
        visited.add(new_node)
        tree_edges.append((u, v))
        # push frontier edges emanating from the newly added node
        for (a, b), c in edge_costs.items():
            if (a == new_node and b not in visited) or (b == new_node and a not in visited):
                heapq.heappush(heap, (c, a, b))

    if len(tree_edges) != len(nodes) - 1:
        raise RuntimeError("Failed to build a spanning tree; graph may be disconnected.")
    return tree_edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny synthetic graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }

    # Each node holds a multiset of string tokens
    node_items = {
        "A": ["apple", "banana", "cherry"],
        "B": ["banana", "date", "elderberry"],
        "C": ["apple", "fig", "grape"],
        "D": ["cherry", "date", "fig"],
    }

    # Fully connected candidate edges
    edges = [(u, v) for u in nodes for v in nodes if u < v]

    # Assign a random epistemic flag to each edge
    random.seed(42)
    edge_flags = {}
    for e in edges:
        label = random.choice(EPISTEMIC_FLAGS)
        flag = certainty_flag(
            label,
            confidence_bps=random.randint(2000, 8000),
            authority_class="synthetic",
            rationale="smoke test",
        )
        edge_flags[e] = flag

    # Build the hybrid spanning tree
    tree = build_hybrid_minimum_spanning_tree(nodes, node_items, edges, edge_flags)

    print("Hybrid Minimum Spanning Tree edges:")
    for u, v in tree:
        print(f"  {u} – {v}")

    # Verify that the tree has exactly |V|-1 edges
    assert len(tree) == len(nodes) - 1, "Tree size mismatch"
    print("Smoke test completed successfully.")