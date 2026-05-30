# DARWIN HAMMER — match 601, survivor 1
# gen: 4
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s2.py (gen3)
# born: 2026-05-29T23:30:04Z

"""
Hybrid Endpoint‑Tree Bayesian‑Tropical Engine
============================================

Parents
-------
* **Algorithm A** – ``minimum_cost_tree.py`` + ``bayes_update.py``  
  Provides deterministic tree geometry (edge lengths, root‑to‑node distances) and a
  Bayesian posterior update for edge beliefs:
  
      p_post = (p_prior·L) / (L·p_prior + FP·(1‑p_prior))                (1)

* **Algorithm B** – ``hybrid_endpoint_circ_state_space_duality`` +  
  ``hybrid_hoeffding_tree_tropical_maxplus``  
  Supplies endpoint health scores, a tropical (max‑plus) linear mapping and a
  Hoeffding‑bound based split decision.

Mathematical Bridge
-------------------
An endpoint is interpreted as a *leaf edge* of the tree.  
The endpoint's **health_score** is taken as the Bayesian *prior* `p_prior` for the
incident edge, while its **recovery_priority** acts as the *likelihood* `L`.  
The posterior `p_e` (1) becomes a probabilistic weight for the geometric edge
length `ℓ(e)`.  

Node beliefs `q_v` are defined as the average posterior of incident edges,
providing a soft weighting of root‑to‑node distances `d(v)`.

A tropical ReLU network maps the vector of edge posteriors `p_e` to a scalar
*split gain* `g`.  The Hoeffding bound decides, with confidence `1‑δ`, whether the
observed gain justifies a tree split.  When a split occurs, a new endpoint is
added and the hybrid cost

    C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)                                 (2)

is recomputed.  Thus the deterministic tree metric, Bayesian belief propagation,
tropical max‑plus evaluation and statistical split test are fused into a single
iterative optimisation loop.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
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
    edge_len : dict mapping edge (ordered as given) → length
    node_dist : dict mapping node → distance from root
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])

    # BFS to compute root distances
    node_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    frontier = [root]
    while frontier:
        cur = frontier.pop()
        for nb in adj[cur]:
            if nb not in visited:
                # distance = parent distance + edge length (choose oriented edge)
                edge = (cur, nb) if (cur, nb) in edge_len else (nb, cur)
                node_dist[nb] = node_dist[cur] + edge_len[edge]
                visited.add(nb)
                frontier.append(nb)

    return adj, edge_len, node_dist


# ----------------------------------------------------------------------
# Algorithm B – endpoint health and tropical max‑plus utilities
# ----------------------------------------------------------------------
class Endpoint:
    """Simple data holder for an endpoint."""
    def __init__(self, node: str, health_score: float, recovery_priority: float):
        self.node = node
        self.health_score = health_score          # interpreted as p_prior ∈ (0,1)
        self.recovery_priority = recovery_priority  # interpreted as likelihood L > 0


def tropical_relu(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Max‑plus (tropical) linear layer followed by ReLU‑like max(·,0).

    For each output dimension i:
        y_i = max_j (weights[i, j] + x_j) + bias_i
    Then apply max(y_i, 0).

    Parameters
    ----------
    x : (d,) input vector
    weights : (k, d) weight matrix (max‑plus)
    bias : (k,) bias vector

    Returns
    -------
    y : (k,) output after tropical ReLU
    """
    # max‑plus multiplication
    max_plus = np.max(weights + x, axis=1) + bias
    return np.maximum(max_plus, 0.0)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a bounded random variable in [0, r].

    Returns the ε such that with probability ≥ 1‑δ the true mean lies within
    ε of the empirical mean after n observations.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return r * math.sqrt(math.log(2.0 / delta) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def bayes_edge_posteriors(
    edges: List[Edge],
    endpoints: List[Endpoint],
    fp_rate: float = 0.01,
) -> Dict[Edge, float]:
    """
    Vectorised Bayesian update for all edges using endpoint data.

    For each edge we locate the endpoint (leaf) that coincides with the edge's
    child node.  The endpoint's health_score supplies `p_prior`, its
    recovery_priority supplies `L`.  The posterior follows equation (1).

    Parameters
    ----------
    edges : list of (parent, child) tuples
    endpoints : list of Endpoint objects (one per leaf node)
    fp_rate : false‑positive rate `FP` used in the denominator

    Returns
    -------
    post : dict mapping Edge → posterior probability p_e
    """
    # Build a quick lookup from node name to endpoint
    ep_by_node = {ep.node: ep for ep in endpoints}
    post: Dict[Edge, float] = {}
    for u, v in edges:
        # Assume v is the child (leaf) for which we may have an endpoint
        ep = ep_by_node.get(v)
        if ep is None:
            # No endpoint data → use a neutral prior of 0.5 and L=1.0
            p_prior, L = 0.5, 1.0
        else:
            p_prior = max(min(ep.health_score, 1.0), 0.0)   # clamp to [0,1]
            L = max(ep.recovery_priority, 0.0) + 1e-9      # avoid zero
        numerator = p_prior * L
        denominator = L * p_prior + fp_rate * (1.0 - p_prior)
        post[(u, v)] = numerator / denominator if denominator != 0 else 0.0
    return post


def node_beliefs(
    adj: Dict[str, List[str]],
    edge_post: Dict[Edge, float],
) -> Dict[str, float]:
    """
    Compute node belief q_v as the average posterior of incident edges.

    For the root node we define q_root = 1.0 (full confidence).
    """
    q: Dict[str, float] = {}
    for node, neighbours in adj.items():
        if not neighbours:
            q[node] = 1.0
            continue
        post_vals = []
        for nb in neighbours:
            e = (node, nb) if (node, nb) in edge_post else (nb, node)
            post_vals.append(edge_post.get(e, 0.0))
        q[node] = sum(post_vals) / len(post_vals) if post_vals else 1.0
    return q


def hybrid_tree_cost(
    edge_len: Dict[Edge, float],
    node_dist: Dict[str, float],
    edge_post: Dict[Edge, float],
    node_bel: Dict[str, float],
    lam: float = 0.5,
) -> float:
    """
    Evaluate the hybrid cost (2).

    Parameters
    ----------
    edge_len : Edge → Euclidean length ℓ(e)
    node_dist : Node → root‑to‑node distance d(v)
    edge_post : Edge → posterior p_e
    node_bel : Node → belief q_v
    lam : path‑weight λ

    Returns
    -------
    C_h : hybrid cost
    """
    edge_term = sum(edge_post[e] * edge_len[e] for e in edge_len)
    node_term = lam * sum(node_bel[n] * node_dist[n] for n in node_dist)
    return edge_term + node_term


def generate_split_gain(
    edge_post: Dict[Edge, float],
    weights: np.ndarray,
    bias: np.ndarray,
) -> float:
    """
    Use a tropical ReLU network to map the vector of edge posteriors to a scalar
    split gain g.  The network has a single output neuron.

    Parameters
    ----------
    edge_post : Edge → posterior probabilities
    weights, bias : parameters of a max‑plus layer (k × m, k)

    Returns
    -------
    g : scalar gain (higher → more promising split)
    """
    x = np.array(list(edge_post.values()))
    # Ensure dimensions match; if not, truncate or pad with zeros
    k, m = weights.shape
    if x.size < m:
        x = np.pad(x, (0, m - x.size))
    elif x.size > m:
        x = x[:m]
    y = tropical_relu(x, weights, bias)
    # Single output assumed
    return float(y[0])


def decide_split(
    gain: float,
    r: float,
    delta: float,
    n_obs: int,
) -> bool:
    """
    Apply Hoeffding bound to decide whether a split should be performed.

    The split is taken if the observed gain exceeds ε, i.e. gain > ε.

    Parameters
    ----------
    gain : observed split gain g
    r : range of the gain variable (max - min). Must be > 0.
    delta : desired failure probability
    n_obs : number of observations (e.g. number of streaming samples)

    Returns
    -------
    split : bool indicating whether to split
    """
    eps = hoeffding_bound(r, delta, n_obs)
    return gain > eps


def hybrid_tree_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    endpoints: List[Endpoint],
    lam: float = 0.5,
    fp_rate: float = 0.01,
    tropical_weights: np.ndarray | None = None,
    tropical_bias: np.ndarray | None = None,
    hoeffding_r: float = 1.0,
    hoeffding_delta: float = 0.05,
    hoeffding_n: int = 100,
) -> Tuple[float, bool]:
    """
    Perform one hybrid iteration:
      1. Compute tree metrics.
      2. Update edge posteriors from endpoint health.
      3. Compute node beliefs.
      4. Evaluate hybrid cost.
      5. Generate a split gain via tropical network.
      6. Decide (Hoeffding) whether to split.

    Returns
    -------
    cost : hybrid cost after this iteration
    split_performed : bool indicating whether a split was triggered
    """
    # 1. Geometry
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)

    # 2. Bayesian edge beliefs
    edge_post = bayes_edge_posteriors(edges, endpoints, fp_rate)

    # 3. Node beliefs
    node_bel = node_beliefs(adj, edge_post)

    # 4. Hybrid cost
    cost = hybrid_tree_cost(edge_len, node_dist, edge_post, node_bel, lam)

    # 5. Tropical gain
    if tropical_weights is None:
        # Simple default: one hidden unit, weight=1 for each edge, bias=0
        tropical_weights = np.ones((1, len(edge_post)), dtype=float)
    if tropical_bias is None:
        tropical_bias = np.zeros((1,), dtype=float)
    gain = generate_split_gain(edge_post, tropical_weights, tropical_bias)

    # 6. Hoeffding split decision
    split = decide_split(gain, hoeffding_r, hoeffding_delta, hoeffding_n)

    return cost, split


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree
    nodes = {
        "A": (0.0, 0.0),   # root
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D")]
    root = "A"

    # Endpoints correspond to leaves B, C, D
    endpoints = [
        Endpoint(node="B", health_score=0.8, recovery_priority=0.9),
        Endpoint(node="C", health_score=0.3, recovery_priority=0.4),
        Endpoint(node="D", health_score=0.6, recovery_priority=0.7),
    ]

    # Random tropical parameters (k=2 hidden units, m = number of edges)
    rng = np.random.default_rng(42)
    tropical_weights = rng.normal(size=(1, len(edges)))
    tropical_bias = rng.normal(size=(1,))

    cost, split = hybrid_tree_update(
        nodes=nodes,
        edges=edges,
        root=root,
        endpoints=endpoints,
        lam=0.4,
        fp_rate=0.02,
        tropical_weights=tropical_weights,
        tropical_bias=tropical_bias,
        hoeffding_r=1.0,
        hoeffding_delta=0.05,
        hoeffding_n=200,
    )

    print(f"Hybrid cost: {cost:.4f}")
    print(f"Split decision: {'YES' if split else 'NO'}")