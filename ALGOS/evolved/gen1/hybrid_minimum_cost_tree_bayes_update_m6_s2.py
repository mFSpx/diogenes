# DARWIN HAMMER — match 6, survivor 2
# gen: 1
# parent_a: minimum_cost_tree.py (gen0)
# parent_b: bayes_update.py (gen0)
# born: 2026-05-29T23:15:26Z

"""Hybrid module combining Minimum‑Cost Tree scoring (Algorithm A) and Bayesian evidence update (Algorithm B).

Mathematical bridge
-------------------
Algorithm A computes a deterministic cost  

    C = Σ_e ℓ(e) + λ Σ_v d(v)                       (1)

where ℓ(e) is Euclidean length of edge *e*, *d(v)* is the root‑to‑*v* distance and *λ* is a path‑weight.

Algorithm B supplies a probabilistic transformation  

    p_post = (p_prior·L) / (L·p_prior + FP·(1‑p_prior))   (2)

for a prior *p_prior*, a likelihood *L* and a false‑positive rate *FP*.

The hybrid replaces the deterministic edge contribution ℓ(e) in (1) by its **expected**
value under the posterior edge belief *p_e* obtained from (2).  
Similarly, node distances are weighted by a node belief *q_v* derived from incident
edge posteriors.  The resulting hybrid cost is

    C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)                (3)

Thus the core topologies of both parents are fused: the tree‑metric supplies the
geometric quantities (ℓ, d) while the Bayesian primitives supply the probabilistic
weights (p_e, q_v).

The module implements:
* `tree_metrics` – builds adjacency, edge lengths and root distances.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `hybrid_tree_cost` – evaluates (3) using the posteriors.
"""

from __future__ import annotations

import math
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
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Algorithm B – Bayesian primitives (vectorised)
# ----------------------------------------------------------------------
def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P(H|E) = p·L / P(E)
    """
    if np.any(marginal <= 0):
        raise ValueError("All marginal probabilities must be > 0")
    return prior * likelihood / marginal


def bayes_edge_posteriors(
    prior_dict: Dict[Edge, float],
    likelihood_dict: Dict[Edge, float],
    false_positive: float,
) -> Dict[Edge, float]:
    """
    Compute posterior probability for each edge using Bayesian update (Eq. 2).

    Parameters
    ----------
    prior_dict, likelihood_dict : dict mapping Edge → probability in [0,1]
    false_positive : scalar false‑positive rate (FP) in [0,1]

    Returns
    -------
    posterior_dict : dict mapping Edge → posterior probability
    """
    edges = list(prior_dict.keys())
    priors = np.array([prior_dict[e] for e in edges], dtype=float)
    likes = np.array([likelihood_dict[e] for e in edges], dtype=float)

    marginal = bayes_marginal(priors, likes, false_positive)
    post = bayes_update(priors, likes, marginal)

    return {e: float(p) for e, p in zip(edges, post)}


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float,
    edge_priors: Dict[Edge, float],
    edge_likelihoods: Dict[Edge, float],
    false_positive: float,
) -> float:
    """
    Compute the hybrid expected cost (Eq. 3).

    Steps
    -----
    1. Obtain deterministic geometry via `tree_metrics`.
    2. Compute posterior edge probabilities via `bayes_edge_posteriors`.
    3. Derive a node belief *q_v* as the complement of the probability that
       **none** of its incident edges are present:
           q_v = 1 - ∏_{e∈inc(v)} (1 - p_e)
    4. Assemble the expected material and path components.

    Returns
    -------
    Expected hybrid cost (float).
    """
    # 1. Geometry
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # 2. Bayesian edge posteriors
    posteriors = bayes_edge_posteriors(edge_priors, edge_likelihoods, false_positive)

    # 3. Node beliefs from incident edge posteriors
    node_belief: Dict[str, float] = {}
    for v in nodes:
        incident = [e for e in edges if v in e]
        prod_no_edge = 1.0
        for e in incident:
            p_e = posteriors[e]
            prod_no_edge *= (1.0 - p_e)
        node_belief[v] = 1.0 - prod_no_edge

    # 4. Expected material term Σ_e p_e·ℓ(e)
    material_exp = sum(posteriors[e] * edge_len[e] for e in edges)

    #    Expected path term λ Σ_v q_v·d(v)
    path_exp = path_weight * sum(node_belief[v] * dist[v] for v in nodes)

    return material_exp + path_exp


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def sample_random_tree(num_nodes: int, seed: int = 0) -> Tuple[Dict[str, Point], List[Edge], str]:
    """
    Generate a random connected tree with `num_nodes` vertices placed uniformly
    in the unit square. Returns (nodes, edges, root).
    """
    rng = np.random.default_rng(seed)
    points = {f"N{i}": (float(rng.random()), float(rng.random())) for i in range(num_nodes)}
    # Create a random spanning tree via a shuffled list of nodes
    node_names = list(points.keys())
    rng.shuffle(node_names)
    edges: List[Edge] = []
    for i in range(1, num_nodes):
        a = node_names[i]
        b = node_names[rng.integers(0, i)]
        edges.append((a, b))
    root = node_names[0]
    return points, edges, root


def uniform_edge_priors(edges: List[Edge], prior: float = 0.5) -> Dict[Edge, float]:
    """Assign the same prior probability to every edge."""
    return {e: prior for e in edges}


def uniform_edge_likelihoods(edges: List[Edge], likelihood: float = 0.8) -> Dict[Edge, float]:
    """Assign the same likelihood to every edge."""
    return {e: likelihood for e in edges}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny deterministic tree
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"

    # Priors / likelihoods (simple uniform case)
    priors = uniform_edge_priors(edges, prior=0.6)
    likes = uniform_edge_likelihoods(edges, likelihood=0.9)
    fp = 0.1
    path_w = 0.3

    cost = hybrid_tree_cost(
        nodes,
        edges,
        root,
        path_weight=path_w,
        edge_priors=priors,
        edge_likelihoods=likes,
        false_positive=fp,
    )
    print(f"Hybrid cost on toy tree: {cost:.6f}")

    # Run on a random larger tree to ensure no runtime errors
    rnd_nodes, rnd_edges, rnd_root = sample_random_tree(15, seed=42)
    rnd_priors = uniform_edge_priors(rnd_edges, prior=0.5)
    rnd_likes = uniform_edge_likelihoods(rnd_edges, likelihood=0.7)
    rnd_cost = hybrid_tree_cost(
        rnd_nodes,
        rnd_edges,
        rnd_root,
        path_weight=0.25,
        edge_priors=rnd_priors,
        edge_likelihoods=rnd_likes,
        false_positive=0.05,
    )
    print(f"Hybrid cost on random tree (15 nodes): {rnd_cost:.6f}")

    sys.exit(0)