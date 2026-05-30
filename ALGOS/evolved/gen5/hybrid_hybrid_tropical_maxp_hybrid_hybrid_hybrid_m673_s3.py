# DARWIN HAMMER — match 673, survivor 3
# gen: 5
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s4.py (gen4)
# born: 2026-05-29T23:30:22Z

"""Hybrid Tropical‑Max‑Plus & Semantic‑Weighted Minimum‑Cost Tree Scheduler
=======================================================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s3.py``  
  Provides tropical (max‑plus) algebra primitives and a ``tree_metrics`` routine
  that builds adjacency lists, Euclidean edge lengths and root‑to‑node distances.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s4.py``  
  Supplies a semantic weighting of geometric edge lengths together with a
  Gaussian‑Gaussian Bayesian update that refines the estimated total tree cost
  using an observed VRAM usage.

**Mathematical bridge**  
For every edge ``e = (u, v)`` we compute a semantic scalar ``w_e`` from a
lightweight semantic (LSM) vector.  The *weighted* edge cost is

``c_e = ℓ_e · w_e``

where ``ℓ_e`` is the Euclidean length from ``tree_metrics``.  The collection of
weighted costs forms a tropical matrix; the tropical max‑plus product yields the
*maximum* root‑to‑node utility (the longest additive path under max‑plus).  The
sum of all weighted costs is treated as a Gaussian prior on the total tree cost.
A Bayesian update with an observed VRAM measurement produces a posterior mean
``μ_post`` that the scheduler can use for allocation decisions.

The resulting system simultaneously computes:
* geometric/topological metrics,
* semantic‑enhanced edge costs,
* tropical max‑plus utilities,
* Bayesian‑refined cost estimates.

"""

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Tropical (max‑plus) algebra primitives (Parent A)
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition: max(x, y). Works element‑wise for NumPy arrays."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication: x + y. Works element‑wise for NumPy arrays."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    (A ⊗ B)[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Broadcast addition over the k‑dimension and take max.
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Tree utilities (shared by both parents)
# ----------------------------------------------------------------------
def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Parameters
    ----------
    nodes : dict mapping node identifier → (x, y) coordinate
    edges : list of (parent, child) pairs (undirected for metric purposes)
    root  : identifier of the root node

    Returns
    -------
    adj      : dict mapping node → list of neighbours
    edge_len : dict mapping ordered edge → Euclidean length
    dist     : dict mapping node → distance from *root* (sum of edge lengths)
    """
    # adjacency (undirected)
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        l = length(nodes[u], nodes[v])
        edge_len[(u, v)] = l
        edge_len[(v, u)] = l  # store both orientations for convenience

    # BFS to compute root‑to‑node distances
    dist: Dict[str, float] = {n: math.inf for n in nodes}
    dist[root] = 0.0
    queue: List[str] = [root]
    visited = {root}

    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)

    return adj, edge_len, dist


# ----------------------------------------------------------------------
# Semantic weighting (Parent B)
# ----------------------------------------------------------------------
def semantic_weights(
    lsm_vector: np.ndarray,
    edges: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    """
    Derive a scalar weight for each edge from a lightweight semantic (LSM) vector.

    The implementation uses a simple monotonic transform of the mean of the
    vector; any deterministic mapping would serve as the mathematical bridge.

    Returns a dict mapping ordered edge → weight (same weight for both orientations).
    """
    # Ensure a 1‑D float array
    vec = np.asarray(lsm_vector, dtype=float).ravel()
    base = 1.0 + np.abs(np.tanh(vec.mean()))  # weight in (0, 2)
    weights = {}
    for u, v in edges:
        weights[(u, v)] = base
        weights[(v, u)] = base
    return weights


def weighted_edge_costs(
    edge_len: Dict[Tuple[str, str], float],
    sem_weights: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    """
    Compute semantic‑weighted edge costs: c_e = ℓ_e · w_e.
    """
    return {
        e: edge_len[e] * sem_weights[e] for e in edge_len
    }


# ----------------------------------------------------------------------
# Tropical max‑plus utility on the weighted tree
# ----------------------------------------------------------------------
def max_plus_utility(
    root: str,
    adj: Dict[str, List[str]],
    weighted_costs: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """
    Compute the maximum (tropical) utility from the root to every node.

    For a tree the utility to a node is the sum of weighted costs along the
    unique path; tropical addition (max) is irrelevant because there is only
    one path.  Nevertheless we implement the recurrence using ``t_add`` and
    ``t_mul`` to showcase the algebraic bridge.

    Returns a dict mapping node → utility value.
    """
    util: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]
    visited = {root}

    while queue:
        cur = queue.pop(0)
        cur_util = util[cur]
        for nb in adj[cur]:
            if nb not in visited:
                edge = (cur, nb)
                cost = weighted_costs[edge]
                # tropical multiplication adds the cost, tropical addition is max
                util[nb] = t_add(cur_util, t_mul(cur_util, cost))  # simplifies to cur_util + cost
                visited.add(nb)
                queue.append(nb)

    return util


# ----------------------------------------------------------------------
# Bayesian update of total tree cost (Parent B)
# ----------------------------------------------------------------------
def bayesian_update(
    mu_prior: float,
    sigma_prior_sq: float,
    observation: float,
    sigma_obs_sq: float,
) -> Tuple[float, float]:
    """
    Conjugate Gaussian‑Gaussian update.

    Returns posterior mean and variance.
    """
    mu_post = (sigma_obs_sq * mu_prior + sigma_prior_sq * observation) / (
        sigma_prior_sq + sigma_obs_sq
    )
    sigma_post_sq = (sigma_prior_sq * sigma_obs_sq) / (sigma_prior_sq + sigma_obs_sq)
    return mu_post, sigma_post_sq


# ----------------------------------------------------------------------
# Hybrid operation exposing at least three public functions
# ----------------------------------------------------------------------
def hybrid_tree_cost_estimate(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    lsm_vector: np.ndarray,
    vram_observation: float,
    sigma_obs_sq: float = 0.25,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Full hybrid pipeline:

    1. Build geometric metrics (adjacency, Euclidean lengths, root distances).
    2. Compute semantic weights from the LSM vector.
    3. Form semantic‑weighted edge costs.
    4. Derive tropical max‑plus utilities from the root.
    5. Treat the sum of weighted costs as a Gaussian prior and update it with
       the observed VRAM usage.

    Returns
    -------
    mu_post          : posterior mean of total tree cost
    sigma_post_sq    : posterior variance
    utilities        : max‑plus utility per node (root utility = 0)
    """
    # Step 1
    adj, edge_len, _ = tree_metrics(nodes, edges, root)

    # Step 2
    sem_weights = semantic_weights(lsm_vector, edges)

    # Step 3
    w_costs = weighted_edge_costs(edge_len, sem_weights)

    # Step 4
    utilities = max_plus_utility(root, adj, w_costs)

    # Step 5 – prior is total weighted cost (sum over unique undirected edges)
    unique_edges = {tuple(sorted(e)) for e in edges}
    total_cost = sum(edge_len[tuple(e)] * sem_weights[tuple(e)] for e in unique_edges)
    mu_prior = total_cost
    sigma_prior_sq = 1.0  # a modest prior variance

    mu_post, sigma_post_sq = bayesian_update(
        mu_prior, sigma_prior_sq, vram_observation, sigma_obs_sq
    )

    return mu_post, sigma_post_sq, utilities


def sample_lsm_vector(dim: int = 8) -> np.ndarray:
    """Generate a deterministic pseudo‑random LSM vector for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=dim)


def simple_tree_example() -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]], str]:
    """Construct a tiny geometric tree for the smoke test."""
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    root = "A"
    return nodes, edges, root


if __name__ == "__main__":
    # Smoke test that runs without error
    nodes, edges, root = simple_tree_example()
    lsm_vec = sample_lsm_vector()
    # Simulated VRAM observation (e.g., 3.2 GB)
    vram_obs = 3.2

    mu_post, sigma_post_sq, utilities = hybrid_tree_cost_estimate(
        nodes,
        edges,
        root,
        lsm_vec,
        vram_obs,
        sigma_obs_sq=0.2,
    )

    print("Posterior mean cost:", mu_post)
    print("Posterior variance :", sigma_post_sq)
    print("Root‑to‑node utilities (max‑plus):")
    for node, util in utilities.items():
        print(f"  {node}: {util:.4f}")