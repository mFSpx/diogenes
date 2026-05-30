# DARWIN HAMMER — match 5012, survivor 4
# gen: 7
# parent_a: hybrid_bayes_claim_kernel_hybrid_hybrid_hybrid_m1718_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s1.py (gen4)
# born: 2026-05-30T00:00:39Z

"""Hybrid Bayesian‑Tree Cost Algorithm
===================================

Parent A: ``bayes_claim_kernel.py`` – provides Bayesian updating, entropy
and simple feature transforms.
Parent B: ``hybrid_hard_truth_math_hybrid_minimum_cost__m12_s0.py`` – provides
tree metric extraction and a minimum‑cost tree formulation.

Mathematical Bridge
-------------------
Both parents manipulate probability distributions over discrete objects.
Parent A updates a prior distribution using Bayes’ rule and modulates it with
Shannon entropy (or KL‑divergence).  
Parent B supplies geometric edge lengths on a tree and later weights those
edges with probabilistic factors.

The fusion consists of:

1. **Tree‑derived edge lengths** ``ℓ(e)`` from ``tree_metrics``.
2. **Bayesian posterior** ``P(v)`` for every node/claim obtained by
   ``bayesian_update(prior, likelihood)``.
3. **Entropy modulation** ``H(P)`` that scales the posterior before it is
   injected into the edge weights.
4. **Hybrid edge weight** ``w(e) = ℓ(e) * (H(P)·P(u)·P(v))`` for edge ``e = (u,v)``.
5. **Minimum‑cost spanning tree** computed on the hybrid weights, yielding a
   cost that respects both geometric resource requirements and probabilistic
   confidence.

The code below implements this pipeline with three core functions:
``tree_metrics``, ``bayesian_update`` and ``hybrid_minimum_spanning_cost``.
Additional utilities from Parent A (lead‑lag transform, Kan basis, pruning)
are retained to demonstrate the full hybrid capability."""


import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Set

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathClaim:
    id: str


@dataclass(frozen=True)
class MathEvidence:
    id: str


@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float
    posterior: float
    evidence_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Concatenate linear (sum) and quadratic (sum of squares) features for each row.
    """
    linear_features = np.sum(X, axis=1, keepdims=True)
    quadratic_features = np.sum(X ** 2, axis=1, keepdims=True)
    return np.concatenate((linear_features, quadratic_features), axis=1)


def kan_basis(grid_size: int) -> np.ndarray:
    """
    Simple exponential basis used in some kernel constructions.
    """
    points = np.linspace(0, 1, grid_size)
    return np.exp(-points[:, None] * np.arange(grid_size)[None, :])


def prune_candidates(signatures: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """
    Element‑wise masking of candidate signatures by a schedule mask.
    """
    return signatures * schedule


def shannon_entropy(prob_dist: np.ndarray) -> float:
    """
    Compute Shannon entropy H(p) = -∑ p_i log p_i.
    """
    p = prob_dist / prob_dist.sum()
    # avoid log(0) by masking zero entries
    mask = p > 0
    return -float(np.sum(p[mask] * np.log(p[mask])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL(p‖q) = ∑ p_i log(p_i / q_i).
    Both vectors must be positive and sum to 1.
    """
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    mask = (p_norm > 0) & (q_norm > 0)
    return float(np.sum(p_norm[mask] * np.log(p_norm[mask] / q_norm[mask])))


# ----------------------------------------------------------------------
# Parent B utilities (tree metrics and MST)
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency list, compute Euclidean edge lengths, and compute
    root‑to‑node distances (sum of edge lengths along the unique path).

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping (u, v) (as supplied) → Euclidean length
    dist : dict mapping node → distance from *root*
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]  # undirected convenience

    # BFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    visited: Set[str] = {root}
    queue: List[str] = [root]

    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                queue.append(nb)

    return adj, edge_len, dist


# Union‑Find structure for Kruskal's algorithm
class UF:
    def __init__(self, elements: List[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def kruskal_mst(
    nodes: List[str],
    weighted_edges: List[Tuple[str, str, float]],
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Simple Kruskal implementation returning total cost and the edges
    that belong to the minimum spanning tree.
    """
    uf = UF(nodes)
    # sort edges by weight
    sorted_edges = sorted(weighted_edges, key=lambda e: e[2])
    total_cost = 0.0
    mst_edges: List[Tuple[str, str, float]] = []
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst_edges.append((u, v, w))
            total_cost += w
            if len(mst_edges) == len(nodes) - 1:
                break
    return total_cost, mst_edges


# ----------------------------------------------------------------------
# Core Hybrid Functions
# ----------------------------------------------------------------------
def bayesian_update(
    prior: Dict[str, float],
    likelihood: Dict[str, float],
) -> Dict[str, float]:
    """
    Perform a Bayesian update for a discrete set of hypotheses.

    posterior(v) = prior(v) * likelihood(v) / Z,
    where Z = Σ_u prior(u) * likelihood(u).

    Parameters
    ----------
    prior : dict mapping hypothesis id → prior probability (must sum to 1)
    likelihood : dict mapping hypothesis id → likelihood value (non‑negative)

    Returns
    -------
    posterior : dict with normalized posterior probabilities.
    """
    unnorm = {h: prior.get(h, 0.0) * likelihood.get(h, 0.0) for h in prior}
    Z = sum(unnorm.values())
    if Z == 0.0:
        # Avoid division by zero – return uniform distribution
        n = len(prior)
        return {h: 1.0 / n for h in prior}
    return {h: v / Z for h, v in unnorm.items()}


def hybrid_edge_weights(
    edge_len: Dict[Tuple[str, str], float],
    posterior: Dict[str, float],
    entropy_scale: float,
) -> List[Tuple[str, str, float]]:
    """
    Compute hybrid edge weights:
        w(u,v) = ℓ(u,v) * (H * P(u) * P(v))

    The entropy factor H = entropy(posterior) is raised to the power
    ``entropy_scale`` to give a tunable influence.

    Returns a list of (u, v, weight) tuples ready for MST computation.
    """
    # Build probability vector for entropy
    prob_vec = np.array(list(posterior.values()))
    H = shannon_entropy(prob_vec)
    scaling = H ** entropy_scale if entropy_scale != 0 else 1.0

    weighted_edges: List[Tuple[str, str, float]] = []
    for (u, v), l in edge_len.items():
        # each undirected edge appears twice in the dict; keep only one orientation
        if (v, u) in weighted_edges:
            continue
        w = l * scaling * posterior.get(u, 0.0) * posterior.get(v, 0.0)
        weighted_edges.append((u, v, w))
    return weighted_edges


def hybrid_minimum_spanning_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: Dict[str, float],
    evidence_likelihood: Dict[str, float],
    entropy_scale: float = 1.0,
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Full hybrid pipeline:
        1. Compute tree metrics (edge lengths).
        2. Bayesian update of node priors using supplied evidence likelihoods.
        3. Modulate posteriors with Shannon entropy.
        4. Build hybrid edge weights.
        5. Return the cost and edge list of the minimum spanning tree.

    Parameters
    ----------
    nodes : dict id → (x, y) coordinates.
    edges : list of (id1, id2) undirected edges.
    root : identifier of the root node (used only for distance metrics, not the MST).
    prior : prior probabilities for each node id (must sum to 1).
    evidence_likelihood : likelihood values for each node id.
    entropy_scale : exponent applied to the entropy term (default 1.0).

    Returns
    -------
    total_cost : float total weight of the hybrid MST.
    mst_edges  : list of (u, v, weight) belonging to the MST.
    """
    # 1. Tree metrics
    _, edge_len, _ = tree_metrics(nodes, edges, root)

    # 2. Bayesian update
    posterior = bayesian_update(prior, evidence_likelihood)

    # 3 & 4. Hybrid edge weights
    weighted_edges = hybrid_edge_weights(edge_len, posterior, entropy_scale)

    # 5. Minimum spanning tree on hybrid weights
    total_cost, mst_edges = kruskal_mst(list(nodes.keys()), weighted_edges)
    return total_cost, mst_edges


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree
    nodes_example = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 0.866),  # equilateral triangle
        "D": (1.5, 0.866),
    }

    edges_example = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]

    root_node = "A"

    # Prior probabilities (must sum to 1)
    prior_example = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    # Simulated evidence: higher likelihood for nodes closer to the root
    # (just for demonstration)
    evidence_likelihood_example = {
        "A": 0.9,
        "B": 0.6,
        "C": 0.4,
        "D": 0.2,
    }

    total, mst = hybrid_minimum_spanning_cost(
        nodes=nodes_example,
        edges=edges_example,
        root=root_node,
        prior=prior_example,
        evidence_likelihood=evidence_likelihood_example,
        entropy_scale=1.0,
    )

    print(f"Hybrid MST total cost: {total:.4f}")
    print("Edges in MST with hybrid weights:")
    for u, v, w in mst:
        print(f"  {u} – {v}: weight = {w:.4f}")

    # Demonstrate lead‑lag transform on a random matrix
    X = np.random.rand(5, 3)
    transformed = lead_lag_transform(X)
    print("\nLead‑lag transformed shape:", transformed.shape)

    # Demonstrate pruning with a random schedule
    signatures = np.random.rand(5, 2)
    schedule = (np.random.rand(5, 2) > 0.5).astype(float)
    pruned = prune_candidates(signatures, schedule)
    print("\nPruned signatures (first row):", pruned[0])