# DARWIN HAMMER — match 249, survivor 5
# gen: 3
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_sketches_rlct_m2_s1.py (gen2)
# born: 2026-05-29T23:27:56Z

"""Hybrid Tree‑Bandit‑Sketch Algorithm
===================================

This module fuses the two parent algorithms:

* **minimum_cost_tree + bayes_update** – provides deterministic tree
  geometry (edge lengths, root‑to‑node distances) and a Bayesian update
  (Eq. 2) that turns a prior edge belief into a posterior probability
  *pₑ*.

* **bandit‑router + sketch‑RLCT** – supplies an online bandit feedback
  loop (Count‑Min sketch of empirical log‑likelihoods) and a
  HyperLogLog‑style sketch that estimates the effective number of
  distinct activation patterns.  The distinct‑count estimate is used as
  the *λ* (RLCT) in the tree cost.

**Mathematical bridge**

1. The likelihood *Lₑ* for an edge *e* is obtained from the
   Count‑Min sketch that aggregates log‑likelihood contributions of
   observed items (tokens).  The sketch yields an unbiased estimate of
   the empirical log‑likelihood sum, which we exponentiate to obtain a
   likelihood value.

2. The prior *p_priorₑ* for an edge is derived from the bandit
   statistics (average reward of the corresponding action).  The
   Bayesian update (Eq. 2) produces the posterior edge weight *pₑ*.

3. The RLCT weight *λ* is set to the distinct‑count estimate supplied by
   a LogLog sketch (a lightweight surrogate for HyperLogLog).  This
   couples the probabilistic belief on edges with the geometric tree
   metric, yielding the hybrid cost

       Cₕ = Σₑ pₑ·ℓ(e) + λ Σᵥ qᵥ·d(v)                     (Eq. 3)

   where *qᵥ* is the normalized product of incident edge posteriors.

The functions below implement the full pipeline:
`tree_metrics`, `count_min_sketch`, `bandit_update_policy`,
`estimate_distinct_loglog`, `bayes_edge_posteriors`,
`hybrid_tree_cost`, and a convenience wrapper
`compute_hybrid_cost`.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    edge_len : dict mapping ordered edge → length
    node_dist : dict mapping node → distance from root
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Edge, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        # store ordered tuple for deterministic lookup
        ordered = (a, b) if a < b else (b, a)
        edge_len[ordered] = length(nodes[a], nodes[b])

    # BFS to compute root distances
    node_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    frontier = [root]

    while frontier:
        cur = frontier.pop()
        for nb in adj[cur]:
            if nb not in visited:
                visited.add(nb)
                # distance accumulates edge length
                ordered = (cur, nb) if cur < nb else (nb, cur)
                node_dist[nb] = node_dist[cur] + edge_len[ordered]
                frontier.append(nb)

    return dict(adj), edge_len, node_dist


# ----------------------------------------------------------------------
# Algorithm B – Bandit router (Count‑Min sketch) utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit action."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    """A single observed reward for a context‑action pair."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# Global mutable policy table (action → [total_reward, count])
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all accumulated bandit statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Mean reward for *action* (0 if never seen)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times *action* has been observed."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incremental online update of the global bandit policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def count_min_sketch(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """
    Classic Count‑Min sketch.

    Parameters
    ----------
    items : iterable of hashable identifiers
    width : number of counters per hash function
    depth : number of independent hash functions

    Returns
    -------
    table : depth × width integer matrix
    """
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    rng = random.Random(0xC0FFEE)  # deterministic seed for reproducibility

    for item in items:
        for d in range(depth):
            # simple deterministic hash per depth
            h = hash((item, d, rng.seed()))
            idx = h % width
            table[d][idx] += 1
    return table


def estimate_count_min(table: List[List[int]], item: str) -> int:
    """
    Query the Count‑Min sketch for an upper‑bound on the frequency of *item*.
    """
    width = len(table[0])
    rng = random.Random(0xC0FFEE)
    estimates = []
    for d, row in enumerate(table):
        h = hash((item, d, rng.seed()))
        idx = h % width
        estimates.append(row[idx])
    return min(estimates)


# ----------------------------------------------------------------------
# Sketch‑RLCT – distinct‑count (LogLog) estimator
# ----------------------------------------------------------------------
def _rho(w: int) -> int:
    """Position of leftmost 1-bit (1‑based)."""
    return (w & -w).bit_length()


def estimate_distinct_loglog(items: Iterable[str], m: int = 64) -> float:
    """
    Very lightweight LogLog distinct‑count estimator (a surrogate for HyperLogLog).

    Parameters
    ----------
    items : iterable of hashable identifiers
    m : number of registers (must be power of two)

    Returns
    -------
    estimate : float, estimated cardinality
    """
    if m & (m - 1):
        raise ValueError("m must be a power of two")
    registers = np.zeros(m, dtype=int)

    for item in items:
        h = hash(item) & 0xFFFFFFFFFFFFFFFF  # 64‑bit hash
        idx = h & (m - 1)                     # low log2(m) bits as register index
        w = h >> int(math.log2(m))            # remaining bits
        rank = _rho(w)                       # position of first 1
        registers[idx] = max(registers[idx], rank)

    # harmonic mean of 2^{-register}
    Z = 1.0 / np.sum(2.0 ** -registers)
    alpha_m = 0.7213 / (1 + 1.079 / m)  # standard constant for LogLog
    estimate = alpha_m * m * m * Z
    return estimate


# ----------------------------------------------------------------------
# Bayesian edge posterior computation (bridge)
# ----------------------------------------------------------------------
def bayes_edge_posteriors(
    edges: List[Edge],
    prior_fp: float,
    cm_sketch: List[List[int]],
    distinct_estimate: float,
) -> Dict[Edge, float]:
    """
    Compute posterior probability p_e for each edge using the Bayesian rule

        p_post = (p_prior·L) / (L·p_prior + FP·(1‑p_prior))

    where
        p_prior  = sigmoid(mean_reward)   (derived from bandit stats)
        L        = exp( log‑likelihood estimate from Count‑Min )
        FP       = prior false‑positive rate (global constant)

    The distinct‑count estimate modulates the likelihood scale to reflect
    the RLCT λ term.
    """
    posteriors: Dict[Edge, float] = {}

    for a, b in edges:
        edge_id = f"{a}<->{b}"
        # Prior from bandit reward (scaled to [0,1] via sigmoid)
        r = _reward(edge_id)
        p_prior = 1.0 / (1.0 + math.exp(-r))

        # Likelihood from Count‑Min sketch (upper‑bound frequency)
        freq_est = estimate_count_min(cm_sketch, edge_id) + 1  # add‑1 smoothing
        # Scale by distinct estimate to keep values in a reasonable range
        L = math.exp(min(math.log(freq_est) / max(distinct_estimate, 1.0), 5.0))

        FP = prior_fp
        numerator = p_prior * L
        denominator = numerator + FP * (1.0 - p_prior)
        p_post = numerator / denominator if denominator != 0 else 0.0
        ordered = (a, b) if a < b else (b, a)
        posteriors[ordered] = p_post

    return posteriors


# ----------------------------------------------------------------------
# Hybrid cost evaluation
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    edge_post: Dict[Edge, float],
    lambda_rlct: float,
) -> float:
    """
    Compute the hybrid cost

        C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)

    where q_v is the normalized product of incident edge posteriors.
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)

    # Edge contribution
    edge_term = sum(edge_post.get(e, 0.0) * edge_len[e] for e in edge_len)

    # Node belief q_v: product of incident posteriors, normalized
    node_belief: Dict[str, float] = {}
    for v, neighbours in adj.items():
        prod = 1.0
        for nb in neighbours:
            ordered = (v, nb) if v < nb else (nb, v)
            prod *= edge_post.get(ordered, 1e-6)  # avoid zero product
        node_belief[v] = prod

    total = sum(node_belief.values())
    if total == 0:
        total = 1.0
    for v in node_belief:
        node_belief[v] /= total

    node_term = sum(node_belief[v] * node_dist[v] for v in node_dist)

    return edge_term + lambda_rlct * node_term


# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def compute_hybrid_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    bandit_updates: List[BanditUpdate],
    sketch_items: Iterable[str],
    prior_fp: float = 0.05,
) -> float:
    """
    Full hybrid workflow:

    1. Update the bandit policy with observed rewards.
    2. Build a Count‑Min sketch from *sketch_items* (log‑likelihood tokens).
    3. Estimate distinct token count via LogLog (RLCT λ).
    4. Compute Bayesian edge posteriors using (2) and (3).
    5. Evaluate the hybrid tree cost (Eq. 3).

    Returns the scalar cost.
    """
    # 1. Bandit policy update
    update_policy(bandit_updates)

    # 2. Count‑Min sketch of log‑likelihood tokens
    cm = count_min_sketch(sketch_items)

    # 3. Distinct‑count estimate → λ
    distinct_est = estimate_distinct_loglog(sketch_items)
    lambda_rlct = math.log1p(distinct_est)  # smooth mapping to positive λ

    # 4. Posterior edge probabilities
    edge_post = bayes_edge_posteriors(edges, prior_fp, cm, distinct_est)

    # 5. Hybrid cost
    return hybrid_tree_cost(nodes, edges, root, edge_post, lambda_rlct)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"

    # Simulated bandit feedback (random rewards)
    random.seed(42)
    bandit_updates = [
        BanditUpdate(context_id="ctx1", action_id="A<->B", reward=random.random(), propensity=0.5),
        BanditUpdate(context_id="ctx2", action_id="B<->C", reward=random.random(), propensity=0.5),
        BanditUpdate(context_id="ctx3", action_id="C<->D", reward=random.random(), propensity=0.5),
        BanditUpdate(context_id="ctx4", action_id="D<->A", reward=random.random(), propensity=0.5),
    ]

    # Sketch items: treat each edge traversal as a token
    sketch_items = [f"tok_{i%3}" for i in range(100)]  # three distinct tokens repeated

    cost = compute_hybrid_cost(nodes, edges, root, bandit_updates, sketch_items)
    print(f"Hybrid cost: {cost:.4f}")