# DARWIN HAMMER — match 527, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s3.py (gen3)
# parent_b: hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (gen4)
# born: 2026-05-29T23:29:29Z

"""Hybrid Algorithm combining Decision Regret (Parent A) and Decreasing‑Rate Pruning with Epistemic Certainty (Parent B).

Mathematical Bridge
-------------------
* Parent A defines a *regret* for each action as a cost‑risk product and measures the
  inequality of these regrets with the Gini coefficient **G**.
* Parent B updates edge weights of a graph using Bayesian marginal probabilities that
  encode *epistemic certainty* and then prunes edges with a decreasing‑rate schedule
  **p(t)=λ·exp(−αt)**.

The hybrid algorithm treats each *action* as a node in a weighted graph.  
For an edge *(i, j)* we build a composite weight  


W_ij = d_ij · (1 + R_i) · (1 + R_j) · (1 − G) · M_ij


where  

* `d_ij` – Euclidean distance (Parent B).  
* `R_i` – normalized regret of node *i* (Parent A).  
* `G` – Gini coefficient of the regret distribution (Parent A).  
* `M_ij` – Bayesian marginal probability derived from epistemic priors
  (Parent B).

The resulting weights are then fed to the decreasing‑rate pruning function.
After pruning, a Fisher‑score‑based localization angle is computed and the actions
are ranked by a hybrid expected value that rewards low regret and high retained
connectivity.

The three public functions demonstrate this pipeline:
`compute_regret_gini`, `compose_edge_weights`, and `hybrid_prune_and_rank`.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regret and Gini utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D array."""
    if values.ndim != 1:
        raise ValueError("gini_coefficient expects a 1‑D array")
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def compute_regret_gini(costs: np.ndarray, risks: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute per‑action regret as cost·risk, normalize it, and return the
    Gini coefficient of the regret distribution.
    """
    if costs.shape != risks.shape:
        raise ValueError("costs and risks must have the same shape")
    raw_regret = costs * risks
    total = raw_regret.sum()
    if total == 0:
        normalized = np.zeros_like(raw_regret)
    else:
        normalized = raw_regret / total
    gini = gini_coefficient(normalized)
    return normalized, gini


# ----------------------------------------------------------------------
# Parent B – Bayesian update and pruning utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for a binary hypothesis."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability p(t)=λ·exp(−αt) capped at 1."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(edges: List[Hashable], t: float, lam: float = 1.0,
               alpha: float = 0.2, seed: int | str | None = None) -> List[Hashable]:
    """Randomly drop edges according to prune_probability."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
@dataclass
class ActionNode:
    """A node representing a decision action."""
    id: int
    position: Tuple[float, float]
    cost: float
    risk: float
    prior: float          # epistemic prior probability of correctness
    likelihood: float     # likelihood of observation given truth
    false_positive: float # false‑positive rate of observation


def compose_edge_weights(nodes: List[ActionNode],
                        edges: List[Tuple[int, int]],
                        regrets: np.ndarray,
                        gini: float) -> Dict[Tuple[int, int], float]:
    """
    Build composite edge weights W_ij = d_ij·(1+R_i)·(1+R_j)·(1−G)·M_ij
    where M_ij is the Bayesian marginal for the weaker node of the pair.
    """
    node_lookup = {node.id: node for node in nodes}
    weight_dict: Dict[Tuple[int, int], float] = {}

    for (u, v) in edges:
        n_u = node_lookup[u]
        n_v = node_lookup[v]

        # geometric distance
        d = euclidean_length(n_u.position, n_v.position)

        # regret factors (add 1 to keep weights >0)
        R_u = regrets[u]
        R_v = regrets[v]

        # epistemic marginal – use the node with lower prior as the conservative estimate
        if n_u.prior <= n_v.prior:
            marginal = bayes_marginal(n_u.prior, n_u.likelihood, n_u.false_positive)
        else:
            marginal = bayes_marginal(n_v.prior, n_v.likelihood, n_v.false_positive)

        # composite weight
        w = d * (1.0 + R_u) * (1.0 + R_v) * (1.0 - gini) * marginal
        weight_dict[(u, v)] = w

    return weight_dict


def hybrid_prune_and_rank(nodes: List[ActionNode],
                          edges: List[Tuple[int, int]],
                          t: float,
                          lam: float = 1.0,
                          alpha: float = 0.2,
                          seed: int | str | None = None) -> List[Tuple[int, float]]:
    """
    Full hybrid pipeline:
    1. Compute regrets and Gini.
    2. Compose edge weights.
    3. Prune edges with decreasing‑rate schedule.
    4. Compute a hybrid expected value for each node:
       EV_i = (1 - regret_i) * connectivity_i
       where connectivity_i = sum of retained incident edge weights.
    5. Return actions sorted by descending EV.
    """
    # ----- 1. Regret & Gini -----
    costs = np.array([n.cost for n in nodes])
    risks = np.array([n.risk for n in nodes])
    regrets, gini = compute_regret_gini(costs, risks)

    # ----- 2. Edge weights -----
    weighted_edges = compose_edge_weights(nodes, edges, regrets, gini)

    # ----- 3. Pruning -----
    # Keep edge identifiers; pruning discards based on probability only.
    edge_ids = list(weighted_edges.keys())
    retained_ids = prune_edges(edge_ids, t, lam, alpha, seed)

    # Build adjacency map of retained weights
    adjacency: Dict[int, float] = {node.id: 0.0 for node in nodes}
    for eid in retained_ids:
        w = weighted_edges[eid]
        u, v = eid
        adjacency[u] += w
        adjacency[v] += w

    # ----- 4. Hybrid Expected Value -----
    ev = {}
    for node in nodes:
        # (1 - regret) rewards low regret; connectivity rewards retained structure
        ev[node.id] = (1.0 - regrets[node.id]) * adjacency[node.id]

    # ----- 5. Ranking -----
    ranked = sorted(ev.items(), key=lambda kv: kv[1], reverse=True)
    return ranked


def fisher_localization_angle(nodes: List[ActionNode],
                              edges: List[Tuple[int, int]],
                              retained_edges: List[Tuple[int, int]]) -> float:
    """
    Simple Fisher‑score based orientation estimator.
    Treat each retained edge as a 2‑D vector; the Fisher score is approximated by
    the variance of angles. The optimal angle is the mean direction of the vectors.
    """
    if not retained_edges:
        return 0.0
    vectors = []
    lookup = {n.id: n for n in nodes}
    for u, v in retained_edges:
        p_u = np.array(lookup[u].position)
        p_v = np.array(lookup[v].position)
        vec = p_v - p_u
        if np.linalg.norm(vec) == 0:
            continue
        vectors.append(vec / np.linalg.norm(vec))
    angles = np.arctan2([v[1] for v in vectors], [v[0] for v in vectors])
    mean_angle = float(np.mean(angles))
    return mean_angle  # radians


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    rng = random.Random(42)

    # generate a small synthetic graph
    N = 6
    nodes: List[ActionNode] = []
    for i in range(N):
        pos = (rng.uniform(0, 10), rng.uniform(0, 10))
        node = ActionNode(
            id=i,
            position=pos,
            cost=rng.uniform(1, 5),
            risk=rng.uniform(0.1, 0.9),
            prior=rng.uniform(0.4, 0.9),
            likelihood=rng.uniform(0.5, 1.0),
            false_positive=rng.uniform(0.0, 0.3),
        )
        nodes.append(node)

    # fully connect the graph
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    # run hybrid pipeline
    t = 2.0  # pruning time step
    ranking = hybrid_prune_and_rank(nodes, edges, t, lam=1.0, alpha=0.3, seed=123)

    print("Hybrid ranking (action_id, hybrid_EV):")
    for aid, ev in ranking:
        print(f"  Action {aid}: EV = {ev:.4f}")

    # demonstrate Fisher localization on the retained edges
    # we need the list of retained edges from the same pruning step
    # recompute retained edges for the angle function
    costs = np.array([n.cost for n in nodes])
    risks = np.array([n.risk for n in nodes])
    regrets, gini = compute_regret_gini(costs, risks)
    weighted = compose_edge_weights(nodes, edges, regrets, gini)
    retained = prune_edges(list(weighted.keys()), t, lam=1.0, alpha=0.3, seed=123)

    angle_rad = fisher_localization_angle(nodes, edges, retained)
    angle_deg = math.degrees(angle_rad)
    print(f"Estimated localization angle: {angle_deg:.2f}°")