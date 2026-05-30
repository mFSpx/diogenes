# DARWIN HAMMER — match 4606, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1465_s2.py (gen5)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py (gen3)
# born: 2026-05-29T23:57:01Z

"""
Hybrid Algorithm: DARWIN HAMMER Fusion of
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1465_s2.py
- Parent B: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py

Mathematical Bridge
-------------------
Parent A provides a *regret‑weighted strategy*: for each possible action it
computes a regret value r_i ∈ [0,1] that measures how far the action is from the
optimal expected reward.  Parent B treats edge existence as a Bayesian
probability p(e) that is updated with new evidence (likelihood, false‑positive
rate) and finally pruned with a decreasing probability.

The fusion treats the *trust* values of edges from Parent A as the Bayesian
*prior* π_e for each edge.  The regret values r_i are interpreted as a
penalty on the prior: π′_e = π_e·(1−r_i).  This adjusted prior is then fed into
the Bayesian update of Parent B, yielding a posterior probability
P(e|data).  The posterior serves both as a *trust‑weighted velocity* for the
edge and as the quantity that drives the decreasing‑probability pruning.

Thus the core mathematical pipeline is:

    1. Compute regret vector r = regret_weighted_strategy(actions, rewards)
    2. Adjust edge trusts τ → τ′ = τ·(1−r)                (regret‑adjusted prior)
    3. Bayesian update:   π = bayes_update(τ′, likelihood, marginal)
    4. Prune edges with probability p(t) = λ·exp(−α·t) using π as a filter.

The following module implements this hybrid pipeline with three public
functions that illustrate the combined dynamics.
"""

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
# Parent A – Regret‑Weighted Strategy (simplified)
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[int],
    rewards: List[float],
    num_actions: int,
) -> np.ndarray:
    """
    Compute a regret value for each possible action.

    Parameters
    ----------
    actions : List[int]
        Sequence of taken actions (integer identifiers in range(num_actions)).
    rewards : List[float]
        Observed reward for each action in ``actions``.
    num_actions : int
        Total number of distinct actions.

    Returns
    -------
    np.ndarray
        Array ``r`` of shape (num_actions,) with values in [0, 1] where
        ``r[i]`` is the regret of action ``i`` relative to the best average
        reward observed.
    """
    if len(actions) != len(rewards):
        raise ValueError("actions and rewards must have the same length")
    # Accumulate total reward per action
    totals = np.zeros(num_actions, dtype=float)
    counts = np.zeros(num_actions, dtype=int)
    for a, r in zip(actions, rewards):
        totals[a] += r
        counts[a] += 1
    # Avoid division by zero
    avg_rewards = np.where(counts > 0, totals / counts, 0.0)
    best_avg = avg_rewards.max()
    # Regret is the gap to the best average, normalized to [0,1]
    regret = best_avg - avg_rewards
    max_gap = best_avg if best_avg > 0 else 1.0
    regret_norm = regret / max_gap
    return np.clip(regret_norm, 0.0, 1.0)


# ----------------------------------------------------------------------
# Parent B – Bayesian Edge Probability Utilities (unchanged)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for a Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be within [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform a Bayesian update given prior, likelihood and marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing pruning probability as a function of time."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def adjust_trust_with_regret(
    trust: Dict[Edge, float],
    regret_vec: np.ndarray,
    action_map: Dict[int, Edge],
) -> Dict[Edge, float]:
    """
    Adjust edge trust values using the regret vector.

    For each edge e that corresponds to an action a,
    τ′_e = τ_e·(1−r_a).

    Parameters
    ----------
    trust : dict[Edge, float]
        Original trust values (∈[0,1]) for each edge.
    regret_vec : np.ndarray
        Regret values for actions as returned by ``compute_regret_weighted_strategy``.
    action_map : dict[int, Edge]
        Mapping from action index to the edge it influences.

    Returns
    -------
    dict[Edge, float]
        Regret‑adjusted trust values.
    """
    adjusted = {}
    for a, edge in action_map.items():
        r = regret_vec[a] if a < len(regret_vec) else 0.0
        τ = trust.get(edge, 0.0)
        adjusted[edge] = np.clip(τ * (1.0 - r), 0.0, 1.0)
    return adjusted


def hybrid_edge_posterior(
    prior: float,
    evidence_likelihood: float,
    false_positive: float = 0.01,
) -> float:
    """
    Compute the Bayesian posterior for an edge using the hybrid prior.

    The prior already incorporates regret‑adjusted trust.
    """
    marginal = bayes_marginal(prior, evidence_likelihood, false_positive)
    return bayes_update(prior, evidence_likelihood, marginal)


def hybrid_prune_and_flow(
    edges: List[Edge],
    nodes: Dict[str, Point],
    trust: Dict[Edge, float],
    regret_vec: np.ndarray,
    action_map: Dict[int, Edge],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> List[Edge]:
    """
    Perform a full hybrid update:
      1. Adjust trust with regret (creates a prior per edge).
      2. Apply a Bayesian update using a synthetic likelihood derived from
         edge length (shorter edges are more plausible).
      3. Prune edges with a decreasing probability that depends on time ``t``.
    """
    rng = random.Random(seed)

    # 1. Regret‑adjusted trust → prior
    prior_dict = adjust_trust_with_regret(trust, regret_vec, action_map)

    # 2. Bayesian posterior per edge
    posterior: Dict[Edge, float] = {}
    for e in edges:
        # Likelihood inversely proportional to geometric length
        p1, p2 = nodes[e[0]], nodes[e[1]]
        length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        # Avoid division by zero; clamp length to a minimum
        length = max(length, 1e-6)
        likelihood = 1.0 / (1.0 + length)  # in (0,1]
        prior = prior_dict.get(e, 0.0)
        posterior[e] = hybrid_edge_posterior(prior, likelihood)

    # 3. Prune based on time‑dependent probability and posterior strength
    p_prune = prune_probability(t, lam, alpha)
    kept_edges = []
    for e in edges:
        keep_chance = (1.0 - p_prune) + p_prune * posterior[e]  # higher posterior → higher chance to keep
        if rng.random() < keep_chance:
            kept_edges.append(e)

    return kept_edges


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with 4 nodes forming a square
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges: List[Edge] = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("A", "C"),  # diagonal
    ]

    # Initial trust values (uniform)
    trust: Dict[Edge, float] = {e: 0.6 for e in edges}

    # Map actions (0..4) to edges
    action_map = {i: e for i, e in enumerate(edges)}

    # Simulated actions and rewards
    actions = [0, 2, 2, 4, 1, 3, 0, 2]          # indices into action_map
    rewards = [0.5, 0.9, 0.8, 0.2, 0.6, 0.4, 0.7, 0.85]

    # Compute regret‑weighted strategy
    regret_vec = compute_regret_weighted_strategy(actions, rewards, num_actions=len(edges))

    # Perform hybrid update at time t = 5.0
    updated_edges = hybrid_prune_and_flow(
        edges=edges,
        nodes=nodes,
        trust=trust,
        regret_vec=regret_vec,
        action_map=action_map,
        t=5.0,
        lam=1.0,
        alpha=0.15,
        seed=42,
    )

    print("Original edges:", edges)
    print("Regret vector:", regret_vec)
    print("Edges after hybrid prune:", updated_edges)