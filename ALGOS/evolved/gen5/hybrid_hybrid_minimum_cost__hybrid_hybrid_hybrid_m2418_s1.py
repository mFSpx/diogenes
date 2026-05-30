# DARWIN HAMMER — match 2418, survivor 1
# gen: 5
# parent_a: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hard_truth_ma_m1004_s0.py (gen4)
# born: 2026-05-29T23:42:12Z

"""Hybrid Minimum-Cost Tree with Bayesian Bandit-Driven Node Rewards.

Parent A: Implements a contextual multi‑armed bandit policy and a minimum‑cost tree
scoring function (tree_cost).  The policy provides empirical mean rewards
(_reward) and selection counts (_count) for actions identified by strings.

Parent B: Provides stylometry‑style feature handling and a Bayesian evidence
update framework (Gaussian conjugate prior) for aggregating noisy observations.

Mathematical Bridge:
The expected reward of a bandit action is interpreted as a *prior mean* for a
Gaussian model of a node’s latent utility.  When a node participates in a
tree, its observed rewards (from bandit updates) are treated as noisy samples.
A Bayesian posterior mean μ̂ is computed and injected into the tree‑cost
objective as a reward‑adjustment term.  Thus the hybrid objective is

    C_hybrid = Σ_edge length
               + λ·Σ_path_distance(root → node)
               – β·Σ_node μ̂(node)

where μ̂(node) is the posterior mean derived from the bandit statistics for the
action associated with that node.  This fuses the combinatorial topology of the
tree with the probabilistic learning dynamics of the bandit/Bayesian components.

The module below implements:
* Bandit policy update (from Parent A).
* Gaussian conjugate Bayesian update (from Parent B).
* Hybrid tree cost that combines material cost, path penalties and Bayesian‑adjusted rewards.
"""

import math
import random
import sys
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit core (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float


_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]


def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def length(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)


# ----------------------------------------------------------------------
# Bayesian update utilities (Parent B)
# ----------------------------------------------------------------------
def bayesian_posterior_gaussian(
    prior_mean: float,
    prior_variance: float,
    observations: List[float],
    observation_variance: float,
) -> Tuple[float, float]:
    """
    Conjugate Gaussian update.
    Prior:   N(prior_mean, prior_variance)
    Likelihood for each obs: N(obs, observation_variance)

    Returns posterior mean and variance.
    """
    if not observations:
        return prior_mean, prior_variance

    n = len(observations)
    obs_sum = sum(observations)

    precision_prior = 1.0 / prior_variance
    precision_likelihood = n / observation_variance

    posterior_variance = 1.0 / (precision_prior + precision_likelihood)
    posterior_mean = posterior_variance * (
        precision_prior * prior_mean + (obs_sum / observation_variance)
    )
    return posterior_mean, posterior_variance


# ----------------------------------------------------------------------
# Hybrid tree cost (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
    reward_weight: float = 1.0,
    prior_variance: float = 1.0,
    observation_variance: float = 1.0,
) -> float:
    """
    Compute the hybrid cost:
        material (sum of edge lengths)
        + path_weight * Σ distances from root
        - reward_weight * Σ posterior mean rewards (Bayesian adjusted)

    The posterior mean for a node is obtained by treating the bandit empirical
    reward as a prior mean and the per‑node reward observations (from the
    bandit policy) as Gaussian samples.
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + length(nodes[cur], nodes[nb])
                stack.append(nb)

    # Accumulate Bayesian‑adjusted reward term
    reward_term = 0.0
    for node_id in nodes:
        # Use node_id as the action identifier for the bandit
        prior_mu = _reward(node_id)          # empirical mean (may be 0)
        prior_n = _count(node_id)
        # Treat each observed reward as an independent observation
        # Retrieve raw observations from the policy store
        total, count = _POLICY.get(node_id, [0.0, 0.0])
        observations = []
        if count > 0:
            # Reconstruct approximate observations assuming equal contribution
            avg = total / count
            observations = [avg] * int(count)  # crude but sufficient for demo
        post_mu, _ = bayesian_posterior_gaussian(
            prior_mean=prior_mu,
            prior_variance=prior_variance,
            observations=observations,
            observation_variance=observation_variance,
        )
        reward_term += post_mu

    total_cost = material + path_weight * sum(dist.values()) - reward_weight * reward_term
    return total_cost


# ----------------------------------------------------------------------
# Example helper: generate random updates for testing
# ----------------------------------------------------------------------
def generate_random_updates(
    node_ids: List[str], num_updates: int = 10
) -> List[BanditUpdate]:
    """Create a list of synthetic BanditUpdate objects for given node identifiers."""
    updates = []
    for _ in range(num_updates):
        node = random.choice(node_ids)
        reward = random.uniform(0.0, 1.0)
        propensity = random.uniform(0.1, 1.0)
        updates.append(
            BanditUpdate(
                context_id="ctx_dummy",
                action_id=node,
                reward=reward,
                propensity=propensity,
            )
        )
    return updates


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes = {
        "A": Point(0.0, 0.0),
        "B": Point(1.0, 0.0),
        "C": Point(1.0, 1.0),
        "D": Point(0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"

    # Reset and populate bandit policy with synthetic data
    reset_policy()
    updates = generate_random_updates(list(nodes.keys()), num_updates=20)
    update_policy(updates)

    # Compute hybrid cost
    cost = hybrid_tree_cost(
        nodes,
        edges,
        root,
        path_weight=0.2,
        reward_weight=1.0,
        prior_variance=1.0,
        observation_variance=0.5,
    )
    print(f"Hybrid tree cost: {cost:.4f}")

    # Verify that the policy dictionary has been populated
    for nid in nodes:
        print(
            f"Node {nid}: count={_count(nid):.0f}, mean_reward={_reward(nid):.3f}"
        )