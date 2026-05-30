# DARWIN HAMMER — match 4113, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1387_s1.py (gen6)
# parent_b: hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s0.py (gen3)
# born: 2026-05-29T23:53:34Z

"""
Hybrid Regret‑Bandit / Path‑Signature & Bayesian Tree Routing Fusion

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret (Algorithm A)
- hybrid_bayes_claim_kernel_hybrid_ternary_route (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A provides a *store dynamics* signal:
    Δ   = α·Σ(inflow) – β·Σ(outflow)
    levelₜ₊₁ = max(0, levelₜ + Δ·dt)
    dance   = tanh(gain·Δ)

Algorithm B supplies a *Bayesian posterior* update and a *minimum‑cost tree* score.
We fuse them by letting the store‑derived **dance** modulate the Bayesian
posterior before it is used as a probability (propensity) in the regret‑bandit
utility and as a weight on edge costs in the tree.  Concretely:

    posterior′_h = dance · Bayes(prior_h , likelihood_h)
    utility_h    = (expected_reward_h – regret_h) · posterior′_h

The weighted posterior also rescales edge lengths:
    weighted_length(e) = length(e) · (posterior′_{head(e)} + posterior′_{tail(e)})/2

Thus the dynamics of the physical store influence uncertainty handling and
routing decisions, completing a mathematically unified hybrid system.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures (light‑weight replicas of the parents)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    gain: float = 0.5   # scaling for the dance signal


@dataclass(frozen=True)
class BanditAction:
    """Action description used by the regret‑bandit side."""
    action_id: str
    expected_reward: float
    regret: float = 0.0   # cumulative regret estimate


# ----------------------------------------------------------------------
# Core mathematics from Algorithm A
# ----------------------------------------------------------------------
def compute_store_delta(inflows: List[float], outflows: List[float], store: StoreState) -> float:
    """Δ = α·Σ(inflow) – β·Σ(outflow)"""
    inflow_sum = sum(inflows)
    outflow_sum = sum(outflows)
    delta = store.alpha * inflow_sum - store.beta * outflow_sum
    return delta


def update_store(store: StoreState, delta: float) -> None:
    """Integrate Δ into the store level using the discrete dynamics."""
    new_level = max(0.0, store.level + delta * store.dt)
    store.level = new_level


def compute_dance(store: StoreState, delta: float) -> float:
    """dance = tanh(gain·Δ) – a bounded signal in (‑1, 1)."""
    return math.tanh(store.gain * delta)


# ----------------------------------------------------------------------
# Core mathematics from Algorithm B
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    posterior_weights: Dict[str, float],
) -> float:
    """
    Minimum‑cost tree cost where each edge length is weighted by the
    average posterior weight of its two incident nodes.
    """
    # Build adjacency list (undirected)
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    weighted_sum = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        w_a = posterior_weights.get(a, 1.0)
        w_b = posterior_weights.get(b, 1.0)
        avg_w = (w_a + w_b) / 2.0
        weighted_sum += length(nodes[a], nodes[b]) * avg_w

    # Simple depth‑first traversal to ensure connectivity (not a true MST)
    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adj[node]:
            if nbr not in visited:
                stack.append(nbr)

    # If the graph is disconnected, penalise heavily
    if len(visited) != len(nodes):
        weighted_sum += 1e6
    return weighted_sum


def bayesian_update(prior: float, likelihood: float) -> float:
    """
    One‑step Bayesian update for a binary hypothesis.
    posterior = (prior * likelihood) / (prior * likelihood + (1‑prior)*(1‑likelihood))
    """
    numerator = prior * likelihood
    denominator = numerator + (1.0 - prior) * (1.0 - likelihood)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid functions (the three required demonstrations)
# ----------------------------------------------------------------------
def hybrid_posterior_update(
    priors: Dict[str, float],
    likelihoods: Dict[str, float],
    dance: float,
) -> Dict[str, float]:
    """
    Update a set of hypothesis posteriors with Bayesian rule and then
    modulate them by the store's dance signal (the mathematical bridge).
    The resulting values are renormalised to sum to 1.
    """
    raw: Dict[str, float] = {}
    for h, prior in priors.items():
        lik = likelihoods.get(h, 0.5)   # default neutral evidence
        post = bayesian_update(prior, lik)
        # Modulation by dance (keeps sign, scales magnitude)
        raw[h] = max(0.0, post * (1.0 + dance))  # ensure non‑negative

    total = sum(raw.values())
    if total == 0.0:
        # fallback to uniform distribution
        n = len(raw)
        return {h: 1.0 / n for h in raw}
    return {h: v / total for h, v in raw.items()}


def regret_weighted_utility(
    action: BanditAction,
    dance: float,
) -> float:
    """
    Compute utility = (expected_reward – regret) * dance.
    The dance signal couples the path‑signature dynamics to the bandit policy.
    """
    base = action.expected_reward - action.regret
    return base * dance


def hybrid_decision_step(
    actions: List[BanditAction],
    store: StoreState,
    inflows: List[float],
    outflows: List[float],
    priors: Dict[str, float],
    likelihoods: Dict[str, float],
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[BanditAction, float]:
    """
    Perform a full hybrid iteration:
      1. Compute store Δ and update level + dance.
      2. Update posterior probabilities with the dance signal.
      3. Score the tree with posterior‑weighted edge costs.
      4. Evaluate each action's regret‑weighted utility, further
         scaled by the (inverse) tree cost to prefer cheaper routing.
      5. Return the action with maximal hybrid utility and the associated utility value.
    """
    # 1. Store dynamics
    delta = compute_store_delta(inflows, outflows, store)
    update_store(store, delta)
    dance = compute_dance(store, delta)

    # 2. Bayesian posterior modulation
    post = hybrid_posterior_update(priors, likelihoods, dance)

    # 3. Tree cost (lower cost → higher desirability)
    cost = tree_cost(nodes, edges, root, post)
    cost_factor = 1.0 / (1.0 + cost)   # map cost ∈ [0,∞) to (0,1]

    # 4. Action utilities
    best_action = None
    best_score = -math.inf
    for act in actions:
        util = regret_weighted_utility(act, dance)
        # Incorporate routing desirability
        hybrid_score = util * cost_factor
        if hybrid_score > best_score:
            best_score = hybrid_score
            best_action = act

    return best_action, best_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a simple store
    store = StoreState(level=10.0, alpha=0.8, beta=0.3, dt=1.0, gain=0.6)

    # Dummy inflow/outflow streams
    inflows = [5.0, 2.0]
    outflows = [1.0, 0.5]

    # Bandit actions
    actions = [
        BanditAction(action_id="A", expected_reward=4.0, regret=1.2),
        BanditAction(action_id="B", expected_reward=3.5, regret=0.8),
        BanditAction(action_id="C", expected_reward=5.0, regret=2.0),
    ]

    # Bayesian priors and synthetic likelihoods for three hypotheses
    priors = {"h1": 0.4, "h2": 0.35, "h3": 0.25}
    likelihoods = {"h1": 0.7, "h2": 0.4, "h3": 0.6}

    # Simple geometric graph (tree‑like)
    nodes = {
        "root": (0.0, 0.0),
        "n1": (1.0, 2.0),
        "n2": (3.0, 1.0),
        "n3": (4.0, 4.0),
    }
    edges = [
        ("root", "n1"),
        ("n1", "n2"),
        ("n2", "n3"),
    ]
    root = "root"

    # Run a single hybrid decision step
    chosen_action, score = hybrid_decision_step(
        actions=actions,
        store=store,
        inflows=inflows,
        outflows=outflows,
        priors=priors,
        likelihoods=likelihoods,
        nodes=nodes,
        edges=edges,
        root=root,
    )

    print(f"Chosen action: {chosen_action.action_id if chosen_action else 'None'}")
    print(f"Hybrid utility score: {score:.4f}")
    print(f"Updated store level: {store.level:.4f}")
    print(f"Final dance signal: {compute_dance(store, compute_store_delta(inflows, outflows, store)):.4f}")