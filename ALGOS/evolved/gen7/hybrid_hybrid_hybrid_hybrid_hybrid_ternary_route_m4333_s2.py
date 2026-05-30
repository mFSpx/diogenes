# DARWIN HAMMER — match 4333, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1876_s2.py (gen6)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (gen2)
# born: 2026-05-29T23:55:04Z

"""Hybrid algorithm merging:
- Parent A: temperature‑dependent Schoolfield developmental rate, bandit action scoring and Shannon entropy of propensities.
- Parent B: ternary router tree‑cost computation with Bayesian marginal update.

Mathematical bridge:
The Schoolfield rate r(T) is used as a multiplicative certainty factor that scales both:
1) Bayesian updates of edge‑uncertainty probabilities in the tree cost model.
2) Confidence bounds of bandit actions (LinUCB style).
Additionally, the Shannon entropy H(p) of the bandit propensities provides an information‑theoretic weight that modulates the routing cost, linking the decision‑hygiene scores of A with the routing decisions of B.
The resulting hybrid system yields temperature‑aware, entropy‑weighted routing and action selection.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0  # cal mol⁻¹ K⁻¹
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol⁻¹ K⁻¹


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float  # probability of being selected
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridLinUCB"


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------


def schoolfield_rate(T: float, p: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Temperature‑dependent developmental rate r(T) using the Schoolfield‑Rollinson
    equation (simplified to a single‐enzyme form).

    r(T) = rho_25 * exp( -ΔH_a / (R * (1/T - 1/298.15)) ) *
           (1 + exp(ΔH_l / (R * (1/T_low - 1/T))))⁻¹ *
           (1 + exp(ΔH_h / (R * (1/T - 1/T_high))))⁻¹
    """
    R = p.r_cal
    inv_T = 1.0 / T
    inv_ref = 1.0 / 298.15
    term_activation = math.exp(-p.delta_h_activation / (R) * (inv_T - inv_ref))

    low_factor = 1.0 + math.exp(p.delta_h_low / R * (1.0 / p.t_low - inv_T))
    high_factor = 1.0 + math.exp(p.delta_h_high / R * (inv_T - 1.0 / p.t_high))

    rate = p.rho_25 * term_activation / (low_factor * high_factor)
    return max(rate, 0.0)


def shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy H = -∑ p_i log2(p_i) (base 2)."""
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]  # ignore zero probabilities
    return -np.sum(probs * np.log2(probs))


def bayes_marginal_weighted(
    prior: float,
    likelihood: float,
    false_positive: float,
    temperature: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> float:
    """
    Bayesian marginal update where the likelihood is scaled by the
    temperature‑dependent rate r(T). This couples Parent A's physiology
    model with Parent B's evidence update.
    """
    rT = schoolfield_rate(temperature, params)
    weighted_likelihood = likelihood * rT
    # standard marginal: P(E) = L * prior + false_positive * (1 - prior)
    return weighted_likelihood * prior + false_positive * (1.0 - prior)


# ----------------------------------------------------------------------
# Tree routing (from Parent B) with hybrid extensions
# ----------------------------------------------------------------------


Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost_hybrid(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    temperature: float,
    entropy_weight: float,
    path_weight: float = 0.2,
) -> float:
    """
    Compute material cost + temperature‑scaled path cost.
    The path cost is multiplied by r(T) and by an entropy‑derived factor
    (1 + H). This creates a bridge between the entropy of bandit propensities
    and the routing cost.
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += euclidean_length(nodes[a], nodes[b])

    # Breadth‑first traversal to obtain distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean_length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    rT = schoolfield_rate(temperature)
    entropy_factor = 1.0 + entropy_weight  # H in [0, ...] → scale >1

    path_cost = path_weight * sum(dist.values()) * rT * entropy_factor
    return material + path_cost


# ----------------------------------------------------------------------
# Bandit action selection (from Parent A) with hybrid scaling
# ----------------------------------------------------------------------


def select_bandit_action(
    actions: List[BanditAction],
    temperature: float,
    time_step: int,
) -> BanditAction:
    """
    LinUCB‑style selection where the confidence term is multiplied by the
    temperature‑dependent rate r(T) and by an entropy‑derived scaling.
    """
    if not actions:
        raise ValueError("No actions provided")

    # Normalise propensities to obtain a probability distribution
    prop_vals = np.array([a.propensity for a in actions], dtype=float)
    if prop_vals.sum() == 0:
        prop_vals = np.ones_like(prop_vals) / len(prop_vals)
    else:
        prop_vals = prop_vals / prop_vals.sum()

    entropy = shannon_entropy(prop_vals.tolist())
    rT = schoolfield_rate(temperature)

    scores = []
    for a, p in zip(actions, prop_vals):
        # classic LinUCB: mu + alpha * sqrt(log(t)/n)
        # Here confidence_bound already encodes alpha; we further scale.
        exploration = a.confidence_bound * math.sqrt(math.log(max(time_step, 1) + 1) / (p + 1e-9))
        hybrid_score = a.expected_reward + exploration * rT * (1.0 + entropy)
        scores.append(hybrid_score)

    best_idx = int(np.argmax(scores))
    return actions[best_idx]


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


def _demo():
    # Temperature (K)
    T = 295.0

    # ----- Bandit actions -----
    actions = [
        BanditAction("a1", propensity=0.2, expected_reward=1.0, confidence_bound=0.5),
        BanditAction("a2", propensity=0.5, expected_reward=0.8, confidence_bound=0.7),
        BanditAction("a3", propensity=0.3, expected_reward=0.6, confidence_bound=0.9),
    ]
    chosen = select_bandit_action(actions, temperature=T, time_step=10)
    print(f"Chosen action: {chosen.action_id}")

    # ----- Tree routing -----
    nodes = {
        "root": (0.0, 0.0),
        "n1": (1.0, 2.0),
        "n2": (3.0, 1.0),
        "n3": (4.0, 4.0),
    }
    edges = [("root", "n1"), ("n1", "n2"), ("n2", "n3")]
    # Use entropy of the bandit propensities as the entropy weight
    prop_vals = [a.propensity for a in actions]
    entropy_w = shannon_entropy(prop_vals)
    cost = tree_cost_hybrid(nodes, edges, root="root", temperature=T, entropy_weight=entropy_w)
    print(f"Hybrid tree cost: {cost:.4f}")

    # ----- Bayesian marginal with temperature scaling -----
    prior = 0.3
    likelihood = 0.6
    false_pos = 0.1
    posterior = bayes_marginal_weighted(prior, likelihood, false_pos, temperature=T)
    print(f"Temperature‑scaled posterior: {posterior:.4f}")


if __name__ == "__main__":
    _demo()