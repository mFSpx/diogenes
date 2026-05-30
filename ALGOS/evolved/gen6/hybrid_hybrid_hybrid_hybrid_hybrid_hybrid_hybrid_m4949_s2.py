# DARWIN HAMMER — match 4949, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2290_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m2082_s1.py (gen4)
# born: 2026-05-29T23:59:00Z

"""
Hybrid Algorithm Fusion of:
- Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2290_s0.py)
  * State‑Space Model (SSM) based expected values, morphology‑driven recovery priority,
    SSIM/entropy weighting, Gini coefficient on weekday distribution.
- Parent B (hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m2082_s1.py)
  * Liquid‑Time‑Constant (LTC) gating, variational free‑energy, Bayesian minimum‑cost routing.

Mathematical Bridge
-------------------
Both parents manipulate probability‑like quantities.  We fuse them by:

1. Computing a **global probability vector** `q` from the normalized expected
   values of actions (Parent A) and a **prior vector** `π` from router edge priors
   (Parent B).

2. Evaluating the **variational free energy** `F = Σ q·log(q/π)` – the bridge
   between the action distribution and the routing priors.

3. Modulating the **Liquid‑Time‑Constant** `τ(g)` of each routing group `g`
   with `F` and a similarity score `s_g` (derived from SSIM‑like morphology
   similarity) :

   
   τ(g) = gating * (1 + α·F) * (1 + β·s_g)
   

   (`α,β` are tunable scalars.)

4. The **effective routing cost** for group `g` becomes

   
   C_g = π_g * τ(g)                     (1)
   

   The group with minimal `C_g` is selected, its priors are updated with Bayes’
   rule using a likelihood proportional to the observed cost, and the
   **recovery priority** derived from morphology scales the expected reward
   of each action.

The resulting module provides three core functions that demonstrate this
fusion and a smoke‑test that runs end‑to‑end.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between the two parent topologies)
# ----------------------------------------------------------------------


class Morphology:
    """Physical characteristics used to derive a recovery priority."""

    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.mass = float(mass)


class MathAction:
    """Base action definition from Parent A."""

    __slots__ = ("id", "expected_value", "cost", "risk")

    def __init__(
        self,
        id: str,
        expected_value: float,
        cost: float = 0.0,
        risk: float = 0.0,
    ):
        self.id = id
        self.expected_value = float(expected_value)
        self.cost = float(cost)
        self.risk = float(risk)


class HybridAction:
    """Resulting action after hybrid processing."""

    __slots__ = (
        "action_id",
        "expected_reward",
        "expected_value",
        "cost",
        "risk",
        "ternary_symbol",
    )

    def __init__(
        self,
        action_id: str,
        expected_reward: float,
        expected_value: float,
        cost: float = 0.0,
        risk: float = 0.0,
        ternary_symbol: int = 0,
    ):
        self.action_id = action_id
        self.expected_reward = float(expected_reward)
        self.expected_value = float(expected_value)
        self.cost = float(cost)
        self.risk = float(risk)
        self.ternary_symbol = int(ternary_symbol)

    def __repr__(self) -> str:
        return (
            f"HybridAction(id={self.action_id!r}, reward={self.expected_reward:.3f}, "
            f"value={self.expected_value:.3f}, cost={self.cost:.3f}, "
            f"risk={self.risk:.3f}, ternary={self.ternary_symbol})"
        )


# ----------------------------------------------------------------------
# Helper mathematics – pieces from both parents
# ----------------------------------------------------------------------


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a non‑negative list."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def variational_free_energy(q: np.ndarray, p: np.ndarray) -> float:
    """
    Compute Σ q·log(q/p).  Both vectors must sum to 1 and contain non‑negative entries.
    """
    if q.shape != p.shape:
        raise ValueError("q and p must have the same shape")
    if np.any(q < 0) or np.any(p <= 0):
        raise ValueError("Probabilities must be non‑negative and p > 0")
    q = q / q.sum()
    p = p / p.sum()
    return float(np.sum(q * np.log(q / p)))


def liquid_time_constant(
    gating: float,
    free_energy: float,
    similarity: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    LTC modulation (Parent B) where `similarity` ∈ [0,1].
    """
    return gating * (1.0 + alpha * free_energy) * (1.0 + beta * similarity)


def morphology_priority(morph: Morphology) -> float:
    """
    Derive a scalar priority from morphology.
    Larger volume‑to‑mass ratio → higher priority (recovery speed).
    """
    volume = morph.length * morph.width * morph.height
    if morph.mass == 0:
        return 0.0
    return float(volume / morph.mass)


def ssim_like_similarity(m1: Morphology, m2: Morphology) -> float:
    """
    Very lightweight SSIM‑inspired similarity between two morphologies.
    Returns a value in [0,1].
    """
    attrs1 = np.array([m1.length, m1.width, m1.height, m1.mass])
    attrs2 = np.array([m2.length, m2.width, m2.height, m2.mass])
    mean1, mean2 = attrs1.mean(), attrs2.mean()
    var1, var2 = attrs1.var(), attrs2.var()
    cov = np.mean((attrs1 - mean1) * (attrs2 - mean2))
    C1, C2 = 1e-4, 9e-4
    numerator = (2 * mean1 * mean2 + C1) * (2 * cov + C2)
    denominator = (mean1 ** 2 + mean2 ** 2 + C1) * (var1 + var2 + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def hybrid_compute_action_distribution(
    actions: List[MathAction],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the action probability vector `q` from normalized expected values
    (Parent A) and a uniform prior `p` for the router.
    Returns (q, p).
    """
    values = np.array([a.expected_value for a in actions], dtype=float)
    if values.sum() == 0:
        q = np.full_like(values, 1.0 / len(values))
    else:
        q = values / values.sum()
    p = np.full_like(q, 1.0 / len(q))  # uniform prior for the router
    return q, p


def hybrid_route(
    groups: Tuple[str, ...],
    priors: Dict[str, float],
    gating: float,
    free_energy: float,
    similarities: Dict[str, float],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> str:
    """
    Bayesian minimum‑cost routing (Parent B) with LTC‑modulated costs.
    Returns the selected group identifier.
    """
    costs = {}
    for g in groups:
        tau = liquid_time_constant(
            gating, free_energy, similarities.get(g, 0.0), alpha, beta
        )
        costs[g] = priors[g] * tau

    # Choose group with minimal cost
    selected = min(costs, key=costs.get)

    # Bayes update of priors using a simple likelihood proportional to 1 / (1 + cost)
    likelihoods = {g: 1.0 / (1.0 + costs[g]) for g in groups}
    total = sum(priors[g] * likelihoods[g] for g in groups)
    for g in groups:
        priors[g] = (priors[g] * likelihoods[g]) / total

    return selected


def hybrid_transform_actions(
    actions: List[MathAction],
    morph: Morphology,
    gini: float,
    selected_group: str,
    group_bonus: float = 1.0,
) -> List[HybridAction]:
    """
    Convert MathAction → HybridAction using:
    - morphology priority (scales reward)
    - Gini coefficient (penalises risk)
    - a group‑specific bonus (e.g., from routing outcome)
    """
    priority = morphology_priority(morph)
    transformed = []
    for a in actions:
        # Base reward is expected value scaled by morphology priority
        reward = a.expected_value * priority

        # Adjust reward by Gini (more uneven weekday distribution → higher penalty)
        reward *= (1.0 - gini)

        # Add a small deterministic ternary symbol based on modulo of hash
        ternary = (hash(a.id) % 3) - 1  # yields -1,0,1

        # Incorporate group bonus (could be a function of selected_group)
        reward *= group_bonus

        transformed.append(
            HybridAction(
                action_id=a.id,
                expected_reward=reward,
                expected_value=a.expected_value,
                cost=a.cost,
                risk=a.risk,
                ternary_symbol=ternary,
            )
        )
    return transformed


def hybrid_pipeline(
    actions: List[MathAction],
    morph: Morphology,
    weekday_counts: List[int],
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
    gating: float = 0.8,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> Tuple[str, List[HybridAction]]:
    """
    End‑to‑end hybrid processing:
    1. Compute Gini over weekday distribution.
    2. Build action distribution q and uniform prior p.
    3. Compute variational free energy F = Σ q·log(q/p).
    4. Derive a similarity score for each group using a dummy reference morphology.
    5. Route to the minimal‑cost group.
    6. Transform actions into HybridAction objects.
    Returns (selected_group, list_of_hybrid_actions).
    """
    # 1. Gini coefficient on weekday counts (Monday=0 … Sunday=6)
    gini = gini_coefficient(weekday_counts)

    # 2. Action distribution
    q, p = hybrid_compute_action_distribution(actions)

    # 3. Free energy
    free_energy = variational_free_energy(q, p)

    # 4. Similarities – we compare each group’s “canonical” morphology
    #    (here we fabricate a simple reference morphology per group)
    ref_morphs = {
        "codex": Morphology(1.0, 1.0, 1.0, 1.0),
        "groq": Morphology(1.2, 0.9, 1.1, 1.0),
        "cohere": Morphology(0.9, 1.1, 1.0, 1.0),
        "local_models": Morphology(1.0, 1.0, 0.8, 1.0),
    }
    similarities = {
        g: ssim_like_similarity(morph, ref_morphs[g]) for g in groups
    }

    # 5. Initialize priors uniformly and route
    priors = {g: 1.0 / len(groups) for g in groups}
    selected = hybrid_route(
        groups, priors, gating, free_energy, similarities, alpha, beta
    )

    # 6. Transform actions – we give a modest bonus to the selected group
    group_bonus = 1.0 + 0.1 * (list(groups).index(selected))
    hybrid_actions = hybrid_transform_actions(
        actions, morph, gini, selected, group_bonus
    )

    return selected, hybrid_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny set of actions
    sample_actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="A2", expected_value=5.0, cost=1.0, risk=0.2),
        MathAction(id="A3", expected_value=8.0, cost=1.5, risk=0.15),
    ]

    # Example morphology (could be a robot arm, etc.)
    sample_morph = Morphology(length=2.0, width=0.5, height=0.3, mass=4.0)

    # Weekday distribution (e.g., number of tasks per weekday)
    weekday_counts = [12, 15, 9, 13, 11, 7, 8]  # Monday … Sunday

    selected_group, hybrid_actions = hybrid_pipeline(
        actions=sample_actions,
        morph=sample_morph,
        weekday_counts=weekday_counts,
        gating=0.9,
        alpha=0.7,
        beta=0.3,
    )

    print(f"Selected routing group: {selected_group}")
    print("Hybrid actions:")
    for ha in hybrid_actions:
        print("  ", ha)