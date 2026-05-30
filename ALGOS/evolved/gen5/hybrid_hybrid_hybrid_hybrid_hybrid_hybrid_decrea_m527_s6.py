# DARWIN HAMMER — match 527, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s3.py (gen3)
# parent_b: hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (gen4)
# born: 2026-05-29T23:29:29Z

"""
Hybrid Algorithm: Regret‑Epistemic Pruning with Fisher Localization

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_regret_engine_m2_s3.py (Regret‑weighted strategy,
  Gini coefficient, decision‑feature expected value)
- hybrid_hybrid_decreasing_pr_hybrid_fisher_locali_m215_s0.py (Decreasing‑rate
  pruning, epistemic certainty flags, Bayesian updates, Fisher angle
  localization, ternary route selection)

Mathematical Bridge:
The regret‑weighted strategy yields a cost vector **r** for actions.  Epistemic
flags provide a certainty factor **c** ∈ [0,1] that can be interpreted as a
likelihood in a Bayesian update.  By treating **r** as a prior “risk” and **c**
as a likelihood we compute a posterior weight **w** = BayesUpdate(r, c, m),
where the marginal *m* = BayesMarginal(r, c, fp) and *fp* is a small false‑
positive rate.  The resulting posterior weights are then fed to the
decreasing‑rate pruning schedule, producing a time‑dependent pruning
probability.  Surviving actions are localized via a Fisher‑score weighted
centroid (the “optimal angle”), and a ternary route is selected by ranking
the remaining actions with a Gini‑adjusted expected value.

The implementation below intertwines the core equations of both parents
instead of merely concatenating them.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from collections.abc import Hashable

import numpy as np

# ----------------------------------------------------------------------
# Helper mathematics (shared between parents)
# ----------------------------------------------------------------------


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1‑D array."""
    if values.ndim != 1:
        raise ValueError("Gini coefficient requires a 1‑D array")
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def bayes_marginal(prior: float, likelihood: float, false_positive: float = 1e-6) -> float:
    """Marginal probability for a simple Bayesian update."""
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability given prior, likelihood and marginal."""
    if marginal == 0:
        return 0.0
    return (likelihood * prior) / marginal


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Decreasing‑rate pruning probability (Parent B)."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def compute_hybrid_weights(
    actions: list[Hashable],
    values: np.ndarray,
    risks: np.ndarray,
    epistemic_flags: list[str],
    false_positive: float = 1e-6,
) -> np.ndarray:
    """
    Combine regret‑weighted strategy (Parent A) with epistemic certainty
    (Parent B) via Bayesian updating.

    Parameters
    ----------
    actions : list[Hashable]
        Identifier for each action.
    values : np.ndarray
        Expected monetary/value payoff for each action.
    risks : np.ndarray
        Regret or cost associated with each action (higher = worse).
    epistemic_flags : list[str]
        One of the predefined certainty strings.
    false_positive : float
        Small probability for the marginal calculation.

    Returns
    -------
    np.ndarray
        Posterior weights normalised to sum to 1.
    """
    if not (len(actions) == values.size == risks.size == len(epistemic_flags)):
        raise ValueError("All input sequences must have the same length")

    # ---- Regret‑weighted strategy (Parent A) ----
    # Convert regrets to a prior “risk” probability via a softmax‑like map.
    # Larger regret → larger prior probability of being undesirable.
    prior = np.exp(risks)  # monotonic transform
    prior /= prior.sum()

    # ---- Epistemic certainty mapping (Parent B) ----
    certainty_map = {
        "FACT": 1.0,
        "PROBABLE": 0.8,
        "POSSIBLE": 0.5,
        "SURE_MAYBE": 0.6,
        "BULLSHIT": 0.0,
    }
    likelihood = np.array([certainty_map.get(f, 0.0) for f in epistemic_flags], dtype=float)

    # ---- Bayesian posterior (mathematical bridge) ----
    marginals = np.vectorize(bayes_marginal)(prior, likelihood, false_positive)
    post = np.vectorize(bayes_update)(prior, likelihood, marginals)

    # ---- Incorporate expected value ----
    # Higher value should increase weight; we multiply by a normalized value term.
    val_norm = values - values.min()
    if val_norm.max() > 0:
        val_norm /= val_norm.max()
    else:
        val_norm = np.zeros_like(values)
    combined = post * (1 + val_norm)  # simple linear blend

    # Normalise to a probability distribution
    combined_sum = combined.sum()
    if combined_sum == 0:
        # fallback to uniform distribution
        combined = np.full_like(combined, 1.0 / combined.size)
    else:
        combined /= combined_sum

    return combined


def prune_hybrid_actions(
    actions: list[Hashable],
    weights: np.ndarray,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> list[Hashable]:
    """
    Apply decreasing‑rate pruning (Parent B) to actions weighted by the hybrid
    posterior weights.

    Parameters
    ----------
    actions : list[Hashable]
        Original actions.
    weights : np.ndarray
        Posterior weights from ``compute_hybrid_weights``.
    t : float
        Current time‑step or iteration index.
    lam, alpha : float
        Pruning schedule parameters.
    seed : optional
        Random seed for reproducibility.

    Returns
    -------
    list[Hashable]
        Subset of actions that survive pruning.
    """
    if len(actions) != weights.size:
        raise ValueError("Length mismatch between actions and weights")
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)

    # Higher weight → lower chance of removal; we invert the probability.
    survivors = [
        act
        for act, w in zip(actions, weights)
        if rng.random() >= p * (1 - w)  # weight‑biased survival
    ]
    return survivors


def fisher_angle_localization(
    points: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute a Fisher‑score weighted centroid and return the angle (in radians)
    from the origin to that centroid.  This mirrors the Fisher localization
    step from Parent B.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        (x, y) coordinates for each action.
    weights : np.ndarray, shape (N,)
        Corresponding importance weights.

    Returns
    -------
    float
        Angle θ ∈ [‑π, π] of the weighted centroid.
    """
    if points.shape[0] != weights.size:
        raise ValueError("Points and weights must have the same length")
    # Weighted centroid
    cx = np.average(points[:, 0], weights=weights)
    cy = np.average(points[:, 1], weights=weights)
    return math.atan2(cy, cx)


def ternary_route_selection(
    actions: list[Hashable],
    values: np.ndarray,
    weights: np.ndarray,
    gini: float,
) -> list[Hashable]:
    """
    Select up to three actions (a ternary route) based on a Gini‑adjusted
    expected value score.  This combines the ternary routing idea (Parent B)
    with the Gini coefficient (Parent A).

    Parameters
    ----------
    actions : list[Hashable]
        Candidate actions after pruning.
    values : np.ndarray
        Expected values for the original action set.
    weights : np.ndarray
        Posterior weights for the original action set.
    gini : float
        Gini coefficient of the weight distribution.

    Returns
    -------
    list[Hashable]
        Up to three chosen actions.
    """
    # Score = weight * value, penalised by Gini (higher inequality → lower score)
    penalty = 1.0 - gini  # larger Gini → smaller penalty factor
    scores = weights * values * penalty

    # Map scores back to the surviving actions
    action_score_map = {act: sc for act, sc in zip(actions, scores) if act in actions}
    # Sort by descending score and pick top three
    selected = sorted(action_score_map, key=action_score_map.get, reverse=True)[:3]
    return selected


def optimize_hybrid_decision(
    actions: list[Hashable],
    points: np.ndarray,
    values: np.ndarray,
    risks: np.ndarray,
    epistemic_flags: list[str],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: int | str | None = None,
) -> dict:
    """
    End‑to‑end hybrid optimisation (demonstrates the fused topology).

    Returns a dictionary with:
        - 'weights': posterior weights
        - 'gini': Gini coefficient of the weights
        - 'survivors': actions after pruning
        - 'angle': Fisher localisation angle (radians)
        - 'route': ternary route (list of actions)
    """
    # 1. Hybrid posterior weights
    weights = compute_hybrid_weights(actions, values, risks, epistemic_flags)

    # 2. Gini coefficient (Parent A)
    gini = gini_coefficient(weights)

    # 3. Prune according to decreasing‑rate schedule (Parent B)
    survivors = prune_hybrid_actions(actions, weights, t, lam, alpha, seed)

    # 4. Fisher localisation on surviving actions
    survivor_mask = np.isin(actions, survivors)
    angle = fisher_angle_localization(points[survivor_mask], weights[survivor_mask])

    # 5. Ternary route selection
    route = ternary_route_selection(survivors, values, weights, gini)

    return {
        "weights": weights,
        "gini": gini,
        "survivors": survivors,
        "angle": angle,
        "route": route,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy dataset
    np.random.seed(0)
    num_actions = 10
    actions = [f"act_{i}" for i in range(num_actions)]
    points = np.random.uniform(-10, 10, size=(num_actions, 2))
    values = np.random.uniform(0, 100, size=num_actions)
    risks = np.random.uniform(0, 5, size=num_actions)  # higher = more regret
    flags = random.choices(
        ["FACT", "PROBABLE", "POSSIBLE", "SURE_MAYBE", "BULLSHIT"], k=num_actions
    )

    result = optimize_hybrid_decision(
        actions=actions,
        points=points,
        values=values,
        risks=risks,
        epistemic_flags=flags,
        t=3.0,
        lam=1.2,
        alpha=0.15,
        seed=42,
    )

    print("Posterior weights:", result["weights"])
    print("Gini coefficient:", result["gini"])
    print("Surviving actions after pruning:", result["survivors"])
    print("Fisher localisation angle (rad):", result["angle"])
    print("Selected ternary route:", result["route"])