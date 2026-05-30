# DARWIN HAMMER — match 19, survivor 9
# gen: 2
# parent_a: regret_engine.py (gen0)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py (gen1)
# born: 2026-05-29T23:25:18Z

from __future__ import annotations

import math
import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Dict

import numpy as np


# ---------- Parent A structures ----------
@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True, slots=True)
class MathCounterfactual:
    """Alternative outcome information for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ---------- Parent B utilities ----------
def weekday_index(year: int, month: int, day: int) -> int:
    """
    Return the ISO weekday index (0 = Monday … 6 = Sunday) for a given date.
    This replaces the buggy ``doomsday`` implementation that shifted the index.
    """
    return dt.date(year, month, day).weekday()


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient for a non‑negative distribution.
    Returns 0 for an empty or all‑zero input.
    """
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ---------- Improved Hybrid core ----------
def _weekday_sequence(year: int, month: int, num_days: int) -> list[int]:
    """
    Produce a list of weekday indices for the first *num_days* of a month.
    The length is exactly *num_days*; each entry is in 0‑6.
    """
    return [weekday_index(year, month, day) for day in range(1, num_days + 1)]


def _balanced_weekday_mapping(actions: List[MathAction]) -> Dict[str, int]:
    """
    Assign each action to a weekday in a way that guarantees the most even
    possible distribution regardless of the number of actions.
    The mapping cycles over the 7 weekdays (0‑6) rather than over the month
    length, eliminating the bias introduced by ``_map_actions_to_weekdays``.
    """
    mapping: Dict[str, int] = {}
    for i, act in enumerate(actions):
        mapping[act.id] = i % 7
    return mapping


def _regret_weighted_probs(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Classic regret‑weighted softmax probabilities (Parent A).
    """
    cf_lookup = {
        c.action_id: c.outcome_value * c.probability for c in counterfactuals
    }
    scores = {
        a.id: a.expected_value - a.cost - a.risk + cf_lookup.get(a.id, 0.0)
        for a in actions
    }
    best = max(scores.values())
    # use exp(score - best) for numerical stability
    raw = {k: math.exp(v - best) for k, v in scores.items()}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def compute_hybrid_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> Dict[str, float]:
    """
    Deeply fused hybrid strategy.

    1. Compute regret‑weighted probabilities (Parent A).
    2. Map actions to weekdays using a *balanced* assignment (improved bridge).
    3. Derive the weekday probability vector and its Gini coefficient.
    4. Use the Gini as a *dynamic regularisation strength* (α) that
       modulates a convex combination between the raw distribution and a
       perfectly uniform distribution over actions.
       α = G^β with β≥1 makes the regulariser more sensitive to high inequality.
    5. Return a properly normalised probability dictionary.
    """
    if not actions:
        return {}

    # ---- Step 1 -----------------------------------------------------------
    raw_probs = _regret_weighted_probs(actions, counterfactuals)

    # ---- Step 2 -----------------------------------------------------------
    mapping = _balanced_weekday_mapping(actions)

    # ---- Step 3 -----------------------------------------------------------
    weekday_probs = np.zeros(7, dtype=float)
    for act_id, prob in raw_probs.items():
        wd = mapping[act_id]
        weekday_probs[wd] += prob
    gini = gini_coefficient(weekday_probs)

    # ---- Step 4 -----------------------------------------------------------
    # β controls how sharply the regulariser reacts to inequality.
    # A value of 2 gives a quadratic response, preserving ordering when G is small.
    beta = 2.0
    alpha = min(gini ** beta, 1.0)  # ensure α∈[0,1]

    n_actions = len(actions)
    uniform_share = 1.0 / n_actions
    regularised = {
        k: prob * (1.0 - alpha) + alpha * uniform_share for k, prob in raw_probs.items()
    }

    # Guard against floating‑point drift
    total = sum(regularised.values()) or 1.0
    return {k: v / total for k, v in regularised.items()}


def rank_hybrid_actions(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> List[MathAction]:
    """
    Return actions sorted by the hybrid probabilities (descending).
    Ties are broken deterministically by action identifier.
    """
    probs = compute_hybrid_regret_weighted_strategy(
        actions, counterfactuals, year, month, num_days
    )
    return sorted(actions, key=lambda a: (-probs.get(a.id, 0.0), a.id))


def gini_of_weighted_weekdays(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> float:
    """
    Diagnostic: Gini coefficient of the weekday distribution derived from the
    *raw* regret‑weighted probabilities (before any regularisation).
    """
    if not actions:
        return 0.0
    raw_probs = _regret_weighted_probs(actions, counterfactuals)
    mapping = _balanced_weekday_mapping(actions)
    weekday_probs = np.zeros(7, dtype=float)
    for act_id, prob in raw_probs.items():
        wd = mapping[act_id]
        weekday_probs[wd] += prob
    return gini_coefficient(weekday_probs)


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Sample actions
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=8.0, cost=1.0, risk=0.5),
        MathAction(id="C", expected_value=6.0, cost=0.5, risk=2.0),
        MathAction(id="D", expected_value=7.0, cost=1.5, risk=1.0),
    ]

    # Sample counterfactuals (some may be missing)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=1.5, probability=0.8),
        MathCounterfactual(action_id="C", outcome_value=-0.5, probability=0.6),
    ]

    year, month, num_days = 2022, 6, 30

    hybrid_probs = compute_hybrid_regret_weighted_strategy(
        actions, counterfactuals, year, month, num_days
    )
    print("Hybrid probabilities:")
    for aid, prob in hybrid_probs.items():
        print(f"  {aid}: {prob:.4f}")

    ranked = rank_hybrid_actions(actions, counterfactuals, year, month, num_days)
    print("\nRanked actions (hybrid):")
    for a in ranked:
        print(f"  {a.id}")

    gini_val = gini_of_weighted_weekdays(actions, counterfactuals, year, month, num_days)
    print(f"\nGini of weekday distribution (pre‑regularisation): {gini_val:.4f}")