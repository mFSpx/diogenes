# DARWIN HAMMER — match 19, survivor 7
# gen: 2
# parent_a: regret_engine.py (gen0)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py (gen1)
# born: 2026-05-29T23:25:18Z

from __future__ import annotations

import math
import random
import sys
import pathlib
import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np

# ---------- Parent A structures ----------
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

# ---------- Parent B utilities ----------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) using Doomsday algorithm."""
    return (dt.date(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

# ---------- Hybrid core ----------
def _weekday_sequence(year: int, month: int, num_days: int) -> list[int]:
    """Generate a list of weekday indices for the first *num_days* of a month."""
    return [doomsday(year, month, day) for day in range(1, num_days + 1)]

def _map_actions_to_weekdays(
    actions: list[MathAction],
    year: int,
    month: int,
    num_days: int,
) -> dict[str, int]:
    """
    Assign each action to a weekday by cycling through the month’s weekday sequence.
    Returns a mapping action_id → weekday_index.
    """
    if not actions:
        return {}
    weekdays = _weekday_sequence(year, month, num_days)
    mapping = {}
    for idx, act in enumerate(actions):
        mapping[act.id] = weekdays[idx % len(weekdays)]
    return mapping

def compute_hybrid_regret_weighted_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """
    1. Compute the classic regret‑weighted probabilities (parent A).
    2. Distribute those probabilities over weekdays using the Doomsday mapping (parent B).
    3. Evaluate the inequality of that weekday distribution via the Gini coefficient.
    4. Regularise the raw probabilities with the Gini factor:
           p'_i = p_i * (1 - G) + G / N   (N = number of actions)
       This keeps the total mass = 1 while damping extreme concentration.
    Returns a normalized probability dictionary keyed by action id.
    """
    if not actions:
        return {}

    # ---- Step 1: regret‑weighted raw probabilities ----
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0)
        for a in actions
    }
    best = max(vals.values())
    raw_weights = {k: math.exp(v - best) for k, v in vals.items()}
    total_raw = sum(raw_weights.values()) or 1.0
    raw_probs = {k: v / total_raw for k, v in raw_weights.items()}

    # ---- Step 2: map actions → weekdays and accumulate weighted counts ----
    mapping = _map_actions_to_weekdays(actions, year, month, num_days)
    weekday_counts = np.zeros(7, dtype=float)
    for act_id, prob in raw_probs.items():
        wd = mapping[act_id]
        weekday_counts[wd] += prob

    # ---- Step 3: Gini of the weekday probability distribution ----
    gini = gini_coefficient(weekday_counts)

    # ---- Step 4: regularise raw probabilities with Gini ----
    n = len(actions)
    regularised = {
        k: prob * (1 - gini) + gini / n for k, prob in raw_probs.items()
    }
    # Clip probabilities to ensure they are within valid range
    regularised = {
        k: max(epsilon, min(1 - epsilon, v)) for k, v in regularised.items()
    }
    # Normalise again to guard against floating‑point drift
    total_reg = sum(regularised.values()) or 1.0
    return {k: v / total_reg for k, v in regularised.items()}

def rank_hybrid_actions(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> list[MathAction]:
    """
    Rank actions by the hybrid probabilities (descending). Ties are broken by action id.
    """
    probs = compute_hybrid_regret_weighted_strategy(
        actions, counterfactuals, year, month, num_days
    )
    prob_lookup = lambda a: (-probs.get(a.id, 0.0), a.id)
    return sorted(actions, key=prob_lookup)

def gini_of_weighted_weekdays(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    year: int,
    month: int,
    num_days: int,
) -> float:
    """
    Expose the Gini coefficient of the weekday distribution derived from the
    regret‑weighted probabilities (before regularisation). Useful for diagnostics.
    """
    if not actions:
        return 0.0
    # raw regret probabilities
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0)
        for a in actions
    }
    best = max(vals.values())
    raw_weights = {k: math.exp(v - best) for k, v in vals.items()}
    total_raw = sum(raw_weights.values()) or 1.0
    raw_probs = {k: v / total_raw for k, v in raw_weights.items()}

    mapping = _map_actions_to_weekdays(actions, year, month, num_days)
    weekday_counts = np.zeros(7, dtype=float)
    for act_id, prob in raw_probs.items():
        wd = mapping[act_id]
        weekday_counts[wd] += prob
    return gini_coefficient(weekday_counts)

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