# DARWIN HAMMER — match 19, survivor 4
# gen: 2
# parent_a: regret_engine.py (gen0)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py (gen1)
# born: 2026-05-29T23:25:18Z

"""Hybrid Regret‑Weighted Gini Calendar
Parent A: regret_engine.py – computes a softmax‑like probability distribution over actions
               using regret‑adjusted expected values.
Parent B: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py – generates weekday
               counts from the Doomsday calendar and measures their inequality with
               the Gini coefficient.

Mathematical bridge:
    • The regret engine outputs a normalized probability vector **p** over a set of
      actions.
    • The Gini coefficient is a scalar functional G(p) that quantifies the
      inequality of any non‑negative distribution that sums to one.
    • By treating each weekday (or any discrete outcome) as an action whose
      expected value is its raw count from the Doomsday calendar, we can feed the
      resulting probability vector into the Gini calculator.  The hybrid therefore
      (i) builds a regret‑weighted strategy from calendar‑derived values and
      (ii) evaluates the fairness/unevenness of that strategy with the Gini
      coefficient.  The two operations are mathematically fused because the
      Gini is applied directly to the output of the regret‑weighting transform.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np

# ----------------------------------------------------------------------
# Data structures – copied from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with an expected value and optional cost/risk penalties."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A counterfactual adjustment for a specific action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Core algorithms from the two parents
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
) -> dict[str, float]:
    """Parent A – returns a softmax‑like distribution over actions."""
    if not actions:
        return {}
    # apply counterfactual adjustments
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    # regret‑adjusted values
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(vals.values())
    # exponentiate the regret (negative regret => larger weight)
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def gini_coefficient(values: Iterable[float]) -> float:
    """Parent B – Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) for a Gregorian date."""
    return (dt.date(year, month, day).weekday() + 1) % 7  # match Parent B's convention


def weekday_distribution(year: int, month: int, num_days: int) -> np.ndarray:
    """Count occurrences of each weekday in the requested month slice."""
    weekdays = [doomsday(year, month, day) for day in range(1, num_days + 1)]
    counts = np.zeros(7, dtype=int)
    for wd in weekdays:
        counts[wd] += 1
    return counts


# ----------------------------------------------------------------------
# Hybrid layer – three demonstrative functions
# ----------------------------------------------------------------------
def build_weekday_actions(
    year: int, month: int, num_days: int
) -> list[MathAction]:
    """
    Create a MathAction for each weekday (0‑6) where the expected value is the
    raw count of that weekday in the calendar slice.  Cost and risk are set to 0.
    """
    counts = weekday_distribution(year, month, num_days)
    actions: list[MathAction] = []
    for wd in range(7):
        actions.append(
            MathAction(
                id=str(wd),
                expected_value=float(counts[wd]),
                cost=0.0,
                risk=0.0,
            )
        )
    return actions


def hybrid_regret_gini_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual] | None = None,
) -> tuple[dict[str, float], float]:
    """
    1. Compute the regret‑weighted probability distribution over the supplied actions.
    2. Evaluate the Gini coefficient of that distribution (a measure of inequality).
    Returns the probability map and its Gini score.
    """
    cf = counterfactuals or []
    probs = compute_regret_weighted_strategy(actions, cf)
    gini = gini_coefficient(probs.values())
    return probs, gini


def evaluate_calendar_regret_gini(
    year: int, month: int, num_days: int, counterfactuals: list[MathCounterfactual] | None = None
) -> None:
    """
    End‑to‑end demo:
        * Build weekday actions from a Doomsday calendar slice.
        * Apply the hybrid regret‑Gini computation.
        * Print a readable report.
    """
    actions = build_weekday_actions(year, month, num_days)
    probs, gini = hybrid_regret_gini_strategy(actions, counterfactuals)
    # Human‑readable ordering by weekday index
    sorted_items = sorted(probs.items(), key=lambda kv: int(kv[0]))
    print(f"Weekday counts (Mon=0 … Sun=6) for {year}-{month:02d} (first {num_days} days):")
    for wd, prob in sorted_items:
        count = next(a.expected_value for a in actions if a.id == wd)
        print(f"  Weekday {wd}: count={int(count):2d}, regret‑prob={prob:.4f}")
    print(f"\nGini coefficient of the regret‑weighted distribution: {gini:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: May 2023, first 31 days (full month)
    YEAR = 2023
    MONTH = 5
    DAYS = 31
    # No counterfactual adjustments for the demo
    evaluate_calendar_regret_gini(YEAR, MONTH, DAYS)
    # Demonstrate that the Gini of a uniform distribution is 0
    uniform_probs = {str(i): 1 / 7 for i in range(7)}
    print("\nGini of uniform distribution (should be 0.0):",
          gini_coefficient(uniform_probs.values()))