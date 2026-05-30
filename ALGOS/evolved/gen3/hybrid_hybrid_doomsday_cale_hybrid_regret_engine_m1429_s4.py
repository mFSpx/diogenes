# DARWIN HAMMER — match 1429, survivor 4
# gen: 3
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s4.py (gen1)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s0.py (gen2)
# born: 2026-05-29T23:36:20Z

import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class Action:
    """Elementary decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class Counterfactual:
    """Alternative outcome used for regret computation."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _aggregate_weekday_counts(year: int, month: int, num_days: int) -> np.ndarray:
    """
    Return an array ``c`` of length 7 where ``c[w]`` is the number of
    occurrences of weekday ``w`` (Monday=0 … Sunday=6) in the
    ``num_days``‑long window starting at ``year‑month‑1``.
    """
    if num_days <= 0:
        raise ValueError("num_days must be positive")
    # Build a list of weekdays for the requested range
    weekdays = [
        dt.date(year, month, day).weekday()
        for day in range(1, num_days + 1)
    ]
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(float)


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient for a non‑negative 1‑D array.
    Returns 0 for a degenerate distribution (all zeros).
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is defined for non‑negative values only")
    total = arr.sum()
    if total == 0:
        return 0.0
    # Sort and apply the classic Gini formula
    sorted_arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_arr)) / (n * total)
    return float(gini)


def compute_regret_weights(
    actions: List[Action],
    counterfactuals: List[Counterfactual],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Return a probability distribution over ``actions`` derived from
    regret‑adjusted utilities.

    The utility of an action ``a`` is
        u(a) = EV(a) – cost(a) – risk(a) + Σ_c outcome_c·p_c
    where the sum runs over counterfactuals that refer to ``a``.
    A soft‑max (with optional temperature) converts utilities into a
    proper distribution.
    """
    if not actions:
        return {}

    # Map counterfactual contributions to actions
    cf_contrib = {
        cf.action_id: cf.outcome_value * cf.probability
        for cf in counterfactuals
    }

    # Compute utilities
    utilities = np.array([
        a.expected_value - a.cost - a.risk + cf_contrib.get(a.id, 0.0)
        for a in actions
    ], dtype=float)

    # Numerical stability: subtract max before exponentiation
    max_u = utilities.max()
    exp_vals = np.exp((utilities - max_u) / max(temperature, 1e-12))
    probs = exp_vals / exp_vals.sum()

    return {a.id: float(p) for a, p in zip(actions, probs)}


def rank_actions_by_ev(actions: List[Action]) -> List[Action]:
    """Return actions sorted descending by net expected value."""
    return sorted(
        actions,
        key=lambda a: (-(a.expected_value - a.cost - a.risk), a.id)
    )


def regret_weighted_gini(
    year: int,
    month: int,
    num_days: int,
    regret_weights: Dict[str, float],
) -> float:
    """
    Compute a Gini coefficient where each weekday count is weighted
    by the regret weight of the corresponding day identifier.

    ``regret_weights`` must contain exactly ``num_days`` entries keyed
    by ``day{i}`` where ``i`` runs from 0 to ``num_days‑1``.
    """
    if len(regret_weights) != num_days:
        raise ValueError(
            f"regret_weights must contain {num_days} entries, got {len(regret_weights)}"
        )

    # Raw weekday frequencies
    raw_counts = _aggregate_weekday_counts(year, month, num_days)  # shape (7,)

    # Build a weight vector aligned with the chronological order of days
    day_weights = np.array(
        [regret_weights[f"day{i}"] for i in range(num_days)], dtype=float
    )

    # Distribute the chronological weights onto weekdays
    weekday_weights = np.zeros(7, dtype=float)
    for i, weight in enumerate(day_weights):
        wd = dt.date(year, month, i + 1).weekday()
        weekday_weights[wd] += weight

    # Weighted counts = raw frequency * aggregated regret weight per weekday
    weighted_counts = raw_counts * weekday_weights

    return gini_coefficient(weighted_counts)


def hybrid_regret_weighted_doomsday(
    year: int,
    month: int,
    num_days: int,
    temperature: float = 1.0,
) -> Tuple[float, List[Action]]:
    """
    End‑to‑end pipeline:

    1. Create a dummy ``Action`` for each calendar day.
    2. Compute a regret‑weighted probability distribution over those actions.
    3. Apply the distribution as weights to the weekday frequency vector.
    4. Return the Gini coefficient of the weighted weekday distribution
       together with the ranked actions (for downstream inspection).

    The function is deliberately deterministic given the same inputs
    because the underlying utilities are all zero; only the soft‑max
    temperature influences the shape of the distribution.
    """
    # 1. Dummy actions – all utilities are zero, allowing the temperature
    #    to shape the distribution.
    actions = [Action(id=f"day{i}", expected_value=0.0) for i in range(num_days)]

    # 2. Counterfactuals mirror the actions with unit outcome.
    counterfactuals = [
        Counterfactual(action_id=f"day{i}", outcome_value=0.0, probability=1.0)
        for i in range(num_days)
    ]

    regret_weights = compute_regret_weights(
        actions, counterfactuals, temperature=temperature
    )

    # 3. Compute weighted Gini.
    gini = regret_weighted_gini(year, month, num_days, regret_weights)

    # 4. Provide a ranking for external use.
    ranked_actions = rank_actions_by_ev(actions)

    return gini, ranked_actions


if __name__ == "__main__":
    # Demo run – prints the Gini coefficient for the current month.
    today = dt.date.today()
    gini_val, _ = hybrid_regret_weighted_doomsday(
        year=today.year,
        month=today.month,
        num_days=30,
        temperature=0.5,
    )
    print(f"Regret‑weighted Gini (30‑day window): {gini_val:.6f}")