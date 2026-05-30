# DARWIN HAMMER — match 3137, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_regret_engine_m1429_s1.py (gen3)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_pherom_m330_s0.py (gen4)
# born: 2026-05-29T23:48:03Z

"""Hybrid Calendar‑Regret‑Hoeffding‑Pheromone Engine

Parents:
- hybrid_hybrid_doomsday_cale_hybrid_regret_engine_m1429_s1.py (weekday Sakamoto + Gini‑based regret weighting)
- hybrid_hybrid_hoeffding_tre_hybrid_hybrid_pherom_m330_s0.py (Hoeffding bound + Gini impurity + pheromone decay)

Mathematical bridge:
Both parents expose a *Gini* measure – in the calendar side it quantifies unevenness (regret),
in the Hoeffding side it quantifies impurity of a split.  This implementation uses the
Gini value of regret‑weighted action scores as the impurity fed to a Hoeffding bound.
A pheromone system supplies a time‑decaying confidence signal that modulates the
splitting decision.  The resulting hybrid can rank weekdays, compute regret‑aware
values, and decide (via Hoeffding‑Gini) whether a further partition of the calendar
is statistically justified, all while pheromones bias the choice toward historically
promising candidates.
"""

from __future__ import annotations

import datetime as _dt
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """An action associated with a calendar day."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class Counterfactual:
    """What could have happened for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class Day:
    """Simple container for a weekday (0=Mon … 6=Sun) and its occurrence count."""
    weekday: int
    count: int


# ----------------------------------------------------------------------
# Core mathematics from Parent A
# ----------------------------------------------------------------------
def weekday_sakamoto(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised Sakamoto algorithm returning weekday numbers 0=Sun … 6=Sat."""
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)
    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    # Convert to ISO Monday‑based (0=Mon … 6=Sun) for easier downstream handling
    return ((w + 6) % 7).astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """Classic Gini coefficient for a non‑negative vector."""
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


def compute_regret_weighted_strategy(
    actions: List[Action],
    counterfactuals: List[Counterfactual],
) -> Dict[str, float]:
    """
    Regret‑weighted value for each action:
        V = EV - cost - risk - Σ_cf(outcome·probability)
    """
    if not actions:
        return {}

    # map counterfactual contribution per action
    cf_map = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}

    result: Dict[str, float] = {}
    for a in actions:
        regret = cf_map.get(a.id, 0.0)
        result[a.id] = a.expected_value - a.cost - a.risk - regret
    return result


# ----------------------------------------------------------------------
# Core mathematics from Parent B
# ----------------------------------------------------------------------
def hoeffding_epsilon(n: int, R: float = 1.0, delta: float = 0.05) -> float:
    """
    Hoeffding bound epsilon = sqrt( (R^2 * ln(1/delta)) / (2 * n) )
    R is the range of the random variable (for Gini impurity R = 1).
    """
    if n <= 0:
        return float("inf")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


class HybridPheromoneSystem:
    """
    Minimal pheromone manager: each surface (identified by a key) holds a
    signal value that decays exponentially with a configurable half‑life.
    """

    def __init__(self):
        self._store: Dict[str, Tuple[float, float, _dt.datetime]] = {}
        # mapping: key → (signal_value, half_life_seconds, last_update)

    def update(self, key: str, signal_value: float, half_life_seconds: float) -> None:
        now = _dt.datetime.utcnow()
        if key in self._store:
            prev_val, prev_hl, prev_time = self._store[key]
            elapsed = (now - prev_time).total_seconds()
            decayed = prev_val * (0.5 ** (elapsed / prev_hl))
            new_val = decayed + signal_value
        else:
            new_val = signal_value
        self._store[key] = (new_val, half_life_seconds, now)

    def get(self, key: str) -> float:
        """Return the current (decayed) signal value; missing keys give 0."""
        now = _dt.datetime.utcnow()
        if key not in self._store:
            return 0.0
        val, half_life, ts = self._store[key]
        elapsed = (now - ts).total_seconds()
        decayed = val * (0.5 ** (elapsed / half_life))
        # store the decayed version for future calls
        self._store[key] = (decayed, half_life, now)
        return decayed


# ----------------------------------------------------------------------
# Hybrid operations (three+ functions)
# ----------------------------------------------------------------------
def evaluate_split_gini(
    scores: np.ndarray,
    n_samples: int,
    pheromone_system: HybridPheromoneSystem,
    surface_key: str,
) -> Tuple[bool, float, float, str]:
    """
    Decide whether to split a set of scores based on:
      * Gini impurity of the scores (treated as a distribution)
      * Hoeffding epsilon for the current sample size
      * A pheromone confidence signal that lowers the required gain gap

    Returns a tuple (should_split, epsilon, gain_gap, reason).
    """
    # 1. Gini impurity (0 = pure, 0.5 = maximally impure for binary)
    # Convert raw scores to a probability distribution via softmax‑like scaling.
    if scores.size == 0:
        return False, 0.0, 0.0, "no data"
    prob = scores - scores.min()  # shift to non‑negative
    if prob.sum() == 0:
        prob = np.ones_like(prob) / prob.size
    else:
        prob = prob / prob.sum()
    gini = 1.0 - np.sum(prob ** 2)

    # 2. Hoeffding epsilon
    eps = hoeffding_epsilon(n_samples)

    # 3. Pheromone signal (higher → more trust → lower required gain)
    pheromone = pheromone_system.get(surface_key)

    # 4. Required gain gap: we demand that impurity reduction exceeds epsilon,
    #    scaled down by pheromone (capped between 0.1 and 1.0)
    pheromone_factor = max(0.1, 1.0 - pheromone)  # pheromone near 1 → factor ~0.1
    required_gap = eps * pheromone_factor

    # 5. Simulated gain (difference between current Gini and an optimistic split)
    #    Here we approximate the best possible split as halving the impurity.
    simulated_gain = gini / 2.0

    should = simulated_gain > required_gap
    reason = (
        f"gini={gini:.4f}, eps={eps:.4f}, pheromone={pheromone:.4f}, "
        f"required_gap={required_gap:.4f}, gain={simulated_gain:.4f}"
    )
    return should, eps, simulated_gain, reason


def hybrid_calendar_regret_tree(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    actions: List[Action],
    counterfactuals: List[Counterfactual],
    pheromone_system: HybridPheromoneSystem,
) -> Dict[int, Dict]:
    """
    End‑to‑end hybrid pipeline:

    1. Compute weekday for each (year, month, day).
    2. Aggregate actions per weekday.
    3. Compute regret‑weighted values per action.
    4. For each weekday, build a score vector (regret‑weighted values of its actions)
       and ask ``evaluate_split_gini`` whether this weekday merits a further split.
    5. Return a dictionary keyed by weekday containing the raw scores,
       Gini, split decision and the explanatory reason string.
    """
    wk = weekday_sakamoto(years, months, days)

    # Group indices by weekday
    weekday_groups: Dict[int, List[int]] = {i: [] for i in range(7)}
    for idx, w in enumerate(wk):
        weekday_groups[int(w)].append(idx)

    # Regret‑weighted values (global, not per date)
    regret_map = compute_regret_weighted_strategy(actions, counterfactuals)

    # Build result container
    result: Dict[int, Dict] = {}

    for wd, idxs in weekday_groups.items():
        # Gather scores of actions that appear on these dates (mocked by random selection)
        # For demonstration we sample a subset of actions proportional to the number of dates.
        n_dates = len(idxs)
        sample_size = min(len(actions), max(1, n_dates))
        sampled_actions = random.sample(actions, sample_size)
        scores = np.array([regret_map[a.id] for a in sampled_actions], dtype=np.float64)

        # Update pheromone signal: each evaluation contributes a unit signal
        surface_key = f"weekday_{wd}"
        pheromone_system.update(surface_key, signal_value=1.0, half_life_seconds=300.0)

        should_split, eps, gain, reason = evaluate_split_gini(
            scores, n_samples=n_dates + 1, pheromone_system=pheromone_system, surface_key=surface_key
        )

        result[wd] = {
            "date_count": n_dates,
            "action_count": len(sampled_actions),
            "scores": scores.tolist(),
            "gini": 1.0 - np.sum((scores - scores.min()) / (scores.max() - scores.min() + 1e-12) ** 2),
            "should_split": should_split,
            "epsilon": eps,
            "gain": gain,
            "reason": reason,
        }

    return result


def aggregate_weekday_counts(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> List[Day]:
    """
    Simple helper that counts occurrences of each weekday in the supplied dates.
    Returns a list of ``Day`` objects sorted by weekday.
    """
    wk = weekday_sakamoto(years, months, days)
    counts = np.bincount(wk, minlength=7)
    return [Day(weekday=i, count=int(c)) for i, c in enumerate(counts)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate a random set of dates (one year span)
    rng = np.random.default_rng(42)
    n_dates = 200
    years = rng.integers(2020, 2023, size=n_dates)
    months = rng.integers(1, 13, size=n_dates)
    days = rng.integers(1, 28, size=n_dates)  # safe for all months

    # 2. Create dummy actions and counterfactuals
    actions = [
        Action(id=f"A{i}", expected_value=rng.random() * 100, cost=rng.random() * 10, risk=rng.random() * 5)
        for i in range(15)
    ]
    counterfactuals = [
        Counterfactual(action_id=a.id, outcome_value=rng.random() * 20, probability=rng.random())
        for a in actions
    ]

    # 3. Initialise pheromone system
    pheromone_system = HybridPheromoneSystem()

    # 4. Run the hybrid pipeline
    calendar_result = hybrid_calendar_regret_tree(
        years=years,
        months=months,
        days=days,
        actions=actions,
        counterfactuals=counterfactuals,
        pheromone_system=pheromone_system,
    )

    # 5. Print a concise summary
    print("Weekday summary (Mon=0 … Sun=6):")
    for wd in range(7):
        info = calendar_result[wd]
        print(
            f"  {wd}: dates={info['date_count']}, actions={info['action_count']}, "
            f"split={info['should_split']}, reason={info['reason']}"
        )

    # 6. Demonstrate the simple count helper
    counts = aggregate_weekday_counts(years, months, days)
    print("\nRaw weekday counts:")
    for d in counts:
        print(f"  Weekday {d.weekday}: {d.count} occurrences")