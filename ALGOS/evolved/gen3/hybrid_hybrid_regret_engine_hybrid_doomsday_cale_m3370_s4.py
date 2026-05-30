# DARWIN HAMMER — match 3370, survivor 4
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_doomsday_cale_m19_s0.py (gen2)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s2.py (gen1)
# born: 2026-05-29T23:49:41Z

"""Hybrid Regret‑Weighted Doomsday‑Gini Engine
Parents:
- hybrid_regret_engine_hybrid_doomsday_cale_m19_s0.py (Regret‑weighted strategy + EV ranking + Doomsday weekday mapping + Gini)
- hybrid_doomsday_calendar_gini_coefficient_m49_s2.py (Vectorised Doomsday + Gini on weekday frequencies)

Mathematical bridge:
The Doomsday mapping converts calendar dates into a discrete action space
W = {0,…,6} (Sunday…Saturday).  Regret‑weighted strategy assigns each action a
utility u_w = EV_w – cost_w – risk_w + counterfactual_w.  The soft‑max‑like
weights p_w = exp(u_w – max(u)) / Σ exp(·) are a probability distribution over
W.  The Gini coefficient G(p) quantifies the inequality of that distribution.
Thus the weekday frequency vector C (from Doomsday) feeds the regret computation,
and the resulting weight vector feeds the Gini computation – a closed loop that
fuses both parent topologies into a single unified system.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    id: str               # identifier, e.g. weekday name or index
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class Counterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Parent B – Vectorised Doomsday
# ----------------------------------------------------------------------
def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Doomsday calculation.
    Returns an array of weekday indices where 0 = Sunday … 6 = Saturday.
    Mirrors (date.weekday() + 1) % 7 for each (year, month, day) triple.
    """
    # Build a datetime64[D] array
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    flat = dates.ravel()
    # Convert each element to Python datetime to obtain weekday
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday()
         for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

# ----------------------------------------------------------------------
# Parent A – Regret‑Weighted Strategy (numpy‑enabled)
# ----------------------------------------------------------------------
def compute_regret_weights_numpy(
    actions: list[Action],
    counterfactuals: list[Counterfactual],
) -> dict[str, float]:
    """
    Compute regret‑weighted probabilities for a list of actions using NumPy
    for vectorised arithmetic.  Mirrors compute_regret_weighted_strategy from
    Parent A but operates on NumPy arrays.
    """
    if not actions:
        return {}
    # Map counterfactuals to a dict for fast lookup
    cf_dict = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    # Extract vectors
    ev = np.array([a.expected_value for a in actions], dtype=np.float64)
    cost = np.array([a.cost for a in actions], dtype=np.float64)
    risk = np.array([a.risk for a in actions], dtype=np.float64)
    cf_vals = np.array([cf_dict.get(a.id, 0.0) for a in actions], dtype=np.float64)
    # Regret‑adjusted values
    vals = ev - cost - risk + cf_vals
    best = np.max(vals)
    # Soft‑max‑like weighting (exp of regret difference)
    w = np.exp(vals - best)
    total = w.sum() if w.sum() != 0 else 1.0
    probs = w / total
    return {a.id: float(p) for a, p in zip(actions, probs)}

# ----------------------------------------------------------------------
# Hybrid core: combine weekday frequencies, regret weighting, and Gini
# ----------------------------------------------------------------------
def gini_coefficient_numpy(values: np.ndarray) -> float:
    """
    Vectorised Gini coefficient for a non‑negative 1‑D array.
    Implements G = Σ (2·i – n – 1)·x_i / (n·Σ x_i) where x_i are sorted.
    """
    if values.size == 0:
        return 0.0
    xs = np.sort(values.astype(np.float64))
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    cumulative = (2 * np.arange(1, n + 1) - n - 1) * xs
    denominator = n * xs.sum()
    return float(cumulative.sum() / denominator) if denominator != 0 else 0.0

def weekday_counts_numpy(year: int, month: int, num_days: int) -> np.ndarray:
    """
    Compute the frequency of each weekday (0=Sunday … 6=Saturday) for the
    first ``num_days`` of a given month using the vectorised Doomsday.
    Returns a length‑7 integer array.
    """
    years = np.full(num_days, year, dtype=np.int32)
    months = np.full(num_days, month, dtype=np.int32)
    days = np.arange(1, num_days + 1, dtype=np.int32)
    weekdays = doomsday_numpy(years, months, days)
    counts = np.bincount(weekdays, minlength=7)
    return counts

def hybrid_regret_gini(
    year: int,
    month: int,
    num_days: int,
    actions: list[Action],
    counterfactuals: list[Counterfactual],
) -> tuple[dict[str, float], float]:
    """
    Full hybrid pipeline:
    1. Derive weekday frequencies for the specified calendar slice.
    2. Treat each weekday as an action; augment its expected value with the
       observed frequency (interpreted as a contextual payoff).
    3. Compute regret‑weighted probabilities over the actions.
    4. Return the Gini coefficient of the resulting probability distribution.

    The returned tuple is (probability_dict, gini_value).
    """
    # Step 1 – weekday frequencies
    freq = weekday_counts_numpy(year, month, num_days)          # shape (7,)
    # Step 2 – build a mapping from weekday index to Action (if missing, create dummy)
    action_map = {a.id: a for a in actions}
    enriched_actions: list[Action] = []
    for w in range(7):
        wid = str(w)                                            # use string keys for compatibility
        base = action_map.get(wid, Action(id=wid, expected_value=0.0))
        # Incorporate frequency as an additive expected value component
        enriched = Action(
            id=base.id,
            expected_value=base.expected_value + float(freq[w]),
            cost=base.cost,
            risk=base.risk,
        )
        enriched_actions.append(enriched)

    # Step 3 – regret‑weighted probabilities
    probs = compute_regret_weights_numpy(enriched_actions, counterfactuals)

    # Step 4 – Gini of the probability vector
    prob_array = np.array([probs[str(w)] for w in range(7)], dtype=np.float64)
    gini_val = gini_coefficient_numpy(prob_array)

    return probs, gini_val

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample month: May 2026 (31 days)
    YEAR = 2026
    MONTH = 5
    DAYS_IN_MONTH = 31

    # Define actions for each weekday (id = weekday index as string)
    base_actions = [
        Action(id=str(w), expected_value=random.uniform(10, 20), cost=random.uniform(0, 3), risk=random.uniform(0, 2))
        for w in range(7)
    ]

    # Define some counterfactuals (optional)
    cf = [
        Counterfactual(action_id=str(w), outcome_value=random.uniform(-5, 5), probability=0.8)
        for w in range(7)
    ]

    probs, gini_val = hybrid_regret_gini(YEAR, MONTH, DAYS_IN_MONTH, base_actions, cf)

    print("Regret‑weighted probabilities per weekday (0=Sun … 6=Sat):")
    for w in range(7):
        print(f"  Weekday {w}: {probs[str(w)]:.4f}")
    print(f"\nGini coefficient of the probability distribution: {gini_val:.4f}")

    # Simple sanity checks
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 1e-9, "Probabilities must sum to 1"
    assert 0.0 <= gini_val <= 1.0, "Gini must be in [0,1]"

    print("\nSmoke test completed successfully.")