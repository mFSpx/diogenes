# DARWIN HAMMER — match 121, survivor 0
# gen: 3
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# born: 2026-05-29T23:25:44Z

"""Hybrid algorithm merging the Doomsday‑Calendar Gini analysis (Parent A) with the
Bandit‑based decision engine (Parent B).

Mathematical bridge:
- Parent A provides a 7‑element weekday count vector `c` and its Gini coefficient
  `G(c)`. From `c` we construct a 7×7 weighted‑difference matrix `W = outer(c,c) *
  |i‑j|`. This matrix (or its flattened vector) serves as a high‑dimensional
  context for the bandit algorithm.
- Parent B uses a context‑aware LinUCB‑style surrogate where the exploration
  bonus is scaled by `sqrt(∑ context_i²)`. By feeding `vec(W)` as the context,
  the bandit’s confidence bounds become a function of the weekday distribution.
- The reward fed back to the bandit is `R = 1 - G(c)`, i.e. higher reward for a
  more uniform weekday spread.

The module therefore fuses both topologies: matrix‑based statistics from the
calendar side drive the context and reward of the bandit side, creating a
single unified learning loop.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from pathlib import Path
from typing import Iterable, Tuple, Union, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Doomsday calendar utilities
# ----------------------------------------------------------------------


def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """Return weekday numbers (Mon=0 … Sun=6) for vectorised dates."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift so that Sunday becomes 0, Monday 1, … Saturday 6 (as in parent)
    return (py_weekday + 1) % 7


def weekday_counts(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
) -> np.ndarray:
    """Count occurrences of each weekday for an iterable of dates."""
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)

    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)

    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)


def gini_coefficient_numpy(values: np.ndarray) -> float:
    """Standard Gini coefficient for a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator


def weekday_gini(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """Gini coefficient of the weekday count distribution."""
    counts = weekday_counts(dates)
    return gini_coefficient_numpy(counts)


def weekday_gini_matrix(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
) -> np.ndarray:
    """
    Produce a 7×7 matrix `W` where

        W[i, j] = c_i * c_j * |i - j|

    with `c` the weekday count vector.
    """
    counts = weekday_counts(dates).astype(float)
    weight = np.outer(counts, counts)                # c_i * c_j
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])       # |i - j|
    weighted_diff = weight * diff
    return weighted_diff


# ----------------------------------------------------------------------
# Parent B – Bandit core (adapted)
# ----------------------------------------------------------------------


# Global mutable stores (mirroring parent B)
_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]
_STORE: Dict[str, float] = {}          # optional virtual VRAM store (unused here)


def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(action_id: str) -> float:
    """Empirical mean reward for an action, 0 if never selected."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: np.ndarray,
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> Tuple[str, float, float, float]:
    """
    Choose an action given a numeric context vector.

    Returns a tuple (action_id, propensity, expected_reward, confidence_bound).
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Exploration / exploitation decision
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Simple Beta‑Bernoulli posterior sampling
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0.0, _reward(a)),
                1 + max(0.0, 1 - _reward(a)),
            ),
        )
    else:  # default LinUCB‑style surrogate
        # Scale factor derived from the context norm (the mathematical bridge)
        scale = math.sqrt(float(np.sum(context.astype(float) ** 2))) if context.size else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return chosen, propensity, _reward(chosen), confidence


def update_policy(action_id: str, reward: float, propensity: float) -> None:
    """Incorporate observed reward for a given action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, n + 1]
    # The virtual store is kept for API compatibility; we simply mirror the reward.
    _STORE[action_id] = _STORE.get(action_id, 0.0) + reward * propensity


# ----------------------------------------------------------------------
# Hybrid operations (demonstrating the fused system)
# ----------------------------------------------------------------------


def hybrid_context_vector(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Build the bandit context vector from calendar data.

    The vector is the flattened weighted‑difference matrix defined in
    `weekday_gini_matrix`. Its L2 norm directly influences the LinUCB exploration
    term.
    """
    mat = weekday_gini_matrix(dates)
    return mat.ravel()


def hybrid_reward_from_dates(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """
    Compute a scalar reward for the bandit based on weekday uniformity.

    Reward = 1 - Gini(weekday_counts).  A perfectly uniform distribution yields
    reward 1, while a highly skewed distribution yields values near 0.
    """
    return 1.0 - weekday_gini(dates)


def hybrid_bandit_step(
    dates: Iterable[Union[dt.date, Tuple[int, int, int]]],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> Tuple[Tuple[str, float, float, float], float]:
    """
    Perform one complete bandit iteration using calendar‑derived context and reward.

    Returns:
        (action_id, propensity, expected_reward, confidence_bound), reward_observed
    """
    ctx = hybrid_context_vector(dates)
    action_id, propensity, exp_reward, confidence = select_action(
        ctx, actions, algorithm=algorithm, epsilon=epsilon, seed=seed
    )
    reward = hybrid_reward_from_dates(dates)
    update_policy(action_id, reward, propensity)
    return (action_id, propensity, exp_reward, confidence), reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate a modest list of dates spanning a few months
    start = dt.date(2023, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(120)]

    # Define a small action space
    actions = ["alpha", "beta", "gamma", "delta"]

    # Reset any prior state
    reset_policy()

    # Run a few hybrid bandit steps
    for step in range(5):
        result, rew = hybrid_bandit_step(dates, actions, algorithm="linucb", seed=step)
        action_id, propensity, exp_reward, confidence = result
        print(
            f"Step {step+1}: chosen={action_id}, prop={propensity:.2f}, "
            f"exp_reward={exp_reward:.3f}, confidence={confidence:.3f}, reward={rew:.3f}"
        )

    # Display final policy statistics
    print("\nFinal policy statistics:")
    for a in actions:
        tot, cnt = _POLICY.get(a, [0.0, 0.0])
        print(f"  {a}: count={int(cnt)}, mean_reward={_reward(a):.3f}")