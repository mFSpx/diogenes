# DARWIN HAMMER — match 1433, survivor 5
# gen: 4
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s6.py (gen3)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py (gen3)
# born: 2026-05-29T23:36:26Z

"""Hybrid Doomsday-FoldChange Bandit Module
Parents:
- hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s6.py
- hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py

Mathematical Bridge:
The fold‑change detection (FCD) subsystem produces a response series (x_t, y_t) that
encodes temporal dynamics of an input stream.  We use the slowly varying component
y_t as a *modulation signal* for the NLMS adaptive filter that predicts the
weekday of a calendar date.  Concretely, the RLCT‑adjusted learning rate μ̂ is
scaled by a factor (1+⟨y⟩) where ⟨y⟩ is the average of the y‑component over the
FCD horizon.  This couples the free‑energy penalty of the Real Log‑Canonical
Threshold (via the weight norm) with the dynamical information of the FCD.
The same response series also biases the hybrid bandit router: action
propensities are multiplied by exp(⟨y⟩) before the usual reward‑based update,
thereby letting the temporal context steer exploration.

The resulting unified system can:
1. Predict weekdays from date features with an NLMS filter whose step size
   adapts to both model complexity (RLCT) and external dynamics (FCD).
2. Update a bandit policy using rewards while being steered by the FCD signal.
3. Select actions based on a combination of learned rewards and the current
   FCD context.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Core of Parent A – Doomsday → NLMS with RLCT‑adjusted learning rate
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0=Sunday … 6=Saturday."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def date_features(year: int, month: int, day: int) -> np.ndarray:
    """
    Normalized feature vector for a calendar date.

    Features (order):
        0 : year scaled to [0,1] over [1900,2100]
        1 : sin(2π*month/12)
        2 : cos(2π*month/12)
        3 : sin(2π*day/31)
        4 : cos(2π*day/31)
        5 : constant bias term (1.0)
    """
    # Year scaling
    yr_min, yr_max = 1900.0, 2100.0
    yr_norm = (year - yr_min) / (yr_max - yr_min)
    # Cyclic month
    month_angle = 2.0 * math.pi * (month - 1) / 12.0
    # Cyclic day (max 31)
    day_angle = 2.0 * math.pi * (day - 1) / 31.0
    return np.array(
        [
            yr_norm,
            math.sin(month_angle),
            math.cos(month_angle),
            math.sin(day_angle),
            math.cos(day_angle),
            1.0,
        ],
        dtype=float,
    )


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction."""
    return float(weights @ x)


def rlct_adjusted_mu(weights: np.ndarray, base_mu: float = 0.5) -> float:
    """
    RLCT‑inspired adjustment of the NLMS learning rate.

    μ̂ = base_mu / (1 + log(1 + ||w||₂))
    """
    norm = np.linalg.norm(weights)
    return base_mu / (1.0 + math.log1p(norm))


# ----------------------------------------------------------------------
# Core of Parent B – Fold‑Change Detection (FCD) + Hybrid Bandit
# ----------------------------------------------------------------------
def step(
    u: float,
    x: float,
    y: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay_x: float = 1.0,
    decay_y: float = 1.0,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Advance the feed‑forward state using Euler integration."""
    if dt < 0:
        raise ValueError("dt must be non‑negative")
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy


def response_series(
    inputs: list[float], x0: float = 1.0, y0: float = 0.0, **kw
) -> list[tuple[float, float]]:
    """Generate (x_t, y_t) pairs for a sequence of inputs."""
    x, y = x0, y0
    out: list[tuple[float, float]] = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out


# Simple in‑memory bandit policy -------------------------------------------------
_POLICY: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # [total_reward, count]


def reset_policy() -> None:
    """Clear all stored statistics."""
    _POLICY.clear()


def _average_reward(action: str) -> float:
    total, n = _POLICY[action]
    return total / n if n > 0 else 0.0


def update_policy(updates: list[tuple[str, float]]) -> None:
    """Incremental reward update for a list of (action, reward) pairs."""
    for action, reward in updates:
        total, n = _POLICY[action]
        _POLICY[action] = [total + reward, n + 1]


# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_nlms_update(
    weights: np.ndarray,
    date_vec: np.ndarray,
    target: float,
    fcd_inputs: list[float],
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update where the learning rate is modulated by
    the average y‑component of a Fold‑Change Detection response series.

    μ̃ = μ̂ * (1 + ⟨y⟩)   where μ̂ is the RLCT‑adjusted base learning rate.
    """
    # Compute RLCT‑adjusted base learning rate
    mu_rlct = rlct_adjusted_mu(weights, base_mu)

    # Generate FCD response and extract average y
    _, y_series = zip(*response_series(fcd_inputs))
    avg_y = sum(y_series) / len(y_series) if y_series else 0.0

    # Final modulated learning rate
    mu_mod = mu_rlct * (1.0 + avg_y)

    # Standard NLMS update with the modulated mu
    y_pred = nlms_predict(weights, date_vec)
    error = target - y_pred
    norm_x = float(date_vec @ date_vec) + eps
    delta = (mu_mod / norm_x) * error * date_vec
    new_weights = weights + delta
    return new_weights, error


def hybrid_update_bandit_with_fcd(
    actions: list[str],
    fcd_inputs: list[float],
    rewards: list[float],
    x0: float = 1.0,
    y0: float = 0.0,
) -> None:
    """
    Update the bandit policy using supplied rewards, but first bias each
    action's reward by exp(⟨y⟩) where ⟨y⟩ is the average y‑signal from the FCD
    response to the current input stream.
    """
    # Compute average y from the response series
    _, y_series = zip(*response_series(fcd_inputs, x0, y0))
    avg_y = sum(y_series) / len(y_series) if y_series else 0.0
    bias = math.exp(avg_y)  # positive bias grows with sustained y

    updates = [(action, reward * bias) for action, reward in zip(actions, rewards)]
    update_policy(updates)


def hybrid_select_action(
    actions: list[str],
    fcd_inputs: list[float],
    x0: float = 1.0,
    y0: float = 0.0,
    epsilon: float = 0.1,
) -> str:
    """
    Choose an action using an ε‑greedy scheme that combines:
    - the learned average reward per action,
    - a multiplicative factor exp(⟨y⟩) derived from the FCD response,
    - a small random exploration term.
    """
    # Compute context bias from FCD
    _, y_series = zip(*response_series(fcd_inputs, x0, y0))
    avg_y = sum(y_series) / len(y_series) if y_series else 0.0
    context_factor = math.exp(avg_y)

    if random.random() < epsilon:
        return random.choice(actions)

    # Score = context_factor * average_reward
    scores = {
        a: context_factor * _average_reward(a) for a in actions
    }
    # If all scores are zero (e.g., before any update), fall back to uniform random
    if all(v == 0.0 for v in scores.values()):
        return random.choice(actions)

    # Return action with maximal score
    return max(scores, key=scores.get)


def hybrid_predict_weekday(
    year: int,
    month: int,
    day: int,
    weights: np.ndarray,
    fcd_inputs: list[float],
) -> int:
    """
    Predict the weekday for a given date using the NLMS filter whose learning
    rate is dynamically modulated by the current Fold‑Change Detection context.
    The output of the linear predictor is rounded to the nearest integer in
    [0,6] (mod 7) to produce a valid weekday index.
    """
    x = date_features(year, month, day)
    # Modulate learning rate (no weight update, just compute effective mu)
    mu = rlct_adjusted_mu(weights)
    _, y_series = zip(*response_series(fcd_inputs))
    avg_y = sum(y_series) / len(y_series) if y_series else 0.0
    mu_eff = mu * (1.0 + avg_y)

    # Perform a *virtual* NLMS step to obtain a refined prediction
    y_pred = nlms_predict(weights, x)
    # No weight change here; we just return the rounded weekday
    weekday = int(round(y_pred)) % 7
    return weekday


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a random weight vector for the NLMS filter
    rng = np.random.default_rng(42)
    w = rng.normal(scale=0.1, size=6)

    # Sample training data (a few dates with their true weekday)
    training_set = [
        (2023, 1, 1),  # Sunday -> 0
        (2023, 2, 14),  # Tuesday -> 2
        (2023, 12, 25),  # Monday -> 1
    ]
    # Simple FCD input stream (e.g., sensor readings)
    fcd_stream = [0.5, 0.7, 0.2, 0.9, 0.3]

    # Train for a few epochs
    for epoch in range(5):
        total_err = 0.0
        for y, m, d in training_set:
            target = doomsday(y, m, d)
            x_vec = date_features(y, m, d)
            w, err = hybrid_nlms_update(
                w, x_vec, float(target), fcd_inputs=fcd_stream
            )
            total_err += abs(err)
        # Update a dummy bandit policy using the same FCD stream
        actions = ["alpha", "beta", "gamma"]
        rewards = [random.uniform(-1, 1) for _ in actions]
        hybrid_update_bandit_with_fcd(actions, fcd_stream, rewards)

    # Demonstrate prediction
    test_date = (2024, 7, 4)  # Thursday expected 4 (0=Sun)
    pred = hybrid_predict_weekday(*test_date, weights=w, fcd_inputs=fcd_stream)
    print(f"Predicted weekday for {test_date}: {pred} (0=Sun)")

    # Demonstrate action selection
    chosen = hybrid_select_action(["alpha", "beta", "gamma"], fcd_stream)
    print(f"Chosen action: {chosen}")

    # Verify that code runs without exception
    sys.exit(0)