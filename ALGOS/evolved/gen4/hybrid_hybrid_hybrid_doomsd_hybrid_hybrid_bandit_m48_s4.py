# DARWIN HAMMER — match 48, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_nlms_omni_cha_m115_s0.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s3.py (gen3)
# born: 2026-05-29T23:26:41Z

"""Hybrid Date‑Bandit NLMS Fusion

This module fuses the two parent algorithms:

* **Parent A** – provides vectorised calendar utilities
  (`weekday_sakamoto`) and a statistical inequality measure
  (`gini_coefficient`). Both operate on NumPy arrays and rely on
  simple arithmetic that can be expressed as matrix‑vector products.

* **Parent B** – implements a contextual multi‑armed bandit with
  immutable data structures, a circuit‑breaker, and a temperature‑
  performance model (`Schoolfield`). The core learning rule is a
  Normalised Least‑Mean‑Squares (NLMS) update that is itself a
  matrix‑vector operation.

**Mathematical bridge**

The bridge is the *feature matrix* built from calendar data.  For each
record we compute


x = [weekday, day_of_year / 365, schoolfield_rate(T)]


where `weekday` comes from Parent A, the normalised day‑of‑year is a
linear scaling, and `schoolfield_rate` (Parent B) injects a biologically
motivated non‑linear term.  The NLMS weight update


e = d – w·x
μ = η / (ε + x·x)
w ← w + μ·e·x


uses the **Gini coefficient** of the recent reward batch as a dynamic
scale for the base step size `η`.  Thus the inequality of the reward
distribution (Parent A) directly modulates the adaptation speed of the
bandit predictor (Parent B), creating a mathematically unified hybrid
system.

The module supplies three high‑level functions that demonstrate this
integration:

* `compute_feature_matrix` – builds the joint calendar/temperature
  feature matrix.
* `nlms_predict` – produces per‑action reward predictions using NLMS
  weights.
* `hybrid_batch_update` – updates both the global bandit policy and the
  NLMS weights, scaling the NLMS learning rate with the Gini coefficient
  of the observed rewards.

All code is pure Python 3 with only the allowed standard‑library
imports and NumPy.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------


def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Tomohiko Sakamoto weekday algorithm.
    Returns 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a 1‑D non‑negative array.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
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


# ----------------------------------------------------------------------
# Parent B data structures and temperature model
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Immutable record of a single interaction."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    rho_25: float = 1.0                # rate at 25 °C (arbitrary units)
    delta_h_activation: float = 12_000.0  # J mol⁻¹
    t_low: float = 283.15              # K  (≈10 °C)
    t_high: float = 307.15             # K  (≈34 °C)
    delta_h_low: float = -45_000.0     # J mol⁻¹
    delta_h_high: float = 65_000.0     # J mol⁻¹
    r_cal: float = 1.987               # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def schoolfield_rate(params: SchoolfieldParams, temperature: np.ndarray) -> np.ndarray:
    """
    Compute the Schoolfield rate for an array of temperatures (K).
    Vectorised implementation based on the equation:

        r(T) = rho_25 *
               exp[ -ΔH_a / R * (1/T - 1/298.15) ] /
               ( 1 + exp[ ΔH_l / R * (1/T_l - 1/T) ] + exp[ ΔH_h / R * (1/T_h - 1/T) ] )
    """
    T = temperature.astype(np.float64)
    R = params.r_cal * 4.184  # convert cal·mol⁻¹·K⁻¹ to J·mol⁻¹·K⁻¹

    num = np.exp(-params.delta_h_activation / R * (1.0 / T - 1.0 / 298.15))
    low = np.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))
    high = np.exp(params.delta_h_high / R * (1.0 / params.t_high - 1.0 / T))

    denominator = 1.0 + low + high
    return params.rho_25 * num / denominator


# ----------------------------------------------------------------------
# Global bandit policy store (same as Parent B)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [cumulative_reward, count]


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy using a batch of observations."""
    for u in updates:
        stats = _POLICY[u.action_id]
        stats[0] += float(u.reward)
        stats[1] += 1.0


def average_reward(action_id: str) -> float:
    """Return the average reward for a given action, or 0.0 if unseen."""
    cum, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return cum / cnt if cnt > 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid feature construction (bridge between parents)
# ----------------------------------------------------------------------


def compute_feature_matrix(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    """
    Build a (N, 3) feature matrix for N date records.

    Columns:
        0 – weekday (0‑6) normalised to [0, 1]
        1 – day‑of‑year scaled to [0, 1]
        2 – schoolfield rate at a temperature derived from day‑of‑year
    """
    # Weekday feature (Parent A)
    wd = weekday_sakamoto(years, months, days).astype(np.float64) / 6.0

    # Day‑of‑year (vectorised, using datetime for correctness)
    # Build a temporary array of datetime objects
    dates = np.empty(years.shape, dtype=object)
    for idx, (y, m, d) in enumerate(zip(years, months, days)):
        dates[idx] = dt.date(int(y), int(m), int(d))
    # NumPy does not have a direct day‑of‑year; fallback to Python loop
    doy = np.array([date.timetuple().tm_yday for date in dates], dtype=np.float64) / 365.0

    # Temperature model: sinusoidal seasonal variation around 298 K
    # T = 298 K + 10 K * sin(2π * doy)
    temperature = 298.0 + 10.0 * np.sin(2.0 * np.pi * doy)

    # Schoolfield rate (Parent B)
    rate = schoolfield_rate(params, temperature)

    # Stack into feature matrix
    return np.column_stack((wd, doy, rate))


# ----------------------------------------------------------------------
# NLMS prediction and hybrid update
# ----------------------------------------------------------------------


def nlms_predict(
    weight_dict: Dict[str, np.ndarray],
    feature_vec: np.ndarray,
) -> Dict[str, float]:
    """
    Predict expected reward for each action using current NLMS weights.
    `weight_dict` maps action_id → weight vector (shape (3,)).
    Returns a dictionary action_id → scalar prediction.
    """
    preds = {}
    for aid, w in weight_dict.items():
        preds[aid] = float(np.dot(w, feature_vec))
    return preds


def hybrid_batch_update(
    updates: List[BanditUpdate],
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    weight_dict: Dict[str, np.ndarray],
    base_step: float = 0.1,
    epsilon: float = 1e-6,
) -> None:
    """
    Perform a hybrid batch update:

    1. Update the global bandit policy (simple cumulative reward).
    2. Build the feature matrix for the associated contexts.
    3. Compute the Gini coefficient of the reward batch.
    4. Scale the NLMS step size by (1 + Gini) and apply the NLMS rule
       to each action present in the batch.

    The function mutates `weight_dict` in‑place.
    """
    if not updates:
        return

    # 1. Policy update (Parent B)
    update_policy(updates)

    # 2. Feature matrix (bridge)
    feats = compute_feature_matrix(years, months, days)  # shape (N, 3)

    # 3. Gini coefficient of observed rewards
    rewards = np.array([u.reward for u in updates], dtype=np.float64)
    gini = gini_coefficient(rewards)

    # Adaptive step size
    mu = base_step * (1.0 + gini)

    # 4. NLMS weight update per action
    # Group updates by action for efficiency
    action_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, u in enumerate(updates):
        action_groups[u.action_id].append(idx)

    for aid, indices in action_groups.items():
        # Current weight vector; initialise if absent
        w = weight_dict.get(aid)
        if w is None:
            w = np.zeros(feats.shape[1], dtype=np.float64)
        # Batch NLMS over the indices belonging to this action
        for i in indices:
            x = feats[i]                      # feature vector (3,)
            d = updates[i].reward             # desired scalar
            y = np.dot(w, x)                  # prediction
            e = d - y                         # error
            norm_factor = epsilon + np.dot(x, x)
            adaptation = (mu / norm_factor) * e * x
            w = w + adaptation
        weight_dict[aid] = w


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Generate a small synthetic batch
    N = 5
    rng = np.random.default_rng(42)

    years = np.full(N, 2024, dtype=np.int64)
    months = rng.integers(1, 13, size=N, dtype=np.int64)
    days = rng.integers(1, 28, size=N, dtype=np.int64)  # safe for all months

    # Two dummy actions
    actions = ["A", "B"]
    updates: List[BanditUpdate] = []
    for i in range(N):
        aid = actions[i % 2]
        reward = rng.normal(loc=1.0 if aid == "A" else 0.5, scale=0.2)
        updates.append(
            BanditUpdate(
                context_id=f"ctx_{i}",
                action_id=aid,
                reward=max(0.0, reward),  # ensure non‑negative for Gini
                propensity=0.5,
            )
        )

    # Initialise NLMS weights (zero)
    nlms_weights: Dict[str, np.ndarray] = {aid: np.zeros(3) for aid in actions}

    # Run the hybrid update
    hybrid_batch_update(updates, years, months, days, nlms_weights)

    # Show results
    print("Policy statistics:")
    for aid in actions:
        print(f"  Action {aid}: avg reward = {average_reward(aid):.4f}")

    print("\nNLMS weights after update:")
    for aid, w in nlms_weights.items():
        print(f"  {aid}: {w}")

    # Predict on a new random date
    new_year = np.array([2024])
    new_month = np.array([rng.integers(1, 13)])
    new_day = np.array([rng.integers(1, 28)])
    new_feat = compute_feature_matrix(new_year, new_month, new_day)[0]
    preds = nlms_predict(nlms_weights, new_feat)
    print("\nPredictions for a fresh context:")
    for aid, p in preds.items():
        print(f"  {aid}: {p:.4f}")