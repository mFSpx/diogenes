# DARWIN HAMMER — match 1433, survivor 4
# gen: 4
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s6.py (gen3)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py (gen3)
# born: 2026-05-29T23:36:26Z

"""Hybrid Doomsday-FoldChange Bandit Module

Parents:
- hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s6.py (date → NLMS predictor with RLCT‑adjusted μ)
- hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py (fold‑change detection feeding a bandit policy)

Mathematical bridge:
The fold‑change detection system produces a scalar modulation m(t)=‖(x_t,y_t)‖₂ that
reflects recent temporal dynamics.  This modulation is injected into the NLMS
weight update as a multiplicative factor on the RLCT‑adjusted learning rate,
and simultaneously it perturbs the reward signal used by the bandit router.
Thus the adaptive filter and the bandit share a common “free‑energy” term:
    μ̂_t = μ₀ / (1+log(1+‖w_t‖₂)) · (1 + m(t))
and the reward for an action a after a prediction is r = −|error|·(1+m(t)).
The unified system can therefore predict weekdays from calendar features while
adapting its action policy based on the same temporal signal."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Global bandit policy storage
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear the internal bandit statistics."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Average reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> int:
    """Number of times *action* has been taken."""
    return int(_POLICY.get(action, [0.0, 0.0])[1])

def update_policy(updates: list[tuple[str, float]]) -> None:
    """Incorporate a list of (action, reward) observations into the policy."""
    for action, reward in updates:
        total, n = _POLICY.get(action, [0.0, 0.0])
        _POLICY[action] = [total + reward, n + 1]

# ----------------------------------------------------------------------
# Fold‑change detection (parent B core)

def step(u: float, x: float, y: float,
         dt: float = 1.0,
         gain: float = 1.0,
         decay_x: float = 1.0,
         decay_y: float = 1.0,
         eps: float = 1e-12) -> tuple[float, float]:
    """Euler integration of the feed‑forward fold‑change detector."""
    if dt < 0:
        raise ValueError('dt must be non‑negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def response_series(inputs: list[float],
                    x0: float = 1.0,
                    y0: float = 0.0,
                    **kw) -> list[tuple[float, float]]:
    """Generate (x_t, y_t) for each input *u*."""
    x, y = x0, y0
    out: list[tuple[float, float]] = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out

def latest_modulation(inputs: list[float]) -> float:
    """
    Return a scalar modulation m = ‖(x_T, y_T)‖₂ derived from the last
    fold‑change detector state.
    """
    if not inputs:
        return 0.0
    x, y = response_series(inputs)[-1]
    return math.hypot(x, y)

# ----------------------------------------------------------------------
# Date → feature vector (parent A core)

def date_features(year: int, month: int, day: int) -> np.ndarray:
    """
    Normalised calendar features.
    - year scaled to [0,1] over 1900‑2100
    - month encoded as sin/cos pair
    - day encoded as sin/cos pair
    """
    # year normalisation
    y_norm = (year - 1900) / 200.0  # 1900→0.0, 2100→1.0
    # cyclic encodings
    month_angle = 2.0 * math.pi * (month - 1) / 12.0
    day_angle = 2.0 * math.pi * (day - 1) / 31.0
    feats = np.array([
        y_norm,
        math.sin(month_angle), math.cos(month_angle),
        math.sin(day_angle),   math.cos(day_angle)
    ], dtype=float)
    return feats

# ----------------------------------------------------------------------
# NLMS with RLCT‑adjusted learning rate (parent A core)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction ŷ = w·x."""
    return float(weights @ x)

def rlct_adjusted_mu(weights: np.ndarray, base_mu: float = 0.5) -> float:
    """
    RLCT‑inspired learning‑rate scaling:
        μ̂ = base_mu / (1 + log(1+‖w‖₂))
    """
    norm = np.linalg.norm(weights)
    return base_mu / (1.0 + math.log1p(norm))

def hybrid_nlms_update(weights: np.ndarray,
                       x: np.ndarray,
                       target: float,
                       modulation: float,
                       base_mu: float = 0.5,
                       eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """
    NLMS weight update where the learning rate is both RLCT‑adjusted
    and modulated by the fold‑change magnitude.

    μ̃ = μ̂ · (1 + modulation)
    """
    # RLCT‑adjusted base rate
    mu_hat = rlct_adjusted_mu(weights, base_mu)
    # incorporate temporal modulation
    mu_t = mu_hat * (1.0 + modulation)

    y = nlms_predict(weights, x)
    e = target - y
    norm_x = float(x @ x) + eps
    delta = (mu_t / norm_x) * e * x
    new_weights = weights + delta
    return new_weights, e

# ----------------------------------------------------------------------
# Hybrid bandit that uses the same modulation

def hybrid_select_action(actions: list[str],
                         inputs: list[float],
                         x0: float = 1.0,
                         y0: float = 0.0) -> str:
    """
    Choose an action using a soft‑max over (average reward + modulation).
    The modulation is the latest fold‑change magnitude.
    """
    if not actions:
        raise ValueError("action list must be non‑empty")
    modulation = latest_modulation(inputs)

    # compute raw scores: reward + modulation (so that a high modulation
    # slightly boosts all actions, preserving relative differences)
    scores = np.array([_reward(a) + modulation for a in actions], dtype=float)

    # numerical stability for soft‑max
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    probs = exp_scores / exp_scores.sum()
    chosen = random.choices(actions, weights=probs, k=1)[0]
    return chosen

# ----------------------------------------------------------------------
# End‑to‑end hybrid operation

def hybrid_predict_and_train(date: tuple[int, int, int],
                             inputs: list[float],
                             actions: list[str],
                             weights: np.ndarray,
                             base_mu: float = 0.5) -> tuple[int, np.ndarray]:
    """
    Perform one hybrid cycle:
    1. Build the feature vector from *date*.
    2. Obtain modulation from *inputs* (fold‑change detector).
    3. Predict weekday with NLMS.
    4. Choose an action via the bandit router.
    5. Compute reward = –|error|·(1+modulation) and update the policy.
    6. Update NLMS weights using the same modulation.

    Returns the integer weekday (0=Sunday … 6=Saturday) and the updated
    weight vector.
    """
    year, month, day = date
    x_feat = date_features(year, month, day)

    # modulation from temporal signal
    modulation = latest_modulation(inputs)

    # NLMS prediction (raw float) → nearest weekday
    y_pred_float = nlms_predict(weights, x_feat)
    y_pred = int(round(y_pred_float)) % 7

    # true weekday (ground truth) using Python's calendar
    true_weekday = (dt.date(year, month, day).weekday() + 1) % 7

    # error for weight update
    error = true_weekday - y_pred_float

    # bandit step
    chosen_action = hybrid_select_action(actions, inputs)
    reward = -abs(error) * (1.0 + modulation)  # negative error as reward
    update_policy([(chosen_action, reward)])

    # NLMS weight update
    new_weights, _ = hybrid_nlms_update(weights, x_feat, float(true_weekday), modulation, base_mu)

    return y_pred, new_weights

# ----------------------------------------------------------------------
# Smoke test

if __name__ == "__main__":
    import datetime as dt

    # initialise
    random.seed(42)
    np.random.seed(0)
    reset_policy()
    actions = ["train", "evaluate", "explore"]
    # start with small random weights (5 features)
    w = np.random.randn(5) * 0.01

    # generate a few random dates and synthetic input streams
    for _ in range(5):
        # random date between 1990 and 2025
        yr = random.randint(1990, 2025)
        mo = random.randint(1, 12)
        # ensure day is valid for month (simple clamp)
        dy = random.randint(1, 28)
        date = (yr, mo, dy)

        # synthetic input series (e.g., sensor readings)
        inputs = [random.uniform(0.0, 2.0) for _ in range(10)]

        pred, w = hybrid_predict_and_train(date, inputs, actions, w)
        print(f"Date {date} → predicted weekday {pred}, updated weights norm {np.linalg.norm(w):.4f}")

    # final policy snapshot
    print("\nFinal policy statistics:")
    for act in actions:
        print(f"  {act}: avg reward={_reward(act):.3f}, count={_count(act)}")