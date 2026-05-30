# DARWIN HAMMER — match 1901, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_hdc_hy_m590_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s2.py (gen5)
# born: 2026-05-29T23:39:44Z

"""
Hybrid Regret‑State‑Space Bandit Engine
=======================================

This module fuses the two parent algorithms:

* **Parent A** – *hybrid_hybrid_regret_engine_hybrid_hdc_hy_m590_s0.py*  
  Provides a regret‑engine style computation based on the ISO weekday index
  and a Gini coefficient derived from a distribution of expected values.

* **Parent B** – *hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s2.py*  
  Implements a state‑space model (SSD) together with a Hybrid Bandit Router
  (HBR) that uses a log‑count ratio and confidence bounds.

**Mathematical bridge**

1. **Weekday‑modulated state transition** – the ISO weekday index `w ∈ {0,…,6}`
   is turned into a scalar modulation factor `μ(w)=1+α·sin(π·w/7)`.  
   This factor multiplies the state‑transition matrix `A` of the SSD, thus
   letting the day of the week influence the dynamics of the hidden state.

2. **Gini‑weighted confidence** – the Gini coefficient `G` of the vector of
   expected values of the bandit actions is used to shrink or enlarge the
   confidence bounds `c_i`:

   `c_i' = c_i·(1 + β·G)`

   A more unequal distribution (high `G`) inflates the confidence bounds,
   encouraging exploration.

3. **Hybrid store factor** – the HBR term
   `S_i = log_count_ratio·N_i` (where `N_i` is the selection count of action
   `i`) is multiplied by the Gini‑weighted confidence and added to the SSD
   output `y`.  The final observation therefore fuses the symbolic
   state‑space evolution with the bandit‑driven regret information.

The three core functions below realise this fusion:
`weekday_modulated_matrix`, `gini_weighted_confidence` and
`hybrid_ssm_bandit_step`.  The high‑level driver `hybrid_regret_ssm_router`
executes the combined process over a sequence of inputs.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime as dt
from typing import List, Dict, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (derived from the parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element used by the regret engine."""
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


@dataclass(frozen=True, slots=True)
class BanditAction:
    """Bandit‑style action used by the HBR component."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True, slots=True)
class BanditUpdate:
    """Result of pulling a bandit action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Utility functions from Parent A (regret engine)
# ----------------------------------------------------------------------


def weekday_index(year: int, month: int, day: int) -> int:
    """
    Return the ISO weekday index (0 = Monday … 6 = Sunday) for a given date.
    """
    return dt(year, month, day).weekday()


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient for a non‑negative distribution.
    Returns 0 for an empty or all‑zero input.
    """
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0 or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0.0:
        raise ValueError("values must be non‑negative")
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Utility functions from Parent B (state‑space + bandit)
# ----------------------------------------------------------------------


_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # total reward, count


def reset_policy() -> None:
    """Reset the global bandit policy."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Mean reward for a given action."""
    total, n = _POLICY[action]
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times an action has been selected."""
    return _POLICY[action][1]


def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Hybrid store factor = log‑count‑ratio × selection count."""
    return log_count_ratio * count


def _fold_change_detection(x: float, eps: float = 1e-12) -> float:
    """Fold‑change detection used by the original HBR."""
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0


def _ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single step of a linear state‑space model:
        h_{t+1} = A·h_t + B·x_t
        y_t     = C·h_{t+1}
    """
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y


# ----------------------------------------------------------------------
# Hybrid building blocks (the mathematical bridge)
# ----------------------------------------------------------------------


def weekday_modulated_matrix(
    base_A: np.ndarray,
    year: int,
    month: int,
    day: int,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Modulate the state‑transition matrix `A` with a scalar that depends on the
    ISO weekday index.

    μ(w) = 1 + α·sin(π·w/7)

    Parameters
    ----------
    base_A : np.ndarray
        Original state‑transition matrix.
    year, month, day : int
        Date used to obtain the weekday index.
    alpha : float, optional
        Modulation amplitude (default 0.1).

    Returns
    -------
    np.ndarray
        The weekday‑modulated matrix.
    """
    w = weekday_index(year, month, day)
    mu = 1.0 + alpha * math.sin(math.pi * w / 7.0)
    return base_A * mu


def gini_weighted_confidence(actions: List[BanditAction], beta: float = 0.2) -> List[BanditAction]:
    """
    Adjust the confidence bounds of a list of bandit actions using the Gini
    coefficient of their expected rewards.

    c_i' = c_i·(1 + β·G)

    Returns a new list with updated confidence bounds (BanditAction is frozen,
    therefore new instances are created).
    """
    expected = [a.expected_reward for a in actions]
    G = gini_coefficient(expected)
    factor = 1.0 + beta * G
    adjusted = [
        BanditAction(
            action_id=a.action_id,
            propensity=a.propensity,
            expected_reward=a.expected_reward,
            confidence_bound=a.confidence_bound * factor,
            algorithm=a.algorithm,
        )
        for a in actions
    ]
    return adjusted


def hybrid_ssm_bandit_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    actions: List[BanditAction],
    log_count_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one combined SSM‑Bandit step.

    1. Execute the linear state‑space update (`_ssm_step`).
    2. For each action compute a hybrid store factor `S_i`.
    3. Weight `S_i` by the (Gini‑adjusted) confidence bound `c_i'`.
    4. Add the aggregated bandit term to the raw observation `y`.

    Returns
    -------
    h_new : np.ndarray
        Updated hidden state.
    y_adj : np.ndarray
        Observation after bandit augmentation.
    """
    # 1. State‑space evolution
    h_new, y = _ssm_step(h, x, A, B, C)

    # 2. Gini‑weighted confidence bounds
    actions_adj = gini_weighted_confidence(actions)

    # 3. Compute the bandit augmentation term
    bandit_term = 0.0
    for a in actions_adj:
        count = _count(a.action_id)
        S_i = _hybrid_store_factor(a.action_id, count, log_count_ratio)
        bandit_term += S_i * a.confidence_bound

    # 4. Broadcast the scalar term to the shape of y (assume y is 1‑D)
    y_adj = y + bandit_term

    return h_new, y_adj


# ----------------------------------------------------------------------
# High‑level hybrid router
# ----------------------------------------------------------------------


def hybrid_regret_ssm_router(
    actions: List[BanditAction],
    dates: List[Tuple[int, int, int]],
    A_base: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray,
    x_seq: np.ndarray,
    log_count_ratio: float = 0.05,
) -> np.ndarray:
    """
    Run the fused regret‑state‑space router over a sequence of inputs.

    For each time step `t`:
        * Modulate `A` with the weekday of `dates[t]`.
        * Execute `hybrid_ssm_bandit_step`.
        * Record the observation.

    The function also updates the global bandit policy with synthetic rewards
    (for demonstration purposes) so that the policy evolves during the run.

    Parameters
    ----------
    actions : List[BanditAction]
        List of candidate actions (shared across all timesteps).
    dates : List[Tuple[int, int, int]]
        Corresponding (year, month, day) for each timestep.
    A_base, B, C : np.ndarray
        Base state‑space matrices.
    h0 : np.ndarray
        Initial hidden state.
    x_seq : np.ndarray
        Input sequence of shape (T, input_dim).
    log_count_ratio : float, optional
        Scaling factor for the hybrid store term.

    Returns
    -------
    np.ndarray
        Matrix `Y` of shape (T, output_dim) containing the adjusted observations.
    """
    T = x_seq.shape[0]
    h = h0.copy()
    Y = np.zeros((T, C.shape[0]))

    for t in range(T):
        year, month, day = dates[t]
        A_mod = weekday_modulated_matrix(A_base, year, month, day)

        # Perform the hybrid step
        h, y_adj = hybrid_ssm_bandit_step(
            h,
            x_seq[t],
            A_mod,
            B,
            C,
            actions,
            log_count_ratio,
        )
        Y[t] = y_adj

        # ------------------------------------------------------------------
        # Dummy policy update (makes the example self‑contained)
        # ------------------------------------------------------------------
        # Randomly pick an action, simulate a reward and update the policy.
        chosen = random.choice(actions)
        reward = chosen.expected_reward + random.gauss(0, 0.1)  # noisy reward
        _POLICY[chosen.action_id][0] += reward
        _POLICY[chosen.action_id][1] += 1.0

    return Y


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple 2‑dimensional linear system
    A_base = np.array([[0.9, 0.1],
                       [0.2, 0.8]])
    B = np.array([[0.05, 0.0],
                  [0.0, 0.05]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])

    h0 = np.zeros(2)

    # Input sequence: 10 timesteps, small random perturbations
    T = 10
    np.random.seed(42)
    x_seq = np.random.randn(T, 2) * 0.1

    # Dates spanning a week (Monday to Wednesday)
    start_date = dt(2026, 5, 26)  # Saturday
    dates = [(d.year, d.month, d.day) for d in [start_date + np.timedelta64(i, 'D') for i in range(T)]]

    # Define three bandit actions
    actions = [
        BanditAction(action_id="A", propensity=0.3, expected_reward=1.0, confidence_bound=0.2),
        BanditAction(action_id="B", propensity=0.5, expected_reward=0.8, confidence_bound=0.15),
        BanditAction(action_id="C", propensity=0.2, expected_reward=1.2, confidence_bound=0.25),
    ]

    # Reset any lingering policy state
    reset_policy()

    # Run the hybrid router
    Y = hybrid_regret_ssm_router(
        actions=actions,
        dates=dates,
        A_base=A_base,
        B=B,
        C=C,
        h0=h0,
        x_seq=x_seq,
        log_count_ratio=0.07,
    )

    # Simple sanity check: Y should have shape (T, 2) and contain finite numbers
    assert Y.shape == (T, 2)
    assert np.all(np.isfinite(Y))

    print("Hybrid Regret‑SSM router executed successfully. Output shape:", Y.shape)