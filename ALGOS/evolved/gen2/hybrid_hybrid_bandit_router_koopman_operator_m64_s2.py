# DARWIN HAMMER — match 64, survivor 2
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s2.py (gen1)
# parent_b: koopman_operator.py (gen0)
# born: 2026-05-29T23:24:13Z

"""
Hybrid Bandit‑Koopman Store

This module fuses the three core ideas of the parent algorithms:

* **Bandit‑Router / Store** – a contextual bandit whose confidence term is
  scaled by a scalar “store” that evolves with inflow (reward) and outflow
  (cost).  The store therefore modulates exploration.
* **Koopman Operator (DMD)** – a linear model of the evolution of a vector of
  observables.  Here we treat the per‑action empirical mean rewards
  ``μ = [μ_a]`` as an observable ψ(x).  By fitting a Koopman operator to the
  time‑series of μ we obtain a *forecast* of future rewards without
  re‑learning the bandit.

**Mathematical bridge**

Let

* ``S_t`` be the store at time *t*,
* ``μ_t`` ∈ ℝ^A the vector of empirical mean rewards for the *A* actions,
* ``c_a(t) = (1 + S_t/(S_t+1)) / √(1+N_a(t)}`` the store‑dependent confidence
  multiplier (parent A).

We approximate the dynamics of ``μ_t`` by a linear Koopman operator ``K``:


μ_{t+1} ≈ K μ_t .


The forecast ``μ̂_{t+h}`` obtained from ``K^h μ_t`` is then used as the
expected reward in the bandit index


U_a(t) = μ̂_a(t) + η·‖context‖·c_a(t) .


Thus the store shapes exploration while the Koopman forecast shapes
exploitation, yielding a tightly coupled hybrid system.

The implementation below provides:

* ``hybrid_select_action`` – bandit selection using the Koopman‑forecasted
  means and store‑scaled confidence.
* ``hybrid_step`` – update of policy, store and history after observing a
  batch of rewards.
* ``fit_policy_koopman`` – fit a Koopman operator to the recorded history of
  mean‑reward vectors.
* ``forecast_rewards`` – obtain a multi‑step forecast of future means.

A small smoke test demonstrates end‑to‑end usage.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures (derived from bandit_router.py)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global policy store (action → [cumulative reward, count])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}
# History of mean‑reward vectors for Koopman fitting
_MEAN_HISTORY: List[np.ndarray] = []


def reset_policy() -> None:
    """Erase all learned statistics and history."""
    _POLICY.clear()
    _MEAN_HISTORY.clear()


def _reward(action: str) -> float:
    """Empirical mean reward for *action* (0 if never observed)."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observations into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def snapshot_mean_vector(actions: List[str]) -> np.ndarray:
    """Return the current mean‑reward vector ordered as ``actions``."""
    return np.array([_reward(a) for a in actions], dtype=float)


def _record_mean(actions: List[str]) -> None:
    """Append the current mean vector to the history (for Koopman)."""
    _MEAN_HISTORY.append(snapshot_mean_vector(actions))


# ----------------------------------------------------------------------
# Store dynamics (derived from honeybee_store.py)
# ----------------------------------------------------------------------


def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Apply the honeybee store equation.

    Δ = α·Σ(inflow) – β·Σ(outflow)
    S' = max(0, S + dt·Δ)

    Returns the new store value and the raw Δ.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    """
    Map a store change Δ to a bounded control signal.
    """
    return max(0.0, min(limit, base + gain * delta_store))


# ----------------------------------------------------------------------
# Koopman (DMD) core – trimmed version of koopman_operator.py
# ----------------------------------------------------------------------


def _dmd(X: np.ndarray, X_prime: np.ndarray, rank: int = 10):
    """Standard Dynamic Mode Decomposition (internal helper)."""
    d, T = X.shape
    r = min(rank, d, T)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]

    S_inv = np.diag(1.0 / S)
    K_tilde = U.T @ X_prime @ Vt.T @ S_inv
    eigenvalues, W = np.linalg.eig(K_tilde)
    Phi = X_prime @ Vt.T @ S_inv @ W
    Phi_pinv = np.linalg.pinv(Phi)
    K_approx = Phi @ np.diag(eigenvalues) @ Phi_pinv
    return eigenvalues, Phi, K_approx


def fit_policy_koopman(rank: int = 5) -> Dict[str, np.ndarray]:
    """
    Fit a Koopman operator to the recorded mean‑reward trajectory.

    Returns a dict with keys:
        'K'          – (A, A) Koopman matrix
        'eigenvalues'– (r,) complex eigenvalues
        'eigenvectors'– (A, r) DMD modes
        'rank'       – effective rank used
    """
    if len(_MEAN_HISTORY) < 2:
        raise ValueError("Insufficient history for Koopman fitting")

    traj = np.stack(_MEAN_HISTORY, axis=0)  # (T, A)
    X = traj[:-1].T          # (A, T-1)
    X_prime = traj[1:].T    # (A, T-1)

    eigenvalues, eigenvectors, K = _dmd(X, X_prime, rank=rank)
    return {"K": K, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "rank": rank}


def forecast_rewards(
    current_means: np.ndarray,
    koopman_model: Dict[str, np.ndarray],
    steps: int = 1,
) -> np.ndarray:
    """
    Forecast future mean‑reward vectors using the fitted Koopman matrix.

    Parameters
    ----------
    current_means : (A,) ndarray
        Latest observed mean‑reward vector.
    koopman_model : dict
        Output of ``fit_policy_koopman``.
    steps : int
        Number of future steps to predict.

    Returns
    -------
    ndarray (steps, A) – forecasted means for each future step.
    """
    K = koopman_model["K"]
    x = current_means.astype(complex)
    forecasts = np.empty((steps, len(x)), dtype=complex)
    for i in range(steps):
        x = K @ x
        forecasts[i] = x
    return forecasts.real


# ----------------------------------------------------------------------
# Hybrid core: action selection that uses Koopman‑forecasted means
# ----------------------------------------------------------------------


def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    koopman_model: Dict[str, np.ndarray] | None = None,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    forecast_horizon: int = 1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit rule whose expected reward is taken from
    a Koopman forecast and whose confidence term is scaled by the current store.

    Parameters
    ----------
    context : dict
        Feature vector for the current decision (used only for its Euclidean norm).
    actions : list[str]
        Candidate action identifiers.
    store : float
        Current store level (non‑negative). Higher store → larger confidence term.
    koopman_model : dict | None
        If provided, the model is used to forecast mean rewards; otherwise
        empirical means are used.
    algorithm : {'linucb', 'epsilon_greedy', 'thompson'}
        Underlying bandit policy (only ``linucb`` is fully supported here).
    epsilon : float
        Exploration probability for epsilon‑greedy.
    eta : float
        Base scaling factor for the confidence term.
    forecast_horizon : int
        How many steps ahead to forecast when ``koopman_model`` is given.
    seed : int|str|None
        Random seed for reproducibility.

    Returns
    -------
    BanditAction
        The selected action together with diagnostic quantities.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Store‑dependent confidence multiplier (lies in (1,2))
    store_factor = 1.0 + store / (store + 1.0)

    # Expected rewards μ̂_a
    if koopman_model is not None:
        # Use the latest empirical means as the seed for the forecast
        current_means = snapshot_mean_vector(actions)
        forecast = forecast_rewards(current_means, koopman_model, steps=forecast_horizon)
        # Take the last forecasted step as the estimate
        mu_hat = forecast[-1]
    else:
        mu_hat = np.array([_reward(a) for a in actions], dtype=float)

    # Confidence term per action
    conf = np.array(
        [(store_factor) / math.sqrt(1.0 + _count(a)) for a in actions],
        dtype=float,
    )

    # Context norm (simple Euclidean)
    ctx_vec = np.array(list(context.values()), dtype=float)
    ctx_norm = np.linalg.norm(ctx_vec) if ctx_vec.size else 1.0

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        idx = rng.randrange(len(actions))
    else:
        # LinUCB‑style index
        ucb = mu_hat + eta * ctx_norm * conf
        idx = int(np.argmax(ucb))

    chosen_action = actions[idx]
    return BanditAction(
        action_id=chosen_action,
        propensity=1.0 / len(actions) if algorithm == "epsilon_greedy" else None,
        expected_reward=float(mu_hat[idx]),
        confidence_bound=float(conf[idx]),
        algorithm=algorithm,
    )


def hybrid_step(
    updates: List[BanditUpdate],
    actions: List[str],
    store: float,
    cost_per_action: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, List[BanditUpdate]]:
    """
    Process a batch of bandit updates, evolve the store, and record the mean‑reward
    vector for later Koopman fitting.

    Parameters
    ----------
    updates : list[BanditUpdate]
        Observations obtained from the last decision round.
    actions : list[str]
        Full list of possible actions (used to order the mean vector).
    store : float
        Current store level.
    cost_per_action : float
        Fixed outflow cost applied to each observed action.
    alpha, beta, dt : float
        Store dynamics parameters.

    Returns
    -------
    new_store : float
        Updated store after applying inflow/outflow.
    processed_updates : list[BanditUpdate]
        The same list (returned for convenience).
    """
    # Update policy statistics
    update_policy(updates)

    # Compute inflow as sum of rewards, outflow as sum of costs
    inflow = [u.reward for u in updates]
    outflow = [cost_per_action for _ in updates]

    new_store, _ = update_store(store, inflow, outflow, alpha=alpha, beta=beta, dt=dt)

    # Record the current mean‑reward vector for Koopman learning
    _record_mean(actions)

    return new_store, updates


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple demo with three actions
    ACTIONS = ["A", "B", "C"]
    store = 0.0
    context = {"x1": 0.5, "x2": -1.2}
    rng = random.Random(42)

    # Run a few rounds
    for t in range(6):
        # Select action
        # No Koopman model yet for the first few steps
        act = hybrid_select_action(
            context=context,
            actions=ACTIONS,
            store=store,
            koopman_model=None,
            algorithm="linucb",
            seed=rng.randint(0, 1000),
        )
        # Simulate stochastic reward
        reward = rng.gauss(mu=1.0 if act.action_id == "B" else 0.0, sigma=0.5)
        upd = BanditUpdate(
            context_id="ctx0",
            action_id=act.action_id,
            reward=reward,
            propensity=act.propensity or 1.0,
        )
        # Update store and policy
        store, _ = hybrid_step([upd], ACTIONS, store, cost_per_action=0.1)

    # Fit Koopman on the collected history
    koop_model = fit_policy_koopman(rank=2)

    # One more decision using the Koopman forecast
    final_action = hybrid_select_action(
        context=context,
        actions=ACTIONS,
        store=store,
        koopman_model=koop_model,
        algorithm="linucb",
        forecast_horizon=2,
        seed=123,
    )
    print("Final selected action:", final_action)
    sys.exit(0)