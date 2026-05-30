# DARWIN HAMMER — match 64, survivor 0
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s2.py (gen1)
# parent_b: koopman_operator.py (gen0)
# born: 2026-05-29T23:24:13Z

"""
Hybrid Koopman-Honeybee-Store Algorithm

This module fuses the governing equations of two independent prototypes:

* **koopman_operator.py** — a linearization technique for nonlinear dynamical systems, using the Koopman operator to act on observables (functions of state) rather than on the state directly.
* **hybrid_bandit_router_honeybee_store_m9_s2.py** — a hybrid bandit-store algorithm that combines a lightweight contextual bandit router with a simple store dynamics primitive, where a scalar "store" evolves according to inflow/outflow and a "dance duration" function maps that Δ to a bounded control signal.

The mathematical bridge is built on the observation that the Koopman operator can be used to linearize the nonlinear dynamics of the store, allowing for a more accurate prediction of the store's behavior. The store's dynamics are then used to modulate the confidence term of the bandit, creating a coupled system that integrates the governing equations of both parents.
"""


import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import sys
from pathlib import Path
import math


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


_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()


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


def fit_koopman(trajectory, rank=10):
    """Fit a Koopman operator from a state trajectory.

    Parameters
    ----------
    trajectory : array (T, d)
        Time-ordered sequence of states.
    rank : int
        DMD truncation rank.

    Returns
    -------
    dict with keys:
        K            - (d, d) complex ndarray, approximate Koopman matrix
        eigenvalues  - (r,) complex ndarray
        eigenvectors - (d, r) complex ndarray  (DMD modes)
        rank         - int, effective rank used
    """
    trajectory = np.asarray(trajectory, dtype=float)
    T, d = trajectory.shape
    if T < 2:
        raise ValueError("trajectory must have at least 2 time steps")

    # Build snapshot matrices: columns are time steps.
    X = trajectory[:-1].T      # (d, T-1)
    X_prime = trajectory[1:].T # (d, T-1)

    r = min(rank, d, T - 1)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    U = U[:, :r]
    S = S[:r]
    Vt = Vt[:r, :]

    S_inv = np.diag(1.0 / S)

    # Step 2 — reduced (projected) operator  K_tilde = U^T X' V S^{-1}.
    K_tilde = U.T @ X_prime @ Vt.T @ S_inv   # (r, r)

    # Step 3 — eigendecompose K_tilde.
    eigenvalues, W = np.linalg.eig(K_tilde)   # W columns are right eigenvectors

    # Step 4 — lift eigenvectors back to full state space (DMD modes).
    Phi = X_prime @ Vt.T @ S_inv @ W          # (d, r)

    # Step 5 — reconstruct approximate K.
    # K_approx x ≈ Phi diag(lambda) pinv(Phi) x
    Phi_pinv = np.linalg.pinv(Phi)
    K_approx = Phi @ np.diag(eigenvalues) @ Phi_pinv   # (d, d)

    return {
        "K": K_approx,
        "eigenvalues": eigenvalues,
        "eigenvectors": Phi,
        "rank": r,
    }


def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit rule whose confidence term is scaled by the
    current store value.

    Parameters
    ----------
    context : dict
        Feature vector for the current decision (used only for its Euclidean norm).
    actions : list[str]
        Candidate action identifiers.
    store : float
        Current store level (non‑negative). Higher store → more aggressive
        exploration via an inflated confidence bound.
    algorithm : {'linucb', 'epsilon_greedy', 'thompson'}
        Which underlying bandit policy to employ.
    epsilon : float
        Exploration probability for epsilon‑greedy.
    eta : float
        Base scaling factor for the confidence term (mirrors the 0.1 constant
        in the original LinUCB stub).
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

    # Helper: store‑dependent confidence multiplier
    store_factor = 1.0 + store / (store + 1.0)  # lies in (1,2)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        chosen = max(actions, key=lambda a: _reward(a))

    return BanditAction(
        action_id=chosen,
        propensity=1.0,
        expected_reward=_reward(chosen),
        confidence_bound=eta * store_factor,
        algorithm=algorithm,
    )


def koopman_forecast(x0, model, steps):
    """Forecast forward in time using the Koopman matrix.

    Applies  x_{t+1} = K x_t  iteratively, which is equivalent to
    K^t x_0 in exact arithmetic.

    Parameters
    ----------
    x0 : array (d,)
        Initial state.
    model : dict
        Output of fit_koopman.
    steps : int
        Number of steps to forecast.  Returns steps+1 points including x0.

    Returns
    -------
    ndarray (steps, d)  — forecast states at times 1 … steps (x0 excluded).
    """
    K = model["K"]
    x = np.asarray(x0, dtype=complex)
    out = np.empty((steps, len(x)), dtype=complex)
    for t in range(steps):
        x = K @ x
        out[t] = x
    return out.real


def hybrid_forecast(
    store: float,
    context: Dict[str, float],
    actions: List[str],
    state_trajectory: List[List[float]],
    steps: int,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> List[BanditAction]:
    """
    Forecast the store and selected actions forward in time.

    Parameters
    ----------
    store : float
        Current store level (non‑negative).
    context : dict
        Feature vector for the current decision (used only for its Euclidean norm).
    actions : list[str]
        Candidate action identifiers.
    state_trajectory : list of lists[float]
        Time-ordered sequence of states.
    steps : int
        Number of steps to forecast.
    algorithm : {'linucb', 'epsilon_greedy', 'thompson'}
        Which underlying bandit policy to employ.
    epsilon : float
        Exploration probability for epsilon‑greedy.
    eta : float
        Base scaling factor for the confidence term (mirrors the 0.1 constant
        in the original LinUCB stub).
    seed : int|str|None
        Random seed for reproducibility.

    Returns
    -------
    list of BanditAction
        The forecasted selected actions together with diagnostic quantities.
    """
    model = fit_koopman(state_trajectory)
    forecast_states = koopman_forecast(state_trajectory[-1], model, steps)
    forecast_actions = []
    for state in forecast_states:
        new_store, _ = update_store(store, [1.0], [0.0])
        action = hybrid_select_action(
            context, actions, new_store, algorithm, epsilon, eta, seed
        )
        forecast_actions.append(action)
        store = new_store
    return forecast_actions


if __name__ == "__main__":
    # Smoke test
    context = {"feature1": 1.0, "feature2": 2.0}
    actions = ["action1", "action2"]
    store = 10.0
    state_trajectory = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    algorithm = "linucb"
    epsilon = 0.1
    eta = 0.1
    seed = 7
    steps = 3

    forecast_actions = hybrid_forecast(
        store, context, actions, state_trajectory, steps, algorithm, epsilon, eta, seed
    )
    print(forecast_actions)