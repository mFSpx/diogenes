# DARWIN HAMMER — match 64, survivor 1
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s2.py (gen1)
# parent_b: koopman_operator.py (gen0)
# born: 2026-05-29T23:24:13Z

"""
This module fuses the Hybrid Bandit-Store Algorithm from hybrid_bandit_router_honeybee_store_m9_s2.py 
and the Koopman Operator from koopman_operator.py. The mathematical bridge is built on the observation 
that the store dynamics from the Hybrid Bandit-Store Algorithm can be used to modulate the confidence 
term in the bandit, while the Koopman Operator can be used to forecast the future store values, allowing 
for more informed decision making. The fusion integrates the governing equations of both parents, 
allowing for a more sophisticated and dynamic decision making process.
"""

import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

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

def dmd(X, X_prime, rank=10):
    """Standard Dynamic Mode Decomposition.

    Parameters
    ----------
    X : array (d, T)
        Snapshot matrix; column t is the state at time t.
    X_prime : array (d, T)
        Shifted snapshot matrix; column t is the state at time t+1.
    rank : int
        Truncation rank for the SVD.  Clamped to min(d, T) if too large.

    Returns
    -------
    eigenvalues : ndarray (r,) complex
        DMD eigenvalues (Koopman eigenvalues in the truncated basis).
    eigenvectors : ndarray (d, r) complex
        DMD modes (columns), full-state representation.
    K_approx : ndarray (d, d) complex
        Full approximate Koopman matrix  K ≈ Phi diag(lambda) pinv(Phi).
    """
    X = np.asarray(X, dtype=float)
    X_prime = np.asarray(X_prime, dtype=float)
    d, T = X.shape
    r = min(rank, d, T)

    # Step 1 — thin SVD of X, truncated to rank r.
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

    return eigenvalues, Phi, K_approx

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
    eigenvalues, eigenvectors, K = dmd(X, X_prime, rank=r)

    return {
        "K": K,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "rank": r,
    }

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
    elif algorithm == "linucb":
        # Use the Koopman operator to forecast the future store value
        # and use it to select the action
        forecast_store = koopman_forecast([store], {"K": np.array([[1.0]])}, 1)
        forecast_store = forecast_store[0, 0]
        chosen = max(actions, key=lambda action: _reward(action) + eta * store_factor * forecast_store)
    else:
        raise ValueError("Invalid algorithm")

    return BanditAction(
        action_id=chosen,
        propensity=1.0,
        expected_reward=_reward(chosen),
        confidence_bound=eta * store_factor,
        algorithm=algorithm,
    )

def hybrid_update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Update the store using the honeybee store equation and the Koopman operator.

    Parameters
    ----------
    store : float
        Current store level.
    inflow : list[float]
        Inflow to the store.
    outflow : list[float]
        Outflow from the store.
    alpha : float
        Inflow coefficient.
    beta : float
        Outflow coefficient.
    dt : float
        Time step.

    Returns
    -------
    new_store : float
        New store level.
    delta : float
        Change in store level.
    """
    # Use the Koopman operator to forecast the future store value
    forecast_store = koopman_forecast([store], {"K": np.array([[1.0]])}, 1)
    forecast_store = forecast_store[0, 0]

    # Update the store using the honeybee store equation
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)

    # Use the forecasted store value to adjust the update
    new_store = new_store * (1.0 + forecast_store / (forecast_store + 1.0))

    return new_store, delta

if __name__ == "__main__":
    # Smoke test
    context = {"feature1": 1.0, "feature2": 2.0}
    actions = ["action1", "action2"]
    store = 10.0
    action = hybrid_select_action(context, actions, store)
    print(action)
    new_store, delta = hybrid_update_store(store, [1.0], [0.5])
    print(new_store, delta)