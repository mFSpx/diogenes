# DARWIN HAMMER — match 3557, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m7_s2.py (gen4)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s3.py (gen1)
# born: 2026-05-29T23:50:47Z

"""Hybrid Algorithm combining Radial‑Basis Function (RBF) surrogate modeling
(PARENT A) with Multi‑Armed Bandit action selection and store dynamics
(PARENT B).

Mathematical bridge:
- PARENT A treats a MinHash signature as a discrete signal and builds a
  surrogate mapping ϕ(x) = Σ_i w_i·exp(-ε²‖x−c_i‖²) from signal vectors x
  to a target (e.g., Chelydrid strike output).
- PARENT B produces a scalar “store delta” (Δ) from inflow/outflow that
  modulates the exploration/exploitation balance of a bandit policy.

In the hybrid we view the store delta Δ as an additional component of the
signal vector fed to the RBF surrogate.  The surrogate predicts a *reward
adjustment* r̂ for each action based on the current context vector
c = [context features…, Δ].  This predicted reward replaces the raw empirical
reward in the Thompson‑sampling step, thereby fusing the continuous‑signal
learning of A with the discrete decision‑making of B in a single mathematical
framework.
"""

import math
import random
import sys
import pathlib
from typing import Iterable, List, Tuple, Dict, Sequence
import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – RBF surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve Ax = b via Gaussian elimination with partial pivoting.
    A must be square and nonsingular.
    """
    n = len(A)
    # Build augmented matrix
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]

    for col in range(n):
        # Pivot selection
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        M[col], M[pivot_row] = M[pivot_row], M[col]

        # Normalize pivot row
        pivot_val = M[col][col]
        M[col] = [v / pivot_val for v in M[col]]

        # Eliminate below
        for r in range(col + 1, n):
            factor = M[r][col]
            M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))
    return x

def rbf_surrogate_train(
    X: List[Vector],
    y: List[float],
    epsilon: float = 1.0,
) -> Tuple[List[Vector], List[float]]:
    """
    Train an RBF surrogate.
    Returns (centers, weights) where centers == X and weights solve Φ w = y,
    with Φ_ij = gaussian(||X_i - X_j||, epsilon).
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    n = len(X)
    Phi = [[gaussian(euclidean(X[i], X[j]), epsilon) for j in range(n)] for i in range(n)]
    w = solve_linear(Phi, y)
    return X, w

def rbf_surrogate_predict(
    centers: List[Vector],
    weights: List[float],
    x: Vector,
    epsilon: float = 1.0,
) -> float:
    """Predict output for a new input vector x using trained surrogate."""
    return sum(
        w * gaussian(euclidean(x, c), epsilon)
        for c, w in zip(centers, weights)
    )

# ----------------------------------------------------------------------
# Parent B – Bandit policy and store dynamics
# ----------------------------------------------------------------------
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
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
    """Simple linear store dynamics."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    """Map store change to a bounded “dance” duration."""
    return max(0.0, min(limit, base + gain * delta_store))

# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
# Mapping each action -> (centers, weights) of its surrogate model
_SURROGATE_REGISTRY: Dict[str, Tuple[List[Vector], List[float]]] = {}

def train_action_surrogate(
    action_id: str,
    contexts: List[Vector],
    rewards: List[float],
    epsilon: float = 1.0,
) -> None:
    """
    Fit an RBF surrogate for a specific action using past (context, reward) pairs.
    The trained model is stored in _SURROGATE_REGISTRY.
    """
    centers, weights = rbf_surrogate_train(contexts, rewards, epsilon)
    _SURROGATE_REGISTRY[action_id] = (centers, weights)

def predict_action_reward(
    action_id: str,
    context_vec: Vector,
    epsilon: float = 1.0,
) -> float:
    """
    Return surrogate‑predicted reward for `action_id`. If no model exists,
    fall back to empirical average reward from the bandit policy.
    """
    if action_id in _SURROGATE_REGISTRY:
        centers, weights = _SURROGATE_REGISTRY[action_id]
        return rbf_surrogate_predict(centers, weights, context_vec, epsilon)
    else:
        return _reward(action_id)

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "thompson",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    epsilon_surrogate: float = 1.0,
) -> BanditAction:
    """
    Choose an action using a Thompson‑sampling style rule where the
    expected reward is supplied by the RBF surrogate (which incorporates
    the store delta as part of the signal vector).
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Convert context dict to vector and append store factor as additional dimension
    ctx_vals = list(context.values())
    store_factor = 1.0 + store / (store + 1.0)  # same formulation as parent B
    signal_vec = ctx_vals + [store_factor]

    # Exploration vs exploitation
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
    elif algorithm == "thompson":
        # Sample from a Beta distribution whose parameters are shaped by
        # surrogate reward and store factor.
        def sample(action: str) -> float:
            r_hat = predict_action_reward(action, signal_vec, epsilon_surrogate)
            n = max(1.0, _count(action))
            # Positive part of surrogate reward drives the first shape parameter
            a = 1.0 + max(0.0, r_hat) * store_factor
            b = 1.0 + max(0.0, 1.0 - r_hat) * store_factor
            return rng.betavariate(a, b)

        chosen_id = max(actions, key=sample)
    else:
        # Default to UCB based on empirical reward and store factor
        scale = np.linalg.norm(ctx_vals) if ctx_vals else 1.0
        def ucb_score(action: str) -> float:
            avg = _reward(action)
            n = _count(action)
            bonus = math.sqrt((2 * math.log(max(1, scale))) / (n + 1e-9))
            return avg + store_factor * bonus

        chosen_id = max(actions, key=ucb_score)

    # Assemble BanditAction result
    propensity = 1.0 / len(actions)  # simplistic uniform propensity
    exp_reward = predict_action_reward(chosen_id, signal_vec, epsilon_surrogate)
    confidence = math.sqrt(1.0 / (1.0 + _count(chosen_id)))  # placeholder
    return BanditAction(
        action_id=chosen_id,
        propensity=propensity,
        expected_reward=exp_reward,
        confidence_bound=confidence,
        algorithm=algorithm,
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset state
    reset_policy()
    _SURROGATE_REGISTRY.clear()

    # Define dummy actions and generate synthetic training data
    actions = ["A", "B", "C"]
    for act in actions:
        # Create 5 random context vectors (2 features) and synthetic rewards
        contexts = [np.random.rand(2).tolist() for _ in range(5)]
        rewards = [random.random() for _ in range(5)]
        # Train surrogate per action
        train_action_surrogate(act, contexts, rewards, epsilon=0.5)

        # Populate bandit policy with the same rewards (as if observed)
        updates = [
            BanditUpdate(context_id=f"{act}_{i}", action_id=act, reward=r, propensity=1.0)
            for i, r in enumerate(rewards)
        ]
        update_policy(updates)

    # Initial store state
    store = 0.0
    # Simulate a step
    context = {"feat1": 0.3, "feat2": 0.7}
    inflow = [0.2, 0.1]
    outflow = [0.05]

    # Update store dynamics
    store, delta = update_store(store, inflow, outflow, alpha=1.0, beta=1.0, dt=1.0)

    # Choose action via hybrid selector
    chosen = hybrid_select_action(
        context=context,
        actions=actions,
        store=store,
        algorithm="thompson",
        epsilon=0.1,
        seed=42,
        epsilon_surrogate=0.5,
    )
    print(f"Chosen action: {chosen}")

    # Demonstrate dance duration based on store change
    dur = dance_duration(delta_store=delta, base=1.0, gain=2.0, limit=10.0)
    print(f"Dance duration: {dur:.2f}s")