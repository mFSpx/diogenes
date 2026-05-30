# DARWIN HAMMER — match 2649, survivor 5
# gen: 5
# parent_a: bandit_router.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s6.py (gen4)
# born: 2026-05-29T23:43:18Z

"""Hybrid Bandit Router with RBF Surrogate.

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a lightweight contextual bandit that tracks empirical
  rewards per action and selects actions using ε‑greedy, Thompson sampling,
  or a LinUCB‑style confidence bound.
* **Parent B** – an RBF (Radial‑Basis‑Function) surrogate that approximates a
  reward surface from observed (context, reward) pairs by solving the linear
  system K w = y, where K is the Gaussian kernel matrix.

The mathematical bridge is the **reward estimate**.  Parent A provides a
scalar empirical mean  r̂_emp (a) for each action a, while Parent B yields a
context‑dependent prediction  r̂_sur (a, x) = ∑_i w_i · exp(‑(ε·‖x‑c_i‖)²).
The hybrid algorithm linearly combines these two estimates and adds the
original LinUCB‑style confidence term, producing a unified score

    score(a, x) = α·r̂_emp(a) + (1‑α)·r̂_sur(a, x) + β·‖x‖ / √(1+N_a),

where α∈[0,1] balances empirical vs. surrogate knowledge, β is a scaling
constant, and N_a is the number of times action a has been played.
The action with the maximal score is selected.

The implementation below integrates both families of equations into a
single, self‑contained Python module.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
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


# Empirical statistics: action_id → [total_reward, count]
_POLICY: Dict[str, List[float]] = {}

# Mapping context identifiers to raw context vectors (used by the surrogate)
_CONTEXT_STORE: Dict[str, Vector] = {}

# ----------------------------------------------------------------------
# RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class RBFSurrogate:
    """
    RBF interpolator for a single action.

    Stores observed contexts (centers) and associated rewards (targets).
    The weight vector w solves K·w = y where K_ij = gaussian(||c_i-c_j||).
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.centers: List[Vector] = []   # list of context vectors
        self.targets: List[float] = []    # observed rewards
        self.weights: np.ndarray = np.array([])  # solved after each insertion

    def _kernel_matrix(self) -> np.ndarray:
        n = len(self.centers)
        if n == 0:
            return np.empty((0, 0))
        K = np.empty((n, n), dtype=float)
        for i, ci in enumerate(self.centers):
            for j, cj in enumerate(self.centers):
                K[i, j] = gaussian(euclidean(ci, cj), self.epsilon)
        return K

    def _solve_weights(self) -> None:
        """Solve K·w = y using NumPy's linear solver (fallback to least‑squares)."""
        K = self._kernel_matrix()
        if K.size == 0:
            self.weights = np.array([])
            return
        y = np.array(self.targets, dtype=float)
        try:
            self.weights = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            # Singular matrix – fall back to least‑squares solution
            self.weights = np.linalg.lstsq(K, y, rcond=None)[0]

    def add_point(self, ctx: Vector, reward: float) -> None:
        """Append a new observation and recompute the weight vector."""
        self.centers.append(tuple(float(v) for v in ctx))
        self.targets.append(float(reward))
        self._solve_weights()

    def predict(self, ctx: Vector) -> float:
        """Return the RBF‑based prediction for a new context."""
        if not self.centers:
            return 0.0
        k_vec = np.array(
            [gaussian(euclidean(ctx, c), self.epsilon) for c in self.centers],
            dtype=float,
        )
        return float(k_vec @ self.weights)


# One surrogate per action
_SURROGATES: Dict[str, RBFSurrogate] = {}

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def reset_hybrid(epsilon: float = 1.0) -> None:
    """
    Clear empirical statistics, stored contexts, and all surrogates.
    """
    _POLICY.clear()
    _CONTEXT_STORE.clear()
    _SURROGATES.clear()


def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def _action_count(a: str) -> int:
    return int(_POLICY.get(a, [0.0, 0.0])[1])


def update_hybrid(updates: List[BanditUpdate]) -> None:
    """
    Process a batch of BanditUpdate objects.

    For each update we:
      1. Update the empirical statistics in _POLICY.
      2. Store the raw context vector (looked up from _CONTEXT_STORE).
      3. Feed the (context, reward) pair to the action‑specific RBF surrogate.
    """
    for u in updates:
        # --- empirical update ---
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

        # --- surrogate update ---
        ctx = _CONTEXT_STORE.get(u.context_id)
        if ctx is None:
            # Missing context – ignore surrogate training for this point.
            continue
        surrogate = _SURROGATES.setdefault(u.action_id, RBFSurrogate(epsilon=1.0))
        surrogate.add_point(ctx, u.reward)


def register_context(context_id: str, vector: Vector) -> None:
    """
    Register a raw context vector that can later be referenced by its identifier.
    """
    _CONTEXT_STORE[context_id] = tuple(float(v) for v in vector)


def select_hybrid_action(
    context: Vector,
    actions: List[str],
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    algorithm: str = "hybrid",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a hybrid score that blends empirical means,
    RBF surrogate predictions, and a LinUCB‑style confidence term.

    Parameters
    ----------
    context : Vector
        Current contextual feature vector.
    actions : List[str]
        Candidate action identifiers.
    alpha : float (default 0.5)
        Weight for the empirical component (0 → only surrogate, 1 → only empirical).
    beta : float (default 0.1)
        Scaling factor for the confidence term.
    algorithm : str
        Retained for compatibility; only 'hybrid' is meaningful here.
    epsilon : float
        Exploration probability for ε‑greedy fallback.
    seed : int | str | None
        Random seed for reproducibility.

    Returns
    -------
    BanditAction
        The selected action together with its diagnostic fields.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # ε‑greedy exploration shortcut
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
        return BanditAction(
            action_id=chosen,
            propensity=1.0 / len(actions),
            expected_reward=_empirical_reward(chosen),
            confidence_bound=0.0,
            algorithm="epsilon_greedy",
        )

    # Compute scale = ||context|| (used in confidence term)
    scale = math.sqrt(sum(float(v) * float(v) for v in context)) if context else 1.0

    # Hybrid scoring for each action
    scores: Dict[str, float] = {}
    for a in actions:
        r_emp = _empirical_reward(a)
        surrogate = _SURROGATES.get(a)
        r_sur = surrogate.predict(context) if surrogate else 0.0
        confidence = beta * scale / math.sqrt(1 + _action_count(a))
        score = alpha * r_emp + (1 - alpha) * r_sur + confidence
        scores[a] = score

    chosen = max(actions, key=lambda a: scores[a])

    # Build BanditAction with diagnostics
    chosen_surrogate = _SURROGATES.get(chosen)
    expected = (
        alpha * _empirical_reward(chosen)
        + (1 - alpha) * (chosen_surrogate.predict(context) if chosen_surrogate else 0.0)
    )
    confidence = beta * scale / math.sqrt(1 + _action_count(chosen))
    return BanditAction(
        action_id=chosen,
        propensity=1.0 / len(actions),
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


# ----------------------------------------------------------------------
# Demonstration Functions (three distinct utilities)
# ----------------------------------------------------------------------
def train_random_batch(num_updates: int = 20) -> None:
    """
    Generate a synthetic batch of updates for two actions ('A' and 'B')
    using random contexts and rewards, then feed them to the hybrid updater.
    """
    actions = ["A", "B"]
    for i in range(num_updates):
        ctx_id = f"ctx_{i}"
        ctx_vec = [random.uniform(-1, 1) for _ in range(3)]
        register_context(ctx_id, ctx_vec)

        # Simulated reward: linear function + noise, differing per action
        reward = (
            sum(ctx_vec) * (1 if random.choice(actions) == "A" else -1)
            + random.gauss(0, 0.1)
        )
        upd = BanditUpdate(
            context_id=ctx_id,
            action_id=random.choice(actions),
            reward=reward,
            propensity=1.0,
        )
        update_hybrid([upd])


def evaluate_policy(num_trials: int = 10) -> List[BanditAction]:
    """
    Run the hybrid selector on fresh random contexts and collect the chosen actions.
    """
    chosen: List[BanditAction] = []
    for _ in range(num_trials):
        ctx = [random.uniform(-1, 1) for _ in range(3)]
        act = select_hybrid_action(
            context=ctx,
            actions=["A", "B"],
            alpha=0.6,
            beta=0.05,
            algorithm="hybrid",
            epsilon=0.0,
            seed=None,
        )
        chosen.append(act)
    return chosen


def report_statistics(actions: List[BanditAction]) -> None:
    """
    Print a concise summary of the actions selected during evaluation.
    """
    from collections import Counter

    cnt = Counter(a.action_id for a in actions)
    print("Selection frequencies:", dict(cnt))
    avg_reward = sum(a.expected_reward for a in actions) / len(actions)
    print(f"Average hybrid expected reward: {avg_reward:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset everything
    reset_hybrid()

    # Train on a synthetic batch
    train_random_batch(num_updates=50)

    # Evaluate the hybrid policy
    results = evaluate_policy(num_trials=15)

    # Report
    report_statistics(results)

    # Ensure that the returned objects are of the correct type
    assert all(isinstance(r, BanditAction) for r in results)
    print("Hybrid bandit router executed successfully.")