# DARWIN HAMMER — match 1158, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py (gen2)
# parent_b: bandit_router.py (gen0)
# born: 2026-05-29T23:33:14Z

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Data structures
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

# ----------------------------------------------------------------------
# Global state for a simple LinUCB implementation
# ----------------------------------------------------------------------
# For each action we store (A, b) where:
#   A = D x D regularized covariance matrix
#   b = D x 1 accumulated reward-weighted context vector
# D = dimension of context vectors (derived from first call to `select_action`)
_POLICY: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_DIM: int = 0
_REG: float = 1.0          # regularization term λ
_ALPHA: float = 1.0        # scaling of Fisher information in exploration term
_BETA: float = 0.5         # base exploration coefficient for UCB

def reset_policy() -> None:
    """Clear all learned statistics."""
    global _POLICY, _DIM
    _POLICY.clear()
    _DIM = 0

def _ensure_action(action_id: str) -> None:
    """Create default (A, b) for a new action if it does not exist."""
    global _POLICY, _DIM
    if action_id not in _POLICY:
        if _DIM == 0:
            raise RuntimeError("Context dimension not set – call `select_action` first.")
        A = _REG * np.identity(_DIM)
        b = np.zeros((_DIM, 1))
        _POLICY[action_id] = (A, b)

def update_policy(updates: List[BanditUpdate]) -> None:
    """
    Incrementally update the LinUCB statistics for each observed (context, action, reward).
    """
    global _POLICY, _DIM
    for upd in updates:
        # Initialise dimension on first call
        if _DIM == 0:
            raise RuntimeError("Context dimension not set – call `select_action` first.")
        _ensure_action(upd.action_id)
        A, b = _POLICY[upd.action_id]
        # Context vector as column
        x = np.array(upd.context_id.split(','), dtype=float).reshape((-1, 1))
        if x.shape[0] != _DIM:
            raise ValueError(f"Context dimension mismatch for action {upd.action_id}")
        A += x @ x.T
        b += upd.reward * x
        _POLICY[upd.action_id] = (A, b)

def _reward_estimate(action_id: str) -> Tuple[float, float]:
    """
    Return (estimated reward, variance term) for the given action.
    If the action has never been seen, return (0, large variance).
    """
    global _POLICY, _DIM
    if action_id not in _POLICY:
        return 0.0, 1.0  # maximal uncertainty for unseen actions
    A, b = _POLICY[action_id]
    A_inv = np.linalg.inv(A)
    theta_hat = A_inv @ b
    est = float(theta_hat.T @ np.ones((_DIM, 1)))  # dot with vector of ones as a simple proxy
    var = float(np.ones((_DIM, 1)).T @ A_inv @ np.ones((_DIM, 1)))
    return est, max(var, 1e-12)

# ----------------------------------------------------------------------
# Gaussian beam & Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a scalar Gaussian beam.
    Returns a non‑negative scalar; eps guards against division by zero.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid scoring function
# ----------------------------------------------------------------------
def hybrid_fisher_bandit(
    theta: float,
    center: float,
    width: float,
    action_id: str,
) -> float:
    """
    Compute a UCB‑style score that blends the LinUCB estimate with a Fisher‑information
    based exploration boost.
    """
    fisher = fisher_score(theta, center, width)
    est_reward, variance = _reward_estimate(action_id)
    # Classic LinUCB term
    ucb = est_reward + _BETA * math.sqrt(variance)
    # Fisher‑information scaling – additive to keep the term bounded
    boost = _ALPHA * fisher
    return ucb + boost

# ----------------------------------------------------------------------
# Action selection
# ----------------------------------------------------------------------
def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action given a context vector and a list of candidate actions.
    Supported algorithms:
        - "linucb"      : Upper‑confidence bound using linear model.
        - "epsilon_greedy": Random exploration with probability epsilon.
        - "thompson"   : Simple Beta‑Bernoulli Thompson sampling (fallback for
                         non‑numeric contexts).
    The function also updates the global dimension `_DIM` on first call.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Convert context dict to a column vector
    global _DIM
    ctx_vals = np.array([context[k] for k in sorted(context)], dtype=float).reshape((-1, 1))
    if _DIM == 0:
        _DIM = ctx_vals.shape[0]
    elif ctx_vals.shape[0] != _DIM:
        raise ValueError("All contexts must have the same dimensionality")

    # Helper to compute a deterministic score for each action
    def linucb_score(a: str) -> float:
        _ensure_action(a)
        est, var = _reward_estimate(a)
        return est + _BETA * math.sqrt(var)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Treat reward as Bernoulli; use Beta posterior with pseudo‑counts
        def thompson_sample(a: str) -> float:
            total, n = _policy_summary(a)
            a_alpha = 1 + max(0.0, total)
            b_beta = 1 + max(0.0, n - total)
            return rng.betavariate(a_alpha, b_beta)
        chosen = max(actions, key=thompson_sample)
    else:  # default to LinUCB
        chosen = max(actions, key=linucb_score)

    # Compute reporting statistics
    total, n = _policy_summary(chosen)
    expected = total / n if n > 0 else 0.0
    confidence = 1.0 / math.sqrt(1.0 + n)
    propensity = 1.0 / len(actions)

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )

def _policy_summary(action_id: str) -> Tuple[float, int]:
    """
    Helper returning (cumulative reward, number of observations) for an action.
    """
    if action_id not in _POLICY:
        return 0.0, 0
    A, b = _POLICY[action_id]
    # Number of updates equals trace(A) - D*λ (since each update adds x xᵀ)
    n = int(round(np.trace(A) - _REG * _DIM))
    total = float(b.sum())
    return total, n

# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()
    # Simulated context: "0.5,1.2" encodes two features
    ctx_str = "0.5,1.2"
    context = {f"f{i}": v for i, v in enumerate(map(float, ctx_str.split(",")))}
    actions = ["a1", "a2", "a3"]
    chosen = select_action(context, actions, algorithm="linucb")
    score = hybrid_fisher_bandit(theta=1.0, center=0.0, width=1.0, action_id=chosen.action_id)
    print(f"Chosen action: {chosen.action_id}, hybrid score: {score:.4f}")