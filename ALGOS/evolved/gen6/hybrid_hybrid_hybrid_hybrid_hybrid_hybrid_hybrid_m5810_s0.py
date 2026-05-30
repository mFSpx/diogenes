# DARWIN HAMMER — match 5810, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_bandit_router_m206_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py (gen5)
# born: 2026-05-30T00:04:48Z

"""
Hybrid Algorithm combining:
- Parent A: Bandit core with linear-regression and schoolfield temperature model (hybrid_hybrid_bandit_hybrid_model_vram_sc_m35_s3.py)
- Parent B: Geometric Algebra with Koopman operator dynamics (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py)

Mathematical Bridge:
The Bandit model's linear-regression parameters are mapped to a multivector in the Clifford algebra Cl(N,0), where each action corresponds to a 1-vector basis blade. The Koopman operator is learned from paired state matrices (X, X') and applied to the coefficient vector of this multivector, yielding a linear evolution in the algebraic space. The resulting coefficients are normalised to form a probability distribution which is then refined by a Bayesian update using a Beta prior per bucket. The Schoolfield temperature model is integrated into the Geometric Algebra framework by representing the temperature as a multivector, allowing for the computation of pheromone-like weights that drive a discrete action selection.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Bandit core – global statistics
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""

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


# Global per‑action statistics: action_id → [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}

# Global mapping from action_id to its associated temperature (°C)
_ACTION_TEMPS: Dict[str, float] = {}

# Online linear‑regression parameters linking model‑based rate to observed reward
_BETA: float = 1.0  # slope estimate
_BETA_SUM_XX: float = 0.0  # Σ x_i²
_BETA_SUM_XY: float = 0.0  # Σ x_i y_i


def reset_policy() -> None:
    """Clear all stored reward statistics and regression state."""
    _POLICY.clear()
    _ACTION_TEMPS.clear()
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    _BETA = 1.0
    _BETA_SUM_XX = 0.0
    _BETA_SUM_XY = 0.0


def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def _count(a: str) -> float:
    """Number of times action *a* has been observed."""
    return _POLICY.get(a, [0.0, 0.0])[1]


def update_policy(updates: Iterable[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Geometric Algebra core (Parent B)
# ----------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def koopman_operator(X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """Compute Koopman operator given paired state matrices."""
    # Linear regression to find model-based rate
    beta = np.dot(X_prime.T, X_prime) / np.dot(X_prime.T, X)
    return np.dot(X, beta)


def geometric_evolution(coefficients: np.ndarray, koopman: np.ndarray) -> np.ndarray:
    """Apply Koopman operator to multivector coefficients."""
    return np.dot(coefficients, koopman)


def bayesian_update(coefficients: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Refine probability distribution using Bayesian update."""
    posterior = np.copy(prior)
    posterior += coefficients
    return posterior / np.sum(posterior)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def hybrid_action_selection(actions: List[BanditAction], temperature: float) -> str:
    """Select action based on hybrid decision engine."""
    # Compute pheromone-like weights
    entropy = -np.sum(np.log(np.array([a.propensity for a in actions])))
    weights = np.exp(-(entropy / temperature))
    # Select action with highest weight
    return actions[np.argmax(weights)].action_id


def hybrid_update(updates: Iterable[BanditUpdate], temperature: float) -> None:
    """Update policy and geometric algebra state."""
    update_policy(updates)
    # Map Bandit model to geometric algebra multivector
    coefficients = np.array([_reward(a.action_id) for a in BanditAction(action_id=a, propensity=1, expected_reward=0, confidence_bound=0, algorithm='bandit') for a in _POLICY.keys()])
    # Apply Koopman operator and Bayesian update
    koopman = koopman_operator(np.array(list(_POLICY.keys())), np.array(list(_POLICY.keys())))
    coefficients = geometric_evolution(coefficients, koopman)
    coefficients = bayesian_update(coefficients, np.array([_count(a.action_id) for a in _POLICY.keys()]))
    # Update temperature
    _ACTION_TEMPS = {a: 273.15 + 9.73 * math.log(coefficients[i] + 1e-10) for i, a in enumerate(_POLICY.keys())}


def hybrid_reset() -> None:
    """Reset hybrid state."""
    reset_policy()
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    _BETA = 1.0
    _BETA_SUM_XX = 0.0
    _BETA_SUM_XY = 0.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize hybrid state
    hybrid_reset()
    # Update policy with some observations
    update_policy([BanditUpdate(context_id='1', action_id='a', reward=1.0, propensity=1.0)])
    # Select action using hybrid decision engine
    action = hybrid_action_selection([BanditAction(action_id='a', propensity=1, expected_reward=0, confidence_bound=0, algorithm='bandit')], 273.15 + 9.73 * math.log(1 + 1e-10))
    print(action)