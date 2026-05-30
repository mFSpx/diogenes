# DARWIN HAMMER — match 5828, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m932_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s0.py (gen4)
# born: 2026-05-30T00:04:53Z

"""Hybrid algorithm merging geometric algebra health scoring with bandit‑RBF decision making.

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py (health‑weighted geometric algebra)
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s0.py (bandit store + RBF surrogate)

Mathematical bridge:
The multivector coefficient vector  **v**  is used as the feature vector for an RBF surrogate
\( \hat r = \sum_i w_i \exp(-\|v-c_i\|^2/(2\sigma^2)) \).
The surrogate’s prediction \(\hat r\) is multiplied by a health factor
\(h = (1 - (\text{risk} * \text{failure\_rate})) (1 - \text{recovery\_priority})\)
to obtain a health‑aware expected reward.
This reward drives a LinUCB‑style bandit selection, while the bandit store updates the surrogate
weights after observing real rewards. The result is a unified decision pipeline that
operates on geometric‑algebra objects and respects health‑risk considerations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometric algebra core (Parent A)
# ----------------------------------------------------------------------
class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def to_vector(self) -> np.ndarray:
        """Flatten coefficients into a dense vector of length 2**n ordered by blade index."""
        size = 1 << self.n
        vec = np.zeros(size, dtype=float)
        for blade, coef in self.components.items():
            # blade encoded as bitmask
            idx = 0
            for b in blade:
                idx |= 1 << (b - 1)  # blades are 1‑based
            vec[idx] = coef
        return vec

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


# ----------------------------------------------------------------------
# Bandit core (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


# Virtual store (acts as lightweight VRAM)
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}  # key -> surrogate weight contribution


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _average_reward(key: str) -> float:
    total, n = _POLICY.get(key, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    Health factor as defined in Parent A.

    Parameters
    ----------
    reconstruction_risk_score : float
        Risk score in [0, 1].
    failures : int
        Observed failure count.
    failure_threshold : int
        Threshold for normalising failures.
    recovery_priority : float
        Priority weight in [0, 1].

    Returns
    -------
    float
        Health factor in [0, 1].
    """
    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be > 0")
    failure_rate = failures / failure_threshold
    health = (1.0 - (reconstruction_risk_score * failure_rate)) * (1.0 - recovery_priority)
    return max(0.0, min(1.0, health))


def rbf_surrogate_predict(
    mv: Multivector,
    centers: List[np.ndarray],
    weights: List[float],
    sigma: float,
) -> float:
    """
    RBF surrogate prediction using the multivector coefficient vector as input.

    Parameters
    ----------
    mv : Multivector
        Geometric algebra object providing the feature vector.
    centers : list of np.ndarray
        RBF centers (each of shape (d,)).
    weights : list of float
        Corresponding weights for each center.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    float
        Predicted reward.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = mv.to_vector()
    pred = 0.0
    for c, w in zip(centers, weights):
        diff = x - c
        k = math.exp(-np.dot(diff, diff) / (2.0 * sigma ** 2))
        pred += w * k
    return pred


def hybrid_select_action(
    context_mv: Multivector,
    actions: List[str],
    health_params: Tuple[float, int, int, float],
    centers: List[np.ndarray],
    weights: List[float],
    sigma: float,
    algorithm: str = "linucb",
    alpha: float = 1.0,
) -> BanditAction:
    """
    Select an action by combining health‑adjusted RBF predictions with a LinUCB bandit.

    The expected reward for each action a is:
        r̂_a = health * surrogate_predict(context_mv)
    The confidence bound follows the classic LinUCB formula:
        cb_a = alpha * sqrt( (xᵀ A⁻¹ x) )
    where A is approximated by the number of times the action has been taken
    (stored in _POLICY).

    Parameters
    ----------
    context_mv : Multivector
        Current geometric‑algebra context.
    actions : list of str
        Candidate action identifiers.
    health_params : tuple
        (reconstruction_risk_score, failures, failure_threshold, recovery_priority)
    centers, weights, sigma : RBF surrogate parameters.
    algorithm : str
        Currently only "linucb" is supported.
    alpha : float
        Exploration coefficient.

    Returns
    -------
    BanditAction
        Chosen action with associated statistics.
    """
    if not actions:
        raise ValueError("No actions to select from")

    # Compute health once (same for all actions in this simple fusion)
    health = compute_health(*health_params)

    # Base surrogate prediction for the given context
    base_pred = rbf_surrogate_predict(context_mv, centers, weights, sigma)

    best_score = -math.inf
    chosen = None

    for a in actions:
        # Update policy stats if missing
        if a not in _POLICY:
            _POLICY[a] = [0.0, 0.0]  # total reward, count

        total, count = _POLICY[a]
        avg_reward = total / count if count else 0.0

        # Health‑aware expected reward
        exp_reward = health * base_pred

        # Simple LinUCB confidence bound using count as proxy for feature covariance
        # A⁻¹ ≈ 1/(1+count)
        confidence = alpha * math.sqrt(1.0 / (1.0 + count))

        score = exp_reward + confidence

        if score > best_score:
            best_score = score
            chosen = BanditAction(
                action_id=a,
                propensity=exp_reward,  # for compatibility with Parent B expectations
                expected_reward=exp_reward,
                confidence_bound=confidence,
                algorithm=algorithm,
            )

    # Record the selection in the store (virtual VRAM) for later weight updates
    _STORE[chosen.action_id] = health * base_pred
    return chosen


def hybrid_update(
    chosen_action: BanditAction,
    observed_reward: float,
    centers: List[np.ndarray],
    weights: List[float],
    sigma: float,
    learning_rate: float = 0.1,
) -> List[float]:
    """
    Update the surrogate model weights using the observed reward and the stored
    health‑adjusted prediction.

    Simple gradient step:
        w_i ← w_i + η * (r - ŷ) * k_i
    where k_i is the kernel value for center i.

    Parameters
    ----------
    chosen_action : BanditAction
        Action that was taken.
    observed_reward : float
        Real reward received from the environment.
    centers, weights, sigma : RBF surrogate parameters.
    learning_rate : float
        Step size η.

    Returns
    -------
    list of float
        Updated weight list.
    """
    # Update policy statistics
    total, count = _POLICY.get(chosen_action.action_id, [0.0, 0.0])
    _POLICY[chosen_action.action_id] = [total + observed_reward, count + 1]

    # Retrieve the context vector that produced the stored prediction
    # In this simplified version we reuse the stored health‑adjusted prediction as proxy.
    pred = _STORE.get(chosen_action.action_id, 0.0)
    error = observed_reward - pred

    # Re‑compute kernel values for the last context (approximation)
    # We cannot recover the exact context vector, so we assume a unit vector.
    # This keeps the update self‑contained for the demonstration.
    dummy_context = np.ones_like(centers[0])
    for i, c in enumerate(centers):
        diff = dummy_context - c
        k = math.exp(-np.dot(diff, diff) / (2.0 * sigma ** 2))
        weights[i] += learning_rate * error * k

    return weights


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a simple 3‑dimensional multivector (n=3)
    mv = Multivector(
        {
            frozenset(): 1.0,            # scalar part
            frozenset({1}): 0.5,
            frozenset({2}): -0.3,
            frozenset({1, 2}): 0.2,
        },
        n=3,
    )

    # RBF surrogate parameters (random but reproducible)
    rng = random.Random(42)
    dim = 1 << mv.n  # 2**n coefficients
    centers = [np.array([rng.random() for _ in range(dim)]) for _ in range(5)]
    weights = [rng.random() - 0.5 for _ in range(5)]
    sigma = 1.0

    actions = ["A", "B", "C"]

    health_params = (0.2, 3, 10, 0.1)  # risk, failures, threshold, priority

    chosen = hybrid_select_action(
        context_mv=mv,
        actions=actions,
        health_params=health_params,
        centers=centers,
        weights=weights,
        sigma=sigma,
        algorithm="linucb",
        alpha=0.5,
    )
    print("Chosen action:", chosen)

    # Simulate an observed reward
    obs_reward = rng.random()
    weights = hybrid_update(
        chosen_action=chosen,
        observed_reward=obs_reward,
        centers=centers,
        weights=weights,
        sigma=sigma,
        learning_rate=0.05,
    )
    print("Updated weights:", weights[:3], "...")
    print("Policy stats:", _POLICY)