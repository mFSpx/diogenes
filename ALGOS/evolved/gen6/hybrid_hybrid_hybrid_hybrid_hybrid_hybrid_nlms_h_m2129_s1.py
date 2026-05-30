# DARWIN HAMMER — match 2129, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_sparse_wta_hy_m884_s1.py (gen3)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m545_s2.py (gen5)
# born: 2026-05-29T23:40:55Z

"""Hybrid Algorithm: Fusion of Bandit Router (Parent A) and NLMS‑RBF‑Hoeffding (Parent B)

Parent A:  hybrid_hybrid_hybrid_bandit_hybrid_sparse_wta_hy_m884_s1.py  
Parent B:  hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m545_s2.py  

Mathematical Bridge
-------------------
Both parents manipulate *vectors* that represent either actions (bandit policy) or
features (kernel‑based surrogate).  The bridge is built on a **similarity‑weighted
bandit signal**:

* The bandit router supplies an *expected reward* 𝑟̂ₐ and a *propensity* πₐ for each
  action 𝑎.
* The NLMS learner needs an adaptive step‑size μ that depends on the confidence of
  the chosen action.  We set  

      μₐ = ε · (1 + 𝑟̂ₐ) / (‖x‖² + δ)  

  where ε is the base NLMS learning‑rate and δ a stability constant.
* Action selection is performed by a kernel similarity matrix K built from stored
  feature prototypes.  For a new context x the similarity vector  

      sₐ = K[x, prototypeₐ]  

  is multiplied element‑wise by the bandit expected reward, yielding a *precision‑modulated*
  score  

      scoreₐ = 𝑟̂ₐ · sₐ .

  The action with maximal score is routed to the NLMS update.

Thus the variational‑free‑energy idea (confidence → precision) of Parent A modulates the
gradient‑descent‑like NLMS update of Parent B, while the RBF kernel of Parent B supplies
the similarity measure that drives the bandit’s decision policy.

The code below implements this fusion with three core functions:
`update_policy`, `nlms_update`, and `select_action`.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit Router core (Parent A)
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


# Global mutable policy store: action_id → [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()


def _average_reward(action_id: str) -> float:
    """Return the empirical mean reward for an action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of observed rewards into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def get_bandit_action(action_id: str) -> BanditAction:
    """Create a BanditAction snapshot from the current policy."""
    exp_reward = _average_reward(action_id)
    # For simplicity we use a Hoeffding bound as confidence.
    n = _POLICY.get(action_id, [0.0, 0.0])[1]
    conf = hoeffding_bound(r=1.0, delta=0.05, n=int(n)) if n else float('inf')
    return BanditAction(
        action_id=action_id,
        propensity=1.0,  # placeholder – could be derived from softmax of rewards
        expected_reward=exp_reward,
        confidence_bound=conf,
        algorithm="HybridBandit"
    )


# ----------------------------------------------------------------------
# NLMS‑RBF‑Hoeffding core (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> np.ndarray:
    """Compute the full RBF kernel matrix for a dict of feature vectors."""
    n = len(features)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[i], features[j])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding confidence bound."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def nlms_update(
    weights: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    epsilon: float,
    bandit_reward: float,
    delta: float = 1e-8,
) -> np.ndarray:
    """
    Perform a single Normalized LMS update where the step size μ is modulated
    by the bandit expected reward.

    μ = ε · (1 + r̂) / (‖x‖² + δ)

    Returns the updated weight vector.
    """
    norm_sq = float(np.dot(input_vec, input_vec)) + delta
    mu = epsilon * (1.0 + bandit_reward) / norm_sq
    error = desired - float(np.dot(weights, input_vec))
    new_weights = weights + mu * error * input_vec
    return new_weights


def haversine_distance(
    loc1: Tuple[float, float], loc2: Tuple[float, float]
) -> float:
    """Great‑circle distance in kilometers between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, loc1)
    lat2, lon2 = map(math.radians, loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def compute_resource_vector(
    entity: dict,
    reference_location: Tuple[float, float],
    beta: float,
    sigma: float,
    bandit_propensity: float,
) -> List[float]:
    """
    Extend the original resource vector with a term proportional to the
    bandit propensity, thereby coupling privacy‑model resources with bandit
    confidence.
    """
    distance = haversine_distance(entity["location"], reference_location)
    pi = beta * sigma
    score = entity["score"]
    # Propensity term scales the distance component.
    weighted_distance = distance * (1.0 + bandit_propensity)
    return [weighted_distance, pi, score]


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
class ModelPool:
    """
    Stores prototype feature vectors for each action.  The pool is sparse:
    only actions that have been selected at least once are kept.
    """

    def __init__(self) -> None:
        self._features: Dict[str, List[float]] = {}
        self._next_idx: int = 0
        self._index_map: Dict[str, int] = {}

    def add(self, action_id: str, feature_vec: List[float]) -> None:
        """Add or replace the prototype for an action."""
        self._features[action_id] = feature_vec
        if action_id not in self._index_map:
            self._index_map[action_id] = self._next_idx
            self._next_idx += 1

    def get_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return the stacked feature matrix and the corresponding action order."""
        ordered_actions = list(self._features.keys())
        matrix = np.vstack([self._features[a] for a in ordered_actions])
        return matrix, ordered_actions

    def get_action_by_index(self, idx: int) -> str:
        """Reverse lookup from matrix row index to action_id."""
        for aid, i in self._index_map.items():
            if i == idx:
                return aid
        raise KeyError(f"No action for index {idx}")


def select_action(
    context_vec: List[float],
    pool: ModelPool,
    epsilon: float = 1.0,
) -> Tuple[str, float]:
    """
    Choose an action by combining RBF similarity with bandit expected reward.

    1. Compute kernel similarity between the context and every prototype.
    2. Multiply each similarity by the action's expected reward.
    3. Return the action with the highest product and the associated score.
    """
    if not pool._features:
        raise ValueError("ModelPool is empty")

    # Build kernel matrix for prototypes.
    proto_mat, actions = pool.get_feature_matrix()
    # Compute similarity of context to each prototype.
    sims = np.array(
        [
            gaussian(euclidean(context_vec, proto.tolist()), epsilon)
            for proto in proto_mat
        ]
    )
    # Retrieve bandit expected rewards.
    rewards = np.array([_average_reward(aid) for aid in actions])
    # Precision‑modulated scores.
    scores = sims * (1.0 + rewards)
    best_idx = int(np.argmax(scores))
    best_action = actions[best_idx]
    return best_action, float(scores[best_idx])


def hybrid_step(
    context_vec: List[float],
    desired_output: float,
    pool: ModelPool,
    weights: np.ndarray,
    nlms_eps: float = 0.1,
) -> Tuple[np.ndarray, str]:
    """
    Perform a full hybrid iteration:
    * Select an action via similarity‑reward fusion.
    * Update NLMS weights using the bandit’s expected reward for that action.
    * Return the updated weights and the chosen action.
    """
    action_id, score = select_action(context_vec, pool)
    bandit = get_bandit_action(action_id)

    # Convert to numpy for NLMS.
    x = np.array(context_vec, dtype=np.float64)
    new_weights = nlms_update(
        weights=weights,
        input_vec=x,
        desired=desired_output,
        epsilon=nlms_eps,
        bandit_reward=bandit.expected_reward,
    )
    # Optionally, update the prototype with the new context (sparse WTA style).
    pool.add(action_id, context_vec)
    return new_weights, action_id


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny policy.
    reset_policy()
    updates = [
        BanditUpdate(context_id="c1", action_id="a1", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="a2", reward=0.2, propensity=0.5),
        BanditUpdate(context_id="c3", action_id="a1", reward=0.8, propensity=0.5),
    ]
    update_policy(updates)

    # Build a model pool with two prototype vectors.
    pool = ModelPool()
    pool.add("a1", [0.2, 0.1, 0.4])
    pool.add("a2", [0.9, 0.8, 0.7])

    # Random context and desired output.
    ctx = [0.3, 0.2, 0.5]
    desired = 1.0

    # Initialise NLMS weights.
    w = np.zeros(len(ctx), dtype=np.float64)

    # Run a hybrid step.
    new_w, chosen = hybrid_step(context_vec=ctx, desired_output=desired, pool=pool, weights=w)

    print(f"Chosen action: {chosen}")
    print(f"Updated weights: {new_w}")

    # Demonstrate resource vector computation.
    entity = {"location": (37.7749, -122.4194), "score": 42.0}
    ref_loc = (34.0522, -118.2437)
    prop = get_bandit_action(chosen).propensity
    rv = compute_resource_vector(entity, ref_loc, beta=0.3, sigma=0.7, bandit_propensity=prop)
    print(f"Resource vector for {chosen}: {rv}")

    sys.exit(0)