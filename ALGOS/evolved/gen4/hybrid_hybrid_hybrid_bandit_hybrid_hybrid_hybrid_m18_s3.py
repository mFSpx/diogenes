# DARWIN HAMMER — match 18, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s2.py (gen3)
# born: 2026-05-29T23:26:28Z

"""Hybrid Bandit‑RBF Model
This module fuses two Darwin‑Hammer parents:

* **Parent A** – a contextual multi‑armed bandit that tracks
  cumulative reward and uses a LinUCB‑style confidence bound.
* **Parent B** – an RBF surrogate that learns a nonlinear mapping
  from a vector (here the concatenation of context and one‑hot
  action encoding) to the observed reward by solving a linear
  system K·w = y where K_ij = exp(‑ε²‖c_i‑c_j‖²).

**Mathematical bridge** – The bandit’s expected reward
`_reward(a)` is no longer a simple empirical average; it is
replaced by the surrogate’s prediction for the vector
`[context, action_one_hot]`.  The surrogate is updated after every
BanditUpdate, keeping the bandit’s confidence term unchanged.
Thus the selection rule combines the LinUCB exploration term with
a nonlinear, data‑driven reward estimate, yielding a single unified
decision‑making system."""
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence

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


_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}                 # placeholder VRAM store (unused)
_SURROGATE = None                             # will hold an RBFSurrogate instance


def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)


def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# RBF surrogate (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel exp(‑(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with Gauss‑Jordan elimination (no external libs)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Return Σ_i w_i·K(x, c_i)."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def _encode_action(action_id: str, action_space: List[str]) -> List[float]:
    """One‑hot encode an action within the given action space."""
    encoding = [0.0] * len(action_space)
    try:
        idx = action_space.index(action_id)
    except ValueError as exc:
        raise ValueError(f"unknown action {action_id}") from exc
    encoding[idx] = 1.0
    return encoding


def _build_feature_vector(context: Dict[str, float],
                          action_id: str,
                          action_space: List[str]) -> List[float]:
    """Concatenate context values (sorted by key) with one‑hot action."""
    ctx_vals = [context[k] for k in sorted(context.keys())]
    return ctx_vals + _encode_action(action_id, action_space)


def update_surrogate(context: Dict[str, float],
                     action_id: str,
                     reward: float,
                     action_space: List[str],
                     epsilon: float = 1.0) -> None:
    """Add a new observation to the RBF surrogate and recompute weights."""
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=epsilon)

    # Build new center
    new_center = tuple(_build_feature_vector(context, action_id, action_space))
    new_centers = _SURROGATE.centers + [new_center]

    # Kernel matrix K_{ij} = gaussian(||c_i - c_j||)
    size = len(new_centers)
    K = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            K[i][j] = gaussian(euclidean(new_centers[i], new_centers[j]), epsilon)

    # Target vector y = observed rewards
    y = (_SURROGATE.weights + [reward]) if _SURROGATE.weights else [reward]

    # Solve for new weights
    new_weights = solve_linear(K, y)

    _SURROGATE = RBFSurrogate(centers=new_centers,
                              weights=new_weights,
                              epsilon=epsilon)


# ----------------------------------------------------------------------
# Hybrid selection (core fusion)
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a LinUCB‑style rule where the expected reward
    is supplied by the RBF surrogate.  The confidence term still depends
    on the empirical visit count stored in _POLICY.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Exploration branch for epsilon‑greedy
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        # Compute surrogate‑based reward estimates
        surrogate_rewards = {}
        for a in actions:
            if _SURROGATE and _SURROGATE.centers:
                feat = _build_feature_vector(context, a, actions)
                surrogate_rewards[a] = _SURROGATE.predict(feat)
            else:
                surrogate_rewards[a] = 0.0

        if algorithm == "thompson":
            # Sample from a Beta posterior centred on surrogate reward
            chosen = max(
                actions,
                key=lambda a: rng.betavariate(
                    1 + max(0.0, surrogate_rewards[a]),
                    1 + max(0.0, 1 - surrogate_rewards[a]),
                ),
            )
        else:  # LinUCB‑style
            scale = math.sqrt(
                sum(float(v) * float(v) for v in context.values())
            ) if context else 1.0
            chosen = max(
                actions,
                key=lambda a: surrogate_rewards[a]
                + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
            )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    expected = surrogate_rewards.get(chosen, _empirical_reward(chosen))
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_update(
    context: Dict[str, float],
    action_id: str,
    reward: float,
    action_space: List[str],
    epsilon: float = 1.0,
) -> None:
    """
    Record the observed reward in the bandit statistics and
    immediately incorporate the sample into the RBF surrogate.
    """
    # Update empirical statistics
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, cnt + 1]

    # Update surrogate model
    update_surrogate(context, action_id, reward, action_space, epsilon)


def social_adjust(
    context: Dict[str, float],
    g_best: Vector,
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> Dict[str, float]:
    """
    Apply the PSO‑style social interaction (Parent B) to the numeric
    context before feeding it to the selector.  Returns a new context
    dictionary with adjusted values.
    """
    ctx_vec = [context[k] for k in sorted(context.keys())]
    if len(ctx_vec) != len(g_best):
        raise ValueError("dimension mismatch between context and g_best")
    adjusted = np.array(
        [xi + (r1 if r1 is not None else random.Random(seed).random()) *
         (gj - k * xi) for xi, gj in zip(ctx_vec, g_best)]
    )
    # Re‑map back to the original keys preserving order
    new_context = dict(zip(sorted(context.keys()), adjusted.tolist()))
    return new_context


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()
    actions = ["A", "B", "C"]
    # Simulated global best vector for social adjustment (dimension = context size)
    g_best_vec = [0.5, 0.5, 0.5]

    for t in range(5):
        # Random context
        raw_context = {
            "x1": random.random(),
            "x2": random.random(),
            "x3": random.random(),
        }

        # Optional social interaction step
        ctx = social_adjust(raw_context, g_best_vec, k=1, seed=t)

        # Select action
        ba = hybrid_select_action(ctx, actions, algorithm="linucb", seed=t)

        # Simulated reward: higher if action matches a hidden rule
        # (e.g., reward = 1 if sum(context) + action_index > 1.5)
        idx = actions.index(ba.action_id)
        reward = 1.0 if sum(ctx.values()) + idx > 1.5 else 0.0

        # Update models
        hybrid_update(ctx, ba.action_id, reward, actions)

        print(
            f"t={t} ctx={list(ctx.values())[:3]} "
            f"chosen={ba.action_id} exp={ba.expected_reward:.3f} "
            f"reward={reward}"
        )