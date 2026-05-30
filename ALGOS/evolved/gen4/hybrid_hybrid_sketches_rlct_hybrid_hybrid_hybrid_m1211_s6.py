# DARWIN HAMMER — match 1211, survivor 6
# gen: 4
# parent_a: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s1.py (gen3)
# born: 2026-05-29T23:34:24Z

"""Hybrid Sketch‑RLCT‑Bandit Router

This module fuses the *sketch + RLCT* machinery from
`hybrid_sketches_rlct_grokking_m5_s0.py` with the *bandit router* logic from
`hybrid_hybrid_ternar_hybrid_bandit_router_m202_s1.py`.

Mathematical bridge
-------------------
* The Count‑Min sketch (and its flattened vector) provides a low‑dimensional
  representation **s** of a data batch.
* The Real Log‑Canonical Threshold (RLCT) is estimated from the frequency
  distribution of the sketch and yields a scalar **γ** that quantifies the
  curvature of the loss surface, i.e. the amount of information loss caused by
  the dimensionality reduction.
* In a stochastic bandit setting the confidence bound of an action is often
  proportional to the inverse square‑root of the number of observations.
  Here we replace the generic variance term with **γ**, thus the bound becomes
  `√γ / √t`.  This directly couples the sketch‑derived RLCT to the exploration
  term of the bandit.
* The reward for an action is taken as the similarity (cosine similarity) between
  consecutive sketch vectors, i.e. the amount of information preserved after a
  new batch arrives.

The resulting hybrid algorithm simultaneously
1. compresses streaming data,
2. measures information loss via RLCT,
3. drives a contextual bandit whose confidence intervals are informed by that
   loss, and
4. routes decisions based on the updated expected rewards.

"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Sketch utilities (from Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch table for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def flatten_sketch(sketch: List[List[int]]) -> np.ndarray:
    """Flatten a 2‑D sketch into a 1‑D float vector."""
    return np.asarray([float(v) for row in sketch for v in row], dtype=np.float64)


def estimate_rlct_from_losses(train_losses_per_n: List[float],
                              n_values: List[int]) -> float:
    """Estimate the Real Log‑Canonical Threshold (RLCT) from loss vs. n data."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


# ----------------------------------------------------------------------
# Bandit utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A derived scalar that can be used as a dynamic scaling factor."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe for zero vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class HybridBanditRouter:
    """A ternary router whose confidence bounds are driven by RLCT."""

    def __init__(self, actions: List[str]):
        self.t = 0  # time step / number of updates
        self.actions: Dict[str, BanditAction] = {
            aid: BanditAction(
                action_id=aid,
                propensity=1.0 / len(actions),
                expected_reward=0.0,
                confidence_bound=0.0,
            )
            for aid in actions
        }
        self.store = StoreState()

    def _update_confidence(self, rlct: float) -> None:
        """Refresh confidence bounds using the latest RLCT estimate."""
        self.t += 1
        for aid, act in self.actions.items():
            # classic UCB term sqrt(2 * log(t) / n) -> replace variance with rlct
            bound = math.sqrt(rlct / max(1, self.t))
            self.actions[aid] = BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=act.expected_reward,
                confidence_bound=bound,
                algorithm=act.algorithm,
            )

    def select_action(self) -> BanditAction:
        """Select the action with highest upper confidence value."""
        best = max(
            self.actions.values(),
            key=lambda a: a.expected_reward + a.confidence_bound,
        )
        return best

    def incorporate_feedback(self, update: BanditUpdate, rlct: float) -> None:
        """Update expected reward for the chosen action and recompute bounds."""
        act = self.actions[update.action_id]
        # Incremental average reward
        n = self.t + 1  # pretend we have seen n samples for this action
        new_exp = ((act.expected_reward * (n - 1)) + update.reward) / n
        self.actions[update.action_id] = BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,
            expected_reward=new_exp,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm,
        )
        # Update the global confidence using the latest RLCT
        self._update_confidence(rlct)


def hybrid_process(
    data_batches: List[List[Any]],
    width: int = 64,
    depth: int = 4,
    action_ids: Tuple[str, str, str] = ("A", "B", "C"),
) -> List[BanditAction]:
    """
    Process a sequence of data batches.

    For each batch:
    1. Build a Count‑Min sketch and flatten it.
    2. Estimate RLCT from the non‑zero frequencies of the sketch.
    3. Compute similarity with the previous sketch (reward).
    4. Feed the reward to a bandit router whose confidence bounds are scaled by RLCT.
    5. Return the selected action after each update.
    """
    router = HybridBanditRouter(list(action_ids))
    previous_vec: np.ndarray | None = None
    selected_actions: List[BanditAction] = []

    for batch_idx, batch in enumerate(data_batches):
        # 1. Sketch
        sketch = count_min_sketch(batch, width=width, depth=depth)
        vec = flatten_sketch(sketch)

        # 2. RLCT estimation from sketch frequencies
        losses = [int(v) for v in vec if v > 0]
        n_vals = list(range(1, len(losses) + 1))
        rlct = (
            estimate_rlct_from_losses(losses, n_vals) if len(losses) > 1 else 0.0
        )

        # 3. Reward = similarity to previous sketch
        reward = (
            cosine_similarity(previous_vec, vec) if previous_vec is not None else 0.0
        )
        previous_vec = vec.copy()

        # 4. Bandit decision
        chosen = router.select_action()
        selected_actions.append(chosen)

        # 5. Update router with feedback
        update = BanditUpdate(
            context_id=f"batch_{batch_idx}",
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity,
        )
        router.incorporate_feedback(update, rlct)

        # Optional store dynamics (demonstrates the StoreState usage)
        inflow = [reward]
        outflow = [chosen.confidence_bound]
        router.store.update(inflow, outflow)

    return selected_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic streaming data: each batch is a list of integers
    random.seed(42)
    synthetic_batches = [
        [random.randint(0, 100) for _ in range(200)] for _ in range(10)
    ]

    actions = hybrid_process(synthetic_batches)

    for i, act in enumerate(actions):
        print(
            f"Batch {i:02d}: selected action = {act.action_id}, "
            f"exp_reward = {act.expected_reward:.4f}, "
            f"conf_bound = {act.confidence_bound:.4f}"
        )