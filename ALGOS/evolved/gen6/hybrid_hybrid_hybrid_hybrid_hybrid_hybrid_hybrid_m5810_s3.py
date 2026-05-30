# DARWIN HAMMER — match 5810, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_bandit_router_m206_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m917_s3.py (gen5)
# born: 2026-05-30T00:04:49Z

"""Hybrid Bandit‑Koopman‑Sketch Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit that keeps global per‑action
  reward statistics, a temperature model, and an online linear‑regression
  linking a model‑based rate to observed reward.

* **Parent B** – a Count‑Min sketch whose bucket frequencies are interpreted as
  coefficients of a multivector in a Clifford algebra.  A learned Koopman
  operator evolves this coefficient vector linearly; a Bayesian Beta update
  converts the evolved coefficients into a posterior probability distribution
  that modulates pheromone‑like weights.

**Mathematical bridge**  
The bridge is the *coefficient vector* **c** ∈ ℝᵈ obtained from the sketch.
We treat **c** as the coordinate representation of a multivector **M**.  The
Koopman operator **K** ∈ ℝ^{d×d} advances the multivector:  

  **c′ = K c**  

The resulting vector **c′** is fed to a per‑bucket Beta posterior
(α_i, β_i) → posterior mean μ_i = α_i/(α_i+β_i).  These means form a
probability distribution **p** over the sketch buckets.  We then map each
bucket back to the corresponding action (via a deterministic hash) and
combine the bucket probability with the bandit’s empirical mean reward
r_a and temperature‑adjusted propensity θ_a:

  propensity_a = p_{bucket(a)} · exp(−|T_a−25|/10)  

  expected_reward_a = r_a · β̂  (β̂ is the online regression slope)

Finally a `BanditAction` is produced for each candidate action.  The
algorithm therefore unifies the linear‑operator dynamics of Parent B with the
reward‑driven exploration of Parent A.

The implementation below contains three public functions that showcase the
hybrid workflow:

1. `update_hybrid` – updates bandit statistics *and* the Count‑Min sketch.
2. `evolve_coefficients` – applies the Koopman operator to the sketch vector.
3. `compute_hybrid_actions` – builds `BanditAction` objects by fusing the
   Bayesian‑sketch probabilities, temperature model, and bandit rewards.

A tiny smoke test is provided under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, FrozenSet, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit core (global statistics)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBanditKoopmanSketch"


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


def update_policy(updates: Sequence[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    global _BETA, _BETA_SUM_XX, _BETA_SUM_XY
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

        # Online regression: we treat propensity as predictor x,
        # reward as response y.
        x = float(u.propensity)
        y = float(u.reward)
        _BETA_SUM_XX += x * x
        _BETA_SUM_XY += x * y
        if _BETA_SUM_XX != 0.0:
            _BETA = _BETA_SUM_XY / _BETA_SUM_XX


# ----------------------------------------------------------------------
# Parent B – Count‑Min Sketch + Koopman + Bayesian bridge
# ----------------------------------------------------------------------


class CountMinSketch:
    """Simple Count‑Min sketch with d hash functions and width w."""

    def __init__(self, d: int = 4, w: int = 256, seed: int = 0):
        self.d = d
        self.w = w
        self.tables = np.zeros((d, w), dtype=np.float64)
        rng = random.Random(seed)
        self._hash_seeds = [rng.randint(1, 2**31 - 1) for _ in range(d)]

    def _hash(self, item: str, i: int) -> int:
        h = hash((item, self._hash_seeds[i]))
        return h % self.w

    def add(self, item: str, increment: float = 1.0) -> None:
        for i in range(self.d):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item: str) -> float:
        """Return the minimum count across hash tables (standard CM estimate)."""
        mins = [self.tables[i, self._hash(item, i)] for i in range(self.d)]
        return min(mins)

    def vector(self) -> np.ndarray:
        """Flatten the sketch into a 1‑D coefficient vector (length d·w)."""
        return self.tables.ravel()


def _initialize_koopman(dim: int, seed: int = 1) -> np.ndarray:
    """Return a random (but reproducible) Koopman operator K ∈ ℝ^{dim×dim}."""
    rng = np.random.default_rng(seed)
    # Small spectral radius to keep evolution stable
    K = rng.normal(loc=0.0, scale=0.05, size=(dim, dim))
    # Add identity component
    K += np.eye(dim) * 0.9
    return K


def _bayesian_posterior_means(counts: np.ndarray, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> np.ndarray:
    """
    For each bucket i we keep a Beta(α_i, β_i) posterior where
        α_i = prior_alpha + count_i
        β_i = prior_beta + (total_counts - count_i)
    The posterior mean μ_i = α_i / (α_i + β_i).
    """
    total = counts.sum()
    alpha = prior_alpha + counts
    beta = prior_beta + (total - counts)
    # Avoid division by zero
    denom = alpha + beta
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(denom > 0, alpha / denom, 0.0)
    return means


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


def evolve_coefficients(
    sketch: CountMinSketch,
    K: np.ndarray,
) -> np.ndarray:
    """
    Apply the Koopman operator to the sketch coefficient vector.
    Returns the evolved coefficient vector c′ = K c.
    """
    c = sketch.vector()
    if K.shape != (c.size, c.size):
        raise ValueError(f"Koopman matrix shape {K.shape} does not match coefficient size {c.size}")
    c_prime = K @ c
    # Ensure non‑negative coefficients (they represent frequencies)
    c_prime = np.maximum(c_prime, 0.0)
    return c_prime


def compute_hybrid_actions(
    action_ids: Sequence[str],
    sketch: CountMinSketch,
    K: np.ndarray,
    temperature_scale: float = 10.0,
) -> List[BanditAction]:
    """
    Build a list of ``BanditAction`` objects by fusing three sources:

    1. **Bayesian‑sketch probabilities** derived from the evolved sketch vector.
    2. **Temperature adjustment** based on the per‑action temperature.
    3. **Bandit reward statistics** together with the online regression slope.
    """
    # 1) Evolve sketch frequencies via Koopman
    evolved = evolve_coefficients(sketch, K)

    # 2) Bayesian posterior means per bucket
    bucket_means = _bayesian_posterior_means(evolved)

    # Normalise to obtain a proper distribution over buckets
    if bucket_means.sum() == 0:
        bucket_probs = np.full_like(bucket_means, 1.0 / bucket_means.size)
    else:
        bucket_probs = bucket_means / bucket_means.sum()

    actions: List[BanditAction] = []
    for a_id in action_ids:
        # Map action to a deterministic bucket (first hash function)
        bucket_idx = sketch._hash(a_id, 0) + sketch.d * sketch.w * 0  # first table offset
        prob_bucket = bucket_probs[bucket_idx]

        # Temperature factor (higher when closer to 25 °C)
        temp = _ACTION_TEMPS.get(a_id, 25.0)  # default 25 °C if unknown
        temp_factor = math.exp(-abs(temp - 25.0) / temperature_scale)

        # Propensity = sketch probability * temperature factor
        propensity = prob_bucket * temp_factor

        # Expected reward uses bandit empirical mean scaled by regression slope
        exp_reward = _reward(a_id) * _BETA

        # Confidence bound – simple sqrt(1/count) scaled by regression slope
        n = _count(a_id)
        conf = (0.0 if n == 0 else math.sqrt(1.0 / n) * _BETA)

        actions.append(
            BanditAction(
                action_id=a_id,
                propensity=propensity,
                expected_reward=exp_reward,
                confidence_bound=conf,
            )
        )
    return actions


def update_hybrid(
    bandit_updates: Sequence[BanditUpdate],
    sketch_updates: Sequence[Tuple[str, float]],
    sketch: CountMinSketch,
) -> None:
    """
    Perform a joint update:

    * Bandit statistics are updated via ``update_policy``.
    * Sketch counts are incremented with ``sketch_updates`` (list of (item, inc)).
    """
    update_policy(bandit_updates)
    for item, inc in sketch_updates:
        sketch.add(item, inc)


# ----------------------------------------------------------------------
# Utility helpers for the demo
# ----------------------------------------------------------------------


def _set_action_temperature(action_id: str, temperature_c: float) -> None:
    """Assign a temperature (in °C) to a specific action."""
    _ACTION_TEMPS[action_id] = temperature_c


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Seed reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define a small set of actions
    actions = ["click_A", "click_B", "click_C"]

    # Initialise temperatures (simulate a biologically‑inspired temperature map)
    for a in actions:
        _set_action_temperature(a, random.uniform(15.0, 35.0))

    # Initialise the Count‑Min sketch and Koopman operator
    sketch = CountMinSketch(d=4, w=64, seed=123)
    K = _initialize_koopman(dim=sketch.d * sketch.w, seed=7)

    # Simulate a batch of observations
    bandit_batch = [
        BanditUpdate(context_id="sess1", action_id="click_A", reward=1.0, propensity=0.3),
        BanditUpdate(context_id="sess2", action_id="click_B", reward=0.0, propensity=0.5),
        BanditUpdate(context_id="sess3", action_id="click_C", reward=1.0, propensity=0.2),
    ]

    sketch_batch = [
        ("click_A", 1.0),
        ("click_B", 2.0),
        ("click_C", 1.5),
        ("click_A", 0.5),
    ]

    # Joint update
    update_hybrid(bandit_batch, sketch_batch, sketch)

    # Compute hybrid actions
    hybrid_actions = compute_hybrid_actions(actions, sketch, K)

    # Simple selection: pick action with highest (propensity + expected_reward)
    best = max(hybrid_actions, key=lambda x: x.propensity + x.expected_reward)
    print("Hybrid actions:")
    for act in hybrid_actions:
        print(
            f"  {act.action_id:8s} | propensity={act.propensity:.4f} | "
            f"exp_reward={act.expected_reward:.4f} | conf={act.confidence_bound:.4f}"
        )
    print(f"\nSelected action: {best.action_id}")

    # Verify that code runs without exception
    sys.exit(0)