# DARWIN HAMMER — match 4708, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2169_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m717_s2.py (gen4)
# born: 2026-05-29T23:57:43Z

"""Hybrid Bandit‑Tree‑Lens Algorithm
=================================

This module fuses the mathematical cores of the two parent algorithms:

* **Parent A** – a count‑min sketch **S** that aggregates (context, action)
  co‑occurrences and a tropical (max‑plus) combination with per‑action reward
  estimates **rₐ**:

      score_tropₐ = max_i ( S[i, h(a)] + rₐ )

* **Parent B** – a ternary “lens” vector **Lₐ**∈{0,1}³, a nine‑dimensional
  regex‑derived count vector **cₐ**∈ℕ⁹ and a learned fusion matrix **F**∈ℝ³ˣ⁹,
  yielding an additive bias

      score_lensₐ = Lₐ · (F · cₐ)

Both families produce a scalar that can be added to a confidence‑bound term.
The **mathematical bridge** is therefore the scalar additive structure:
we simply sum the tropical score, the lens bias, and a Hoeffding‑based
confidence interval (or classic UCB term).  The resulting hybrid selection
criterion for action *a* is

    H(a) = score_tropₐ
           + score_lensₐ
           + √(α·log N / Nₐ)               # Hoeffding/UCB confidence
           + λ̂·log n̂                       # RLCT regularisation

where the symbols retain the meaning of the original parents.
The implementation below provides the sketch, tropical product, lens fusion,
RLCT estimation, and a unified `select_hybrid_action` routine.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives (Count‑Min Sketch)
# ----------------------------------------------------------------------


class CountMinSketch:
    """Count‑Min sketch for non‑negative integer streams.

    Provides an unbiased estimate of the total count for any key.
    """

    def __init__(self, width: int = 2000, depth: int = 5, seed: int = 0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, key: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(key.encode("utf-8"))
        h.update(self.seeds[i].to_bytes(4, "little"))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, key: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: str) -> int:
        """Return the minimum count over all hash rows."""
        return min(self.tables[i, self._hash(key, i)] for i in range(self.depth))


# ----------------------------------------------------------------------
# Tropical (max‑plus) score
# ----------------------------------------------------------------------


def tropical_score(sketch: CountMinSketch, action_id: str, reward_est: float) -> float:
    """Compute max_i ( S[i, h(action)] + reward_est ).

    The sketch stores integer counts; we cast them to float before the
    max‑plus operation.
    """
    max_val = -math.inf
    for i in range(sketch.depth):
        idx = sketch._hash(action_id, i)
        val = float(sketch.tables[i, idx]) + reward_est
        if val > max_val:
            max_val = val
    return max_val


# ----------------------------------------------------------------------
# Lens‑fusion score
# ----------------------------------------------------------------------


def lens_score(lens_vec: np.ndarray, count_vec: np.ndarray, F: np.ndarray) -> float:
    """Compute L · (F · c) = Σ_i Σ_j L_i c_j F_{ij}."""
    return float(lens_vec @ (F @ count_vec))


# ----------------------------------------------------------------------
# RLCT (Real‑Log‑Canonical‑Threshold) estimator
# ----------------------------------------------------------------------


class RLCTEstimator:
    """Online linear regression on (log n, loss) to estimate λ̂.

    We fit loss ≈ λ̂ * log n  (no intercept) using a simple running average.
    """

    def __init__(self):
        self._sum_log_n = 0.0
        self._sum_loss = 0.0
        self._count = 0

    def update(self, n: int, loss: float) -> None:
        if n <= 0:
            return
        self._sum_log_n += math.log(n)
        self._sum_loss += loss
        self._count += 1

    @property
    def lambda_hat(self) -> float:
        if self._sum_log_n == 0.0:
            return 0.0
        return self._sum_loss / self._sum_log_n


# ----------------------------------------------------------------------
# Hybrid bandit‑tree‑lens core
# ----------------------------------------------------------------------


class HybridBanditTreeLens:
    """Container for all state needed by the hybrid algorithm."""

    def __init__(
        self,
        actions: List[str],
        width: int = 2000,
        depth: int = 5,
        lens_matrix: np.ndarray = None,
        alpha: float = 1.0,
    ):
        self.actions = actions
        self.sketch = CountMinSketch(width=width, depth=depth)
        self.total_pulls = 0
        self.action_counts: Dict[str, int] = {a: 0 for a in actions}
        self.reward_sums: Dict[str, float] = {a: 0.0 for a in actions}
        self.distinct_contexts = set()  # simple surrogate for HyperLogLog
        self.rlct = RLCTEstimator()
        self.alpha = alpha

        # Lens fusion matrix F (3×9).  If not supplied we initialise randomly.
        if lens_matrix is None:
            rng = np.random.default_rng(0)
            self.F = rng.normal(scale=0.1, size=(3, 9))
        else:
            self.F = lens_matrix

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _reward_estimate(self, a: str) -> float:
        """Mean reward estimate μ̂_a (sketch‑based)."""
        count = self.action_counts[a]
        if count == 0:
            return 0.0
        return self.reward_sums[a] / count

    def _confidence(self, a: str) -> float:
        """Hoeffding/UCB style confidence term."""
        n_a = self.action_counts[a]
        if n_a == 0:
            return float("inf")
        return math.sqrt(self.alpha * math.log(self.total_pulls + 1) / n_a)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        action: str,
        reward: float,
        context: str,
        lens_vec: np.ndarray,
        count_vec: np.ndarray,
    ) -> None:
        """Incorporate a new (action, reward, context) observation.

        Parameters
        ----------
        action : str
            Identifier of the taken action.
        reward : float
            Observed scalar reward.
        context : str
            Arbitrary string describing the context (used for distinct‑count).
        lens_vec : np.ndarray shape (3,)
            Ternary lens vector for the current context.
        count_vec : np.ndarray shape (9,)
            Regex‑derived count vector for the current context.
        """
        # Update global counters
        self.total_pulls += 1
        self.action_counts[action] += 1
        self.reward_sums[action] += reward

        # Update sketches
        self.sketch.add(f"{action}:{context}", increment=1)

        # Distinct‑context surrogate
        self.distinct_contexts.add(context)

        # RLCT update (using loss = 1 - reward as a simple proxy)
        loss = 1.0 - reward
        self.rlct.update(len(self.distinct_contexts), loss)

        # Store latest lens/count vectors per action for scoring
        # (simple dicts; in a real system we would keep a rolling buffer)
        if not hasattr(self, "_lens_store"):
            self._lens_store = {}
        if not hasattr(self, "_count_store"):
            self._count_store = {}
        self._lens_store[action] = lens_vec
        self._count_store[action] = count_vec

    def hybrid_score(self, action: str) -> float:
        """Compute the unified hybrid score H(a)."""
        # Tropical component
        r_hat = self._reward_estimate(action)
        trop = tropical_score(self.sketch, action, r_hat)

        # Lens component (fallback to zeros if not yet observed)
        lens_vec = getattr(self, "_lens_store", {}).get(action, np.zeros(3))
        count_vec = getattr(self, "_count_store", {}).get(action, np.zeros(9))
        lens = lens_score(lens_vec, count_vec, self.F)

        # Confidence and RLCT terms
        conf = self._confidence(action)
        rlct_term = self.rlct.lambda_hat * math.log(max(len(self.distinct_contexts), 1))

        return trop + lens + conf + rlct_term

    def select_hybrid_action(self, epsilon: float = 0.1) -> str:
        """ε‑greedy selection based on the hybrid score."""
        if random.random() < epsilon:
            return random.choice(self.actions)

        scores = {a: self.hybrid_score(a) for a in self.actions}
        # Higher score = more promising (since tropical and lens are additive)
        return max(scores, key=scores.get)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny action set
    actions = ["A", "B", "C"]
    hybrid = HybridBanditTreeLens(actions)

    # Synthetic context generator
    contexts = ["ctx1", "ctx2", "ctx3"]
    rng = np.random.default_rng(42)

    for t in range(100):
        # Randomly pick a context
        ctx = random.choice(contexts)

        # Random lens (binary ternary) and count vector (small integers)
        lens = rng.integers(0, 2, size=3).astype(float)
        count = rng.integers(0, 5, size=9).astype(float)

        # Choose action via ε‑greedy hybrid policy
        act = hybrid.select_hybrid_action(epsilon=0.2)

        # Simulate a stochastic reward (higher for action "A")
        base = {"A": 0.7, "B": 0.4, "C": 0.2}[act]
        reward = rng.random() < base
        reward = 1.0 if reward else 0.0

        # Feed observation back
        hybrid.observe(act, reward, ctx, lens, count)

    # Final selection (pure exploitation)
    best = hybrid.select_hybrid_action(epsilon=0.0)
    print(f"Best action after training: {best}")
    print("Hybrid scores:")
    for a in actions:
        print(f"  {a}: {hybrid.hybrid_score(a):.4f}")