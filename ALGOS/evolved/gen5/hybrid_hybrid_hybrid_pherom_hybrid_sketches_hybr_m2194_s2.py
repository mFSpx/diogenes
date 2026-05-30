# DARWIN HAMMER — match 2194, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3.py (gen2)
# parent_b: hybrid_sketches_hybrid_hybrid_bandit_m850_s0.py (gen4)
# born: 2026-05-29T23:41:24Z

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np
from typing import List, Tuple, Iterable, Callable

# ----------------------------------------------------------------------
# Epistemic certainty flag (parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True, slots=True)
class CertaintyFlag:
    """Immutable container for an epistemic certainty flag."""
    label: str
    confidence_bps: int  # basis points, 0..10000 → 0%..100%
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ----------------------------------------------------------------------
# Count‑Min Sketch with optional decay (parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Fixed‑size Count‑Min sketch supporting exponential decay.
    Decay is applied lazily on each `add` call to keep counts bounded.
    """

    def __init__(self, width: int = 2_048, depth: int = 4, decay_rate: float = 0.0):
        """
        Parameters
        ----------
        width : int
            Number of columns per hash table row.
        depth : int
            Number of independent hash functions (rows).
        decay_rate : float
            Exponential decay factor applied per update (0 ≤ decay_rate < 1).
            0 means no decay; 0.01 corresponds to ~1% decay each update.
        """
        if not (0.0 <= decay_rate < 1.0):
            raise ValueError("decay_rate must be in [0, 1)")
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.float64)
        self.seeds = [i * 0x9e3779b9 for i in range(depth)]
        self.decay_rate = decay_rate
        self._last_decay = 0  # count of updates since last decay

    def _hash(self, item: str, seed: int) -> int:
        h = hashlib.blake2b(digest_size=8, key=seed.to_bytes(8, "little"))
        h.update(item.encode("utf-8"))
        return int.from_bytes(h.digest(), "little") % self.width

    def _apply_decay(self) -> None:
        """Apply exponential decay to the entire table."""
        if self.decay_rate == 0.0:
            return
        factor = 1.0 - self.decay_rate
        self.table *= factor
        self._last_decay = 0

    def add(self, item: str, count: float = 1.0) -> None:
        """
        Increment the sketch for `item` by `count`.
        Decay is triggered every `width` updates to amortize cost.
        """
        self._last_decay += 1
        if self._last_decay >= self.width:
            self._apply_decay()
        for row, seed in enumerate(self.seeds):
            col = self._hash(item, seed)
            self.table[row, col] += count

    def estimate(self, item: str) -> float:
        """Return the minimum count across hash rows for `item`."""
        mins = []
        for row, seed in enumerate(self.seeds):
            col = self._hash(item, seed)
            mins.append(self.table[row, col])
        return float(min(mins))


# ----------------------------------------------------------------------
# Core mathematical helpers (parent A)
# ----------------------------------------------------------------------
def shannon_entropy(scores: np.ndarray) -> float:
    """
    Compute Shannon entropy (bits) of a non‑negative score vector.
    Scores are first normalised to a probability distribution.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be a 1‑D array")
    total = scores.sum()
    if total == 0:
        return 0.0
    probs = scores / total
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a probability vector (unit‑sum)."""
    total = vector.sum()
    if total == 0:
        return np.full_like(vector, 1.0 / len(vector), dtype=np.float64)
    return vector / total


def _entropy_normalized(entropy: float, max_entropy: float) -> float:
    """Scale entropy to [0,1] using the theoretical maximum for the given action set."""
    if max_entropy == 0:
        return 0.0
    return min(1.0, max(0.0, entropy / max_entropy))


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
class HybridPheromoneBandit:
    """
    Encapsulates the hybrid pheromone‑bandit algorithm with deeper mathematical coupling.
    """

    def __init__(
        self,
        actions: List[str],
        width: int = 2_048,
        depth: int = 4,
        decay_rate: float = 0.001,
        rng: random.Random | None = None,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.sketch = CountMinSketch(width=width, depth=depth, decay_rate=decay_rate)
        self.propensity = np.zeros(self.n_actions, dtype=np.float64)  # estimated rewards
        self.pulls = np.zeros(self.n_actions, dtype=np.int64)        # arm pull counts
        self.rng = rng or random.Random()
        self._max_entropy = math.log2(self.n_actions) if self.n_actions > 1 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def observe_scores(self, scores: np.ndarray) -> None:
        """Update internal propensity estimates with fresh decision‑hygiene scores."""
        if scores.shape != (self.n_actions,):
            raise ValueError("scores shape mismatch")
        # Simple exponential moving average to keep propensity responsive
        alpha = 0.2
        self.propensity = (1 - alpha) * self.propensity + alpha * scores

    def select_action(
        self,
        scores: np.ndarray,
        flag: CertaintyFlag,
    ) -> str:
        """
        Perform a single hybrid decision step.
        Returns the chosen action identifier.
        """
        # 1. Compute normalized entropy
        raw_entropy = shannon_entropy(scores)
        entropy_norm = _entropy_normalized(raw_entropy, self._max_entropy)

        # 2. Entropy‑scaled sketch inflow (smooth, non‑integer)
        self._inflow_sketch(scores, entropy_norm)

        # 3. Pheromone proxy (probability vector)
        pheromone_probs = self._pheromone_distribution()

        # 4. UCB values with epistemic modulation
        ucb_vals = self._ucb_values(flag, entropy_norm)

        # 5. Blend using entropy‑aware weighting
        blended = self._blend_distributions(pheromone_probs, ucb_vals, entropy_norm)

        # 6. Sample according to blended distribution
        chosen_idx = self._sample_index(blended)
        chosen_action = self.actions[chosen_idx]

        # 7. Update bandit statistics
        self.pulls[chosen_idx] += 1

        return chosen_action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inflow_sketch(self, scores: np.ndarray, entropy_norm: float) -> None:
        """
        Add weighted counts to the sketch.
        The multiplier is a smooth exponential of normalized entropy:
            m = exp(β * entropy_norm)
        with β controlling sensitivity (default β=2).
        This avoids unbounded linear scaling while still rewarding uncertainty.
        """
        beta = 2.0
        multiplier = math.exp(beta * entropy_norm)  # ∈ [1, e^β]
        for act, sc in zip(self.actions, scores):
            # Ensure non‑negative contribution; zero scores still get a minimal epsilon
            base = max(sc, 1e-9)
            increment = base * multiplier
            self.sketch.add(act, increment)

    def _pheromone_distribution(self) -> np.ndarray:
        """Estimate pheromone frequencies and normalise to a probability vector."""
        raw_counts = np.array([self.sketch.estimate(a) for a in self.actions], dtype=np.float64)
        return normalize(raw_counts)

    def _ucb_values(self, flag: CertaintyFlag, entropy_norm: float) -> np.ndarray:
        """
        Compute UCB values with two layers of epistemic influence:
        1. Confidence factor reduces exploration for high‑confidence flags.
        2. Entropy‑dependent scaling of the exploration term (more entropy → more exploration).
        """
        total_pulls = int(self.pulls.sum())
        total_pulls = max(1, total_pulls)  # avoid log(0)

        # Optimistic initialization for never‑pulled arms
        safe_pulls = np.where(self.pulls == 0, 1, self.pulls)

        # Classic exploration term
        exploration = np.sqrt(2.0 * np.log(total_pulls) / safe_pulls)

        # Epistemic confidence factor (0 → no extra exploration, 1 → full)
        confidence = flag.confidence_bps / 10000.0
        epistemic_factor = 1.0 - confidence

        # Entropy‑driven scaling: higher entropy amplifies exploration up to factor (1+γ)
        gamma = 1.5
        entropy_factor = 1.0 + gamma * entropy_norm

        scaled_exploration = epistemic_factor * entropy_factor * exploration
        return self.propensity + scaled_exploration

    def _blend_distributions(
        self,
        pheromone: np.ndarray,
        ucb_vals: np.ndarray,
        entropy_norm: float,
    ) -> np.ndarray:
        """
        Blend pheromone and bandit signals.
        Weight α is a convex combination that favours pheromone when entropy is low
        and favours bandit when entropy is high.
        """
        alpha = 1.0 - entropy_norm  # low entropy → α≈1 (pheromone), high entropy → α≈0
        bandit_probs = normalize(ucb_vals)
        return alpha * pheromone + (1.0 - alpha) * bandit_probs

    def _sample_index(self, probs: np.ndarray) -> int:
        """Sample an index according to a probability vector using the internal RNG."""
        cumulative = np.cumsum(probs)
        r = self.rng.random()
        return int(np.searchsorted(cumulative, r, side="right"))

    # ------------------------------------------------------------------
    # Diagnostic utilities
    # ------------------------------------------------------------------
    def get_state_snapshot(self) -> dict:
        """Return a shallow snapshot of internal state for debugging / logging."""
        return {
            "propensity": self.propensity.copy(),
            "pulls": self.pulls.copy(),
            "sketch_snapshot": self.sketch.table.copy(),
        }

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    rng = random.Random(42)
    actions = ["search", "store", "analyze", "report"]
    hybrid = HybridPheromoneBandit(actions, rng=rng)

    # Simulate 100 decision steps
    for _ in range(100):
        # Random decision‑hygiene scores (positive)
        scores = np.random.rand(len(actions)) * 10 + 1

        # Random epistemic flag (varying confidence)
        flag = CertaintyFlag(
            label="PROBABLE",
            confidence_bps=rng.randint(2000, 8000),
            authority_class="simulated",
            rationale="test run",
        )

        hybrid.observe_scores(scores)
        chosen = hybrid.select_action(scores, flag)
        # In a real system we would feed back reward information here

    # Print a brief diagnostic
    snapshot = hybrid.get_state_snapshot()
    print("Final propensity:", snapshot["propensity"])
    print("Pull counts:", snapshot["pulls"])
    print("Sketch row sums (approx total mass):", snapshot["sketch_snapshot"].sum(axis=1))

if __name__ == "__main__":
    _smoke_test()