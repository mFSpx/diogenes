# DARWIN HAMMER — match 2160, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_bandit_m1146_s0.py (gen3)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:41:04Z

"""Hybrid algorithm merging Hoeffding‑tree statistical split testing with
circuit‑breaker state management and morphological geometry.

Mathematical bridge:
- The Hoeffding bound ε = sqrt(r²·ln(1/δ)/(2·n)) provides a confidence interval
  for the difference between the best and second‑best gain.
- Gain for a candidate split is defined as a weighted combination of
  *geometric gain* (sphericity_index of a Morphology object) and
  *statistical gain* derived from the reward stream:
      g = α·sphericity + β·log_likelihood + γ·log(cardinality)
- The circuit‑breaker supplies a binary “open/closed” flag that can be
  interpreted as a hard constraint on split execution.
- The hybrid decision therefore fuses:
      split_allowed = (not circuit.open) and
                      (best_gain - second_best_gain ≥ ε)
  where the gains embed both morphology and sketch‑based statistics.

The module implements this unified decision process and demonstrates it
through three core functions."""

import math
import random
import sys
import hashlib
from dataclasses import dataclass, field, asdict, frozen
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hoeffding helpers (Parent A)
# ----------------------------------------------------------------------


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range *r*, confidence *δ* and sample count *n*."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Standard Hoeffding split test."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    if gap >= eps:
        return SplitDecision(True, eps, gap, "Gap significant")
    else:
        return SplitDecision(False, eps, gap, "Gap not significant")


# ----------------------------------------------------------------------
# Sketch primitives (Parent A – simplified HyperLogLog)
# ----------------------------------------------------------------------


def estimate_log_likelihood(reward_stream: Iterable[float]) -> float:
    """Approximate the log‑likelihood contribution of the reward stream."""
    rewards = list(reward_stream)
    if not rewards:
        return 0.0
    total = np.sum(rewards)
    # Guard against zero or negative values for log
    if total <= 0 or any(r <= 0 for r in rewards):
        return -np.inf
    return np.log(total) - np.sum(np.log(rewards))


class SimpleHyperLogLog:
    """Very light‑weight cardinality estimator using a Python set.
    This replaces the incomplete HyperLogLog from the original source."""
    def __init__(self):
        self._set = set()

    def add(self, x: Any) -> None:
        self._set.add(hash(x))

    def estimate_cardinality(self) -> int:
        return len(self._set)


def estimate_cardinality(reward_stream: Iterable[float]) -> int:
    """Estimate the number of distinct contexts using SimpleHyperLogLog."""
    sketch = SimpleHyperLogLog()
    for reward in reward_stream:
        sketch.add(reward)
    return sketch.estimate_cardinality()


# ----------------------------------------------------------------------
# Circuit‑breaker primitives (Parent B)
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Morphology utilities (Parent B)
# ----------------------------------------------------------------------


@frozen
@dataclass
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def compute_gain(
    reward_stream: Iterable[float],
    morphology: Morphology,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> float:
    """
    Compute a composite gain for a candidate split.

    The gain mixes:
      * geometric gain (sphericity),
      * statistical gain (log‑likelihood of rewards),
      * combinatorial gain (log of distinct reward contexts).

    Parameters
    ----------
    reward_stream: Iterable[float]
        Observed reward values for the current node/leaf.
    morphology: Morphology
        Geometric description influencing the gain.
    alpha, beta, gamma: float
        Non‑negative weights that sum to 1 (not enforced).

    Returns
    -------
    float
        Weighted gain value.
    """
    # Geometric component
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)

    # Statistical components
    ll = estimate_log_likelihood(reward_stream)
    card = estimate_cardinality(reward_stream)
    log_card = math.log(card + 1)  # +1 to avoid log(0)

    # Weighted sum
    gain = alpha * sph + beta * ll + gamma * log_card
    return gain


def evaluate_candidate_splits(
    reward_stream: Iterable[float],
    morphology: Morphology,
    delta: float = 1e-5,
) -> SplitDecision:
    """
    Generate two simple candidate splits (mean‑based and median‑based) and
    evaluate whether a split should be performed using the Hoeffding test.

    The function demonstrates the mathematical fusion:
      * Gains incorporate morphology and sketch statistics.
      * Hoeffding bound supplies the confidence interval.
      * The circuit‑breaker state (checked externally) gates the result.
    """
    rewards = list(reward_stream)
    if not rewards:
        return SplitDecision(False, 0.0, 0.0, "Empty reward stream")

    # Candidate 1: split on mean threshold
    mean_thr = np.mean(rewards)
    left1 = [r for r in rewards if r <= mean_thr]
    right1 = [r for r in rewards if r > mean_thr]
    gain1 = compute_gain(left1, morphology) + compute_gain(right1, morphology)

    # Candidate 2: split on median threshold
    median_thr = np.median(rewards)
    left2 = [r for r in rewards if r <= median_thr]
    right2 = [r for r in rewards if r > median_thr]
    gain2 = compute_gain(left2, morphology) + compute_gain(right2, morphology)

    # Determine which is better
    best_gain, second_gain = (gain1, gain2) if gain1 >= gain2 else (gain2, gain1)

    # Range r for Hoeffding: we use the observed gain span as a proxy
    observed_gains = [gain1, gain2]
    r = max(observed_gains) - min(observed_gains) if len(observed_gains) > 1 else 1.0
    n = len(rewards)

    return should_split(best_gain, second_gain, r, delta, n)


def update_circuit_breaker_with_hybrid(
    cb: EndpointCircuitBreaker,
    reward: float,
    morphology: Morphology,
    recent_rewards: List[float],
    delta: float = 1e-5,
) -> Tuple[bool, SplitDecision]:
    """
    Process a single reward, update the circuit‑breaker and, if the circuit is
    closed, decide whether a split should occur.

    Returns
    -------
    (allow_split, decision) where:
        allow_split – bool indicating whether a split is permissible (circuit closed).
        decision – SplitDecision object from the Hoeffding test.
    """
    # Simple success/failure definition: reward > 0 is success
    if reward > 0:
        cb.record_success()
    else:
        cb.record_failure()

    # Update the rolling window of recent rewards
    recent_rewards.append(reward)
    if len(recent_rewards) > 100:
        recent_rewards.pop(0)

    # If circuit is open, we never split
    if not cb.allow():
        return False, SplitDecision(False, 0.0, 0.0, "Circuit open")

    # Evaluate split on the accumulated window
    decision = evaluate_candidate_splits(recent_rewards, morphology, delta)
    return decision.should_split and cb.allow(), decision


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    # Synthetic reward stream: mixture of positives and occasional zeros/negatives
    random.seed(42)
    rewards = [random.expovariate(1.0) for _ in range(200)]
    # Inject some failures
    for i in range(0, 200, 25):
        rewards[i] = -0.5  # failure signal

    cb = EndpointCircuitBreaker(failure_threshold=5)
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    recent: List[float] = []

    split_occurred = False
    for idx, r in enumerate(rewards, 1):
        allow, decision = update_circuit_breaker_with_hybrid(cb, r, morph, recent)
        if allow:
            print(f"[{idx}] Split permitted – decision: {decision}")
            split_occurred = True
            break

    if not split_occurred:
        print("No split triggered during the demo run.")
    print("Final circuit‑breaker state:", cb.as_dict())


if __name__ == "__main__":
    _demo()