# DARWIN HAMMER — match 3856, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1480_s1.py (gen6)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s2.py (gen2)
# born: 2026-05-29T23:52:05Z

"""Hybrid Allocation, Bandit, and Circuit‑Breaker Fusion
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1480_s1.py (weekday weight & bandit core)
- hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s2.py (lead‑lag transform & circuit breaker)

Mathematical Bridge:
The lead‑lag transform from Parent B converts the temporal sequence of bandit
propensities into an interleaved hypervector.  Parent A treats a similarity
score (here a cosine‑like SSIM proxy) as an exponent for a fractional‑power
binding of a random hypervector.  In the hybrid we raise the lead‑lag vector
element‑wise to the similarity exponent, yielding a *bound hypervector*.
This bound hypervector is then used to modulate the weekday‑weight vector
from Parent A, producing day‑aware allocation scores that also drive the
failure‑threshold adaptation of the EndpointCircuitBreaker from Parent B.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Deterministic pseudo‑day‑of‑week index (0‑6)."""
    return (datetime(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Smooth weight vector that varies with the day of week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Parent B – Lead‑lag transform & circuit breaker
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = self._now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (endpoint usable)."""
        return not self.open

    def adapt_threshold(self, delta: int) -> None:
        """Adjust the failure threshold by delta (kept >=1)."""
        self.failure_threshold = max(1, self.failure_threshold + delta)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

    @staticmethod
    def _now_z() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def lead_lag_transform(seq: Sequence[float]) -> np.ndarray:
    """
    Lead‑lag transform: interleave the original sequence with its cumulative sum.
    For a sequence x₀,…,xₙ‑1 the output is
        [x₀, c₀, x₁, c₁, …] where cᵢ = Σⱼ≤ᵢ xⱼ.
    """
    seq_arr = np.asarray(seq, dtype=np.float64)
    cum = np.cumsum(seq_arr)
    interleaved = np.empty(seq_arr.size + cum.size, dtype=np.float64)
    interleaved[0::2] = seq_arr
    interleaved[1::2] = cum
    return interleaved


# ----------------------------------------------------------------------
# Parent A – Bandit core (simplified) and similarity proxy
# ----------------------------------------------------------------------
class BanditAction:
    __slots__ = ("action_id", "propensity", "expected_reward")

    def __init__(self, action_id: int, propensity: float, expected_reward: float = 0.0):
        self.action_id = action_id
        self.propensity = float(propensity)
        self.expected_reward = float(expected_reward)


class BanditCore:
    """Very small multi‑armed bandit with epsilon‑greedy selection."""

    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.epsilon = float(epsilon)
        self.actions: List[BanditAction] = [
            BanditAction(i, propensity=1.0 / n_actions) for i in range(n_actions)
        ]

    def select_action(self) -> BanditAction:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # greedy: highest propensity
        return max(self.actions, key=lambda a: a.propensity)

    def update(self, action_id: int, reward: float) -> None:
        """Simple incremental update of expected reward and propensity."""
        act = self.actions[action_id]
        # update expected reward with a learning rate α=0.1
        α = 0.1
        act.expected_reward = (1 - α) * act.expected_reward + α * reward
        # propensities are softmax of expected rewards
        rewards = np.array([a.expected_reward for a in self.actions])
        exp_vals = np.exp(rewards - np.max(rewards))
        props = exp_vals / exp_vals.sum()
        for a, p in zip(self.actions, props):
            a.propensity = p


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity proxy used as the exponent (range 0‑1)."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same shape")
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def bind_hypervector(similarity: float, lead_lag_vec: np.ndarray) -> np.ndarray:
    """
    Fractional‑power binding: raise each component of the lead‑lag vector to the
    similarity exponent, then L2‑normalize.
    """
    if not (0.0 <= similarity <= 1.0):
        raise ValueError("similarity must be in [0,1]")
    # Avoid zero‑to‑zero issues
    bound = np.where(lead_lag_vec > 0, lead_lag_vec ** similarity, 0.0)
    norm = np.linalg.norm(bound)
    if norm == 0:
        return bound
    return bound / norm


def hybrid_allocation(
    groups: Sequence[str],
    dow: int,
    bound_vec: np.ndarray,
) -> np.ndarray:
    """
    Modulate the weekday weight vector with the bound hypervector.
    The bound vector is trimmed or tiled to match the number of groups.
    """
    w_vec = weekday_weight_vector(groups, dow)
    # Align sizes
    if bound_vec.size < w_vec.size:
        # tile the bound vector
        repeats = math.ceil(w_vec.size / bound_vec.size)
        bound_aligned = np.tile(bound_vec, repeats)[: w_vec.size]
    else:
        bound_aligned = bound_vec[: w_vec.size]
    allocation = w_vec * bound_aligned
    # Re‑normalize so allocations sum to 1
    total = allocation.sum()
    if total == 0:
        return allocation
    return allocation / total


def adapt_circuit_breaker(
    cb: EndpointCircuitBreaker,
    allocation_scores: np.ndarray,
    flag: str,
) -> None:
    """
    Use epistemic flags as confidence bounds:
    - FACT / SURE_MAYBE → lower threshold (more tolerant)
    - BULLSHIT → raise threshold (more strict)
    The magnitude of adaptation is proportional to the max allocation score.
    """
    confidence = {
        "FACT": -1,
        "PROBABLE": -0.5,
        "POSSIBLE": 0,
        "SURE_MAYBE": -0.5,
        "BULLSHIT": 1,
    }.get(flag, 0)

    delta = int(math.copysign(1, confidence) * max(1, int(confidence * allocation_scores.max() * 5)))
    cb.adapt_threshold(delta)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    # 1. Temporal context – day of week
    today = datetime.utcnow()
    dow = doomsday(today.year, today.month, today.day)

    # 2. Simulated textual context vectors (random but reproducible)
    rng = np.random.default_rng(42)
    context_vec = rng.random(8)
    reference_vec = rng.random(8)

    # 3. Similarity exponent (SSIM‑like proxy)
    sim = cosine_similarity(context_vec, reference_vec)

    # 4. Bandit core produces a sequence of propensities
    bandit = BanditCore(n_actions=5, epsilon=0.2)
    prop_seq = [act.propensity for act in bandit.actions]

    # 5. Lead‑lag transform creates the hypervector
    ll_vec = lead_lag_transform(prop_seq)

    # 6. Bind using similarity exponent
    bound = bind_hypervector(sim, ll_vec)

    # 7. Allocate resources across groups
    allocation = hybrid_allocation(GROUPS, dow, bound)

    # 8. Circuit breaker adapts based on allocation and a random epistemic flag
    cb = EndpointCircuitBreaker(failure_threshold=3)
    flag = random.choice(EPISTEMIC_FLAGS)
    adapt_circuit_breaker(cb, allocation, flag)

    # 9. Print results
    print("Day of week index:", dow)
    print("Similarity exponent:", _pct(sim))
    print("Lead‑lag vector:", np.round(ll_vec, 4))
    print("Bound hypervector (norm≈1):", np.round(bound, 4))
    print("Allocation per group:", {g: _pct(a) for g, a in zip(GROUPS, allocation)})
    print("Chosen epistemic flag:", flag)
    print("Circuit breaker state:", cb.as_dict())


if __name__ == "__main__":
    _demo()