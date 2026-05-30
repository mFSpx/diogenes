# DARWIN HAMMER — match 1480, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m486_s0.py (gen5)
# born: 2026-05-29T23:36:43Z

"""Hybrid Allocation and Bandit Fusion
Parent A: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s2.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m486_s0.py

Mathematical Bridge:
The similarity score produced by the SSIM‑like function (Parent B) is used as the
exponent in a fractional‑power binding of a random hypervector that represents the
current textual context.  The resulting bound hypervector modulates the
weekday‑weight vector (Parent A) that drives deterministic resource allocation.
Epistemic certainty flags from Parent A are interpreted as confidence bounds for
the bandit actions, closing the loop between allocation, reward, and policy
update.
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and simple utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a deterministic pseudo‑day‑of‑week index (0‑6)."""
    return (datetime(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Generate a smooth weight vector that varies with the day of week."""
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
# Bandit core (adapted from Parent B)
# ----------------------------------------------------------------------
class BanditAction:
    __slots__ = ("action_id", "propensity", "expected_reward", "confidence_bound", "algorithm")

    def __init__(self, action_id: str, propensity: float, expected_reward: float,
                 confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm


class BanditUpdate:
    __slots__ = ("context_id", "action_id", "reward", "propensity")

    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


# Global mutable state (mirrors Parent B)
_POLICY: Dict[str, List[float]] = {}          # action_id -> [cumulative_reward, count]
_STORE: Dict[str, float] = {}                # context_id -> last reward
_SURROGATE = None                            # placeholder for future RBF surrogate


def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = None


def _empirical_reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _update_policy(action_id: str, reward: float) -> None:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, n + 1]


# ----------------------------------------------------------------------
# Similarity & Hypervector binding (Parent B side)
# ----------------------------------------------------------------------
def simple_token_overlap(a: str, b: str) -> float:
    """Return a Jaccard‑like similarity in [0,1] based on token overlap."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _minhash_signature(text: str, num_perm: int = 64) -> int:
    """Very lightweight MinHash‑style signature: xor of hashed tokens."""
    h = 0
    for token in text.lower().split():
        token_hash = hash(token) & MAX64
        h ^= token_hash
    # Mix with number of permutations to obtain a reproducible seed
    return (h * num_perm) & MAX64


def generate_hypervector(text: str, dim: int = 128) -> np.ndarray:
    """Generate a random complex hypervector seeded by a MinHash signature."""
    seed = _minhash_signature(text)
    rng = np.random.RandomState(seed % (2 ** 32))
    # Real and imaginary parts are drawn from N(0,1)
    real = rng.normal(size=dim)
    imag = rng.normal(size=dim)
    return real + 1j * imag


def bind_hypervector(base: np.ndarray, similarity: float) -> np.ndarray:
    """
    Fractional‑power binding: raise magnitude to `similarity` and keep phase.
    This implements the “similarity score as exponent” bridge.
    """
    if not (0.0 <= similarity <= 1.0):
        raise ValueError("similarity must be in [0,1]")
    magnitude = np.abs(base)
    phase = np.angle(base)
    # fractional power on magnitude
    bound_mag = magnitude ** similarity
    return bound_mag * np.exp(1j * phase)


# ----------------------------------------------------------------------
# Hybrid allocation that merges both parents
# ----------------------------------------------------------------------
def hybrid_allocate(
    *,
    total_units: float,
    date: datetime,
    groups: Tuple[str, ...] = GROUPS,
    epistemic_flag: str = "FACT",
    context_text: str = "",
    reference_text: str = "",
    deterministic_target_pct: float = 90.0,
) -> Dict[str, Any]:
    """
    Allocate `total_units` across `groups` using a blend of:
      * Weekday‑weight vector (Parent A)
      * Bandit‑derived propensities (Parent B)
      * Hypervector binding driven by similarity between `context_text` and
        `reference_text` (the mathematical bridge).

    Returns a dict with per‑group allocation and diagnostic fields.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if epistemic_flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {epistemic_flag}")

    # 1️⃣ Weekday weight vector (Parent A)
    dow = date.weekday()  # Monday=0 … Sunday=6
    base_weights = weekday_weight_vector(groups, dow)  # sums to 1

    # 2️⃣ Similarity‑driven hypervector binding (Parent B bridge)
    sim = simple_token_overlap(context_text, reference_text)  # ∈[0,1]
    hv = generate_hhypervector = generate_hypervector(context_text, dim=128)
    bound_hv = bind_hypervector(hv, sim)
    # Use the magnitude of the bound hypervector as a modulation factor per group
    # (we simply reshape to match number of groups)
    modulation = np.abs(bound_hv[: len(groups)])
    modulation = modulation / modulation.sum()  # normalise

    # 3️⃣ Bandit propensities (Parent B)
    # The epistemic flag maps to a confidence bound multiplier
    flag_to_confidence = {
        "FACT": 1.0,
        "PROBABLE": 0.9,
        "POSSIBLE": 0.7,
        "BULLSHIT": 0.3,
        "SURE_MAYBE": 0.5,
    }
    confidence = flag_to_confidence[epistemic_flag]

    # For each group we treat the group name as an action_id
    propensities = []
    for g in groups:
        # Expected reward from policy (empirical)
        exp_reward = _empirical_reward(g)
        # Simple UCB‑style confidence bound
        cb = confidence * math.sqrt(2 * math.log(max(1, total_units)) / (1 + _POLICY.get(g, [0, 0])[1] + 1e-9))
        propensity = exp_reward + cb
        propensities.append(propensity)

    propensities = np.array(propensities)
    # Normalise propensities to a probability simplex
    if propensities.sum() == 0:
        propensities = np.ones_like(propensities) / len(propensities)
    else:
        propensities = propensities / propensities.sum()

    # 4️⃣ Fuse the three ingredients
    deterministic_units = total_units * deterministic_target_pct / 100.0
    stochastic_units = total_units - deterministic_units

    deterministic_allocation = deterministic_units * base_weights * modulation
    stochastic_allocation = stochastic_units * propensities * modulation

    allocation = deterministic_allocation + stochastic_allocation

    # 5️⃣ Update bandit policy with a synthetic reward derived from similarity
    reward = sim * 100  # scale to a convenient range
    for g, alloc in zip(groups, allocation):
        # The reward is shared proportionally to allocation size
        _update_policy(g, reward * (alloc / total_units))

    result = {
        "date": date.isoformat(),
        "total_units": total_units,
        "deterministic_target_pct": deterministic_target_pct,
        "epistemic_flag": epistemic_flag,
        "similarity": _pct(sim),
        "allocation": {g: _pct(a) for g, a in zip(groups, allocation)},
        "propensities": {g: _pct(p) for g, p in zip(groups, propensities)},
        "base_weights": {g: _pct(w) for g, w in zip(groups, base_weights)},
        "modulation": {g: _pct(m) for g, m in zip(groups, modulation)},
    }
    return result


# ----------------------------------------------------------------------
# Additional helper functions to showcase the hybrid system
# ----------------------------------------------------------------------
def compute_similarity_report(text_a: str, text_b: str) -> Dict[str, float]:
    """Return a detailed similarity breakdown."""
    sim = simple_token_overlap(text_a, text_b)
    return {
        "jaccard_similarity": _pct(sim),
        "hypervector_norm_ratio": _pct(
            np.linalg.norm(generate_hypervector(text_a)) / np.linalg.norm(generate_hypervector(text_b))
        ),
    }


def bandit_snapshot() -> Dict[str, Tuple[float, int]]:
    """Expose the current empirical rewards and counts per action."""
    return {aid: (round(total, 4), int(cnt)) for aid, (total, cnt) in _POLICY.items()}


def reset_system() -> None:
    """Convenient wrapper to reset all mutable state."""
    reset_policy()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    today = datetime.now(timezone.utc)
    allocation_result = hybrid_allocate(
        total_units=1000.0,
        date=today,
        groups=GROUPS,
        epistemic_flag="PROBABLE",
        context_text="The quick brown fox jumps over the lazy dog",
        reference_text="A fast dark-colored fox leaps above a sleepy canine",
        deterministic_target_pct=85.0,
    )
    print("Hybrid Allocation Result:")
    for k, v in allocation_result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for subk, subv in v.items():
                print(f"    {subk}: {subv}")
        else:
            print(f"  {k}: {v}")

    print("\nBandit Policy Snapshot:")
    for aid, (total, cnt) in bandit_snapshot().items():
        print(f"  {aid}: total_reward={total}, count={cnt}")

    print("\nSimilarity Report:")
    print(compute_similarity_report(
        "The quick brown fox jumps over the lazy dog",
        "A fast dark-colored fox leaps above a sleepy canine"
    ))