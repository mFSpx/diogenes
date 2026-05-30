# DARWIN HAMMER — match 5416, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1200_s3.py (gen5)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s7.py (gen4)
# born: 2026-05-30T00:01:44Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A provides a dynamic store state, health‑score based regret gains, and a
bandit router / work‑share allocator.  
Parent B supplies text‑centric MinHash signatures, shingle generation and
Shannon entropy utilities.

Mathematical Bridge
-------------------
Let  

* **h** ∈ ℝⁿ be the vector of health scores for *n* endpoints (Parent A).  
* **gₐ** = tropical_regret_gains(h, actions) ∈ ℝⁿ be the regret‑based gains
  (Parent A).  

From a textual corpus we obtain  

* **e** = entropy_for_text(text) ∈ ℝ₊ (scalar entropy, Parent B).  
* **m** ∈ ℝᵏ be the MinHash signature (k‑dimensional integer vector,
  Parent B).  

We define a *scalar modulation factor*  


ϕ = (1 + e / 10) * (1 + mean(m) / MAX_HASH)


where MAX_HASH = 2⁶⁴‑1 is the maximum possible hash value.  
The *combined gain vector* is then  


g = ϕ * gₐ


Thus the text‑derived statistics uniformly scale the regret‑based gains,
creating a single unified gain vector that feeds both the bandit router
and the work‑share allocator.  The store state’s “dance” value modulates the
propensity of the selected action, preserving the dynamics of Parent A while
being driven by the text‑aware factor ϕ from Parent B.

The following module implements this fusion and provides three core functions
that demonstrate the hybrid operation.
"""

import sys
import math
import random
import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared)
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1
MAX_HASH = 2 ** 64 - 1  # Upper bound for 64‑bit hash used in MinHash

# ----------------------------------------------------------------------
# Data structures (Parent A)
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
        """Update the store level using linear inflow/outflow dynamics."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded version of the current level used as propensity."""
        return max(0.0, min(self.level, self.limit))

# ----------------------------------------------------------------------
# Parent A core functions
# ----------------------------------------------------------------------
def compute_health_scores(endpoints: List[Dict]) -> np.ndarray:
    """Extract health_score field from a list of endpoint dicts."""
    return np.array([ep.get('health_score', 0.0) for ep in endpoints], dtype=float)


def tropical_regret_gains(health_scores: np.ndarray, actions: List[Dict]) -> np.ndarray:
    """Regret gain = max health – intrinsic cost for each action."""
    max_h = np.max(health_scores) if health_scores.size else 0.0
    gains = np.array([max_h - act.get('intrinsic_cost', 0.0) for act in actions], dtype=float)
    return gains


def bandit_router(store_state: StoreState, health_scores: np.ndarray) -> BanditAction:
    """Select the action with highest health score, using store_state.dance as propensity."""
    if health_scores.size == 0:
        raise ValueError("health_scores cannot be empty")
    action_id = int(np.argmax(health_scores))
    propensity = store_state.dance
    expected_reward = float(health_scores[action_id])
    confidence_bound = 1.0
    return BanditAction(str(action_id), propensity, expected_reward, confidence_bound, 'bandit_router')


def workshare_allocator(store_state: StoreState, gains: np.ndarray) -> np.ndarray:
    """Allocate resources proportionally to gains; fallback to uniform allocation."""
    total = np.sum(gains)
    if total == 0.0:
        return np.full_like(gains, 1.0 / gains.size, dtype=float)
    return gains / total


def update_stats_and_maybe_split(stats: Dict, delta: float, gini_thr: float, gains: np.ndarray) -> bool:
    """Update Hoeffding bound and Gini coefficient; decide on split."""
    stats['hoeffding_bound'] = stats.get('hoeffding_bound', 0.0) + delta
    mean_gain = np.mean(gains) if gains.size else 0.0
    std_gain = np.std(gains) if gains.size else 0.0
    stats['gini_coefficient'] = std_gain / mean_gain if mean_gain != 0 else 0.0
    return stats['hoeffding_bound'] > delta and stats['gini_coefficient'] < gini_thr

# ----------------------------------------------------------------------
# Parent B core functions (text processing)
# ----------------------------------------------------------------------
def _shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i: i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k
    seeds = [random.randrange(0, 2 ** 32) for _ in range(k)]
    signature = []
    for seed in seeds:
        min_hash = min(_hash_token(seed, t) for t in token_set)
        signature.append(min_hash)
    return signature


def minhash_for_text(text: str, k: int = 64, width: int = 5) -> List[int]:
    """Convenient wrapper: shingle the text then compute MinHash."""
    return minhash_signature(_shingles(text, width=width), k=k)


def shannon_entropy(chars: List[str]) -> float:
    """Shannon entropy over a list of characters."""
    if not chars:
        return 0.0
    freq: Dict[str, int] = {}
    for c in chars:
        freq[c] = freq.get(c, 0) + 1
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


def entropy_for_text(text: str, limit: int = 10000) -> float:
    """Entropy of the first *limit* characters of *text*."""
    return shannon_entropy(list((text or "")[:limit])) if text else 0.0

# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------
def compute_modulation_factor(text: str) -> float:
    """
    Compute the scalar factor ϕ that bridges text statistics with regret gains.

    ϕ = (1 + e/10) * (1 + mean(m)/MAX_HASH)
    where e is Shannon entropy and m is the MinHash signature.
    """
    e = entropy_for_text(text)
    m = np.array(minhash_for_text(text, k=64), dtype=float)
    mean_m = np.mean(m) if m.size else 0.0
    phi = (1.0 + e / 10.0) * (1.0 + mean_m / MAX_HASH)
    return phi


def hybrid_combined_gains(store_state: StoreState,
                          endpoints: List[Dict],
                          actions: List[Dict],
                          text: str) -> np.ndarray:
    """
    Produce the combined gain vector g = ϕ * gₐ.

    Steps:
    1. Compute health scores h from endpoints.
    2. Compute regret‑based gains gₐ = tropical_regret_gains(h, actions).
    3. Compute modulation factor ϕ from the supplied text.
    4. Return g = ϕ * gₐ.
    """
    h = compute_health_scores(endpoints)
    g_a = tropical_regret_gains(h, actions)
    phi = compute_modulation_factor(text)
    combined = phi * g_a
    return combined


def hybrid_allocate_and_route(store_state: StoreState,
                              endpoints: List[Dict],
                              actions: List[Dict],
                              text: str) -> Tuple[np.ndarray, BanditAction]:
    """
    Perform allocation using the hybrid gains and then select an action
    via the bandit router (which still uses raw health scores for selection).

    Returns:
        allocation – resource fractions for each action,
        selected_action – BanditAction instance.
    """
    # Combined gains drive allocation
    combined_gains = hybrid_combined_gains(store_state, endpoints, actions, text)
    allocation = workshare_allocator(store_state, combined_gains)

    # Routing still uses health scores (original decision logic)
    health_scores = compute_health_scores(endpoints)
    selected = bandit_router(store_state, health_scores)

    # Update store state with allocation as inflow and a dummy outflow
    outflow = [0.0] * len(allocation)  # placeholder: no consumption yet
    store_state.update(list(allocation), outflow)

    return allocation, selected


def hybrid_update_statistics(stats: Dict,
                             store_state: StoreState,
                             endpoints: List[Dict],
                             actions: List[Dict],
                             text: str,
                             gini_thr: float = 0.3) -> bool:
    """
    Update statistical counters based on the latest store dynamics and decide
    whether a split (e.g., model branching) should occur.

    Returns True if split condition is met.
    """
    # Compute combined gains to obtain a representative delta
    combined_gains = hybrid_combined_gains(store_state, endpoints, actions, text)
    # Use the last delta from the store as the regret‑related delta
    delta = store_state._last_delta
    # Update and evaluate split condition
    return update_stats_and_maybe_split(stats, delta, gini_thr, combined_gains)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data
    endpoints = [
        {"id": "ep1", "health_score": 0.8},
        {"id": "ep2", "health_score": 0.6},
        {"id": "ep3", "health_score": 0.9},
    ]

    actions = [
        {"action_id": "a1", "intrinsic_cost": 0.2},
        {"action_id": "a2", "intrinsic_cost": 0.5},
        {"action_id": "a3", "intrinsic_cost": 0.1},
    ]

    sample_text = "The quick brown fox jumps over the lazy dog. " * 10

    # Initialize components
    store = StoreState(level=5.0)
    stats = {"hoeffding_bound": 0.0, "gini_coefficient": 0.0}

    # Run hybrid allocation and routing
    allocation, selected_action = hybrid_allocate_and_route(store, endpoints, actions, sample_text)

    # Print results (simple verification)
    print("Allocation fractions:", allocation)
    print("Selected action:", selected_action)

    # Update statistics and possibly trigger a split
    split_needed = hybrid_update_statistics(stats, store, endpoints, actions, sample_text)
    print("Split needed?", split_needed)
    print("Updated stats:", stats)
    print("Store level after update:", store.level)