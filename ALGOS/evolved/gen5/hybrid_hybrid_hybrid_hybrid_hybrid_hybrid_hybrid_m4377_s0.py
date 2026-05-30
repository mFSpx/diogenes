# DARWIN HAMMER — match 4377, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_infota_hybrid_hybrid_regret_m2127_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_shap_attribution_m430_s0.py (gen3)
# born: 2026-05-29T23:55:14Z

"""Hybrid Algorithm: StoreState‑SHAP‑Bandit Fusion

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a dynamical StoreState that updates its level from
  inflow/outflow streams, together with signature hashing and similarity
  measures for actions.
* **Parent B** – deterministic/stochastic feature extraction, feature fusion,
  and a SHAP‑style kernel weighting that yields attribution values for
  features.

The mathematical bridge is built by interpreting SHAP attributions as signed
quantities that drive the StoreState dynamics (positive attributions → inflow,
negative → outflow).  The resulting StoreState “dance” value modulates the
propensity of bandit actions, and action signatures are compared via the
original similarity metric.  This creates a single unified system that
simultaneously performs feature‑based attribution, dynamical state tracking,
and contextual bandit selection.

The public API provides three representative functions:
`fuse_features_and_shap`, `store_update_from_text`, and `select_bandit_action`,
demonstrating the hybrid operation.  A smoke test is executed when the module
is run as a script.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


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
    """Dynamical store from Parent A."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Update the store level using signed inflow/outflow streams."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded, monotonic transform of the last delta (used to modulate bandits)."""
        delta = self._last_delta
        return max(0.0, min(self.limit, self.base + self.gain * delta))


# ----------------------------------------------------------------------
# Helper functions (from Parent A)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0.0
    )


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑hash style signature for a set of tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2 ** 64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on min‑hash equality."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must be of equal length")
    intersection = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return intersection / len(sig_a)


# ----------------------------------------------------------------------
# Feature extraction & SHAP‑style attribution (from Parent B)
# ----------------------------------------------------------------------


def _deterministic_features(text: str) -> Dict[str, float]:
    """Deterministic feature extraction via a hash‑seeded PRNG."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
    ]
    return {key: rnd.random() for key in keys}


def _stochastic_features(text: str) -> Dict[str, float]:
    """Stochastic feature extraction via the global random state."""
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
    ]
    return {key: random.random() for key in keys}


def feature_fusion(text: str, alpha: float = 0.5) -> Dict[str, float]:
    """Linear interpolation between deterministic and stochastic features."""
    det = _deterministic_features(text)
    sto = _stochastic_features(text)
    fused = {}
    for key in det:
        fused[key] = alpha * det[key] + (1.0 - alpha) * sto[key]
    return fused


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Exact SHAP kernel weight for a given subset size."""
    if feature_count <= 1:
        return 1.0
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def _approximate_shap_values(features: Dict[str, float]) -> Dict[str, float]:
    """
    Very lightweight SHAP‑like attribution:
    each feature's contribution is its value scaled by a kernel weight that
    depends on its rank in the sorted list.
    """
    n = len(features)
    if n == 0:
        return {}
    # Sort features by absolute magnitude to obtain a deterministic ordering
    sorted_items = sorted(features.items(), key=lambda kv: -abs(kv[1]))
    shap_vals: Dict[str, float] = {}
    for idx, (key, val) in enumerate(sorted_items):
        weight = shapley_kernel_weight(idx, n)  # idx acts as subset size
        shap_vals[key] = val * weight
    return shap_vals


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def fuse_features_and_shap(text: str, alpha: float = 0.5) -> Dict[str, float]:
    """
    Combine feature fusion (Parent B) with a SHAP‑style attribution.
    Returns a dictionary mapping feature names to signed SHAP values.
    """
    fused = feature_fusion(text, alpha)
    shap_vals = _approximate_shap_values(fused)
    return shap_vals


def store_update_from_text(store: StoreState, text: str, alpha: float = 0.5) -> Tuple[float, float]:
    """
    Drive the StoreState dynamics using SHAP attributions derived from `text`.

    Positive SHAP values are treated as inflow, negative values as outflow.
    The function returns the new store level and the raw delta.
    """
    shap_vals = fuse_features_and_shap(text, alpha)
    inflow = [v for v in shap_vals.values() if v > 0]
    outflow = [-v for v in shap_vals.values() if v < 0]  # make them positive magnitudes
    level, delta = store.update(inflow, outflow)
    return level, delta


def select_bandit_action(
    actions: List[BanditAction],
    store: StoreState,
    temperature: float = 1.0,
) -> BanditAction:
    """
    Choose a BanditAction using a score that mixes the original expected reward,
    the confidence bound, and the StoreState's `dance` value (which reflects recent
    SHAP‑driven dynamics).

    The score is:
        score = expected_reward + confidence_bound * f(dance)

    where f(dance) = log(1 + dance / temperature) to keep the influence bounded.
    """
    if not actions:
        raise ValueError("action list must not be empty")
    dance_factor = math.log1p(store.dance / max(temperature, 1e-9))
    best = max(
        actions,
        key=lambda a: a.expected_reward + a.confidence_bound * dance_factor,
    )
    return best


def actions_signature(actions: List[BanditAction], k: int = 128) -> List[int]:
    """
    Compute a min‑hash signature for a collection of BanditAction identifiers.
    This reuses the `signature` routine from Parent A.
    """
    tokens = [a.action_id for a in actions]
    return signature(tokens, k)


def actions_similarity(
    actions_a: List[BanditAction],
    actions_b: List[BanditAction],
    k: int = 128,
) -> float:
    """
    Compute similarity between two sets of actions via their signatures.
    """
    sig_a = actions_signature(actions_a, k)
    sig_b = actions_signature(actions_b, k)
    return similarity(sig_a, sig_b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise deterministic random seed for reproducibility
    random.seed(42)

    # Create a StoreState instance
    store = StoreState(alpha=0.8, beta=0.5, dt=0.5, base=1.0, gain=2.0, limit=5.0)

    # Example text that will flow through the hybrid pipeline
    example_text = "The quick brown fox jumps over the lazy dog."

    # Update the store using SHAP‑driven inflow/outflow
    level, delta = store_update_from_text(store, example_text, alpha=0.6)
    print(f"Store level after update: {level:.4f} (delta={delta:.4f})")
    print(f"Store dance value: {store.dance:.4f}")

    # Define a few mock bandit actions
    actions = [
        BanditAction(
            action_id="A1",
            propensity=0.3,
            expected_reward=1.2,
            confidence_bound=0.4,
            algorithm="UCB",
        ),
        BanditAction(
            action_id="A2",
            propensity=0.5,
            expected_reward=0.9,
            confidence_bound=0.6,
            algorithm="Thompson",
        ),
        BanditAction(
            action_id="A3",
            propensity=0.2,
            expected_reward=1.5,
            confidence_bound=0.3,
            algorithm="EG",
        ),
    ]

    # Select the best action according to the hybrid scoring rule
    chosen = select_bandit_action(actions, store, temperature=0.8)
    print(f"Chosen action: {chosen.action_id} (algorithm={chosen.algorithm})")

    # Compute similarity between two subsets of actions
    subset1 = actions[:2]
    subset2 = actions[1:]
    sim = actions_similarity(subset1, subset2, k=64)
    print(f"Similarity between subsets: {sim:.4f}")

    # Verify entropy utility works on a simple probability vector
    probs = [0.2, 0.5, 0.3]
    print(f"Entropy of {probs}: {entropy(probs):.4f}")

    sys.exit(0)