# DARWIN HAMMER — match 1407, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_endpoi_m1071_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s2.py (gen5)
# born: 2026-05-29T23:36:13Z

"""Hybrid Morphology‑Regret Algorithm
===================================

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – morphology based recovery priority, perceptual hashing and
  geometric indices (sphericity, flatness, righting‑time).
* **Parent B** – lead‑lag transformed path signatures, regret‑weighted utility
  and bandit style action selection.

**Mathematical bridge** – The scalar *recovery priority* (∈[0,1]) derived from a
`Morphology` instance is used as a *scaling factor* for the regret‑weighted
utility.  In addition the binary perceptual hash of the morphology’s geometric
features forms the token set of a `MathAction`.  A time‑series of recovery
priorities is fed through the lead‑lag transform, producing a path whose
signature (simple cumulative moments) modulates the regret term.  Thus the
geometry‑driven priority directly influences the bandit decision process,
creating a unified hybrid system.

The public API provides three representative functions:
`compute_morphology_vector`, `lead_lag_path_signature`, and
`regret_weighted_bandit`.  An executable smoke test demonstrates end‑to‑end
operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting‑time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def compute_dhash(values: List[float]) -> int:
    """Difference hash – 1 bit per adjacent pair."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Perceptual hash – threshold against mean, up to 64 bits."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_morphology_vector(m: Morphology) -> List[float]:
    """
    Returns a feature vector that aggregates the geometric indices and the
    normalized recovery priority.  The vector is used both for hashing and as
    a time‑series input to the lead‑lag path signature.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rti = righting_time_index(m)
    rp = recovery_priority(m)
    return [sph, flat, rti, rp]


# ----------------------------------------------------------------------
# Parent B – lead‑lag, path signature and regret‑weighted bandit
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash style hashing
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape (T, d). Output shape (2T‑1, 2d).
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time, dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # lead = lag = X_t
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lead advances
    out[-1] = np.concatenate([path[-1], path[-1]])               # final point
    return out


def simple_signature(lead_lag_path: np.ndarray) -> np.ndarray:
    """
    Very lightweight signature: first‑order moment (mean) and second‑order
    moment (sum of outer products) flattened.  This mimics a level‑2 signature
    without external libraries.
    """
    # First order: mean of each coordinate
    mean = lead_lag_path.mean(axis=0)                     # (2d,)
    # Second order: element‑wise product summed across time
    second = (lead_lag_path[:, :, None] * lead_lag_path[:, None, :]).sum(axis=0)
    # Flatten upper triangular (including diagonal) to avoid redundancy
    idx = np.triu_indices_from(second)
    second_flat = second[idx]
    return np.concatenate([mean, second_flat])


def regret_weighted_utility(
    action: MathAction,
    signature: np.ndarray,
    scaling: float,
    alpha: float = 0.5,
) -> float:
    """
    Computes a regret‑weighted utility:

        U = scaling * (EV - cost) * (1 - risk) * f(signature)

    where f(signature) = alpha * mean(sig) + (1‑alpha) * std(sig).
    """
    if scaling < 0:
        raise ValueError("scaling must be non‑negative")
    base = (action.expected_value - action.cost) * (1.0 - action.risk)
    if base <= 0:
        return 0.0
    sig_mean = signature.mean()
    sig_std = signature.std()
    f_sig = alpha * sig_mean + (1.0 - alpha) * sig_std
    return scaling * base * max(0.0, f_sig)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def regret_weighted_bandit(
    actions: List[MathAction],
    signature: np.ndarray,
    scaling: float,
    time_step: int = 1,
) -> BanditAction:
    """
    Performs a single bandit selection step.
    1. Compute utilities for all actions.
    2. Convert utilities to propensities via softmax.
    3. Estimate a simple confidence bound (UCB‑style).
    """
    utilities = np.array(
        [regret_weighted_utility(a, signature, scaling) for a in actions],
        dtype=float,
    )
    if np.all(utilities == 0):
        # fall back to uniform random selection
        propensities = np.full_like(utilities, 1.0 / len(utilities))
    else:
        propensities = softmax(utilities)

    # Upper confidence bound (Hoeffding style)
    confidence = np.sqrt(np.log(max(time_step, 2)) / (2 * np.maximum(propensities, 1e-12)))

    chosen_idx = int(np.random.choice(len(actions), p=propensities))
    chosen = actions[chosen_idx]
    return BanditAction(
        action_id=chosen.id,
        propensity=propensities[chosen_idx],
        expected_reward=utilities[chosen_idx],
        confidence_bound=confidence[chosen_idx],
    )


# ----------------------------------------------------------------------
# Hybrid operation demonstration functions
# ----------------------------------------------------------------------
def encode_morphology_tokens(morph_vec: List[float]) -> Tuple[str, ...]:
    """
    Convert a morphology feature vector into a token tuple using both
    difference‑hash and perceptual‑hash bits.  Each bit is turned into a
    string token ``'b0'``, ``'b1'`` ….
    """
    dh = compute_dhash(morph_vec)
    ph = compute_phash(morph_vec)
    # combine bits (up to 64 bits total)
    combined = (dh << 64) | ph
    tokens = tuple(f"b{idx}" for idx in range(combined.bit_length()) if (combined >> idx) & 1)
    return tokens


def lead_lag_path_signature(morph_series: List[float]) -> np.ndarray:
    """
    Build a (T,1) path from a series of recovery priorities, apply the
    lead‑lag transform, and return its simple signature.
    """
    path = np.asarray(morph_series, dtype=float).reshape(-1, 1)  # (T,1)
    ll = lead_lag_transform(path)                                 # (2T‑1,2)
    sig = simple_signature(ll)
    return sig


def hybrid_regret_bandit_step(
    morphology: Morphology,
    actions: List[MathAction],
    time_step: int = 1,
) -> BanditAction:
    """
    End‑to‑end hybrid step:
    1. Compute morphology feature vector and recovery priority series (single
       value repeated to emulate a short horizon).
    2. Encode tokens → attach to actions.
    3. Build lead‑lag signature from the priority series.
    4. Use the priority as the scaling factor in the regret‑weighted bandit.
    """
    # 1. Feature vector and priority
    vec = compute_morphology_vector(morphology)
    priority = vec[-1]                     # last element is recovery priority

    # 2. Token encoding (mutates actions only for demonstration)
    tokens = encode_morphology_tokens(vec)
    enriched_actions = [
        MathAction(
            id=a.id,
            tokens=tokens,
            expected_value=a.expected_value,
            cost=a.cost,
            risk=a.risk,
        )
        for a in actions
    ]

    # 3. Signature from a short constant series (could be a sliding window)
    series = [priority] * 5                # length‑5 dummy series
    signature = lead_lag_path_signature(series)

    # 4. Bandit selection
    return regret_weighted_bandit(enriched_actions, signature, scaling=priority, time_step=time_step)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a sample morphology
    morph = Morphology(length=2.0, width=1.5, height=0.8, mass=3.4)

    # Define a few candidate actions
    candidate_actions = [
        MathAction(id="A", tokens=(), expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="B", tokens=(), expected_value=8.0, cost=1.5, risk=0.05),
        MathAction(id="C", tokens=(), expected_value=12.0, cost=3.0, risk=0.2),
    ]

    # Perform a hybrid bandit step
    result = hybrid_regret_bandit_step(morph, candidate_actions, time_step=7)

    # Print results – should run without exception
    print("Selected Action:", result.action_id)
    print("Propensity:", result.propensity)
    print("Expected Reward (utility):", result.expected_reward)
    print("Confidence Bound:", result.confidence_bound)