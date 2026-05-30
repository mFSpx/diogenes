# DARWIN HAMMER — match 5615, survivor 5
# gen: 5
# parent_a: hybrid_variational_free_ene_hybrid_hybrid_label__m1556_s2.py (gen4)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_hybrid_m1211_s7.py (gen4)
# born: 2026-05-30T00:03:27Z

"""Hybrid Variational-Free-Energy & Sketch-Bandit Router

Parents:
- Parent A: hybrid_variational_free_ene_hybrid_hybrid_label__m1556_s2.py
- Parent B: hybrid_hybrid_sketches_rlct_hybrid_hybrid_hybrid_m1211_s7.py

Mathematical Bridge:
The variational free‑energy (VFE) core of Parent A provides a scalar KL divergence
that quantifies the mismatch between a variational posterior q and a prior p.
Parent B supplies a scalar Real Log‑Canonical‑Threshold (RLCT) λ̂ obtained from a
Count‑Min sketch of the data.  We let λ̂ modulate the *pruning probability* that
weights the KL term, while the *recovery priority* derived from object morphology
scales the same weight.  Consequently the VFE term becomes

    VFE = KL(q‖p) · (1 – α·priority) · (1 – β·entropy_mod) + γ·λ̂

where α,β,γ are tunable coefficients.  The resulting VFE value is then used as
the expected reward in the contextual bandit of Parent B, whose confidence bound
is multiplied by √λ̂ and finally scaled by a slowly varying resource level ℓ.

The module implements this fused pipeline in three high‑level functions:
`variational_free_energy`, `estimate_rlct`, and `hybrid_decision`.  The
`hybrid_decision` function demonstrates the full hybrid operation from raw
items to a bandit score for each arm."""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class TextFeatures:
    evidence_count: int
    planning_count: int
    delay_count: int

# ----------------------------------------------------------------------
# Variational Free Energy utilities (Parent A)
# ----------------------------------------------------------------------
def kl_gaussian(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
) -> float:
    """Closed‑form KL divergence KL[N(q)||N(p)] for univariate Gaussians."""
    mu_q = np.asarray(mu_q, dtype=float)
    sigma_q = np.asarray(sigma_q, dtype=float)
    mu_p = np.asarray(mu_p, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float)

    term1 = np.log(sigma_p / sigma_q)
    term2 = (sigma_q**2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p**2)
    kl = np.sum(term1 + term2 - 0.5)
    return float(kl)


def recovery_priority(morph: Morphology) -> float:
    """Higher priority for larger, lighter objects."""
    volume = morph.length * morph.width * morph.height
    if morph.mass == 0:
        return 0.0
    return volume / morph.mass


def entropy_modulation(txt: TextFeatures) -> float:
    """Entropy‑like modulation based on textual feature richness."""
    total = txt.evidence_count + txt.planning_count + txt.delay_count
    if total == 0:
        return 0.0
    # Simple Shannon‑like term (log‑scaled)
    probs = np.array([
        txt.evidence_count,
        txt.planning_count,
        txt.delay_count
    ]) / total
    return -np.sum(probs * np.log(probs + 1e-12))


def variational_free_energy(
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
    priority: float,
    entropy_mod: float,
    lambda_hat: float,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.1,
) -> float:
    """
    Hybrid VFE = KL * (1‑α·priority) * (1‑β·entropy_mod) + γ·λ̂
    """
    kl = kl_gaussian(mu_q, sigma_q, mu_p, sigma_p)
    weight = (1.0 - alpha * priority) * (1.0 - beta * entropy_mod)
    vfe = kl * weight + gamma * lambda_hat
    return vfe

# ----------------------------------------------------------------------
# Sketching & RLCT estimation (Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Count‑Min sketch matrix of shape (depth, width)."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return slope and intercept of simple least‑squares regression."""
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def estimate_rlct(sketch: List[List[int]]) -> float:
    """
    Estimate the Real Log‑Canonical‑Threshold (RLCT) λ̂.
    Procedure:
    1. Flatten sketch and keep non‑zero counts.
    2. For each count c_i at position i (1‑based), compute
       x_i = log(log(i + 2))   (avoid log(0))
       y_i = log(c_i + 1)      (avoid log(0))
    3. Perform linear regression y = λ̂ * x + const.
    4. Return the slope λ̂ (non‑negative).
    """
    flat = np.array(sketch).flatten()
    nonzero_idx = np.nonzero(flat)[0]
    if len(nonzero_idx) == 0:
        return 0.0
    counts = flat[nonzero_idx].astype(float)

    x = np.log(np.log(nonzero_idx + 2.0) + 1e-12)
    y = np.log(counts + 1.0)

    slope, _ = _linear_regression(x, y)
    return max(slope, 0.0)


# ----------------------------------------------------------------------
# Bandit scoring (Parent B) augmented with VFE
# ----------------------------------------------------------------------
def bandit_score(
    expected_reward: float,
    confidence: float,
    lambda_hat: float,
    resource_level: float,
) -> float:
    """
    Unified decision equation:
        score = (expected_reward + confidence * sqrt(λ̂)) * ℓ
    """
    return (expected_reward + confidence * math.sqrt(lambda_hat)) * resource_level


# ----------------------------------------------------------------------
# Hybrid pipeline demonstration
# ----------------------------------------------------------------------
def hybrid_decision(
    items: List[Any],
    morphology: Morphology,
    text_feat: TextFeatures,
    mu_q: float | np.ndarray,
    sigma_q: float | np.ndarray,
    mu_p: float | np.ndarray,
    sigma_p: float | np.ndarray,
    expected_rewards: List[float],
    confidences: List[float],
    resource_level: float,
) -> List[float]:
    """
    Perform a full hybrid step:
    1. Sketch the items and estimate RLCT λ̂.
    2. Compute recovery priority and entropy modulation.
    3. Evaluate the hybrid variational free energy (VFE) using λ̂.
    4. Use the VFE as the base expected reward for each arm and compute
       bandit scores.
    Returns a list of scores, one per arm.
    """
    # 1. Sketch & RLCT
    sketch = count_min_sketch(items)
    lambda_hat = estimate_rlct(sketch)

    # 2. Modulators
    priority = recovery_priority(morphology)
    entropy_mod = entropy_modulation(text_feat)

    # 3. Hybrid VFE (acts as a global reward bias)
    vfe_bias = variational_free_energy(
        mu_q, sigma_q, mu_p, sigma_p,
        priority, entropy_mod,
        lambda_hat
    )

    # 4. Bandit scores per arm
    scores = []
    for er, cf in zip(expected_rewards, confidences):
        # Blend the VFE bias with the arm‑specific expected reward
        blended_reward = er + vfe_bias
        score = bandit_score(blended_reward, cf, lambda_hat, resource_level)
        scores.append(score)
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    random.seed(42)
    np.random.seed(42)

    # Items to sketch (e.g., token ids)
    items = [f"item_{i}" for i in range(200)]

    # Morphology & text features
    morph = Morphology(length=2.5, width=1.2, height=0.8, mass=3.0)
    txt_feat = TextFeatures(evidence_count=15, planning_count=7, delay_count=3)

    # Gaussian parameters for VFE
    mu_q = np.array([0.0, 1.0])
    sigma_q = np.array([1.0, 0.5])
    mu_p = np.array([0.5, 0.5])
    sigma_p = np.array([1.2, 0.8])

    # Bandit arm parameters
    expected_rewards = [0.2, 0.5, 0.1]          # base expected reward per arm
    confidences = [0.1, 0.05, 0.2]             # confidence (inverse variance)
    resource_level = 0.9                       # slowly varying ℓ

    scores = hybrid_decision(
        items,
        morph,
        txt_feat,
        mu_q,
        sigma_q,
        mu_p,
        sigma_p,
        expected_rewards,
        confidences,
        resource_level,
    )

    print("Hybrid bandit scores per arm:", scores)