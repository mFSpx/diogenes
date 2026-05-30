# DARWIN HAMMER — match 1555, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1154_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m914_s0.py (gen5)
# born: 2026-05-29T23:37:21Z

"""Hybrid Fusion of TTT‑Linear Compression/Bandit Learning with Tropical Regret‑VRAM Scheduling.

Parent A contributes:
- A weight matrix **W** that compresses the observed input distribution.
- A contextual bandit update that modifies **W** using observed rewards.
- A structural similarity (SSIM) measure that evaluates the fidelity of the compressed representation.

Parent B contributes:
- A tropical max‑plus evaluation `max_i (c_i + g)` where the coefficients **c_i** are drawn from a matrix.
- A reconstruction risk score `r = unique_qi / total_records` that quantifies privacy‑related uncertainty.
- A Hoeffding bound that supplies a confidence interval for observed gains.

**Mathematical bridge** – The flattened weight matrix **W** is interpreted as the coefficient vector **c** of the tropical polynomial. The bandit‑driven reward updates reshape **c**, while the reconstruction risk rescales **c** before the tropical evaluation. The SSIM score of the compression is fed as the “gain” *g* in the tropical max‑plus function, thereby linking the two parent topologies into a single decision‑making pipeline.
"""

import json
import os
import sys
import math
import random
import pathlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit structures (from Parent A)
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

# Global mutable stores used by the bandit component
_POLICY: dict[str, List[float]] = {}
_STORE: dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n != 0 else 0.0

# ----------------------------------------------------------------------
# Model tier definition (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """A minimal SSIM implementation for 2‑D grayscale arrays."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    return float(numerator / denominator)

def tropical_max_plus(coeffs: np.ndarray, gain: float) -> float:
    """Evaluate max_i (coeff_i + gain) for a vector of coefficients."""
    return float(np.max(coeffs + gain))

def hoeffding_bound(observed: List[float], confidence: float = 0.95) -> float:
    """Hoeffding bound for the empirical mean of observed values."""
    if not observed:
        return 0.0
    epsilon = math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(observed)))
    return epsilon

def reconstruction_risk(unique_qi: int, total_records: int) -> float:
    """Risk score in [0,1] proportional to the fraction of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_qi / total_records))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compress_with_weights(data: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a linear compression using the weight matrix (TTT‑Linear style).
    The matrix is assumed to have shape (k, d) where d = data.shape[-1].
    """
    if data.ndim != 2:
        raise ValueError("data must be 2‑D (samples × features)")
    if weight_matrix.shape[1] != data.shape[1]:
        raise ValueError("weight matrix inner dimension must match data features")
    return data @ weight_matrix.T  # shape (samples, k)

def bandit_update_weights(
    weight_matrix: np.ndarray,
    updates: List[BanditUpdate],
    lr: float = 0.01,
) -> np.ndarray:
    """
    Adjust the weight matrix using bandit rewards.
    Each update contributes a gradient proportional to the reward and the
    propensity (inverse‑propensity weighting).
    """
    grad = np.zeros_like(weight_matrix)
    for upd in updates:
        # Retrieve current estimate for the action
        est = _reward(upd.action_id)
        # Simple gradient: (reward - estimate) * propensity
        delta = (upd.reward - est) * upd.propensity
        # Map the scalar delta onto the matrix via outer product with a random vector
        # (acts as a stochastic approximation of the true gradient)
        rand_vec = np.random.randn(weight_matrix.shape[0], 1)
        grad += delta * rand_vec @ np.random.randn(1, weight_matrix.shape[1])
        # Update the policy store
        total, n = _POLICY.get(upd.action_id, [0.0, 0.0])
        _POLICY[upd.action_id] = [total + upd.reward, n + 1]
    return weight_matrix + lr * grad

def hybrid_schedule(
    compressed: np.ndarray,
    weight_matrix: np.ndarray,
    model_tiers: List[ModelTier],
    unique_qi: int,
    total_records: int,
    ram_ceiling_mb: int,
) -> Tuple[ModelTier, float]:
    """
    Combine SSIM‑based gain, tropical max‑plus evaluation and reconstruction risk
    to pick a model tier that respects the VRAM ceiling.
    Returns the chosen ModelTier and the associated decision score.
    """
    # 1. Compute a reference reconstruction (identity) and similarity score per sample
    recon = compressed @ np.linalg.pinv(weight_matrix)  # pseudo‑inverse reconstruction
    ssim_scores = np.array([ssim(compressed[i], recon[i]) for i in range(compressed.shape[0])])
    gain = ssim_scores.mean()  # scalar gain for tropical evaluation

    # 2. Flatten weight matrix to coefficient vector and rescale by privacy risk
    coeffs = weight_matrix.flatten()
    risk = reconstruction_risk(unique_qi, total_records)
    coeffs = coeffs * (1.0 - risk)  # higher risk shrinks influence

    # 3. Tropical max‑plus evaluation yields a base decision value
    base_val = tropical_max_plus(coeffs, gain)

    # 4. Adjust by Hoeffding confidence on the SSIM gain
    conf = hoeffding_bound(ssim_scores.tolist())
    decision_score = base_val - conf  # conservative adjustment

    # 5. Choose the highest‑tier model that fits within the remaining VRAM budget
    remaining = ram_ceiling_mb
    feasible = [mt for mt in sorted(model_tiers, key=lambda m: -m.ram_mb) if mt.ram_mb <= remaining]
    chosen = feasible[0] if feasible else ModelTier(name="fallback", ram_mb=0, tier="none")
    return chosen, decision_score

# ----------------------------------------------------------------------
# Example driver demonstrating the hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_step(
    raw_data: np.ndarray,
    weight_matrix: np.ndarray,
    bandit_updates: List[BanditUpdate],
    model_tiers: List[ModelTier],
    unique_qi: int,
    total_records: int,
    ram_ceiling_mb: int,
) -> Tuple[ModelTier, float, np.ndarray]:
    """
    End‑to‑end hybrid step:
    1. Compress data with current weights.
    2. Update weights via bandit feedback.
    3. Schedule a model tier using the updated weights and privacy risk.
    Returns the selected ModelTier, the decision score, and the updated weight matrix.
    """
    compressed = compress_with_weights(raw_data, weight_matrix)
    updated_weights = bandit_update_weights(weight_matrix, bandit_updates)
    chosen_tier, score = hybrid_schedule(
        compressed,
        updated_weights,
        model_tiers,
        unique_qi,
        total_records,
        ram_ceiling_mb,
    )
    return chosen_tier, score, updated_weights

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 10 samples, 5 features
    np.random.seed(42)
    data = np.random.rand(10, 5).astype(np.float32)

    # Initial weight matrix (compress to 3 dimensions)
    W = np.random.randn(3, 5).astype(np.float32)

    # Dummy bandit updates
    updates = [
        BanditUpdate(context_id="c1", action_id="a1", reward=1.2, propensity=0.8),
        BanditUpdate(context_id="c2", action_id="a2", reward=0.4, propensity=0.5),
    ]

    # Model tiers
    tiers = [
        ModelTier(name="large", ram_mb=8000, tier="L"),
        ModelTier(name="medium", ram_mb=4000, tier="M"),
        ModelTier(name="small", ram_mb=2000, tier="S"),
    ]

    # Privacy parameters
    unique_qi = 123
    total_records = 1000
    ram_ceiling = 5000  # MB

    # Run the hybrid pipeline
    chosen, decision, new_W = hybrid_step(
        raw_data=data,
        weight_matrix=W,
        bandit_updates=updates,
        model_tiers=tiers,
        unique_qi=unique_qi,
        total_records=total_records,
        ram_ceiling_mb=ram_ceiling,
    )

    print(f"Chosen tier: {chosen.name} ({chosen.tier}), RAM {chosen.ram_mb} MB")
    print(f"Decision score: {decision:.4f}")
    print(f"Updated weight matrix shape: {new_W.shape}")