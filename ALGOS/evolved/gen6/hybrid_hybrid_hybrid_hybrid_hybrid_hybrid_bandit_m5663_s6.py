# DARWIN HAMMER — match 5663, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py (gen2)
# born: 2026-05-30T00:03:59Z

"""Hybrid algorithm merging:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s0.py
- Parent B: hybrid_hybrid_bandit_router_workshare_allocator_m60_s1.py

Mathematical bridge:
The Structural Similarity Index (SSIM) computed on two input vectors drives the
selection of the *k* most similar dimensions. Those dimensions are hashed into a
high‑dimensional sparse expansion. The sparse vector is summed, perturbed with
Laplace noise (scale = ε₁) and normalised. The resulting privacy‑preserving
aggregate is interpreted as a *risk* score (unique identifiers / total records).
That risk score modulates the gain of a StoreState (honey‑bee store) which in turn
produces a bounded control signal *dance*. The dance value is used as the scale
ε₂ of a second Laplace noise that is added to the expected reward of each
BanditAction, thereby coupling the privacy‑aware similarity computation with the
adaptive work‑share allocation mechanism. The final action selection respects
both the bandit confidence bounds and the privacy‑adjusted rewards."""


import hashlib
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length vectors."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def hybrid_sparse_expansion(values: List[float], m: int, salt: str = "") -> np.ndarray:
    """
    Hash‑based sparse expansion of `values` into a dense vector of length `m`.
    Each input value contributes three signed entries determined by SHA‑256.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = np.zeros(m, dtype=np.float64)

    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) == 0 else -1.0
            out[j] += sign * v
    return out


def top_k_similar_indices(x: List[float], y: List[float], k: int) -> List[int]:
    """
    Return indices of the `k` dimensions with the largest absolute
    contribution to the SSIM numerator (2*mx*my + c1)*(2*cov + c2).
    """
    if len(x) != len(y):
        raise ValueError("vectors must have same length")
    if k <= 0:
        raise ValueError("k must be positive")
    # Simple proxy: absolute difference; smaller diff → higher similarity
    diffs = np.abs(np.asarray(x) - np.asarray(y))
    return list(np.argsort(diffs)[:k])


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honey‑bee style store whose dance signal modulates privacy noise."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply store dynamics and store the last Δ."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Fusion functions
# ----------------------------------------------------------------------
def privacy_preserving_aggregate(
    vectors: List[List[float]],
    top_k: int,
    expansion_dim: int,
    epsilon1: float,
) -> Tuple[np.ndarray, float]:
    """
    1. Compute pairwise SSIM between the first vector and each other vector.
    2. Pick `top_k` most similar vectors (including the reference).
    3. Expand each selected vector with `hybrid_sparse_expansion`.
    4. Sum the expansions, add Laplace noise (scale=1/epsilon1), and normalise.
    5. Return the noisy normalised aggregate and the derived risk score
       (unique identifiers / total records) – here simulated as
       `len(top_k) / len(vectors)`.
    """
    if not vectors:
        raise ValueError("vectors list cannot be empty")
    ref = vectors[0]
    ssim_scores = [compute_ssim(ref, v) for v in vectors]
    # Get indices of top_k highest SSIM scores
    top_indices = sorted(range(len(ssim_scores)), key=lambda i: ssim_scores[i], reverse=True)[:top_k]

    expanded = np.zeros(expansion_dim, dtype=np.float64)
    for idx in top_indices:
        exp_vec = hybrid_sparse_expansion(vectors[idx], expansion_dim, salt=str(idx))
        expanded += exp_vec

    # Laplace noise for differential privacy
    noise = np.random.laplace(loc=0.0, scale=1.0 / epsilon1, size=expansion_dim)
    noisy = expanded + noise

    # Normalisation to unit L2 norm (avoid division by zero)
    norm = np.linalg.norm(noisy)
    normalized = noisy / norm if norm > 0 else noisy

    # Simulated risk: proportion of selected vectors
    risk = len(top_indices) / len(vectors)
    return normalized, risk


def allocate_workshare(
    store: StoreState,
    actions: List[BanditAction],
    risk: float,
    epsilon2: float,
) -> BanditAction:
    """
    Adjust each action's expected reward with Laplace noise whose scale is
    `epsilon2 * store.dance * risk`. Then pick the action with the highest
    (reward + confidence_bound) * propensity product.
    """
    scale = epsilon2 * store.dance * risk
    noisy_actions = []
    for act in actions:
        noisy_reward = act.expected_reward + np.random.laplace(loc=0.0, scale=scale)
        noisy_actions.append(
            BanditAction(
                action_id=act.action_id,
                propensity=act.propensity,
                expected_reward=noisy_reward,
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )
        )
    # Scoring function blending bandit metrics
    def score(a: BanditAction) -> float:
        return a.propensity * (a.expected_reward + a.confidence_bound)

    chosen = max(noisy_actions, key=score)
    return chosen


def hybrid_step(
    vectors: List[List[float]],
    store: StoreState,
    actions: List[BanditAction],
    top_k: int = 3,
    expansion_dim: int = 128,
    epsilon1: float = 0.5,
    epsilon2: float = 0.3,
) -> Tuple[BanditAction, np.ndarray]:
    """
    One unified iteration:
    * Produce a privacy‑preserving aggregate and a risk score.
    * Update the store using the aggregate as inflow and a dummy outflow.
    * Select a bandit action using the updated store and the risk.
    Returns the selected action and the aggregate vector.
    """
    aggregate, risk = privacy_preserving_aggregate(
        vectors=vectors,
        top_k=top_k,
        expansion_dim=expansion_dim,
        epsilon1=epsilon1,
    )
    # For illustration, treat the L1 norm of the aggregate as inflow magnitude
    inflow = [float(np.abs(aggregate).sum())]
    outflow = [0.0]  # no outflow in this simple demo
    store.update(inflow, outflow)

    chosen_action = allocate_workshare(store, actions, risk, epsilon2)
    return chosen_action, aggregate


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic vectors (5‑dimensional)
    rng = np.random.default_rng(seed=42)
    vectors = [rng.random(5).tolist() for _ in range(10)]

    # Initialise store
    store = StoreState(level=5.0, alpha=0.8, beta=0.5, gain=0.3, limit=8.0)

    # Define a few dummy bandit actions
    actions = [
        BanditAction(
            action_id="A",
            propensity=0.6,
            expected_reward=1.2,
            confidence_bound=0.4,
            algorithm="HybridBandit",
        ),
        BanditAction(
            action_id="B",
            propensity=0.3,
            expected_reward=0.9,
            confidence_bound=0.6,
            algorithm="HybridBandit",
        ),
        BanditAction(
            action_id="C",
            propensity=0.1,
            expected_reward=1.5,
            confidence_bound=0.2,
            algorithm="HybridBandit",
        ),
    ]

    selected, agg = hybrid_step(vectors, store, actions)

    print("Selected action:", selected)
    print("Aggregate norm:", np.linalg.norm(agg))
    print("Store level after update:", store.level)
    print("Store dance signal:", store.dance)