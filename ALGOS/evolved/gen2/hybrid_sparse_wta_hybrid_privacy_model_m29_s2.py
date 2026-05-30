# DARWIN HAMMER — match 29, survivor 2
# gen: 2
# parent_a: sparse_wta.py (gen0)
# parent_b: hybrid_privacy_model_pool_m7_s0.py (gen1)
# born: 2026-05-29T23:23:06Z

"""Hybrid algorithm merging Sparse Winner-Take-All (sparse_wta.py) and
Privacy‑Aware Model Pool (hybrid_privacy_model_pool_m7_s0.py).

Mathematical bridge:
1. The sparse expansion maps an input vector `v ∈ ℝⁿ` to a high‑dimensional
   space `ℝᵐ` using locality‑sensitive hashing (parent A).  
2. The resulting expanded vector `e` is treated as a *query* whose
   aggregate (sum) is perturbed with Laplace noise to satisfy differential
   privacy (parent B).  
3. The noisy aggregate is normalised and fed into the reconstruction‑risk
   function `risk = unique_quasi_identifiers / total_records`.  This risk
   score is then used as the scale of a second Laplace noise term that
   governs whether a model may be admitted to the pool.  

Thus the core topology of both parents is fused: hash‑based sparse projection
→ DP‑noised aggregation → risk‑scaled privacy budget → model‑pool admission
decisions. The following module implements three representative hybrid
operations that showcase this pipeline."""
from __future__ import annotations

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Sparse Winner‑Take‑All utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two equal‑length binary vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must be same length")
    return sum(x != y for x, y in zip(a, b))


# ----------------------------------------------------------------------
# Parent B – Privacy‑aware Model Pool
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction for simplicity
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk ∈ [0,1] proportional to the fraction of unique identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Laplace‑noised sum of `values` (ε‑DP)."""
    total = sum(values)
    noise = np.random.laplace(0.0, sensitivity / epsilon)
    return total + noise


def load_model_with_hybrid_privacy(
    model: ModelTier,
    pool: ModelPool,
    expanded_mask: List[int],
    epsilon: float = 1.0,
) -> None:
    """
    Hybrid admission rule:
      1. Compute a DP‑noised aggregate of the binary mask.
      2. Derive a reconstruction risk from the current pool size.
      3. Add a second Laplace term scaled by that risk.
      4. Admit the model iff the noisy total RAM usage stays within the ceiling.
    """
    # Step 1 – DP aggregate of the mask (acts as a proxy for query “size”)
    noisy_sum = dp_aggregate(expanded_mask, epsilon=epsilon, sensitivity=1.0)

    # Step 2 – risk based on current pool occupancy
    risk = reconstruction_risk_score(len(pool.loaded), pool.ram_ceiling_mb)

    # Step 3 – additional privacy noise proportional to risk
    risk_noise = np.random.laplace(0.0, risk / epsilon)

    # Step 4 – decide admission
    projected_usage = pool._used() + model.ram_mb + noisy_sum + risk_noise
    if projected_usage <= pool.ram_ceiling_mb:
        try:
            pool.load(model)
        except RuntimeError:
            # Fallback to eviction strategy if direct load fails
            pool.load_with_eviction(model)


# ----------------------------------------------------------------------
# Hybrid Functions Demonstrating the Unified System
# ----------------------------------------------------------------------
def hybrid_expand_and_mask(
    raw_values: List[float],
    dim: int,
    top_k: int,
    salt: str = "",
) -> List[int]:
    """
    Expand `raw_values` into a sparse high‑dimensional vector,
    then produce a top‑k binary mask.
    """
    expanded = expand(raw_values, dim, salt)
    mask = top_k_mask(expanded, top_k)
    return mask


def hybrid_model_admission(
    model: ModelTier,
    pool: ModelPool,
    raw_query: List[float],
    dim: int,
    top_k: int,
    epsilon: float = 1.0,
) -> None:
    """
    Full pipeline:
      * Convert a query vector into a sparse mask,
      * Use the mask to compute a privacy‑aware admission decision,
      * Load the model into the pool if permitted.
    """
    mask = hybrid_expand_and_mask(raw_query, dim, top_k)
    load_model_with_hybrid_privacy(model, pool, mask, epsilon=epsilon)


def hybrid_hamming_similarity(
    query_values: List[float],
    stored_mask: List[int],
    dim: int,
    top_k: int,
    salt: str = "",
) -> float:
    """
    Compute a privacy‑preserving similarity score between a query vector
    and a stored binary mask using Hamming distance on top‑k masks.
    The raw Hamming distance is normalised to [0,1] and Laplace‑noised.
    """
    query_mask = hybrid_expand_and_mask(query_values, dim, top_k, salt)
    raw_dist = hamming(query_mask, stored_mask)
    max_dist = len(query_mask)
    norm_dist = raw_dist / max_dist if max_dist else 0.0
    # Add DP noise to the similarity (lower distance → higher similarity)
    noisy_similarity = 1.0 - norm_dist + np.random.laplace(0.0, 1.0 / 1.0)
    return max(0.0, min(1.0, noisy_similarity))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic query vector
    query = [random.random() for _ in range(10)]

    # Parameters for the hybrid pipeline
    DIM = 256
    TOP_K = 20
    EPS = 0.8
    SALT = "demo"

    # Build a model pool and a candidate model
    pool = ModelPool(ram_ceiling_mb=4000)
    candidate = ModelTier(name="demo-model", ram_mb=512, tier="T1")

    # Run the hybrid admission process
    hybrid_model_admission(candidate, pool, query, DIM, TOP_K, epsilon=EPS)

    # Verify that the model is loaded
    assert pool.is_loaded(candidate.name), "Model should have been loaded"

    # Create a stored mask (e.g., from a previously admitted model)
    stored_mask = hybrid_expand_and_mask([0.5] * 10, DIM, TOP_K, SALT)

    # Compute a privacy‑preserving similarity score
    sim = hybrid_hamming_similarity(query, stored_mask, DIM, TOP_K, SALT)
    print(f"Privacy‑preserving similarity: {sim:.3f}")

    # Print pool status
    print(f"Pool usage: {pool._used()} MB / {pool.ram_ceiling_mb} MB")
    print(f"Loaded models: {list(pool.loaded.keys())}")