# DARWIN HAMMER — match 62, survivor 1
# gen: 2
# parent_a: sparse_wta.py (gen0)
# parent_b: hybrid_privacy_model_pool_m7_s0.py (gen1)
# born: 2026-05-29T23:25:30Z

"""Hybrid algorithm combining Sparse Winner‑Take‑All (WTA) encoding with
privacy‑aware model‑pool management.

Parent A (sparse_wta.py) provides:
- `expand`   : projects a low‑dimensional real vector into a high‑dimensional
               sparse space using deterministic hash‑based indexing.
- `top_k_mask`: selects the k largest entries and returns a binary mask.
- `hamming`  : Hamming distance between two binary masks.

Parent B (hybrid_privacy_model_pool_m7_s0.py) provides:
- `ModelPool` and `ModelTier` for RAM‑constrained model loading.
- `reconstruction_risk_score` and Laplace noise injection for differential
  privacy.

**Mathematical bridge**  
The current pool state is represented as a real vector `s` (e.g., RAM usage per
model tier). `expand(s, m)` maps `s` into a high‑dimensional sparse vector
`e ∈ ℝ^m`. Applying `top_k_mask(e, k)` yields a binary mask `b ∈ {0,1}^m`. The
Hamming distance `h = hamming(b, b_ref)` between `b` and a fixed reference mask
`b_ref` quantifies how “atypical’’ the pool configuration is. Normalising `h`
produces a privacy‑risk factor `r ∈ [0,1]` that replaces the simplistic
`unique_quasi_identifiers / total_records` used in the original privacy model.
This risk factor directly scales the Laplace noise added to the RAM budget when
deciding whether to load a new model, thereby tightly coupling the sparse
encoding with the differential‑privacy mechanism.

The resulting hybrid functions demonstrate this integration.
"""

from __future__ import annotations
import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Sparse WTA utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """Deterministically project `values` into an m‑dimensional sparse vector."""
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 at the positions of the k largest entries."""
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two equal‑length binary masks."""
    if len(a) != len(b):
        raise ValueError('vectors must be same length')
    return sum(x != y for x, y in zip(a, b))


# ----------------------------------------------------------------------
# Parent B – Model pool with privacy‑aware loading
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


class ModelLoadError(RuntimeError):
    pass


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
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the oldest entry (FIFO policy)
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)


def reconstruction_risk_score(normalised_hamming: float) -> float:
    """Clamp a normalised Hamming distance to [0,1] as a risk score."""
    return max(0.0, min(1.0, normalised_hamming))


def dp_laplace_noise(scale: float) -> float:
    """Draw Laplace noise with zero mean and given scale."""
    return np.random.laplace(0.0, scale)


# ----------------------------------------------------------------------
# Hybrid layer – linking sparse WTA to privacy scoring
# ----------------------------------------------------------------------
def pool_state_vector(pool: ModelPool) -> List[float]:
    """
    Encode the current pool state as a fixed‑length numeric vector.
    For simplicity we use three dimensions: total RAM used, count of T1 models,
    and count of T2 models (T3 is mutually exclusive with T2, so its count is
    inferred).
    """
    total_ram = pool._used()
    t1_cnt = sum(1 for m in pool.loaded.values() if m.tier == "T1")
    t2_cnt = sum(1 for m in pool.loaded.values() if m.tier == "T2")
    return [float(total_ram), float(t1_cnt), float(t2_cnt)]


def hybrid_privacy_risk(pool: ModelPool,
                        embed_dim: int = 128,
                        top_k: int = 10,
                        reference_mask: List[int] | None = None) -> float:
    """
    Compute a privacy risk score from the pool state using the WTA pipeline.
    Steps:
    1. Encode pool state → vector `s`.
    2. Expand `s` to high‑dimensional sparse vector `e = expand(s, embed_dim)`.
    3. Derive binary mask `b = top_k_mask(e, top_k)`.
    4. Compute Hamming distance to a reference mask (default: all‑zero mask).
    5. Normalise by `embed_dim` and map to [0,1] via `reconstruction_risk_score`.
    """
    s = pool_state_vector(pool)
    e = expand(s, embed_dim, salt='hybrid')
    b = top_k_mask(e, top_k)

    if reference_mask is None:
        reference_mask = [0] * embed_dim
    elif len(reference_mask) != embed_dim:
        raise ValueError('reference_mask length must equal embed_dim')

    raw_hamming = hamming(b, reference_mask)
    normalised = raw_hamming / embed_dim
    return reconstruction_risk_score(normalised)


def load_model_hybrid(model: ModelTier,
                      pool: ModelPool,
                      epsilon: float = 1.0,
                      embed_dim: int = 128,
                      top_k: int = 10) -> None:
    """
    Attempt to load `model` into `pool` using a privacy‑aware decision rule.
    The allowable RAM budget is perturbed by Laplace noise whose scale is
    proportional to the hybrid risk score derived from the current pool state.
    """
    # Step 1: compute risk‑based scale
    risk = hybrid_privacy_risk(pool, embed_dim=embed_dim, top_k=top_k)
    noise_scale = risk / max(epsilon, 1e-9)  # avoid division by zero

    # Step 2: draw noise and compute a noisy ceiling
    noise = dp_laplace_noise(noise_scale)
    noisy_ceiling = pool.ram_ceiling_mb + noise

    # Step 3: decide based on noisy ceiling
    if model.ram_mb + pool._used() <= noisy_ceiling:
        try:
            pool.load(model)
        except ModelLoadError:
            # fallback to eviction policy if direct load fails
            pool.load_with_eviction(model)
    else:
        # If the noisy budget is insufficient, we simply skip loading.
        pass


def hybrid_top_k_representation(pool: ModelPool,
                                embed_dim: int = 128,
                                top_k: int = 10) -> List[int]:
    """
    Return the binary top‑k mask that represents the current pool state.
    This function is useful for introspection or downstream similarity queries.
    """
    s = pool_state_vector(pool)
    e = expand(s, embed_dim, salt='repr')
    return top_k_mask(e, top_k)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    np.random.seed(0)
    random.seed(0)

    # Create a model pool with a modest RAM ceiling
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define a few models of varying tiers and RAM footprints
    models = [
        ModelTier(name="alpha", ram_mb=300, tier="T1"),
        ModelTier(name="beta", ram_mb=500, tier="T2"),
        ModelTier(name="gamma", ram_mb=800, tier="T1"),
        ModelTier(name="delta", ram_mb=600, tier="T3"),
    ]

    # Attempt hybrid loading for each model
    for m in models:
        load_model_hybrid(m, pool, epsilon=0.5, embed_dim=64, top_k=5)
        print(f"After trying to load {m.name}: loaded = {list(pool.loaded.keys())}")

    # Show the binary representation of the final pool state
    mask = hybrid_top_k_representation(pool, embed_dim=64, top_k=5)
    print(f"Final top‑k binary mask (len={len(mask)}):", mask)