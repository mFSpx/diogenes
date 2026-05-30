# DARWIN HAMMER — match 117, survivor 1
# gen: 3
# parent_a: hybrid_hdc_serpentina_self_righ_m50_s2.py (gen1)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# born: 2026-05-29T23:25:55Z

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives
# ----------------------------------------------------------------------
Vector = List[int]  # bipolar hypervector


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    dim = len(vecs[0])
    for v in vecs:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    # element‑wise sum then binarize by sign
    summed = [sum(comp) for comp in zip(*vecs)]
    return [1 if s >= 0 else -1 for s in summed]


def similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity for bipolar vectors (identical to normalized dot)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)  # because |a| = |b| = sqrt(dim) for bipolar vectors


# ----------------------------------------------------------------------
# Sparse WTA primitives
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse projection of a short list into an m‑dimensional vector."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three hash collisions per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask (1 for winner, 0 otherwise) of length len(values)."""
    k = max(0, min(k, len(values)))
    # indices of the k largest values (ties broken by index)
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]


# ----------------------------------------------------------------------
# Model pool
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._eviction_order = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self._eviction_order.append(model.name)

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the oldest entry (FIFO)
            oldest_key = self._eviction_order.pop(0)
            del self.loaded[oldest_key]
        self.load(model)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def morphology_hv(
    attributes: Dict[str, float],
    reference: Dict[str, float] | None = None,
    dim: int = 10000,
) -> Vector:
    """
    Encode a dictionary of scalar morphological attributes into a single bipolar hypervector.
    Each attribute name gets a symbolic hypervector; the scalar value is turned into a
    bipolar scaling vector by comparing to a reference (or to zero if None) and then bound.
    All bound vectors are bundled.
    """
    bound_vectors: List[Vector] = []
    for name, value in attributes.items():
        sym = symbol_vector(name, dim)
        # reference value for sign determination
        ref_val = reference.get(name, 0.0) if reference else 0.0
        sign = 1 if (value - ref_val) >= 0 else -1
        scale_vec = [sign] * dim  # pure bipolar scaling
        bound = bind(sym, scale_vec)
        bound_vectors.append(bound)
    return bundle(bound_vectors)


def sparse_wta_hv(scores: List[float], k: int, dim: int = 10000, salt: str = "") -> Vector:
    """
    Convert a list of real‑valued scores into a sparse WTA hypervector.
    1. Expand to a dense `dim`‑dimensional vector via `expand`.
    2. Apply `top_k_mask` to keep only the `k` most salient dimensions.
    3. Binarize the kept dimensions to bipolar (+1 / -1) using the sign of the expanded value.
    """
    expanded = expand(scores, dim, salt)
    mask = top_k_mask(expanded, k)
    hv = []
    for val, m in zip(expanded, mask):
        if m:
            hv.append(1 if val >= 0 else -1)
        else:
            hv.append(0)  # zero for non‑winners (sparse)
    return hv  # type: ignore[return-value]


def hybrid_priority(
    morph_attrs: Dict[str, float],
    model_scores: List[float],
    k_winners: int,
    dim: int = 10000,
    b: float = 0.7,
    k_scale: float = 1.3,
    max_index: float = 10.0,
) -> float:
    """
    Fuse hyperdimensional similarity with sparse WTA saliency.
    - `M` = morphology hypervector (bipolar).
    - `S` = sparse WTA hypervector (bipolar with zeros).
    - Similarity `sim = similarity(M, S)`.
    - The hybrid priority is a linear map of `sim` into the analytic range
      used by the original right‑ing time index:  `priority = b + k_scale * sim`
      clipped to `[0, max_index]`.
    """
    M = morphology_hv(morph_attrs, dim=dim)
    S = sparse_wta_hv(model_scores, k_winners, dim=dim)
    # zero-pad S to match length of M
    S += [0] * (len(M) - len(S))
    sim = similarity(M, S)
    # Clip to ensure priority is within valid range
    priority = max(0.0, min(max_index, b + k_scale * sim))
    return priority


def main():
    model_pool = ModelPool()
    morph_attrs = {"attr1": 1.0, "attr2": 2.0}
    model_scores = [0.5, 0.7, 0.3]
    k_winners = 2
    priority = hybrid_priority(morph_attrs, model_scores, k_winners)
    model_tier = ModelTier("model1", 1000, "T1")
    model_pool.load_with_eviction(model_tier)
    print(priority)


if __name__ == "__main__":
    main()