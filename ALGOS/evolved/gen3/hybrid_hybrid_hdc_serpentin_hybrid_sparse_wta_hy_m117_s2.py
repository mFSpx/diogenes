# DARWIN HAMMER — match 117, survivor 2
# gen: 3
# parent_a: hybrid_hdc_serpentina_self_righ_m50_s2.py (gen1)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# born: 2026-05-29T23:25:55Z

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (Parent A)
# ----------------------------------------------------------------------
Vector = np.ndarray  # bipolar hypervector of dtype int8


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    """Generate a bipolar random hypervector."""
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.int8, count=dim
    )
    return data


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministically map a symbol to a bipolar hypervector."""
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], byteorder="big"
    )
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (multiplication) of two hypervectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    return a * b


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition of hypervectors followed by binarization (sign)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("bundle requires at least one vector")
    stacked = np.stack(vecs, axis=0).astype(np.int32)
    summed = stacked.sum(axis=0)
    return np.where(summed >= 0, 1, -1).astype(np.int8)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """True cosine similarity handling sparse (zero) entries."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal shape")
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# Sparse WTA primitives (Parent B)
# ----------------------------------------------------------------------
def expand(values: Sequence[float], m: int, salt: str = "") -> np.ndarray:
    """
    Hash‑based sparse projection of a short list into an m‑dimensional dense vector.
    Each input value contributes to three pseudo‑random positions with random sign.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = np.zeros(m, dtype=np.float64)
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: np.ndarray, k: int) -> np.ndarray:
    """
    Return a boolean mask where the k largest absolute values are True.
    Ties are broken by index order.
    """
    k = max(0, min(k, values.size))
    if k == 0:
        return np.zeros_like(values, dtype=bool)
    # argsort of negative absolute values gives descending order
    idx = np.argpartition(-np.abs(values), k - 1)[:k]
    mask = np.zeros(values.shape, dtype=bool)
    mask[idx] = True
    return mask


# ----------------------------------------------------------------------
# Model pool (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """
    RAM‑constrained pool with FIFO eviction.
    Uses an OrderedDict to guarantee deterministic oldest‑entry removal.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: "OrderedDict[str, ModelTier]" = OrderedDict()

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

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            oldest_key, _ = self.loaded.popitem(last=False)  # FIFO eviction
            del oldest_key  # noqa: F841
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
    Each attribute name is mapped to a symbolic hypervector; the scalar value is
    turned into a bipolar scaling (+1 / -1) based on a reference and then bound.
    All bound vectors are bundled.
    """
    bound_vectors: List[Vector] = []
    for name, value in attributes.items():
        sym = symbol_vector(name, dim)
        ref_val = reference.get(name, 0.0) if reference else 0.0
        sign = 1 if (value - ref_val) >= 0 else -1
        scale_vec = np.full(dim, sign, dtype=np.int8)
        bound_vectors.append(bind(sym, scale_vec))
    return bundle(bound_vectors)


def sparse_wta_hv(
    scores: Sequence[float],
    k: int,
    dim: int = 10000,
    salt: str = "",
    zero_for_non_winner: bool = False,
) -> Vector:
    """
    Convert a list of real‑valued scores into a sparse WTA hypervector.

    Steps:
    1. Expand the scores into a dense `dim`‑dimensional vector.
    2. Select the `k` most salient dimensions (by absolute magnitude).
    3. Binarize the selected dimensions to +1 / -1 using the sign of the expanded value.
       Non‑selected dimensions become 0 (if `zero_for_non_winner`) or -1 otherwise,
       preserving a fully bipolar vector when desired.
    """
    expanded = expand(scores, dim, salt)
    mask = top_k_mask(expanded, k)

    hv = np.empty(dim, dtype=np.int8)
    hv[mask] = np.where(expanded[mask] >= 0, 1, -1).astype(np.int8)

    if zero_for_non_winner:
        hv[~mask] = 0
    else:
        hv[~mask] = -1  # keep bipolar property

    return hv


def hybrid_priority(
    morph_attrs: Dict[str, float],
    model_scores: Sequence[float],
    k_winners: int,
    dim: int = 10000,
    b: float = 0.7,
    k_scale: float = 1.3,
    max_index: float = 10.0,
    salt: str = "",
) -> float:
    """
    Fuse hyperdimensional similarity with sparse WTA saliency into a single priority value.

    - `M` = morphology hypervector (bipolar).
    - `S` = sparse WTA hypervector (bipolar, optional zeros).
    - `sim` = cosine similarity(M, S) ∈ [-1, 1].
    - Priority is a linear map of `sim` into the analytic range,
      then clipped to `[0, max_index]`.

    Returns:
        float: the clipped priority.
    """
    M = morphology_hv(morph_attrs, dim=dim)
    S = sparse_wta_hv(model_scores, k_winners, dim=dim, salt=salt, zero_for_non_winner=True)
    sim = cosine_similarity(M, S)  # now properly normalized
    priority = b + k_scale * sim
    return float(max(0.0, min(priority, max_index)))