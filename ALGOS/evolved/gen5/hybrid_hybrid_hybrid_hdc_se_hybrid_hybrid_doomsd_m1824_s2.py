# DARWIN HAMMER — match 1824, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s1.py (gen3)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_regret_m890_s1.py (gen4)
# born: 2026-05-29T23:39:04Z

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union
import datetime as dt
import numpy as np

# Hyperdimensional primitives
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

# MinHash Regret Engine primitives
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Return weekday index (0=Mon … 6=Sun) for each (year, month, day)."""
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Shift so that 0 = Monday, 6 = Sunday (same as Python's weekday)
    return py_weekday

def weekday_counts(dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> np.ndarray:
    """Count occurrences of each weekday (Mon=0 … Sun=6) in *dates*."""
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)
    return doomsday_numpy(years_np, months_np, days_np)

def minhash_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """Jaccard similarity between two MinHash signatures."""
    intersection = np.logical_and(s1, s2).sum()
    union = np.logical_or(s1, s2).sum()
    return intersection / union if union != 0 else 0

# Hybrid functions
def hybrid_similarity(v1: Vector, v2: Vector, dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """Compute hybrid similarity between two hypervectors and a set of dates."""
    # Compute hyperdimensional similarity
    hdc_similarity = similarity(v1, v2)
    
    # Compute MinHash regret engine similarity
    weekday_counts_np = weekday_counts(dates)
    probabilities = weekday_counts_np / len(dates)
    gini_coefficient = 1 - np.sum(probabilities ** 2)
    minhash_sim = minhash_similarity((weekday_counts_np > 0).astype(int), (weekday_counts_np > 0).astype(int))
    regret_similarity = gini_coefficient * minhash_sim
    
    # Combine similarities using geometric mean
    return np.sqrt(hdc_similarity * regret_similarity)

def hybrid_regret(v1: Vector, v2: Vector, dates: Iterable[Union[dt.date, Tuple[int, int, int]]]) -> float:
    """Compute hybrid regret between two hypervectors and a set of dates."""
    hybrid_sim = hybrid_similarity(v1, v2, dates)
    return 1 - hybrid_sim

# Smoke test
if __name__ == "__main__":
    v1 = random_vector()
    v2 = random_vector()
    dates = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    hybrid_sim = hybrid_similarity(v1, v2, dates)
    hybrid_regret_score = hybrid_regret(v1, v2, dates)
    print(f"Hybrid similarity: {hybrid_sim:.4f}")
    print(f"Hybrid regret: {hybrid_regret_score:.4f}")