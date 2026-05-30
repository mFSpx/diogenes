# DARWIN HAMMER — match 5155, survivor 1
# gen: 7
# parent_a: hybrid_sparse_wta_hybrid_hybrid_hybrid_m1937_s3.py (gen6)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s3.py (gen2)
# born: 2026-05-30T00:00:08Z

"""Hybrid Sparse-WTA & Hyperdimensional Gini-Doomsday Fusion
Parent A: sparse_wta + multivector utilities.
Parent B: hyperdimensional encoding + Gini coefficient + Doomsday weekday.
Mathematical Bridge:
- The sparse winner‑take‑all (WTA) mask generated from the LTC activation is
  weighted by the Gini coefficient of the same activation distribution.
- The weighted mask is expanded into a high‑dimensional (+/-1) hypervector
  via the hash‑based `expand` routine (Parent A).
- The Doomsday algorithm (Parent B) yields a weekday for a supplied date;
  this weekday is turned into a symbolic hypervector with `symbol_vector`.
- Finally the expanded hypervector is *bound* to the symbolic hypervector,
  producing a fused representation that carries both the sparsity/Gini
  information and the temporal (weekday) context.
The module provides three public functions that showcase this fusion.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
import datetime as dt
from typing import List, Iterable, Tuple

# ----------------------------------------------------------------------
# Parent A utilities (sparse WTA, expansion, sigmoid)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Linear‑threshold cell with sigmoid non‑linearity.
    Concatenates external input I to internal state x,
    applies affine transform, and passes through sigmoid.
    """
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 for the k largest values (ties broken by index)."""
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda p: (-p[1], p[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]

def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """
    Hash‑based sparse expansion (Count‑Min sketch style).
    Each input value contributes to three random positions in an m‑dimensional
    output vector, with a random sign.
    """
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out

# ----------------------------------------------------------------------
# Parent B utilities (hyperdimensional primitives, Gini, Doomsday)
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    """Arithmetic mean of a collection of hypervectors (produces a real‑valued vector)."""
    vecs = list(vectors)
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini coefficient of a non‑negative sequence."""
    xs = sorted([float(v) for v in values])
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += i * x
    total = sum(xs)
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))

def doomsday_weekday(year: int, month: int, day: int) -> int:
    """
    Compute weekday (0=Monday … 6=Sunday) using the Doomsday algorithm.
    Works for Gregorian calendar years >= 1900.
    """
    # Anchor days for centuries
    century = year // 100
    anchor = (5 * (century % 4) + 2) % 7  # 0 = Tuesday in Doomsday notation
    # Year’s doomsday
    y = year % 100
    doomsday = (y // 12 + (y % 12) + (y % 12) // 4 + anchor) % 7
    # Month offsets (for non‑leap years)
    month_offsets = [0, 3, 0, 3, 2, 5, 0, 3, 6, 1, 4, 6]  # Jan..Dec
    if month < 3:
        year_adj = year - 1
    else:
        year_adj = year
    # Convert Doomsday (Tuesday=0) to ISO weekday (Monday=0)
    iso_anchor = (doomsday + 5) % 7
    offset = month_offsets[month - 1]
    weekday = (iso_anchor + offset + day - 1) % 7
    return weekday

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_sparse_wta(input_vec: np.ndarray,
                      external_input: np.ndarray,
                      weight_mat: np.ndarray,
                      bias_vec: np.ndarray,
                      k: int,
                      dim_hd: int = 10000,
                      salt: str = '') -> Vector:
    """
    1. Compute LTC activation.
    2. Apply sparse WTA (top‑k) to obtain a binary mask.
    3. Expand the masked activation to a high‑dimensional (+/-1) hypervector.
    """
    activation = ltc_f(input_vec, external_input, weight_mat, bias_vec)
    mask = top_k_mask(activation.tolist(), k)
    masked_vals = [a * m for a, m in zip(activation.tolist(), mask)]
    expanded = expand(masked_vals, dim_hd, salt=salt)
    # Convert to +/-1 hypervector by sign
    hd_vec = [1 if v >= 0 else -1 for v in expanded]
    return hd_vec

def hybrid_gini_weighted_binding(input_vec: np.ndarray,
                                 external_input: np.ndarray,
                                 weight_mat: np.ndarray,
                                 bias_vec: np.ndarray,
                                 k: int,
                                 date_obj: dt.date,
                                 dim_hd: int = 10000,
                                 salt: str = '') -> Vector:
    """
    Full fusion:
    - Produce a sparse‑WTA hypervector (as above).
    - Compute Gini coefficient of the raw activation and use it to scale the hypervector.
    - Generate a symbolic hypervector from the weekday of `date_obj`.
    - Bind the scaled hypervector with the symbolic hypervector.
    Returns the bound hypervector.
    """
    # Step 1: raw activation and mask
    activation = ltc_f(input_vec, external_input, weight_mat, bias_vec)
    mask = top_k_mask(activation.tolist(), k)
    masked_vals = [a * m for a, m in zip(activation.tolist(), mask)]

    # Step 2: Gini weighting
    gini = gini_coefficient(activation.tolist())
    weighted = [v * gini for v in masked_vals]

    # Step 3: high‑dimensional expansion
    expanded = expand(weighted, dim_hd, salt=salt)
    hd_vec = [1 if v >= 0 else -1 for v in expanded]

    # Step 4: symbolic weekday vector
    weekday = doomsday_weekday(date_obj.year, date_obj.month, date_obj.day)
    symbol = f'weekday-{weekday}'
    sym_vec = symbol_vector(symbol, dim=dim_hd)

    # Step 5: bind
    bound = bind(hd_vec, sym_vec)
    return bound

def hybrid_bundle_multiple(date_vec_pairs: List[Tuple[dt.date, Vector]]) -> Vector:
    """
    Given a list of (date, hypervector) pairs, generate a bundled hypervector.
    Each hypervector is first bound to its weekday symbol, then all are bundled.
    """
    bound_vectors = []
    for date_obj, vec in date_vec_pairs:
        weekday = doomsday_weekday(date_obj.year, date_obj.month, date_obj.day)
        symbol = f'weekday-{weekday}'
        sym_vec = symbol_vector(symbol, dim=len(vec))
        bound = bind(vec, sym_vec)
        bound_vectors.append(bound)
    return bundle(bound_vectors)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dimensions
    dim_input = 20
    dim_external = 5
    dim_hd = 8192
    k = 5

    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    # Random inputs
    x = rng.normal(size=dim_input)
    I = rng.normal(size=dim_external)
    W = rng.normal(size=(dim_input + dim_external, dim_input + dim_external))
    b = rng.normal(size=dim_input + dim_external)

    # Date for Doomsday component
    today = dt.date.today()

    # Hybrid sparse‑WTA hypervector
    hv1 = hybrid_sparse_wta(x, I, W, b, k, dim_hd=dim_hd, salt='test')
    print(f'Hybrid sparse‑WTA vector length: {len(hv1)} (first 5): {hv1[:5]}')

    # Full Gini‑weighted binding
    hv2 = hybrid_gini_weighted_binding(x, I, W, b, k, today, dim_hd=dim_hd, salt='test')
    print(f'Gini‑weighted bound vector length: {len(hv2)} (first 5): {hv2[:5]}')

    # Bundle a few examples
    pairs = [(today, hv1),
             (today - dt.timedelta(days=1), hv2),
             (today - dt.timedelta(days=2), hv1)]
    bundled = hybrid_bundle_multiple(pairs)
    print(f'Bundled vector length: {len(bundled)} (first 5): {bundled[:5]}')