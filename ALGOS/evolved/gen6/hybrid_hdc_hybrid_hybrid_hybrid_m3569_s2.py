# DARWIN HAMMER — match 3569, survivor 2
# gen: 6
# parent_a: hdc.py (gen0)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_vorono_m1379_s4.py (gen5)
# born: 2026-05-29T23:50:51Z

"""Hybrid Hyperdimensional‑Geometric‑Algebra Doomsday Engine

This module fuses two previously independent algorithms:

* **Parent A** – hyper‑dimensional (HD) computing primitives (random bipolar
  vectors, binding, bundling, similarity) together with a deterministic scalar
  time‑series produced by the Sakamoto weekday algorithm.
* **Parent B** – a minimal Euclidean Clifford geometric‑algebra core that
  supplies a geometric product and multivector addition.

**Mathematical bridge**

A scalar value `xₜ` (the weekday index) is lifted to a *scalar blade* (grade 0)
of a multivector.  Each weekday is also encoded as a random bipolar HD vector,
which plays the role of a *vector blade* (grade 1).  The geometric product
between a multivector and a scalar‑blade reduces to scaling of the vector part,
while the product between two vector blades reduces (for bipolar vectors) to
their dot‑product plus a scaled vector part.  Consequently the classic linear
state‑space update

    hₜ = A·hₜ₋₁ + B·xₜ
    yₜ = C·hₜ

can be rewritten in the GA domain as

    hₜ   = 𝔄 ⊗ hₜ₋₁  ⊕  𝔅 ⊗ xₜ
    yₜ   = 𝔠 ⊗ hₜ

where `⊗` is the geometric product defined below and `⊕` is multivector
addition.  After processing the whole request sequence we obtain the scalar
output series `Y = (y₁,…,y_T)` and evaluate its Gini coefficient, thus
preserving the statistical analysis of Parent A while operating on HD‑encoded
vectors inside a geometric‑algebraic framework.

The three public functions below illustrate the end‑to‑end hybrid workflow:

* `weekday_series` – vectorised Sakamoto weekday computation.
* `geometric_product` – GA product for multivectors limited to scalar and
  vector grades (the latter are bipolar HD vectors).
* `hybrid_doomsday` – builds the state‑space model, runs the update loop,
  and returns the output series together with its Gini coefficient.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyper‑dimensional (HD) utilities – derived from Parent A (hd.py)
# ----------------------------------------------------------------------
Vector = np.ndarray  # bipolar vector of shape (dim,) with values {1, -1}
Dim = int


def random_vector(dim: Dim = 10_000, seed: str | int | None = None) -> Vector:
    """Return a bipolar HD vector of length ``dim`` using ``seed`` for reproducibility."""
    rng = random.Random(seed)
    data = [1 if rng.getrandbits(1) else -1 for _ in range(dim)]
    return np.array(data, dtype=np.int8)


def symbol_vector(symbol: str, dim: Dim = 10_000) -> Vector:
    """Deterministically map ``symbol`` to a bipolar vector via SHA‑256 hashing."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication – the HD analogue of the geometric product for vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal length")
    return a * b  # bipolar multiplication stays bipolar


def bundle(vectors: Iterable[Vector]) -> Vector:
    """Majority‑vote bundling (component‑wise sign of the sum)."""
    vecs = list(vectors)
    if not vecs:
        raise ValueError("at least one vector is required")
    sums = np.sum(np.stack(vecs, axis=0), axis=0)
    return np.where(sums >= 0, 1, -1).astype(np.int8)


def similarity(a: Vector, b: Vector) -> float:
    """Normalised inner product (range [-1, 1])."""
    if a.shape != b.shape:
        raise ValueError("vectors must have equal length")
    return float(np.dot(a, b) / a.size)


# ----------------------------------------------------------------------
# Sakamoto weekday series – derived from Parent A (hybrid_hybrid_…_doom…)
# ----------------------------------------------------------------------
def weekday_series(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Sakamoto algorithm.
    Returns an int8 array where 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


# ----------------------------------------------------------------------
# Minimal Clifford geometric‑algebra core – derived from Parent B
# ----------------------------------------------------------------------
Multivector = Tuple[float, Vector]  # (scalar_part, vector_part)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Geometric product for multivectors limited to grades 0 (scalar) and 1 (vector).
    For bipolar vectors the dot product is the usual inner product.
    Returns (scalar, vector) where:
        scalar = a0*b0 + <a1, b1>
        vector = a0*b1 + b0*a1
    """
    a0, a1 = a
    b0, b1 = b

    # Ensure vector parts are NumPy arrays of the same shape
    if a1.shape != b1.shape:
        raise ValueError("vector parts must have equal dimensions")

    scalar = a0 * b0 + float(np.dot(a1, b1))
    vector = a0 * b1 + b0 * a1
    return (scalar, vector)


def mv_add(a: Multivector, b: Multivector) -> Multivector:
    """Component‑wise addition of two multivectors."""
    a0, a1 = a
    b0, b1 = b
    if a1.shape != b1.shape:
        raise ValueError("vector parts must have equal dimensions")
    return (a0 + b0, a1 + b1)


# ----------------------------------------------------------------------
# Gini coefficient – statistical bridge from Parent A
# ----------------------------------------------------------------------
def gini(arr: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D numeric array.
    Value is 0 for perfect equality and 1 for maximal inequality.
    """
    if arr.ndim != 1:
        raise ValueError("Gini is defined for 1‑D arrays")
    if np.any(arr < 0):
        raise ValueError("Gini requires non‑negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr, dtype=np.float64)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini_index = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return float(gini_index)


# ----------------------------------------------------------------------
# Hybrid Doomsday‑HD engine – the fused algorithm
# ----------------------------------------------------------------------
def hybrid_doomsday(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    dim: Dim = 2048,
) -> Tuple[np.ndarray, float]:
    """
    End‑to‑end hybrid workflow:

    1. Compute the weekday scalar series ``xₜ``.
    2. Encode each weekday as a bipolar HD vector (grade‑1 blade).
    3. Initialise random multivectors 𝔄, 𝔅, 𝔠 (each with scalar and vector parts).
    4. Run the GA‑based state update ``hₜ = 𝔄⊗hₜ₋₁ ⊕ 𝔅⊗xₜ``.
    5. Produce scalar outputs ``yₜ = (𝔠⊗hₜ).scalar``.
    6. Return the output series and its Gini coefficient.
    """
    # 1. Weekday scalar series (0‑6)
    w_series = weekday_series(years, months, days)  # shape (T,)

    # 2. HD encoding of each weekday (grade‑1 blades)
    unique_symbols = [f"weekday_{i}" for i in range(7)]
    hd_lookup = {i: symbol_vector(unique_symbols[i], dim) for i in range(7)}
    x_vectors = np.stack([hd_lookup[int(w)] for w in w_series], axis=0)  # (T, dim)

    # 3. Random multivectors 𝔄, 𝔅, 𝔠
    rng = random.Random(42)  # deterministic seed for reproducibility
    def rand_mv() -> Multivector:
        scalar = rng.uniform(-1.0, 1.0)
        vector = random_vector(dim, rng.randint(0, 2**63 - 1)).astype(np.float64)
        return (scalar, vector)

    A_mv = rand_mv()
    B_mv = rand_mv()
    C_mv = rand_mv()

    # 4. Initialise state h₀ (scalar 0, random vector)
    h_state: Multivector = (0.0, random_vector(dim, rng.randint(0, 2**63 - 1)).astype(np.float64))

    # 5. Iterate over time steps
    y_outputs: List[float] = []
    zero_vec = np.zeros(dim, dtype=np.float64)

    for idx in range(w_series.size):
        # scalar blade for xₜ (weekday index as a float)
        x_scalar = float(int(w_series[idx]))
        x_mv: Multivector = (x_scalar, zero_vec)

        # term from A
        term_a = geometric_product(A_mv, h_state)

        # term from B (B ⊗ xₜ)
        term_b = geometric_product(B_mv, x_mv)

        # state update
        h_state = mv_add(term_a, term_b)

        # output yₜ