# DARWIN HAMMER — match 134, survivor 3
# gen: 4
# parent_a: ssim.py (gen0)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py (gen3)
# born: 2026-05-29T23:27:03Z

"""Hybrid SSIM‑Geometric Algebra Module.

This module fuses two parent algorithms:
* **ssim.py** – computes the Structural Similarity Index (SSIM) from two equal‑length
  grayscale sample sequences using means, variances and covariance.
* **hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py** – defines a
  `Multivector` class implementing a Clifford (geometric) algebra and uses it to
  encode decision‑hygiene scores.

**Mathematical bridge**

The statistical moments required by SSIM (mean, variance, covariance) are mapped
to grades of a multivector:


scalar (grade‑0)          → mean
vector  (grade‑1, e0)     → variance
bivector (grade‑2, e0∧e1) → covariance


Thus each signal becomes a multivector `M = μ·1 + σ²·e0`.  The covariance between
two signals is represented as a bivector `C = cov·e0∧e1`.  The geometric product
`M₁ * M₂` mixes these components; its scalar part contains a term proportional to
`μ₁·μ₂ + σ₁²·σ₂²`.  By extracting the scalar part of the product and combining it
with the classic SSIM value we obtain a **Hybrid Similarity** that respects both
statistical and geometric‑algebraic relationships.

The module provides three public functions that showcase the hybrid operation:

* `basic_ssim(x, y, ...)` – classic SSIM.
* `stats_to_multivector(seq)` – converts a 1‑D sequence into a multivector of
  moments.
* `geometric_ssim(x, y, ...)` – hybrid similarity using the geometric product of
  the moment multivectors.

All code is pure Python 3 and depends only on the standard library and NumPy."""

import math
import random
import sys
import pathlib
from typing import Sequence, Dict, Tuple, FrozenSet
import numpy as np


class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # Remove near‑zero components for cleanliness
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # -----------------------------------------------------------------
    # Geometric product (Euclidean metric, e_i·e_i = 1, e_i·e_j = 0 for i≠j)
    # -----------------------------------------------------------------
    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = Multivector._blade_product(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    @staticmethod
    def _blade_product(
        a: FrozenSet[int], b: FrozenSet[int]
    ) -> Tuple[FrozenSet[int], int]:
        """Geometric product of two basis blades.

        Returns (resulting_blade, sign), where sign is ±1.
        The implementation assumes an orthonormal Euclidean basis.
        """
        # Convert to sorted lists for deterministic ordering
        list_a = sorted(a)
        list_b = sorted(b)

        # Determine sign from swaps needed to bring concatenated list into order
        sign = 1
        for i in list_a:
            # Count how many elements in list_b are smaller than i
            smaller = sum(1 for j in list_b if j < i)
            if smaller % 2:
                sign *= -1

        # Cancel duplicate indices (e_i * e_i = 1)
        combined = list_a + list_b
        counts: Dict[int, int] = {}
        for idx in combined:
            counts[idx] = counts.get(idx, 0) + 1
        # Keep indices that appear an odd number of times
        result_set = {idx for idx, cnt in counts.items() if cnt % 2 == 1}
        return frozenset(sorted(result_set)), sign


# -----------------------------------------------------------------
# Parent A – classic SSIM
# -----------------------------------------------------------------
def basic_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for equal‑length 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )


# -----------------------------------------------------------------
# Helper – convert a 1‑D signal into a moment multivector
# -----------------------------------------------------------------
def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    """
    Encode the first‑order statistics of *seq* into a multivector:

    * scalar (grade‑0)  → mean μ
    * vector (grade‑1)  → variance σ² attached to basis e0
    """
    if not seq:
        raise ValueError("sequence must not be empty")
    n = len(seq)
    mu = sum(seq) / n
    var = sum((v - mu) ** 2 for v in seq) / n
    components: Dict[FrozenSet[int], float] = {
        frozenset(): mu,          # scalar part = mean
        frozenset({0}): var,      # e0 coefficient = variance
    }
    return Multivector(components, n=2)


# -----------------------------------------------------------------
# Hybrid similarity – combines SSIM with geometric‑algebraic interaction
# -----------------------------------------------------------------
def geometric_ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tuple[float, float, float]:
    """
    Compute a hybrid similarity score.

    Returns a tuple ``(ssim, geo_scalar, hybrid)`` where:

    * *ssim* – classic SSIM value.
    * *geo_scalar* – scalar part of the geometric product of the moment
      multivectors of *x* and *y* (includes covariance as a bivector).
    * *hybrid* – combined score defined as ``ssim * (1 + geo_scalar)``.
    """
    # Classic SSIM
    ssim_val = basic_ssim(x, y, dynamic_range, k1, k2)

    # Moment multivectors for each signal
    mx = stats_to_multivector(x)
    my = stats_to_multivector(y)

    # Covariance bivector (grade‑2, basis e0∧e1)
    n = len(x)
    mx_mean = sum(x) / n
    my_mean = sum(y) / n
    cov = sum((a - mx_mean) * (b - my_mean) for a, b in zip(x, y)) / n
    cov_mv = Multivector({frozenset({0, 1}): cov}, n=2)

    # Geometric interaction: (mx + cov) * (my + cov)
    geo_product = (mx + cov_mv) * (my + cov_mv)

    geo_scalar = geo_product.scalar_part()
    hybrid_score = ssim_val * (1.0 + geo_scalar)

    return ssim_val, geo_scalar, hybrid_score


# -----------------------------------------------------------------
# Additional demonstration: compare two multivectors directly
# -----------------------------------------------------------------
def multivector_similarity(a: Multivector, b: Multivector) -> float:
    """
    A pure‑geometric similarity based on the normalized scalar part of the
    geometric product.
    """
    prod = a * b
    scalar = prod.scalar_part()
    norm_a = math.sqrt(abs(a * a).scalar_part())
    norm_b = math.sqrt(abs(b * b).scalar_part())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return scalar / (norm_a * norm_b)


# -----------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------
if __name__ == "__main__":
    # Generate two random grayscale signals
    random.seed(0)
    length = 128
    sig1 = [random.uniform(0, 255) for _ in range(length)]
    sig2 = [v + random.gauss(0, 10) for v in sig1]  # add mild noise

    ssim_val = basic_ssim(sig1, sig2)
    print(f"Classic SSIM: {ssim_val:.6f}")

    mv1 = stats_to_multivector(sig1)
    mv2 = stats_to_multivector(sig2)
    print(f"Multivector 1: {mv1}")
    print(f"Multivector 2: {mv2}")

    ssim_geo, geo_scalar, hybrid = geometric_ssim(sig1, sig2)
    print(f"Geometric scalar part: {geo_scalar:.6f}")
    print(f"Hybrid SSIM: {hybrid:.6f}")

    direct_sim = multivector_similarity(mv1, mv2)
    print(f"Direct multivector similarity: {direct_sim:.6f}")