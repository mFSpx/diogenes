# DARWIN HAMMER — match 4561, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_vorono_m1379_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s3.py (gen3)
# born: 2026-05-29T23:56:32Z

"""Hybrid Doomsday‑Voronoi Engine

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_vorono_m1379_s2.py``  
  Provides a vectorised Sakamoto weekday calculator and a Gini‑coefficient
  evaluator for a one‑dimensional score series.

* **Parent B** – ``hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s3.py``  
  Supplies a Clifford‑algebra multivector implementation together with a
  geometric product routine.

Mathematical Bridge
-------------------
The health/state vector ``h ∈ ℝⁿ`` produced by the SSM (or any numeric
process) is re‑interpreted as a multivector ``M`` by assigning the *i*‑th
component of ``h`` to the basis blade ``e_i``.  The geometric product
``Mₜ = Mₜ₋₁ ⊛ MV(hₜ)`` propagates the state in the Clifford algebra.
Scalar statistics extracted from the evolving multivector (scalar part,
norm, blade‑wise magnitudes) are then fed back to the statistical
machinery of Parent A: the Gini coefficient of the weekday series and the
Shannon entropy of a set of decision‑hygiene scores weighted by the scalar
part of the multivector.

The three core functions below demonstrate this fusion:
``weekday_sakamoto`` → ``gini_coefficient`` (Parent A) and
``Multivector`` / ``multivector_from_vector`` / ``hybrid_step`` (Parent B)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – vectorised Sakamoto weekday and Gini coefficient
# ----------------------------------------------------------------------


def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Sakamoto algorithm.
    Returns 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(y: np.ndarray) -> float:
    """
    Gini coefficient for a 1‑D numeric array.
    """
    if y.size == 0:
        return 0.0

    arr = np.sort(y.astype(np.float64).flatten())
    n = arr.size
    cumulative = np.cumsum(arr)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0

    # Gini = (2*Σ i*y_i) / (n*Σ y_i) - (n+1)/n
    i = np.arange(1, n + 1)
    gini = (2.0 * np.sum(i * arr)) / (n * sum_y) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Parent B – Clifford algebra multivector with geometric product
# ----------------------------------------------------------------------


def _blade_sign(indices: Iterable[int]) -> Tuple[List[int], int]:
    """
    Sort a list of basis indices while tracking the sign change due to swaps.
    Duplicate indices cancel (e_i * e_i = 1) and are removed.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
                sign *= -1
            elif lst[i] == lst[j]:
                # e_i * e_i = 1 → remove the pair
                del lst[j]
                del lst[i]
                n -= 2
                i -= 1  # step back to re‑evaluate at new position
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """
    Geometric product of two basis blades.
    Returns (result_blade, sign).
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
    """
    Simple multivector for the Euclidean Clifford algebra Cl(n,0).

    Internally stored as a mapping ``blade → coefficient`` where ``blade`` is a
    ``frozenset`` of integer basis indices.  The empty frozenset represents the
    scalar part.
    """

    def __init__(self, blades: Dict[frozenset[int], float] | None = None):
        self.blades: Dict[frozenset[int], float] = dict()
        if blades:
            for b, c in blades.items():
                if abs(c) > 1e-15:
                    self.blades[b] = float(c)

    @staticmethod
    def basis_vector(idx: int, coeff: float = 1.0) -> "Multivector":
        """Create a single‑blade multivector e_idx."""
        return Multivector({frozenset({idx}): coeff})

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.blades.copy())
        for b, c in other.blades.items():
            result.blades[b] = result.blades.get(b, 0.0) + c
            if abs(result.blades[b]) < 1e-15:
                del result.blades[b]
        return result

    __radd__ = __add__

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.blades.items()})

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __rmul__(self, scalar: float) -> "Multivector":
        return Multivector({b: scalar * c for b, c in self.blades.items()})

    __mul__ = None  # placeholder; defined after class body for type hints

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """
        Compute the geometric product (Clifford product) of two multivectors.
        """
        result: Dict[frozenset[int], float] = {}
        for ba, ca in self.blades.items():
            for bb, cb in other.blades.items():
                blade_res, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[blade_res] = result.get(blade_res, 0.0) + coeff
        # prune near‑zero coefficients
        result = {b: c for b, c in result.items() if abs(c) > 1e-15}
        return Multivector(result)

    def __repr__(self) -> str:
        if not self.blades:
            return "0"
        terms = []
        for blade, coeff in sorted(self.blades.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "∧".join(f"e{i}" for i in sorted(blade))
                terms.append(f"{coeff:.3g}{basis}")
            else:
                terms.append(f"{coeff:.3g}")
        return " + ".join(terms)


# Bind the operator to use geometric_product
Multivector.__mul__ = Multivector.geometric_product  # type: ignore


def multivector_from_vector(v: np.ndarray) -> Multivector:
    """
    Convert a real‑valued 1‑D array into a multivector.
    Component ``v[i]`` becomes the coefficient of basis blade ``e_i``.
    """
    blades: Dict[frozenset[int], float] = {}
    for idx, coeff in enumerate(v):
        if abs(coeff) > 1e-15:
            blades[frozenset({idx})] = float(coeff)
    return Multivector(blades)


def hybrid_step(
    prev_mv: Multivector,
    input_vec: np.ndarray,
) -> Tuple[Multivector, float, float]:
    """
    Perform one hybrid evolution step.

    1. Convert ``input_vec`` to a multivector ``mv_input``.
    2. Geometric‑product evolve: ``mv_next = prev_mv * mv_input``.
    3. Extract the scalar part (empty blade) as ``weight``.
    4. Compute Shannon entropy of a synthetic decision‑hygiene score
       distribution weighted by ``weight``.

    Returns ``(mv_next, weight, entropy)``.
    """
    mv_input = multivector_from_vector(input_vec)
    mv_next = prev_mv * mv_input

    # scalar part (empty blade) – default to 1.0 if missing
    weight = mv_next.blades.get(frozenset(), 1.0)

    # synthetic decision hygiene scores: normalized absolute values of all blades
    coeffs = np.array([abs(c) for c in mv_next.blades.values()], dtype=np.float64)
    if coeffs.sum() == 0:
        probs = np.ones_like(coeffs) / coeffs.size
    else:
        probs = coeffs / coeffs.sum()

    entropy = -float(np.sum(probs * np.log2(probs + 1e-15)))
    return mv_next, float(weight), float(entropy)


# ----------------------------------------------------------------------
# Demonstration functions that combine both parents
# ----------------------------------------------------------------------


def compute_weekday_gini(dates: List[Tuple[int, int, int]]) -> float:
    """
    Given a list of (year, month, day) tuples, compute the Gini coefficient
    of the resulting weekday distribution.
    """
    arr = np.array(dates, dtype=np.int64)
    w = weekday_sakamoto(arr[:, 0], arr[:, 1], arr[:, 2])
    # count occurrences of each weekday (0‑6)
    counts = np.bincount(w, minlength=7)
    return gini_coefficient(counts)


def simulate_hybrid_process(
    init_state: np.ndarray,
    input_series: Iterable[np.ndarray],
    init_mv: Multivector | None = None,
) -> List[Tuple[Multivector, float, float]]:
    """
    Run a sequence of hybrid steps.
    ``init_state`` is a numeric vector that seeds the first multivector
    (if ``init_mv`` is None).  ``input_series`` yields successive input vectors.
    Returns a list of the step results ``(mv, weight, entropy)``.
    """
    if init_mv is None:
        current_mv = multivector_from_vector(init_state)
    else:
        current_mv = init_mv

    history: List[Tuple[Multivector, float, float]] = []
    for vec in input_series:
        current_mv, weight, entropy = hybrid_step(current_mv, vec)
        history.append((current_mv, weight, entropy))
    return history


def hybrid_metrics_for_dates(
    dates: List[Tuple[int, int, int]],
    health_vectors: List[np.ndarray],
) -> Dict[str, float]:
    """
    High‑level wrapper that:
    * computes the weekday Gini coefficient,
    * evolves a multivector over the supplied ``health_vectors``,
    * returns the final scalar weight and the average Shannon entropy.
    """
    gini = compute_weekday_gini(dates)

    # initialise with the first health vector
    init_vec = health_vectors[0]
    steps = simulate_hybrid_process(init_vec, health_vectors[1:])

    final_weight = steps[-1][1] if steps else 1.0
    avg_entropy = float(np.mean([s[2] for s in steps])) if steps else 0.0

    return {
        "weekday_gini": gini,
        "final_scalar_weight": final_weight,
        "average_entropy": avg_entropy,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate 100 random dates between 2000‑01‑01 and 2025‑12‑31
    rng = np.random.default_rng(42)
    years = rng.integers(2000, 2026, size=100)
    months = rng.integers(1, 13, size=100)
    # ensure valid day numbers per month (simple approximation)
    days = rng.integers(1, 28, size=100)

    dates = list(zip(years.tolist(), months.tolist(), days.tolist()))

    # random health vectors of dimension 5
    health_vectors = [rng.normal(loc=0.0, scale=1.0, size=5) for _ in range(20)]

    metrics = hybrid_metrics_for_dates(dates, health_vectors)

    print("Hybrid metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    # quick sanity check: the final multivector should be printable
    init_mv = multivector_from_vector(health_vectors[0])
    final_mv, weight, entropy = hybrid_step(init_mv, health_vectors[1])
    print("\nSample multivector after one step:", final_mv)
    print("Scalar weight:", weight)
    print("Shannon entropy:", entropy)