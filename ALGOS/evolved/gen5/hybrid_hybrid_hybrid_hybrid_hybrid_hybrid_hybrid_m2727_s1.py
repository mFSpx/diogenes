# DARWIN HAMMER — match 2727, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_sheaf_cohomol_m179_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_fisher_m31_s4.py (gen4)
# born: 2026-05-29T23:43:45Z

"""Hybrid Allocation‑Geometric‑Sheaf Module.

This module unites two distinct linearly‑structured algorithms:

* **Parent A** – computes a weekday‑dependent weight vector and allocates a total
  resource across a set of groups.  The allocation vector can be regarded as a
  grade‑1 element (a vector) of a Clifford algebra.

* **Parent B** – provides a minimal Clifford‑algebra implementation (`Multivector`)
  together with Gaussian‑beam and Fisher‑information utilities.

**Mathematical bridge** – The allocation vector `a∈ℝ^n` is embedded as a
grade‑1 multivector `A = Σ_i a_i e_i`.  A scalar “geometric weight’’ `w(θ)` is
obtained from the product of a Gaussian beam and a Fisher‑information score.
The geometric product `W·A` (with `W = w(θ)·1`) scales the allocation while
preserving its algebraic nature.  Consistency of the allocation over a graph of
groups is examined via the sheaf coboundary: differences on edges form a
grade‑2 bivector `B = Σ_{(i,j)} (a_j‑a_i) e_i∧e_j`.  The norm of `B` together
with the magnitude of `W·A` yields a hybrid coherence metric.

The functions below demonstrate this fusion.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (Parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a normalised weight vector that depends on the day of week.
    The rule is arbitrary but deterministic: each group receives a base weight
    of 1, plus an extra bump of 0.1·sin( (dow+idx)·π/4 ).
    """
    base = np.ones(len(groups), dtype=float)
    bumps = np.array(
        [0.1 * math.sin((dow + i) * math.pi / 4) for i in range(len(groups))],
        dtype=float,
    )
    raw = base + bumps
    # Normalise to sum‑to‑one
    return raw / raw.sum()


def allocate_resources(total: float, date: dt.date) -> Dict[str, float]:
    """
    Allocate `total` across GROUPS using the weekday‑dependent weight vector.
    Returns a dict {group: allocation}.
    """
    dow = doomsday(date.year, date.month, date.day)
    w = weekday_weight_vector(GROUPS, dow)
    allocations = {g: _pct(total * wi) for g, wi in zip(GROUPS, w)}
    return allocations

# ----------------------------------------------------------------------
# Clifford algebra utilities (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset[int], blade_b: frozenset[int]
) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {
            k: v for k, v in components.items() if abs(v) > 1e-15
        }  # prune near‑zero
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other):
        """Geometric product with a scalar or another multivector."""
        if isinstance(other, (int, float)):
            return Multivector(
                {blade: coef * other for blade, coef in self.components.items()}, self.n
            )
        if isinstance(other, Multivector):
            result: Dict[frozenset[int], float] = {}
            for ba, ca in self.components.items():
                for bb, cb in other.components.items():
                    rc_blade, sign = _multiply_blades(ba, bb)
                    result[rc_blade] = result.get(rc_blade, 0.0) + sign * ca * cb
            return Multivector(result, self.n)
        raise TypeError("Unsupported operand type for *: '{}'".format(type(other)))

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Norm utilities
    # ------------------------------------------------------------------
    def squared_norm(self) -> float:
        """Return ⟨M·M̃⟩_0, the squared Euclidean norm of the coefficient vector."""
        return sum(coef * coef for coef in self.components.values())

    def norm(self) -> float:
        return math.sqrt(self.squared_norm())

    def __repr__(self) -> str:
        terms = [f"{coef:.3g}*e{sorted(list(b))}" if b else f"{coef:.3g}" for b, coef in self.components.items()]
        return " + ".join(terms) if terms else "0"


def vector_to_multivector(vec: Sequence[float]) -> Multivector:
    """
    Embed a real vector as a grade‑1 multivector:
        v = Σ_i vec[i] e_i
    """
    comps = {frozenset({i}): float(v) for i, v in enumerate(vec)}
    return Multivector(comps, n=len(vec))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a normal distribution N(center, width^2) evaluated at theta.
    I(θ) = ((θ‑center)^2 / width^4) * exp(- (θ‑center)^2 / (2·width^2))
    The eps parameter avoids division by zero.
    """
    if width <= eps:
        raise ValueError("width must be positive")
    delta = theta - center
    return (delta * delta) / (width ** 4 + eps) * gaussian_beam(theta, center, width)


def geometric_weight(theta: float, center: float, width: float) -> float:
    """
    Combine Gaussian beam intensity with Fisher information into a single scalar weight.
    """
    return gaussian_beam(theta, center, width) * fisher_score(theta, center, width)

# ----------------------------------------------------------------------
# Sheaf‑like coboundary expressed in Clifford algebra (bridge)
# ----------------------------------------------------------------------
def coboundary_bivector(allocation: Sequence[float], edges: List[Tuple[int, int]]) -> Multivector:
    """
    Given a list of scalar allocations on the vertices of a graph,
    produce a grade‑2 bivector representing the coboundary:
        B = Σ_{(i,j)∈E} (a_j - a_i) e_i ∧ e_j
    """
    biv_components: Dict[frozenset[int], float] = {}
    for i, j in edges:
        diff = allocation[j] - allocation[i]
        blade = frozenset({i, j})
        biv_components[blade] = biv_components.get(blade, 0.0) + diff
    return Multivector(biv_components, n=len(allocation))

def hybrid_coherence_metric(allocations: Dict[str, float], theta: float, center: float, width: float) -> Tuple[float, float]:
    """
    Compute two complementary coherence measures:

    1. `geom_norm` – the norm of the scaled allocation multivector `W·A`,
       where `W = geometric_weight(theta,…)` and `A` encodes the allocations.

    2. `sheaf_norm` – the norm of the coboundary bivector on the canonical
       line graph of GROUPS.

    Returns `(geom_norm, sheaf_norm)`.
    """
    # 1️⃣ Embed allocations as a grade‑1 multivector
    vec = np.array([allocations[g] for g in GROUPS], dtype=float)
    A = vector_to_multivector(vec)

    # 2️⃣ Scalar geometric weight
    W = geometric_weight(theta, center, width)

    # 3️⃣ Scaled multivector and its norm
    scaled = W * A
    geom_norm = scaled.norm()

    # 4️⃣ Coboundary on a simple line graph (i,i+1)
    edges = [(i, i + 1) for i in range(len(GROUPS) - 1)]
    B = coboundary_bivector(vec, edges)
    sheaf_norm = B.norm()

    return geom_norm, sheaf_norm

# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def demo_allocation(date: dt.date, total: float) -> Dict[str, float]:
    """Allocate `total` on `date` and print the weight vector."""
    allocations = allocate_resources(total, date)
    dow = doomsday(date.year, date.month, date.day)
    w = weekday_weight_vector(GROUPS, dow)
    print(f"Weekday {dow} weight vector: {w}")
    print(f"Allocations: {allocations}")
    return allocations


def demo_geometric_product(allocations: Dict[str, float], theta: float, center: float, width: float) -> Multivector:
    """Create the allocation multivector, scale it geometrically and return the result."""
    vec = np.array([allocations[g] for g in GROUPS], dtype=float)
    A = vector_to_multivector(vec)
    W = geometric_weight(theta, center, width)
    result = W * A
    print(f"Geometric weight W={W:.6g}")
    print(f"W·A = {result}")
    return result


def demo_hybrid_metric(date: dt.date, total: float, theta: float, center: float, width: float) -> None:
    """
    Full pipeline: allocate resources, embed into Clifford algebra,
    compute geometric norm and sheaf coboundary norm, and display them.
    """
    allocations = allocate_resources(total, date)
    geom_norm, sheaf_norm = hybrid_coherence_metric(allocations, theta, center, width)
    print(f"Hybrid coherence metric → geom_norm={geom_norm:.6g}, sheaf_norm={sheaf_norm:.6g}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    today = dt.date.today()
    total_funds = 1_000.0

    print("\n--- Demo Allocation ---")
    allocs = demo_allocation(today, total_funds)

    print("\n--- Demo Geometric Product ---")
    demo_geometric_product(allocs, theta=0.75, center=0.5, width=0.2)

    print("\n--- Demo Hybrid Metric ---")
    demo_hybrid_metric(today, total_funds, theta=0.75, center=0.5, width=0.2)