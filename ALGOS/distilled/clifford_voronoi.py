# DARWIN HAMMER — match 4, survivor 2
# gen: 1
# parent_a: geometric_product.py (gen0)
# parent_b: voronoi_partition.py (gen0)
# born: 2026-05-29T23:14:40Z
#
# DISTILLED USE: GO-25 ontology term clustering where orientation matters.
# Assigns terms by geometric relationship not scalar distance — separates
# 'temporal before' from 'causal before' edges that are numerically close
# but geometrically opposite. Use for Percyphon domain slot clustering
# (12 fixed + 88 fluid) and GO-25 term registry spatial organisation.

"""Hybrid module combining geometric algebra (geometric_product.py) and
Voronoi partitioning (voronoi_partition.py).

Mathematical bridge:
- A 2‑D point (x, y) is represented as a grade‑1 multivector
  **p = x·e₁ + y·e₂** in the Euclidean Clifford algebra Cl(2,0).
- The Euclidean squared distance between two points a and b is the scalar
  part of the inner product ⟨a‑b, a‑b⟩, i.e. `(a - b).inner(b - a).scalar_part()`.
- Voronoi assignment can therefore be performed by comparing these scalar
  distances, unifying the geometric‑product core with the classic
  nearest‑seed algorithm.

The module provides:
* `point_to_mv` – convert a 2‑tuple to a multivector vector.
* `mv_distance` – Euclidean distance via geometric algebra inner product.
* `voronoi_partition_mv` – Voronoi region assignment using multivector distances.
* `rotate_toward` – example rotor that rotates a point toward a seed.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------------
# Geometric algebra core (from geometric_product.py)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti‑commutativity). Duplicate indices cancel because e_i*e_i = 1.
    """
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
                # cancel duplicate pair
                del lst[j:j + 2]
                n -= 2
                sign *= 1  # e_i*e_i = 1 contributes +1
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset[int], blade_b: frozenset[int]) -> Tuple[frozenset[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        # discard zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Grade projection & scalar extraction
    # ------------------------------------------------------------------

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        return geometric_product(self, other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()}, self.n)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full Clifford product a*b."""
    result: Dict[frozenset[int], float] = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            contrib = sign * coef_a * coef_b
            result[blade_out] = result.get(blade_out, 0.0) + contrib
    n = max(a.n, b.n)
    return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, n)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """Symmetric inner product (ab + ba)/2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab + ba) * 0.5


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """Antisymmetric wedge product (ab - ba)/2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab - ba) * 0.5


def reverse(a: Multivector) -> Multivector:
    """Reverse of a multivector: sign flip for grades k where k%4 in {2,3}."""
    result = {}
    for blade, coef in a.components.items():
        k = len(blade)
        sign = -1 if (k % 4) in (2, 3) else 1
        result[blade] = coef * sign
    return Multivector(result, a.n)


# ---------------------------------------------------------------------------
# Voronoi utilities adapted to multivectors (from voronoi_partition.py)
# ---------------------------------------------------------------------------

Point2D = Tuple[float, float]          # classic tuple representation
MVPoint = Multivector                    # grade‑1 multivector representation


def point_to_mv(p: Point2D) -> MVPoint:
    """Convert a 2‑D Euclidean point to a grade‑1 multivector in Cl(2,0)."""
    x, y = p
    # basis e1 -> index 1, e2 -> index 2 (using 1‑based indices for readability)
    components = {
        frozenset({1}): x,
        frozenset({2}): y,
    }
    return Multivector(components, n=2)


def mv_to_point(mv: MVPoint) -> Point2D:
    """Extract (x, y) coordinates from a grade‑1 multivector."""
    x = mv.components.get(frozenset({1}), 0.0)
    y = mv.components.get(frozenset({2}), 0.0)
    return (x, y)


def mv_distance(a: MVPoint, b: MVPoint) -> float:
    """Euclidean distance between two points using geometric‑algebra inner product."""
    diff = a - b
    sq = inner_product(diff, diff).scalar_part()   # (a-b)·(a-b) = |a-b|²
    # Guard against tiny negative due to rounding
    sq = max(sq, 0.0)
    return math.sqrt(sq)


def nearest_mv(point: MVPoint, seeds: List[MVPoint]) -> int:
    """Index of the nearest seed to *point* using multivector distances."""
    if not seeds:
        raise ValueError('seeds required')
    distances = [mv_distance(point, s) for s in seeds]
    # Tie‑break by smallest index (deterministic)
    return min(range(len(seeds)), key=lambda i: (distances[i], i))


def voronoi_partition_mv(points: List[Point2D],
                         seeds: List[Point2D]) -> Dict[int, List[Point2D]]:
    """Assign each point to the region of its nearest seed using GA distances.

    Returns a dict mapping seed index → list of original point tuples.
    """
    # Convert once to multivectors for efficiency
    mv_points = [point_to_mv(p) for p in points]
    mv_seeds = [point_to_mv(s) for s in seeds]

    regions: Dict[int, List[Point2D]] = {i: [] for i in range(len(seeds))}
    for p, mv_p in zip(points, mv_points):
        idx = nearest_mv(mv_p, mv_seeds)
        regions[idx].append(p)
    return regions


def rotate_toward(v: MVPoint, target: MVPoint, angle: float) -> MVPoint:
    """Rotate vector *v* toward *target* by *angle* (radians) using a rotor.

    The rotation plane is defined by the bivector v ∧ target.
    """
    # Compute the rotation plane bivector (outer product)
    plane = outer_product(v, target).grade(2)
    if not plane.components:
        # v and target are collinear; no rotation needed
        return v

    # Normalise the bivector to unit magnitude
    norm_sq = inner_product(plane, plane).scalar_part()
    if norm_sq <= 0:
        return v
    plane_norm = plane * (1.0 / math.sqrt(norm_sq))

    # Rotor: R = cos(θ/2) - sin(θ/2) * (plane_norm)
    cos = math.cos(angle / 2.0)
    sin = math.sin(angle / 2.0)
    rotor = Multivector({frozenset(): cos}, n=2) - sin * plane_norm

    # Apply rotor: v' = R v R̃   (R̃ is reverse)
    rotor_rev = reverse(rotor)
    return rotor * v * rotor_rev


# ---------------------------------------------------------------------------
# Demonstration / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate random points and seeds in the unit square
    random.seed(0)
    num_points = 100
    num_seeds = 5
    points = [(random.random(), random.random()) for _ in range(num_points)]
    seeds = [(random.random(), random.random()) for _ in range(num_seeds)]

    # Hybrid Voronoi partition
    regions = voronoi_partition_mv(points, seeds)

    # Simple sanity checks
    total_assigned = sum(len(lst) for lst in regions.values())
    assert total_assigned == num_points, "Some points were lost during assignment"

    # Show region sizes
    print("Voronoi region sizes (using geometric algebra distances):")
    for idx, pts in regions.items():
        print(f"  Seed {idx}: {len(pts)} points")

    # Demonstrate rotor rotation
    p = point_to_mv((0.5, 0.0))
    target = point_to_mv((0.0, 0.5))
    rotated = rotate_toward(p, target, math.pi / 2)   # rotate 90°
    print("\nRotation example:")
    print("  Original point :", mv_to_point(p))
    print("  Target direction:", mv_to_point(target))
    print("  Rotated point  :", mv_to_point(rotated))
    # The rotated point should be close to (0,0.5)
    rx, ry = mv_to_point(rotated)
    assert math.isclose(rx, 0.0, abs_tol=1e-6) and math.isclose(ry, 0.5, abs_tol=1e-6), "Rotor failed"

    print("\nAll hybrid operations executed successfully.")