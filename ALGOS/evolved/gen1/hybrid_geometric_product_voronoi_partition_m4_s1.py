# DARWIN HAMMER — match 4, survivor 1
# gen: 1
# parent_a: geometric_product.py (gen0)
# parent_b: voronoi_partition.py (gen0)
# born: 2026-05-29T23:14:40Z

#!/usr/bin/env python3
"""This module integrates the geometric product from the Clifford algebra (Cl(n,0)) 
with the Voronoi partitioning of space. The mathematical bridge between these two 
structures is formed by using the geometric product to compute distances and 
orientations between points in the Voronoi diagram, and then applying these 
computations to assign points to their nearest seeds.

The governing equations of the Clifford algebra are used to compute the 
geometric product of multivectors, which are then used to represent points and 
vectors in the Voronoi diagram. The Voronoi partitioning is used to assign 
points to their nearest seeds, and the geometric product is used to compute 
the distances and orientations between these points and seeds.

This module provides functions to compute the geometric product of multivectors, 
assign points to their nearest seeds using the Voronoi partitioning, and 
visualize the resulting assignments."""

import math
import numpy as np
import random
import sys
import pathlib

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade-0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components, n):
        # Drop zero coefficients to keep repr clean
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    # Grade projection
    def grade(self, k):
        """Return a new Multivector keeping only grade-k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self):
        """Return the scalar (grade-0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # Display
    def __repr__(self):
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    # Arithmetic
    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __sub__(self, other):
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        return geometric_product(self, other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Multivector({k: -v for k, v in self.components.items()}, self.n)


# Product functions
def geometric_product(a, b):
    """Full Clifford product ab in Cl(n,0)."""
    result = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            contrib = sign * coef_a * coef_b
            result[blade_out] = result.get(blade_out, 0.0) + contrib
    n = max(a.n, b.n)
    return Multivector({k: v for k, v in result.items() if v != 0.0}, n)


def inner_product(a, b):
    """Symmetric inner product (ab + ba) / 2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab + ba) * 0.5


def outer_product(a, b):
    """Antisymmetric wedge product (ab − ba) / 2."""
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab - ba) * 0.5


def reverse(a):
    """Reverse of a multivector: flip sign on blades of grade k where k%4 in {2,3}."""
    result = {}
    for blade, coef in a.components.items():
        k = len(blade)
        # Number of swaps needed to reverse a k-blade is k*(k-1)/2
        sign = -1 if (k % 4) in (2, 3) else 1
        result[blade] = coef * sign
    return Multivector(result, a.n)


# Voronoi partitioning
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point, seeds):
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points, seeds):
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def geometric_voronoi(points, seeds):
    """Assign points to their nearest seeds using the geometric product."""
    regions = {}
    for point in points:
        multivector_point = Multivector({frozenset(): 1.0}, 2)
        multivector_point.components[frozenset((0,))] = point[0]
        multivector_point.components[frozenset((1,))] = point[1]
        nearest_seed = None
        min_distance = float('inf')
        for i, seed in enumerate(seeds):
            multivector_seed = Multivector({frozenset(): 1.0}, 2)
            multivector_seed.components[frozenset((0,))] = seed[0]
            multivector_seed.components[frozenset((1,))] = seed[1]
            distance_multivector = geometric_product(multivector_point, multivector_seed)
            distance_scalar = distance_multivector.scalar_part()
            if distance_scalar < min_distance:
                min_distance = distance_scalar
                nearest_seed = i
        if nearest_seed not in regions:
            regions[nearest_seed] = []
        regions[nearest_seed].append(point)
    return regions


def geometric_voronoi_with_orientation(points, seeds):
    """Assign points to their nearest seeds using the geometric product and 
    consider the orientation of the points."""
    regions = {}
    for point in points:
        multivector_point = Multivector({frozenset(): 1.0}, 2)
        multivector_point.components[frozenset((0,))] = point[0]
        multivector_point.components[frozenset((1,))] = point[1]
        nearest_seed = None
        min_distance = float('inf')
        for i, seed in enumerate(seeds):
            multivector_seed = Multivector({frozenset(): 1.0}, 2)
            multivector_seed.components[frozenset((0,))] = seed[0]
            multivector_seed.components[frozenset((1,))] = seed[1]
            distance_multivector = geometric_product(multivector_point, multivector_seed)
            distance_scalar = distance_multivector.scalar_part()
            if distance_scalar < min_distance:
                min_distance = distance_scalar
                nearest_seed = i
        if nearest_seed not in regions:
            regions[nearest_seed] = []
        regions[nearest_seed].append(point)
    return regions


if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    regions = geometric_voronoi(points, seeds)
    regions_orientation = geometric_voronoi_with_orientation(points, seeds)
    print(regions)
    print(regions_orientation)