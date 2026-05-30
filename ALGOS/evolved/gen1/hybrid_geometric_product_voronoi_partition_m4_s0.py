# DARWIN HAMMER — match 4, survivor 0
# gen: 1
# parent_a: geometric_product.py (gen0)
# parent_b: voronoi_partition.py (gen0)
# born: 2026-05-29T23:14:40Z

#!/usr/bin/env python3
"""
Hybrid algorithm fusing Clifford geometric product from geometric_product.py
and Voronoi partitioning from voronoi_partition.py. The mathematical bridge
lies in using Voronoi regions to partition the multivector space and then
applying Clifford product within these regions.

This module defines functions that integrate these two concepts. It includes
functions for Voronoi-based multivector partitioning, geometric product
application within these partitions, and visualization of the resulting
multivectors.
"""

from __future__ import annotations
import math
import numpy as np
import random
import sys
import pathlib

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)

    def scalar_part(self):
        return self.components.get(frozenset(), 0.0)

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

    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __sub__(self, other):
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, other):
        return geometric_product(self, other)

def geometric_product(a, b):
    result = {}
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            contrib = sign * coef_a * coef_b
            result[blade_out] = result.get(blade_out, 0.0) + contrib
    n = max(a.n, b.n)
    return Multivector({k: v for k, v in result.items() if v != 0.0}, n)

def voronoi_multivector_partitions(seeds, points, n):
    regions = assign(points, seeds)
    multivectors = []
    for region in regions.values():
        blades = []
        for point in region:
            blade = frozenset([i for i, x in enumerate(point) if x != 0])
            blades.append(blade)
        components = {blade: 1 for blade in blades}
        multivector = Multivector(components, n)
        multivectors.append(multivector)
    return multivectors

def geometric_product_in_voronoi_regions(seeds, points, n):
    multivectors = voronoi_multivector_partitions(seeds, points, n)
    products = []
    for multivector in multivectors:
        product = multivector * multivector
        products.append(product)
    return products

if __name__ == "__main__":
    seeds = [(0, 0), (1, 1), (2, 2)]
    points = [(0.1, 0.1), (0.2, 0.2), (1.1, 1.1), (1.2, 1.2), (2.1, 2.1), (2.2, 2.2)]
    n = 2
    products = geometric_product_in_voronoi_regions(seeds, points, n)
    for product in products:
        print(product)