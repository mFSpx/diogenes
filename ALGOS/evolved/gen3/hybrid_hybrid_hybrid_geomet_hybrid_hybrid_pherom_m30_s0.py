# DARWIN HAMMER — match 30, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# parent_b: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s2.py (gen2)
# born: 2026-05-29T23:26:21Z

"""Hybrid module combining the geometric product, Voronoi partitioning, 
and Ollivier-Ricci curvature calculation from 'hybrid_geometric_product_voronoi_partition_m4_s0.py' 
and the pheromone-based surface usage tracking and Shannon entropy calculation from 
'hybrid_pheromone_infotaxis_m3_s0.py' and 'hybrid_decision_hygiene_shannon_entropy_m12_s0.py'.

The mathematical bridge lies in applying the Shannon entropy calculation to the pheromone probabilities 
obtained from the surface usage tracking, and then using the Ollivier-Ricci curvature calculation to 
quantify the connectivity between the pheromone signal distributions.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter

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
                terms.append(f'{coef}*{blade}')
        return f'Multivector({", ".join(terms)})'

def pheromone_entropy(probabilities):
    """Calculate the Shannon entropy of the pheromone probabilities."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def ollivier_ricci_curvature(graph, pheromone_entropy_values):
    """Calculate the Ollivier-Ricci curvature between graph nodes based on pheromone entropy values."""
    curvature = np.zeros((len(graph), len(graph)))
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i != j:
                curvature[i, j] = pheromone_entropy_values[i] - pheromone_entropy_values[j]
    return curvature

def geometric_product_voronoi_partition(points, seeds, pheromone_probabilities):
    """Apply the geometric product and Voronoi partitioning to the points and seeds, 
    and then calculate the Shannon entropy of the pheromone probabilities."""
    regions = assign(points, seeds)
    multivectors = []
    for region in regions.values():
        multivector = Multivector({}, len(region))
        for i in range(len(region)):
            multivector.components[frozenset([i])] = 1.0
        multivectors.append(multivector)
    pheromone_entropy_values = [pheromone_entropy(p) for p in pheromone_probabilities]
    return multivectors, pheromone_entropy_values

def hybrid_operation(points, seeds, pheromone_probabilities):
    """Perform the hybrid operation by applying the geometric product and Voronoi partitioning, 
    calculating the Shannon entropy of the pheromone probabilities, and then calculating the Ollivier-Ricci curvature."""
    multivectors, pheromone_entropy_values = geometric_product_voronoi_partition(points, seeds, pheromone_probabilities)
    curvature = ollivier_ricci_curvature(regions.keys(), pheromone_entropy_values)
    return multivectors, curvature

def smoke_test():
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    seeds = [(1.0, 1.0), (2.0, 2.0)]
    pheromone_probabilities = [0.5, 0.5]
    multivectors, curvature = hybrid_operation(points, seeds, pheromone_probabilities)
    print(multivectors)
    print(curvature)

if __name__ == "__main__":
    smoke_test()