# DARWIN HAMMER — match 30, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py (gen2)
# parent_b: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s2.py (gen2)
# born: 2026-05-29T23:26:21Z

"""
Hybrid module combining the geometric product and Voronoi partitioning from 
'hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s1.py' and the pheromone-based surface usage tracking 
with Shannon entropy calculation from 'hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s2.py'.

The mathematical bridge lies in applying the Shannon entropy calculation to the probability distribution of 
the Voronoi partitions obtained from the geometric product, and then using the pheromone signals to 
modulate the entropy calculation, allowing for a more detailed understanding of the surface usage patterns 
in the context of the geometric structure.

The mathematical interface is established by using the pheromone probabilities to weight the Voronoi 
partitions, and then applying the Shannon entropy calculation to these weighted partitions.
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
                terms.append(f"{coef}*{' ^ '.join(map(str, sorted(blade)))}")

def shannon_entropy(probabilities):
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log(p, 2)
    return entropy

def modulate_entropy(pheromone_signals, voronoi_partitions):
    probabilities = [len(points) / sum(len(points) for points in voronoi_partitions.values()) for points in voronoi_partitions.values()]
    modulated_probabilities = [p * pheromone_signals[i] for i, p in enumerate(probabilities)]
    modulated_probabilities = [p / sum(modulated_probabilities) for p in modulated_probabilities]
    return modulated_probabilities

def hybrid_operation(points, seeds, pheromone_signals):
    voronoi_partitions = assign(points, seeds)
    probabilities = modulate_entropy(pheromone_signals, voronoi_partitions)
    entropy = shannon_entropy(probabilities)
    return entropy, voronoi_partitions

def generate_points(num_points, x_range, y_range):
    points = []
    for _ in range(num_points):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        points.append((x, y))
    return points

def generate_seeds(num_seeds, x_range, y_range):
    seeds = []
    for _ in range(num_seeds):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        seeds.append((x, y))
    return seeds

def generate_pheromone_signals(num_signals):
    pheromone_signals = []
    for _ in range(num_signals):
        pheromone_signals.append(random.uniform(0, 1))
    return pheromone_signals

if __name__ == "__main__":
    points = generate_points(100, (0, 10), (0, 10))
    seeds = generate_seeds(5, (0, 10), (0, 10))
    pheromone_signals = generate_pheromone_signals(len(seeds))
    entropy, voronoi_partitions = hybrid_operation(points, seeds, pheromone_signals)
    print(f"Shannon Entropy: {entropy}")
    print("Voronoi Partitions:")
    for i, points in voronoi_partitions.items():
        print(f"Seed {i}: {len(points)} points")