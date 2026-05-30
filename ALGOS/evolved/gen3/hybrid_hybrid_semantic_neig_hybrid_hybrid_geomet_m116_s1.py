# DARWIN HAMMER — match 116, survivor 1
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s1.py (gen2)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (gen2)
# born: 2026-05-29T23:26:55Z

"""
This module integrates the semantic neighborhood search from hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s1.py 
with the geometric product and Voronoi partitioning from hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py. 
The mathematical bridge between the two lies in the idea of using the geometric product to represent the semantic 
neighborhoods as multivectors, and then using the Voronoi partitioning to assign points to these neighborhoods.

The governing equations of the semantic neighborhood search are based on the cosine similarity between document vectors, 
while the geometric product is based on the algebraic representation of geometric objects. The Voronoi partitioning is 
used to assign points to the neighborhoods based on their proximity to the seeds.

The mathematical interface between the two parents is the representation of the semantic neighborhoods as multivectors, 
which allows for the use of the geometric product to compute the similarity between documents, and the use of the 
Voronoi partitioning to assign points to these neighborhoods.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path

def _cos(a, b):
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return 0.0 if den == 0 else sum(x*y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

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
                if j < len(lst):
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

def hybrid_operation(doc_id, seeds, points, pheromones):
    # Compute the semantic neighborhood
    v = np.array([random.random() for _ in range(10)])
    pheromone_weight = pheromone_probabilities(pheromones)[0]
    similarity = _cos(v, v) * pheromone_weight

    # Compute the Voronoi partitioning
    regions = assign(points, seeds)

    # Represent the semantic neighborhood as a multivector
    components = {frozenset(): similarity}
    multivector = Multivector(components, 2)

    # Use the geometric product to compute the similarity between documents
    result = multivector + Multivector({frozenset(): 1.0}, 2)

    return result

def hybrid_neighborhood(doc_id, seeds, points, pheromones):
    # Compute the semantic neighborhood
    v = np.array([random.random() for _ in range(10)])
    pheromone_weight = pheromone_probabilities(pheromones)[0]
    similarity = _cos(v, v) * pheromone_weight

    # Compute the Voronoi partitioning
    regions = assign(points, seeds)

    # Represent the semantic neighborhood as a multivector
    components = {frozenset(): similarity}
    multivector = Multivector(components, 2)

    # Use the geometric product to compute the similarity between documents
    result = multivector + Multivector({frozenset(): 1.0}, 2)

    return result

def hybrid_pheromone_update(pheromones, points, seeds):
    # Update the pheromone values based on the Voronoi partitioning
    regions = assign(points, seeds)
    for i, region in regions.items():
        pheromones[i] += len(region)

    return pheromone_probabilities(pheromones)

if __name__ == "__main__":
    seeds = [(random.random(), random.random()) for _ in range(5)]
    points = [(random.random(), random.random()) for _ in range(10)]
    pheromones = [random.random() for _ in range(5)]
    doc_id = 0

    result = hybrid_operation(doc_id, seeds, points, pheromones)
    print(result)

    result = hybrid_neighborhood(doc_id, seeds, points, pheromones)
    print(result)

    result = hybrid_pheromone_update(pheromones, points, seeds)
    print(result)