# DARWIN HAMMER — match 2702, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py (gen4)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s2.py (gen2)
# born: 2026-05-29T23:43:55Z

"""
This module integrates the radial basis functions from hybrid_hybrid_rbf_surrogate_hybrid_hybrid_fisher_m317_s0.py 
and the geometric product from hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py. 
The mathematical bridge between the two structures is the application of geometric product to model uncertainty in the 
feature vectors, similar to the uncertainty modeling in radial basis functions. 
In this hybrid algorithm, we use geometric product to model the uncertainty of the feature vectors over a graph structure.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

Node = int
Graph = dict[Node, set[Node]]
FeatureVec = tuple[float, float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: dict[Node, FeatureVec]) -> tuple[np.ndarray, list[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

class Multivector:
    def __init__(self, components: dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
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

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            if blade in result:
                result[blade] += coef
            else:
                result[blade] = coef
        return Multivector(result, self.n)

    def __mul__(self, other: 'Multivector') -> 'Multivector':
        result = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                combined = list(blade_a) + list(blade_b)
                result,_ = _blade_sign(combined)
                result[frozenset(result)] = coef_a * coef_b
        return Multivector(result, self.n)

def _blade_sign(indices: list[int]) -> tuple[list[int], int]:
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

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    return a * b

def similarity_with_geometric_product(features: dict[Node, FeatureVec], a: Multivector, b: Multivector) -> np.ndarray:
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
                if i == 0 and j == 1:
                    S[i, j] *= a.scalar_part() * b.scalar_part()
                elif i == 1 and j == 0:
                    S[i, j] *= a.scalar_part() * b.scalar_part()
                elif i == 0 and j == 1:
                    S[i, j] *= geometric_product(a, b).scalar_part()
                elif i == 1 and j == 0:
                    S[i, j] *= geometric_product(a, b).scalar_part()
    return S

def test_similarity_with_geometric_product():
    features = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    a = Multivector({frozenset(): 1.0}, 2)
    b = Multivector({frozenset(): 1.0}, 2)
    S = similarity_with_geometric_product(features, a, b)
    print(S)

if __name__ == "__main__":
    test_similarity_with_geometric_product()