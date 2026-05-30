# DARWIN HAMMER — match 2072, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_perceptual_de_hybrid_semantic_neig_m968_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m932_s1.py (gen5)
# born: 2026-05-29T23:40:43Z

import math
import random
import sys
import pathlib
import numpy as np

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

class RBFSurrogate:
    def __init__(self, centers: list[tuple[float, ...]], weights: np.ndarray, epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: list[float]) -> float:
        return float(
            sum(
                w * gaussian(euclidean(x, c), self.epsilon)
                for w, c in zip(self.weights, self.centers)
            )
        )

def fit_rbf(points: list[list[float]],
            values: list[float],
            epsilon: float = 1.0,
            ridge: float = 1e-9) -> RBFSurrogate:
    n = len(points)
    if n == 0:
        raise ValueError("No points to fit.")
    phi = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            phi[i, j] = gaussian(euclidean(points[i], points[j]), epsilon)
    phi += ridge * np.eye(n)
    y = np.asarray(values, dtype=float)
    weights, *_ = np.linalg.lstsq(phi, y, rcond=None)
    centers = [tuple(map(float, p)) for p in points]
    return RBFSurrogate(centers, weights, epsilon)

class Multivector:
    def __init__(self, components: dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        result = {}
        for blade1, coef1 in self.components.items():
            for blade2, coef2 in other.components.items():
                blade = blade1.symmetric_difference(blade2)
                result[blade] = result.get(blade, 0.0) + coef1 * coef2 * (-1)**sum(1 for i in blade1 & blade2 for j in blade2 if i > j)
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

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items()})

def hybrid_predict(x: list[float], rbf: RBFSurrogate) -> Multivector:
    scalar_part = rbf.predict(x)
    return Multivector({frozenset(): scalar_part}, 1)

def hybrid_grade(x: list[float], rbf: RBFSurrogate, k: int) -> Multivector:
    mv = hybrid_predict(x, rbf)
    return mv.grade(k)

def hybrid_geometric_product(x1: list[float], x2: list[float], rbf1: RBFSurrogate, rbf2: RBFSurrogate) -> Multivector:
    mv1 = hybrid_predict(x1, rbf1)
    mv2 = hybrid_predict(x2, rbf2)
    return mv1.geometric_product(mv2)

def hybrid_scalar_part(x: list[float], rbf: RBFSurrogate) -> float:
    mv = hybrid_predict(x, rbf)
    return mv.scalar_part()

if __name__ == "__main__":
    points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    values = [1.0, 2.0, 3.0]
    rbf = fit_rbf(points, values)
    x = [2.0, 3.0]
    print(hybrid_predict(x, rbf))
    print(hybrid_grade(x, rbf, 0))
    print(hybrid_scalar_part(x, rbf))
    print(hybrid_geometric_product([2.0, 3.0], [4.0, 5.0], rbf, rbf))