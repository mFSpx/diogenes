# DARWIN HAMMER — match 1412, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m687_s0.py (gen5)
# born: 2026-05-29T23:36:11Z

import math
import numpy as np

Point = tuple[float, float]
Edge = tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())

class Multivector:
    def __init__(self, components: dict[frozenset, float] = None):
        self.components: dict[frozenset, float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                combined = list(ba) + list(bb)
                result[frozenset(combined)] = result.get(frozenset(combined), 0.0) + ca * cb
        return Multivector(result)

def _blade_sign(indices: list[int]) -> tuple[list[int], int]:
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
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def hybrid_score(theta: float, center: float, width: float, multivector: Multivector = None) -> float:
    intensity = max(gaussian_beam(theta, center, width), 1e-12)
    derivative = intensity * (-(theta - center) / (width * width))
    score = (derivative * derivative) / intensity
    if multivector:
        score += multivector.components.get(frozenset(), 0.0)
    return score

def hybrid_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2, multivector: Multivector = None) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    score = material + path_weight * sum(dist.values())
    if multivector:
        score += multivector.components.get(frozenset(), 0.0)
    return score

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return prior * likelihood / (likelihood + false_positive)

def bayes_marginal_hybrid(prior: float, likelihood: float, false_positive: float, multivector: Multivector = None) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    if multivector:
        prior += multivector.components.get(frozenset(), 0.0)
        likelihood += multivector.components.get(frozenset(), 0.0)
        false_positive += multivector.components.get(frozenset(), 0.0)
    return prior * likelihood / max(likelihood + false_positive, 1e-12)

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    path_weight = 0.2
    multivector = Multivector({frozenset(): 1.0})
    print(hybrid_cost(nodes, edges, root, path_weight, multivector))
    print(bayes_marginal_hybrid(0.5, 0.5, 0.5, multivector))
    print(hybrid_score(1.0, 1.0, 1.0, multivector=Multivector({frozenset(): 1.0})))