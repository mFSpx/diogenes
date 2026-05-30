# DARWIN HAMMER — match 35, survivor 6
# gen: 4
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:26:34Z

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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


def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = dict(components or {})

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
        result: Dict[frozenset, float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[blade] = result.get(blade, 0.0) + coeff
        result = {k: v for k, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def vector_to_mv(x: float, y: float) -> Multivector:
    return Multivector({frozenset({0}): x, frozenset({1}): y})


def clifford_dot(a: Multivector, b: Multivector) -> float:
    return (a * b).scalar_part()


def clifford_norm(a: Multivector) -> float:
    return math.sqrt(abs(clifford_dot(a, a)))


def clifford_distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    mv_p = vector_to_mv(*p)
    mv_q = vector_to_mv(*q)
    diff = mv_p - mv_q
    return clifford_norm(diff)


Point = Tuple[float, float]
Edge = Tuple[str, str]


def build_clifford_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge]
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Edge]]:
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []
    edge_list: List[Edge] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = clifford_distance(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list


def voronoi_partition(
    nodes: Dict[str, Point], seeds: List[str]
) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    seed_points = {s: nodes[s] for s in seeds}
    for name, pt in nodes.items():
        nearest = min(
            seeds,
            key=lambda s: clifford_distance(pt, seed_points[s]),
        )
        assignment[name] = nearest
    return assignment


def hybrid_analysis(
    nodes: Dict[str, Point],
    edges: List[Edge],
    seeds: List[str],
) -> Tuple[np.ndarray, Dict[str, str]]:
    L, _, _ = build_clifford_length_matrix(nodes, edges)
    partition = voronoi_partition(nodes, seeds)
    return L, partition


def improved_hybrid_analysis(
    nodes: Dict[str, Point],
    edges: List[Edge],
    seeds: List[str],
) -> Tuple[np.ndarray, Dict[str, str], Dict[str, float]]:
    L, _, _ = build_clifford_length_matrix(nodes, edges)
    partition = voronoi_partition(nodes, seeds)
    node_distances = {node: clifford_distance(nodes[node], nodes[partition[node]]) for node in nodes}
    return L, partition, node_distances


if __name__ == "__main__":
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("A", "C"),  
    ]
    seeds = ["A", "C"]

    L, part, dist = improved_hybrid_analysis(nodes, edges, seeds)

    print("Clifford length matrix (rounded):")
    print(np.round(L, 4))
    print("\nVoronoi assignment:")
    for node, seed in part.items():
        print(f"  {node} → {seed}")
    print("\nNode distances to their assigned seeds:")
    for node, distance in dist.items():
        print(f"  {node}: {distance}")

    assert np.allclose(L, L.T, atol=1e-12)
    assert np.all(L >= 0.0)