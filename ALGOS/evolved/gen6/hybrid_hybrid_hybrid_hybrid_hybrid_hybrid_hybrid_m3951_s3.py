# DARWIN HAMMER — match 3951, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py (gen3)
# born: 2026-05-29T23:52:42Z

"""
This module integrates the governing equations of 
'hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s2.py' and 
'hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py'. 

The mathematical bridge lies in the use of the Gini coefficient to inform 
the geometric product of multivectors in the decision-making process. 

By evaluating the Gini coefficient of the edge costs at each step, 
we can leverage the geometric product to guide the decision-making 
process in a way that minimizes the impact of noise in the data.

The hybrid algorithm fuses the core topologies of both parents by 
using the Gini coefficient to inform the geometric product, 
creating a more robust and adaptive decision-making process.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def gini_coefficient(values: List[float]) -> float:
    """Compute the Gini coefficient."""
    values = sorted(values)
    index = np.arange(1, len(values)+1)
    n = len(values)
    return ((np.sum((2 * index - n  - 1) * values)) / (n * np.sum(values)))

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
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
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

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

    def __add__(self, other):
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

    def __mul__(self, other):
        result = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_c, sign = _multiply_blades(blade_a, blade_b)
                coef_c = coef_a * coef_b * sign
                result[blade_c] = result.get(blade_c, 0.0) + coef_c
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    material = 0
    edge_costs = []
    for u, v in edges:
        material += length(nodes[u], nodes[v])
        edge_costs.append(length(nodes[u], nodes[v]))

    gini = gini_coefficient(edge_costs)

    path_cost = 0
    visited = set()
    stack = [(root, 0)]
    while stack:
        node, dist = stack.pop()
        if node not in visited:
            visited.add(node)
            path_cost += dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append((neighbor, dist + length(nodes[node], nodes[neighbor])))

    mv = Multivector({frozenset(): 1.0}, 3)
    mv = mv * Multivector({frozenset({0}): gini}, 3)

    return material + path_weight * path_cost

def hybrid_operation(nodes: Dict[str, Point],
                     edges: List[Tuple[str, str]],
                     root: str) -> Multivector:
    """
    Perform the hybrid operation.
    """
    cost = tree_cost(nodes, edges, root)
    gini = gini_coefficient([length(nodes[u], nodes[v]) for u, v in edges])
    mv = Multivector({frozenset(): cost}, 3)
    mv = mv * Multivector({frozenset({0}): gini}, 3)
    return mv

def smoke_test():
    nodes = {
        'A': Point(0, 0),
        'B': Point(1, 0),
        'C': Point(1, 1),
        'D': Point(0, 1)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    mv = hybrid_operation(nodes, edges, root)
    print(mv.components)

if __name__ == "__main__":
    smoke_test()