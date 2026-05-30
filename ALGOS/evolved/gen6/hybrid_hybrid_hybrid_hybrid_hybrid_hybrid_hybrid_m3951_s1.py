# DARWIN HAMMER — match 3951, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py (gen3)
# born: 2026-05-29T23:52:42Z

"""
This module integrates the governing equations of 'hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s2.py' 
and 'hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py'. The mathematical bridge lies in the use of 
the Gini coefficient to inform the geometric product of multivectors. By evaluating the Gini coefficient of the 
edge costs at each step, we can leverage the geometric product to guide the decision-making process in a way 
that minimizes the impact of noise in the data.

The Gini coefficient is used to calculate the inequality of the edge costs. This inequality measure is then used 
to adjust the geometric product, which in turn guides the decision to incorporate or not incorporate the edge 
into the tree.
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
                blade_c, sign = _blade_sign(list(blade_a) + list(blade_b))
                result[blade_c] = result.get(blade_c, 0.0) + sign * coef_a * coef_b
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

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

def gini_coefficient(edge_costs):
    """
    Calculate the Gini coefficient of a list of edge costs.
    """
    edge_costs = np.array(edge_costs)
    mean = np.mean(edge_costs)
    num = np.sum(np.abs(edge_costs - mean))
    den = 2 * len(edge_costs) * mean
    return num / den if den != 0 else 0

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
    for u, v in edges:
        material += length(nodes[u], nodes[v])

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

    return material + path_weight * path_cost

def hybrid_tree_cost(nodes: Dict[str, Point],
                      edges: List[Tuple[str, str]],
                      root: str,
                      path_weight: float = 0.2,
                      multivector: Multivector = None) -> float:
    """
    Compute the total cost of a tree, taking into account the geometric product of multivectors.
    """
    if multivector is None:
        multivector = Multivector({frozenset(): 1.0}, len(nodes))

    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    material = 0
    for u, v in edges:
        material += length(nodes[u], nodes[v])

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

    gini = gini_coefficient([length(nodes[u], nodes[v]) for u, v in edges])
    multivector_product = multivector * Multivector({frozenset(): gini}, len(nodes))

    return material + path_weight * path_cost + multivector_product.scalar_part()

if __name__ == "__main__":
    nodes = {
        'A': Point(0, 0),
        'B': Point(1, 0),
        'C': Point(1, 1),
        'D': Point(0, 1)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    print(tree_cost(nodes, edges, root))
    print(hybrid_tree_cost(nodes, edges, root))