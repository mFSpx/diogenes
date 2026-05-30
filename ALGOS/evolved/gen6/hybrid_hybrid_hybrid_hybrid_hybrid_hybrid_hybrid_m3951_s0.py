# DARWIN HAMMER — match 3951, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py (gen3)
# born: 2026-05-29T23:52:42Z

"""
This module integrates the governing equations of 'hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s2.py' and 
'hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s7.py'. The mathematical bridge lies in the use of 
the Gini coefficient to inform the geometric product in the Multivector class. By evaluating the Gini coefficient 
of the edge costs at each step, we can leverage the geometric product to guide the decision-making process in a way 
that minimizes the impact of noise in the data.

The Gini coefficient is used to calculate the inequality of the edge costs. This inequality measure 
is then used to adjust the geometric product, which in turn guides the decision to incorporate or not 
incorporate the edge into the tree.
"""

import numpy as np
import math
import random
import sys
import pathlib

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: dict,
              edges: list,
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    adj: dict = {n: [] for n in nodes}
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
                combined = list(blade_a) + list(blade_b)
                result_tuple = tuple(sorted(combined))
                if len(result_tuple) == len(set(result_tuple)):
                    result[frozenset(result_tuple)] = result.get(frozenset(result_tuple), 0.0) + coef_a * coef_b
        return Multivector({k: v for k, v in result.items() if v != 0.0}, self.n)

def gini_coefficient(values):
    """Calculate the Gini coefficient of a list of values."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    coefficient = 0
    for i in range(n):
        coefficient += np.sum(np.abs(values - values[i]))
    return coefficient / (2 * n * n * mean)

def hybrid_tree_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2):
    """Calculate the hybrid tree cost using the Gini coefficient to inform the geometric product."""
    edge_lengths = [length(nodes[u], nodes[v]) for u, v in edges]
    gini = gini_coefficient(edge_lengths)
    multivector = Multivector({frozenset([i]): length for i, length in enumerate(edge_lengths)}, len(edge_lengths))
    result = multivector * multivector
    return tree_cost(nodes, edges, root, path_weight) + gini * result.scalar_part()

def test_hybrid_tree_cost():
    nodes = {'a': Point(0, 0), 'b': Point(1, 0), 'c': Point(0, 1)}
    edges = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    root = 'a'
    return hybrid_tree_cost(nodes, edges, root)

def main():
    print(test_hybrid_tree_cost())

if __name__ == "__main__":
    main()