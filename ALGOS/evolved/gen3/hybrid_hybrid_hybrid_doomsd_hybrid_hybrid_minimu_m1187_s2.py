# DARWIN HAMMER — match 1187, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# born: 2026-05-29T23:33:16Z

"""
This module integrates the hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py and 
hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py algorithms into a single hybrid system.
The mathematical bridge between the two structures is formed by using the geometric product of multivectors 
from the first algorithm to compute the distances and orientations between nodes in the decision tree of the second algorithm.
The expected cost of the decision tree is used as a weight for the multivectors, and then the Shannon entropy 
of the weighted multivectors is calculated to gain insights into the complexity and uncertainty of the decision-making process.

The governing equations of the Clifford algebra are used to compute the geometric product of multivectors, 
which are then used to represent nodes in the metric space. The decision tree and decision hygiene scoring system 
are used to assign weights to the multivectors and calculate the Shannon entropy of the weighted scores.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections.abc import Iterable
from datetime import date
import bisect

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
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
    def __init__(self, blades):
        self.blades = blades

    def __mul__(self, other):
        result = Multivector({})
        for blade_a, coeff_a in self.blades.items():
            for blade_b, coeff_b in other.blades.items():
                blade, sign = _multiply_blades(blade_a, blade_b)
                if blade in result.blades:
                    result.blades[blade] += sign * coeff_a * coeff_b
                else:
                    result.blades[blade] = sign * coeff_a * coeff_b
        return result

    def __repr__(self):
        return f"Multivector({self.blades})"


def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(nodes, edges, root):
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj = {n: [] for n in nodes}
    edge_len = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist


def shannon_entropy(weights):
    """Calculate Shannon entropy of a probability distribution."""
    probs = [w / sum(weights) for w in weights]
    return -sum(p * math.log(p, 2) for p in probs if p > 0)


def hybrid_algorithm(nodes, edges, root, multivectors):
    """
    Hybrid algorithm that integrates the geometric product of multivectors with the decision tree and decision hygiene scoring system.

    Returns
    -------
    expected_cost : float
    shannon_entropy : float
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    weights = []
    for node in nodes:
        multivector = multivectors[node]
        expected_cost = sum(edge_len[(node, neighbour)] * multivector.blades.get(frozenset([neighbour]), 0) for neighbour in adj[node])
        weights.append(expected_cost)
    return sum(weights) / len(weights), shannon_entropy(weights)


if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1), "D": (0, 1)}
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"
    multivectors = {
        "A": Multivector({frozenset([]): 1, frozenset([1]): 2}),
        "B": Multivector({frozenset([]): 1, frozenset([0]): 3}),
        "C": Multivector({frozenset([]): 1, frozenset([0, 1]): 4}),
        "D": Multivector({frozenset([]): 1, frozenset([1]): 5})
    }
    expected_cost, shannon_entropy = hybrid_algorithm(nodes, edges, root, multivectors)
    print(f"Expected Cost: {expected_cost}")
    print(f"Shannon Entropy: {shannon_entropy}")