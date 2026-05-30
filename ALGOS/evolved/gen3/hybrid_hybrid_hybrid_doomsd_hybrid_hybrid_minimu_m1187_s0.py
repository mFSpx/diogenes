# DARWIN HAMMER — match 1187, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# born: 2026-05-29T23:33:15Z

"""
This module integrates the hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0 and 
hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1 algorithms into a single hybrid system.
The bridge between the two structures is the concept of distance and entropy, 
which can be applied to both the geometric product and decision hygiene scoring systems.
By calculating the geometric product of multivectors and the entropy of the decision hygiene feature counts, 
we can gain insights into the complexity and uncertainty of the decision-making process.
The mathematical bridge is formed by using the geometric product to compute distances and orientations 
between days and seeds, and the entropy to calculate the uncertainty of the decision-making process.
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
    def __init__(self, blades: dict):
        self.blades = blades


def length(a: tuple, b: tuple) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: dict,
    edges: list,
    root: str,
) -> tuple:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: dict = {n: [] for n in nodes}
    edge_len: dict = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: dict = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the 
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist


def geometric_product_entropy(multivector: Multivector, nodes: dict, edges: list, root: str) -> float:
    """
    Calculate the geometric product of multivectors and the entropy of the decision hygiene feature counts.

    Returns
    -------
    entropy : the entropy of the decision hygiene feature counts
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    # calculate the geometric product of multivectors
    blades = multivector.blades
    product = 0
    for blade_a, blade_b in blades.items():
        result, sign = _multiply_blades(blade_a, blade_b)
        product += sign * blade_b
    # calculate the entropy of the decision hygiene feature counts
    feature_counts = [len(adj[node]) for node in nodes]
    probabilities = [count / sum(feature_counts) for count in feature_counts]
    entropy = -sum([probability * math.log2(probability) for probability in probabilities])
    return entropy


def decision_hygiene_scoring(nodes: dict, edges: list, root: str) -> dict:
    """
    Calculate the decision hygiene scores for each node.

    Returns
    -------
    scores : dict mapping node → decision hygiene score
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    scores = {}
    for node in nodes:
        # calculate the decision hygiene score
        score = len(adj[node]) / sum(len(adj[n]) for n in nodes)
        scores[node] = score
    return scores


def main():
    # define the nodes and edges
    nodes = {
        'A': (0, 0),
        'B': (3, 4),
        'C': (6, 8),
        'D': (9, 12)
    }
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D')
    ]
    root = 'A'
    # define the multivector
    multivector = Multivector({
        frozenset([1, 2, 3]): 4,
        frozenset([4, 5, 6]): 7
    })
    # calculate the geometric product entropy
    entropy = geometric_product_entropy(multivector, nodes, edges, root)
    print("Geometric product entropy:", entropy)
    # calculate the decision hygiene scores
    scores = decision_hygiene_scoring(nodes, edges, root)
    print("Decision hygiene scores:", scores)


if __name__ == "__main__":
    main()