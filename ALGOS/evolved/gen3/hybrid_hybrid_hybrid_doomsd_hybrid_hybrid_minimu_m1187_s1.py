# DARWIN HAMMER — match 1187, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py (gen2)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1.py (gen2)
# born: 2026-05-29T23:33:15Z

"""
This module integrates the hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0 and 
hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s1 algorithms into a single hybrid system.
The mathematical bridge is formed by using the geometric product of multivectors to represent 
the distances and orientations between decision nodes in the minimum cost tree, and then using 
the expected cost as a weight for the decision hygiene scores based on the Gini coefficient 
and Voronoi partitioning. This allows for a unified framework to analyze the complexity and 
inequality of decision-making processes.
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


def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(nodes, edges, root):
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

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
                # identify the 
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)
    return adj, edge_len, dist


def hybrid_geometric_product(node_a, node_b, blades):
    """
    Compute the geometric product of two nodes in the decision tree.

    Parameters
    ----------
    node_a : Point
        The first node
    node_b : Point
        The second node
    blades : List[frozenset]
        The basis blades for the geometric product

    Returns
    -------
    product : Multivector
        The geometric product of the two nodes
    """
    combined = list(node_a) + list(node_b)
    result, sign = _blade_sign(combined)
    return Multivector([frozenset(result)]), sign


def decision_hygiene_score(node, blades, edges, nodes):
    """
    Compute the decision hygiene score for a given node.

    Parameters
    ----------
    node : str
        The node to compute the score for
    blades : List[frozenset]
        The basis blades for the geometric product
    edges : List[Edge]
        The edges of the decision tree
    nodes : Dict[str, Point]
        The nodes of the decision tree

    Returns
    -------
    score : float
        The decision hygiene score for the node
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, node)
    scores = []
    for nxt in adj[node]:
        product, sign = hybrid_geometric_product(nodes[node], nodes[nxt], blades)
        scores.append(sign * length(nodes[node], nodes[nxt]))
    return np.mean(scores)


def expected_cost(tree_metrics, blades):
    """
    Compute the expected cost of the decision tree.

    Parameters
    ----------
    tree_metrics : Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]
        The tree metrics computed by tree_metrics
    blades : List[frozenset]
        The basis blades for the geometric product

    Returns
    -------
    cost : float
        The expected cost of the decision tree
    """
    adj, edge_len, dist = tree_metrics
    cost = 0
    for edge in edge_len:
        product, sign = hybrid_geometric_product(edge[0], edge[1], blades)
        cost += sign * edge_len[edge]
    return cost


if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (0, 1)}
    edges = [("A", "B"), ("A", "C")]
    blades = [frozenset([0, 1])]
    adj, edge_len, dist = tree_metrics(nodes, edges, "A")
    product, sign = hybrid_geometric_product(nodes["A"], nodes["B"], blades)
    score = decision_hygiene_score("A", blades, edges, nodes)
    cost = expected_cost((adj, edge_len, dist), blades)
    print("Decision hygiene score:", score)
    print("Expected cost:", cost)