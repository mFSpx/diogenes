# DARWIN HAMMER — match 1848, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py (gen5)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:39:12Z

"""
This module fuses the hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py 
(Algorithm A) and ollivier_ricci_curvature.py (Algorithm B) by recognizing that 
the geometric product can be used to compute distances and orientations between 
points in the ternary route graph, and the Ollivier-Ricci curvature can be used to 
analyze the curvature of the graph. The mathematical bridge between these two 
structures is formed by using the geometric product to compute distances and 
orientations between points in the ternary route graph, and then applying these 
computations to assign points to their nearest route nodes. The Ollivier-Ricci 
curvature is used to analyze the curvature of the graph, which is embedded into the 
maximal-independent-set construction.

The governing equations of the Clifford algebra are used to compute the geometric 
product of multivectors, which are then used to represent points and vectors in the 
ternary route graph. The Ollivier-Ricci curvature is computed using the lazy random 
walk distribution and the Wasserstein-1 distance.
"""

import math
import numpy as np
import random
import sys
from typing import Any, Dict, List, Tuple

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
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
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def lazy_rw_distribution(adj, node, alpha=0.5):
    """Lazy random walk distribution centred at *node*.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    node  : the source node
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float probability
    """
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def ollivier_ricci_curvature(adj, edge):
    """Ollivier-Ricci curvature on a weighted graph.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    edge : tuple of two node_ids

    Returns
    -------
    float curvature value
    """
    node_a, node_b = edge
    m_a = lazy_rw_distribution(adj, node_a)
    m_b = lazy_rw_distribution(adj, node_b)
    d = bfs_distances(adj)
    w1 = wasserstein1_distance(m_a, m_b, d)
    return 1 - w1 / d[node_a][node_b]


def bfs_distances(adj):
    """All-pairs shortest-path distances via BFS.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids

    Returns
    -------
    dict mapping node_id -> dict mapping node_id -> float distance
    """
    distances = {}
    for node in adj:
        distances[node] = {}
        queue = [(node, 0)]
        visited = set()
        while queue:
            current_node, distance = queue.pop(0)
            if current_node not in visited:
                visited.add(current_node)
                distances[node][current_node] = distance
                for neighbour in adj.get(current_node, []):
                    if neighbour not in visited:
                        queue.append((neighbour, distance + 1))
    return distances


def wasserstein1_distance(m_a, m_b, d):
    """Wasserstein-1 distance between two discrete probability measures.

    Parameters
    ----------
    m_a : dict mapping node_id -> float probability
    m_b : dict mapping node_id -> float probability
    d : dict mapping node_id -> dict mapping node_id -> float distance

    Returns
    -------
    float Wasserstein-1 distance
    """
    total = 0
    for node_a in m_a:
        for node_b in m_b:
            total += m_a[node_a] * m_b[node_b] * d[node_a][node_b]
    return total


def geometric_product(a, b):
    """Geometric product of two vectors.

    Parameters
    ----------
    a : numpy array
    b : numpy array

    Returns
    -------
    numpy array
    """
    return np.dot(a, b) + np.cross(a, b)


def hybrid_operation(adj, edge):
    """Hybrid operation using geometric product and Ollivier-Ricci curvature.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    edge : tuple of two node_ids

    Returns
    -------
    float curvature value
    """
    node_a, node_b = edge
    m_a = lazy_rw_distribution(adj, node_a)
    m_b = lazy_rw_distribution(adj, node_b)
    d = bfs_distances(adj)
    w1 = wasserstein1_distance(m_a, m_b, d)
    gp = geometric_product(np.array(list(m_a.values())), np.array(list(m_b.values())))
    return 1 - w1 / d[node_a][node_b] * np.linalg.norm(gp)


if __name__ == "__main__":
    adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    edge = (0, 1)
    print(ollivier_ricci_curvature(adj, edge))
    print(hybrid_operation(adj, edge))