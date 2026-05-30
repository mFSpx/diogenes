# DARWIN HAMMER — match 1848, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py (gen5)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:39:12Z

"""
This module fuses the hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py 
and ollivier_ricci_curvature.py by recognizing that the geometric product can be 
used to compute distances and orientations between points in the ternary route 
graph, and the Ollivier-Ricci curvature can be used to analyze the connectivity 
of the graph.

The mathematical bridge between these two structures is formed by using the 
geometric product to compute distances and orientations between points in the 
ternary route graph, and then applying the Ollivier-Ricci curvature to analyze 
the connectivity of the graph. The lazy random walk distribution is used to 
compute the mass kept at each node, and the geometric product is used to compute 
the distances and orientations between points in the graph.
"""

import math
import numpy as np
import random
import sys
from collections import deque
from typing import Any, Dict, List, Tuple

def _blade_sign(indices):
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
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def lazy_rw_distribution(adj, node, alpha=0.5):
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist

def bfs_distances(adj):
    distances = {node: {node: 0 for node in adj} for node in adj}
    for start in adj:
        queue = deque([(start, 0)])
        visited = set()
        while queue:
            node, dist = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbour in adj[node]:
                    if neighbour not in visited:
                        queue.append((neighbour, dist + 1))
                        distances[start][neighbour] = dist + 1
    return distances

def ollivier_ricci_curvature(adj, node1, node2, alpha=0.5):
    distances = bfs_distances(adj)
    mx = lazy_rw_distribution(adj, node1, alpha)
    my = lazy_rw_distribution(adj, node2, alpha)
    wx = sum([mx[node] * distances[node1][node] for node in mx])
    wy = sum([my[node] * distances[node2][node] for node in my])
    kappa = 1 - wx / wy
    return kappa

def hybrid_geometric_product(adj, node1, node2):
    blade_a = frozenset([node1])
    blade_b = frozenset([node2])
    result, sign = _multiply_blades(blade_a, blade_b)
    return result, sign

def hybrid_ollivier_ricci_curvature(adj, node1, node2, alpha=0.5):
    result, sign = hybrid_geometric_product(adj, node1, node2)
    kappa = ollivier_ricci_curvature(adj, node1, node2, alpha)
    return kappa, result, sign

if __name__ == "__main__":
    adj = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    node1 = 0
    node2 = 1
    kappa, result, sign = hybrid_ollivier_ricci_curvature(adj, node1, node2)
    print(f"Ollivier-Ricci Curvature: {kappa}")
    print(f"Geometric Product: {result}")
    print(f"Sign: {sign}")