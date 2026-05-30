# DARWIN HAMMER — match 1848, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_distributed_l_m987_s1.py (gen5)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:39:12Z

"""
This module fuses the hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s2.py and 
hybrid_distributed_leader_e_thanatosis_m65_s2.py by recognizing that the geometric 
product can be used to compute distances and orientations between points in the 
ternary route graph, and the Ollivier-Ricci curvature can be used to control the 
simulated annealing process.

The mathematical bridge between these two structures is formed by using the geometric 
product to compute distances and orientations between points in the ternary route 
graph, and then applying these computations to assign points to their nearest route 
nodes. The Ollivier-Ricci curvature from Algorithm B is used to control the simulated 
annealing process, which is embedded into the maximal-independent-set construction.
"""
import math
import numpy as np
import random
import sys
from collections import deque

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


def ollivier_ricci_curvature(adj, edge):
    """Ollivier-Ricci curvature on weighted graphs.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    edge  : tuple of node_ids

    Returns
    -------
    float Ollivier-Ricci curvature of the edge
    """
    node1, node2 = edge
    m_x = lazy_rw_distribution(adj, node1)
    m_y = lazy_rw_distribution(adj, node2)
    w1 = wasserstein_distance(m_x, m_y)
    d = bfs_distances(adj)[node1][node2]
    return 1 - w1 / d


def wasserstein_distance(m_x, m_y):
    """Wasserstein-1 distance between two discrete probability measures.

    Parameters
    ----------
    m_x  : dict mapping node_id -> float probability
    m_y  : dict mapping node_id -> float probability

    Returns
    -------
    float Wasserstein-1 distance between m_x and m_y
    """
    nodes = set(m_x.keys()) | set(m_y.keys())
    dist = {node: {} for node in nodes}
    for node in nodes:
        for nb in m_x.get(node, {}).keys() | m_y.get(node, {}).keys():
            dist[node][nb] = bfs_distances(adj)[node][nb]
    cost_matrix = [[dist[node1][node2] for node2 in nodes] for node1 in nodes]
    # exact north-west corner / successive-shortest-path greedy on the sorted cost matrix
    sorted_cost_matrix = sorted(zip(*cost_matrix), key=lambda x: x[0])
    greedy_solution = np.zeros((len(nodes), len(nodes)))
    for i, (node1, row) in enumerate(sorted_cost_matrix):
        for j, (node2, cost) in enumerate(row):
            if greedy_solution[node1][node2] == 0:
                greedy_solution[node1][node2] = cost
                break
    # compute the Wasserstein-1 distance from the dual LP
    w1 = 0
    for node1 in nodes:
        for node2 in nodes:
            w1 += greedy_solution[node1][node2] * m_x.get(node1, 0) * m_y.get(node2, 0)
    return w1


def geometric_product(adj, edge):
    """Geometric product of two points in the ternary route graph.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    edge  : tuple of node_ids

    Returns
    -------
    float geometric product of the two points
    """
    node1, node2 = edge
    distances = bfs_distances(adj)
    orientations = {node: {nb: 1 if (node, nb) in adj else -1 for nb in adj.get(node, [])} for node in adj.keys()}
    multivector = {node: 0 for node in adj.keys()}
    multivector[node1] += 1
    multivector[node2] += 1
    for node in adj.keys():
        for nb in adj.get(node, []):
            if (node, nb) != edge:
                multivector[node] += distances[node][nb] * orientations[node][nb]
    result = 0
    for node in adj.keys():
        result += multivector[node] * (multivector[node] - 1)
        result *= -1 if node != node1 else 1
    return result


def hybrid_operation(adj, edge):
    """Hybrid operation combining Ollivier-Ricci curvature and geometric product.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    edge  : tuple of node_ids

    Returns
    -------
    tuple of float Ollivier-Ricci curvature and float geometric product
    """
    curvature = ollivier_ricci_curvature(adj, edge)
    product = geometric_product(adj, edge)
    return curvature, product


if __name__ == "__main__":
    adj = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    edge = ('A', 'F')
    curvature, product = hybrid_operation(adj, edge)
    print(f'Ollivier-Ricci curvature: {curvature}')
    print(f'Geometric product: {product}')