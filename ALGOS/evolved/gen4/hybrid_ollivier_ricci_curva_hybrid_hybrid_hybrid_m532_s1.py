# DARWIN HAMMER — match 532, survivor 1
# gen: 4
# parent_a: ollivier_ricci_curvature.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m309_s1.py (gen3)
# born: 2026-05-29T23:29:26Z

"""
Module ollivier_ricci_rbf_hybrid: A fusion of the Ollivier-Ricci curvature 
algorithm from ollivier_ricci_curvature.py with the radial-basis surrogate 
model from hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m309_s1.py. 
The mathematical bridge between the two structures lies in the use of 
Wasserstein-1 distance as a ground cost for the Ollivier-Ricci curvature 
calculation and the application of radial basis functions to model 
the signal scores and noise scores from the surrogate model.

The hybrid algorithm therefore:

1. **Computes** the Ollivier-Ricci curvature for a given graph.
2. **Models** the signal scores and noise scores using radial basis functions.
3. **Injects** the curvature-derived term into the radial basis function model.

The result is a unified system where graph structure and 
singular-learning-theory asymptotics are combined to guide 
exploration-exploitation balances and enhance robustness to 
duplicate or similar data.
"""

import numpy as np
import math
import random
from collections import defaultdict

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


def wasserstein1_distance(mu, nu, D):
    """Wasserstein-1 distance between two discrete measures.

    Parameters
    ----------
    mu   : dict mapping node_id -> float probability
    nu   : dict mapping node_id -> float probability
    D    : dict mapping (node_id, node_id) -> float distance

    Returns
    -------
    float
    """
    T = {}
    for i in mu:
        for j in nu:
            T[(i, j)] = min(mu.get(i, 0), nu.get(j, 0))
    w1 = 0
    for (i, j), mass in T.items():
        w1 += D[(i, j)] * mass
    return w1


def ollivier_ricci_curvature(adj, alpha=0.5):
    """Ollivier-Ricci curvature for a given graph.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping (node_id, node_id) -> float curvature
    """
    kappa = {}
    D = {}
    for node in adj:
        D[node] = {node: 0}
        queue = [(node, 0)]
        visited = set()
        while queue:
            (current, dist) = queue.pop(0)
            if current not in visited:
                visited.add(current)
                for neighbour in adj[current]:
                    if neighbour not in D[node]:
                        D[node][neighbour] = dist + 1
                        queue.append((neighbour, dist + 1))
    for x in adj:
        for y in adj:
            if x != y:
                mx = lazy_rw_distribution(adj, x, alpha)
                my = lazy_rw_distribution(adj, y, alpha)
                w1 = wasserstein1_distance(mx, my, D[x])
                d = D[x].get(y, float('inf'))
                kappa[(x, y)] = 1 - w1 / d
    return kappa


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def radial_basis_model(x: list[float], centres: list[list[float]], widths: list[float], weights: list[float]) -> float:
    result = 0
    for centre, width, weight in zip(centres, widths, weights):
        result += weight * gaussian(euclidean(x, centre) / width)
    return result


def hybrid_ollivier_ricci_rbf(adj, alpha, centres, widths, weights):
    kappa = ollivier_ricci_curvature(adj, alpha)
    model = lambda x: radial_basis_model(x, centres, widths, weights)
    def curvature_informed_model(x):
        curvature = 0
        for y in adj:
            curvature += kappa.get((x, y), 0)
        return model(x) * (1 + curvature)
    return curvature_informed_model


if __name__ == "__main__":
    adj = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    alpha = 0.5
    centres = [[0, 0], [1, 1], [2, 2]]
    widths = [1.0, 1.0, 1.0]
    weights = [1.0, 1.0, 1.0]
    model = hybrid_ollivier_ricci_rbf(adj, alpha, centres, widths, weights)
    print(model([0.5, 0.5]))