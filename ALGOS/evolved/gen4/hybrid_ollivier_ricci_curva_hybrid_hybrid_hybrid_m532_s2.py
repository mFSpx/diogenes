# DARWIN HAMMER — match 532, survivor 2
# gen: 4
# parent_a: ollivier_ricci_curvature.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m309_s1.py (gen3)
# born: 2026-05-29T23:29:26Z

"""
Module hybrid_ollivier_ricci_rbf: A fusion of the Ollivier-Ricci
curvature algorithm from ollivier_ricci_curvature.py with the radial-basis
surrogate model from hybrid_hybrid_bandit_rbf_router.py. The mathematical bridge
between the two structures lies in the use of the Wasserstein-1 (Earth Mover
Distance) as a loss function in the radial basis function surrogate model, and
the application of the Ollivier-Ricci curvature to guide the exploration of the
surrogate model's confidence bounds.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import pathlib
from collections import defaultdict

Vector = Sequence[float]


def ollivier_rw_distribution(adj, node, alpha=0.5):
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


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            m[row][col] = -m[row][col] / div
            for i in range(col + 1, n):
                m[row][i] += m[col][i] * m[row][col]
    for row in range(n):
        for i in range(n + 1):
            if i == n:
                b[row] /= m[row][n]
            else:
                m[row][i] /= m[row][n]
    return [m[row][n] for row in range(n)]


def calculate_ollivier_ricci_curvature(adj, alpha=0.5):
    """Compute the Ollivier-Ricci curvature on a graph.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping edge (x, y) -> float Ollivier-Ricci curvature
    """
    bfs_dist = bfs_distances(adj)
    curvatures = {}
    for x in adj:
        dist_x = lazy_rw_distribution(adj, x, alpha)
        for y in adj:
            if y == x:
                continue
            d_xy = bfs_dist[x][y]
            m_x = lazy_rw_distribution(adj, x, alpha)
            m_y = lazy_rw_distribution(adj, y, alpha)
            w_1 = wasserstein_1(m_x, m_y, bfs_dist)
            curvatures[(x, y)] = 1 - w_1 / d_xy
    return curvatures


def radial_basis_function(x: Vector, centers: list[Vector], sigmas: list[float]) -> float:
    """Radial basis function surrogate model.

    Parameters
    ----------
    x : input vector
    centers : list of centers for the radial basis functions
    sigmas : list of standard deviations for the radial basis functions

    Returns
    -------
    float value of the radial basis function
    """
    phi = 0.0
    for center, sigma in zip(centers, sigmas):
        phi += gaussian(euclidean(x, center), sigma)
    return phi


def hybrid_ollivier_rbf(adj, alpha=0.5, centers=None, sigmas=None):
    """Hybrid Ollivier-Ricci curvature and radial basis function surrogate model.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)
    centers : list of centers for the radial basis functions
    sigmas : list of standard deviations for the radial basis functions

    Returns
    -------
    dict mapping node_id -> float value of the hybrid model
    """
    curvatures = calculate_ollivier_ricci_curvature(adj, alpha)
    scores = {}
    for x in adj:
        dist_x = lazy_rw_distribution(adj, x, alpha)
        score = 0.0
        for y in adj:
            if y == x:
                continue
            d_xy = bfs_distances(adj)[x][y]
            m_x = lazy_rw_distribution(adj, x, alpha)
            m_y = lazy_rw_distribution(adj, y, alpha)
            w_1 = wasserstein_1(m_x, m_y, bfs_distances(adj))
            curv = curvatures[(x, y)]
            if curv > 0:
                score += radial_basis_function(dist_x, centers, sigmas)
            else:
                score -= radial_basis_function(dist_x, centers, sigmas)
        scores[x] = score
    return scores


def wasserstein_1(m_x: dict, m_y: dict, dist: dict) -> float:
    """Wasserstein-1 (Earth Mover Distance) between two discrete probability measures.

    Parameters
    ----------
    m_x : dict mapping node_id -> float probability
    m_y : dict mapping node_id -> float probability
    dist : dict mapping node_id -> list of node_ids with distances

    Returns
    -------
    float value of the Wasserstein-1 distance
    """
    sorted_nodes = sorted(m_x, key=lambda x: m_x[x], reverse=True)
    cost_matrix = np.zeros((len(sorted_nodes), len(m_y)))
    for i, x in enumerate(sorted_nodes):
        for j, y in enumerate(m_y):
            cost_matrix[i, j] = dist[x][y]
    sorted_m_y = sorted(m_y, key=lambda y: m_y[y], reverse=True)
    T = np.zeros((len(sorted_nodes), len(m_y)))
    for i in range(len(sorted_nodes)):
        for j in range(len(m_y)):
            if i == 0 and j == 0:
                T[i, j] = 1.0
            else:
                best_j = np.argmax(cost_matrix[i, :])
                T[i, best_j] = min(1.0, T[i - 1, best_j] + cost_matrix[i, best_j] - cost_matrix[i - 1, best_j])
    return np.sum(cost_matrix * T)


def bfs_distances(adj):
    """All-pairs shortest-path distances via BFS.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids

    Returns
    -------
    dict mapping node_id -> dict mapping node_id -> int distance
    """
    dist = {node: {node: 0} for node in adj}
    for node in adj:
        queue = deque([(node, 0)])
        while queue:
            x, d = queue.popleft()
            for y in adj[x]:
                if y not in dist[x] or dist[x][y] > d + 1:
                    dist[x][y] = d + 1
                    queue.append((y, d + 1))
    return dist


if __name__ == "__main__":
    adj = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A', 'D'],
        'D': ['B', 'C']
    }
    alpha = 0.5
    centers = [[0.0, 0.0], [1.0, 1.0]]
    sigmas = [0.1, 0.1]
    print(hybrid_ollivier_rbf(adj, alpha, centers, sigmas))