# DARWIN HAMMER — match 4530, survivor 0
# gen: 5
# parent_a: hybrid_ollivier_ricci_curva_hybrid_hybrid_hybrid_m532_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fracti_hybrid_hybrid_hybrid_m1318_s0.py (gen4)
# born: 2026-05-29T23:56:17Z

"""
Module fusing the Ollivier-Ricci Curvature Algorithm from ollivier_ricci_curvature.py
with the Hybrid Fractional-Hoeffding Algorithm from hybrid_hybrid_fractional_hd_hybrid_hoeffding_tre_m12_s1.py.
The mathematical bridge lies in integrating the Hoeffding bound with the Ollivier-Ricci curvature calculation 
through the use of the Gini coefficient as a scaling factor for the Hoeffding bound and the Wasserstein-1 distance 
as a ground cost for the Ollivier-Ricci curvature calculation.
"""

import numpy as np
import math
import random
import sys
import pathlib

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

    .. note::
        This function uses the Wasserstein-1 distance as a ground cost for the calculation.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float curvature
    """
    mu = lazy_rw_distribution(adj, adj.keys()[0])
    nu = lazy_rw_distribution(adj, adj.keys()[1])
    D = {edge: math.sqrt(1) for edge in adj}
    w1 = wasserstein1_distance(mu, nu, D)
    curv = {}
    for node in adj:
        curv[node] = w1 / (math.sqrt(len(adj[node])) + math.sqrt(len(adj)))
    return curv


def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag**2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs)) / (n * sum(xs))


def hoeffding_bound_gini(r: float, delta: float, n: int, gini: float) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r + (gini / (2 * math.sqrt(n)))**2) / (2 * math.log(1 / delta)))


def hybrid_curvature(adj, alpha=0.5):
    """Hybrid curvature calculation combining Ollivier-Ricci and Hoeffding bound.

    .. note::
        This function uses the Gini coefficient as a scaling factor for the Hoeffding bound.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    alpha : mass kept at the node itself (laziness parameter)

    Returns
    -------
    dict mapping node_id -> float curvature
    """
    curv = ollivier_ricci_curvature(adj, alpha)
    gini = gini_coefficient([curv[node] for node in adj])
    r = math.sqrt(sum([curv[node] for node in adj]) / len(adj))
    delta = 0.1
    n = len(adj)
    bound = hoeffding_bound_gini(r, delta, n, gini)
    for node in adj:
        curv[node] = min(curv[node], bound)
    return curv


if __name__ == "__main__":
    adj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    alpha = 0.5
    curvature = hybrid_curvature(adj, alpha)
    print(curvature)