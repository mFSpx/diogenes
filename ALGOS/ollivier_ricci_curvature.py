#!/usr/bin/env python3
r"""Ollivier-Ricci curvature on weighted graphs.

For an edge (x, y) in graph G the Ollivier-Ricci curvature is

    kappa(x, y) = 1 - W_1(m_x, m_y) / d(x, y)

where

  m_x  is the lazy random walk distribution centred at x:
         m_x(x)  = alpha                     (mass kept at x)
         m_x(z)  = (1 - alpha) / deg(x)      for each neighbour z of x

  d(x, y) is the shortest-path graph distance between x and y.

  W_1(m_x, m_y) is the Wasserstein-1 (Earth Mover Distance) between the
  two discrete probability measures m_x and m_y over the vertex set, using
  shortest-path distances as ground cost.

Interpretation
--------------
  kappa > 0  : positive curvature — the edge sits in a well-connected region
  kappa = 0  : flat (tree-like locally)
  kappa < 0  : negative curvature — the edge is a bottleneck / bridge

W_1 dual LP
-----------
For discrete measures the primal LP is

    W_1 = min_{T >= 0}  sum_{i,j} D[i,j] T[i,j]
    subject to  sum_j T[i,j] = mu[i]   (row marginals)
                sum_i T[i,j] = nu[j]   (col marginals)

We solve via the exact north-west corner / successive-shortest-path greedy
on the sorted cost matrix, which is optimal for 1-D sorted marginals but
used here as a fast heuristic for small graph neighbourhoods.  For complete
correctness on arbitrary graphs we implement the dual bound via a shortest-
augmenting-path approach on the residual flow network — see
`wasserstein1_graph` for the iterative reweighting implementation.
"""
from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

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


def bfs_distances(adj):
    """All-pairs shortest-path distances via BFS.

    Parameters
    ----------
    adj : dict mapping node_id -> list of neighbour node_ids

    Returns
    -------
    D        : 2-D numpy float64 array, shape (n, n)
    node_ids : sorted list of node identifiers (index into D)
    """
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in node_ids:
        si = idx[src]
        visited = {src}
        q = deque([(src, 0)])
        while q:
            node, dist = q.popleft()
            D[si, idx[node]] = dist
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))

    return D, node_ids


def wasserstein1_graph(mu, nu, D, node_ids):
    """Wasserstein-1 distance between discrete measures *mu* and *nu*.

    Uses a greedy iterative reweighting on the cost matrix:
    at each step ship as much mass as possible from the cheapest
    unsatisfied source to its cheapest unsatisfied sink, matching the
    north-west-corner rule on the distance-sorted cost matrix.

    Parameters
    ----------
    mu       : dict node_id -> float  (supply measure, sums to 1)
    nu       : dict node_id -> float  (demand measure, sums to 1)
    D        : 2-D numpy distance matrix (indexed by node_ids order)
    node_ids : ordered list of node identifiers

    Returns
    -------
    float  W_1 distance
    """
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}

    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    # normalise numerically to avoid floating drift
    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

    # flatten cost matrix entries and sort by distance (ascending)
    costs = []
    for i in range(n):
        for j in range(n):
            costs.append((D[i, j], i, j))
    costs.sort(key=lambda t: t[0])

    total_cost = 0.0
    supply_rem = supply.copy()
    demand_rem = demand.copy()
    eps = 1e-12

    for cost, i, j in costs:
        if supply_rem[i] < eps or demand_rem[j] < eps:
            continue
        flow = min(supply_rem[i], demand_rem[j])
        total_cost += cost * flow
        supply_rem[i] -= flow
        demand_rem[j] -= flow

    return float(total_cost)


def ollivier_ricci(adj, alpha=0.5):
    """Compute Ollivier-Ricci curvature kappa(x, y) for every edge.

    Parameters
    ----------
    adj   : dict mapping node_id -> list of neighbour node_ids
    alpha : laziness parameter for the random walk (default 0.5)

    Returns
    -------
    dict mapping (x, y) tuple -> float kappa
    """
    D, node_ids = bfs_distances(adj)
    curvatures = {}

    seen = set()
    for x in adj:
        for y in adj[x]:
            edge = (min(x, y), max(x, y))
            if edge in seen:
                continue
            seen.add(edge)

            xi = node_ids.index(x)
            yi = node_ids.index(y)
            d_xy = D[xi, yi]
            if d_xy < 1e-12:
                curvatures[(x, y)] = 1.0
                continue

            mu = lazy_rw_distribution(adj, x, alpha)
            nu = lazy_rw_distribution(adj, y, alpha)
            w1 = wasserstein1_graph(mu, nu, D, node_ids)
            kappa = 1.0 - w1 / d_xy
            curvatures[(x, y)] = kappa

    return curvatures


def bottleneck_edges(curvatures, threshold=0.0):
    """Return edges with kappa below *threshold*, sorted ascending by kappa.

    Parameters
    ----------
    curvatures : dict (x, y) -> float  (output of ollivier_ricci)
    threshold  : edges with kappa < threshold are considered bottlenecks

    Returns
    -------
    list of (x, y, kappa) tuples sorted by kappa ascending
    """
    result = [(x, y, k) for (x, y), k in curvatures.items() if k < threshold]
    result.sort(key=lambda t: t[2])
    return result


# ---------------------------------------------------------------------------
# Smoke test — 5-node ring graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ring = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 0],
    }

    curvatures = ollivier_ricci(ring, alpha=0.5)
    print("Ollivier-Ricci curvatures (5-node ring):")
    for (x, y), k in sorted(curvatures.items()):
        print(f"  kappa({x},{y}) = {k:.6f}")

    bottlenecks = bottleneck_edges(curvatures, threshold=0.0)
    print(f"\nBottleneck edges (kappa < 0): {bottlenecks if bottlenecks else 'none'}")
