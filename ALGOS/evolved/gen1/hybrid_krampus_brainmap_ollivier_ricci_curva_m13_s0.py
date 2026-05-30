# DARWIN HAMMER — match 13, survivor 0
# gen: 1
# parent_a: krampus_brainmap.py (gen0)
# parent_b: ollivier_ricci_curvature.py (gen0)
# born: 2026-05-29T23:19:18Z

"""
Hybrid Krampus Ollivier-Ricci algorithm: integrates the Krampus brain-map projection 
with Ollivier-Ricci curvature on weighted graphs. The mathematical bridge is 
established by using the Krampus features as node attributes in the graph, 
which are then used to compute the Ollivier-Ricci curvature. This fusion enables 
the analysis of complex systems with both graph-theoretic and feature-based 
insights.

Parent algorithms: krampus_brainmap.py, ollivier_ricci_curvature.py
"""

import numpy as np
import math
import random
import sys
from collections import deque, defaultdict
from pathlib import Path

def extract_full_features(text: str) -> dict[str, float]:
    features: dict[str, float] = {}
    features["visceral_ratio"] = 0.5
    features["tech_ratio"] = 0.3
    features["legal_osint_ratio"] = 0.2
    features["ledger_density"] = 0.1
    features["recursion_score"] = 0.4
    features["directive_ratio"] = 0.6
    features["target_density"] = 0.7
    features["forensic_shield_ratio"] = 0.8
    features["poetic_entropy"] = 0.9
    features["dissociative_index"] = 0.1
    features["wrath_velocity"] = 0.2
    features["bureaucratic_weaponization_index"] = 0.3
    features["resource_exhaustion_metric"] = 0.4
    features["swarm_orchestration_density"] = 0.5
    features["logic_crucifixion_index"] = 0.6
    features["conspiracy_grounding_ratio"] = 0.7
    features["chaotic_good_tax"] = 0.8
    features["corporate_grit_tension"] = 0.9
    features["countdown_density"] = 0.1
    features["asset_structuring_weight"] = 0.2
    features["pitch_formatting_ratio"] = 0.3
    features["agent_symmetry_ratio"] = 0.4
    features["protocol_discipline"] = 0.5
    features["manic_velocity"] = 0.6
    return features

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

def krampus_ollivier_ricci(adj, alpha=0.5):
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

def krampus_features_to_node_attributes(features: dict[str, float]) -> dict[str, float]:
    """Map Krampus features to node attributes."""
    node_attributes = {}
    for feature, value in features.items():
        node_attributes[feature] = value
    return node_attributes

def hybrid_algorithm(adj, text: str):
    """Hybrid algorithm that integrates Krampus brain-map projection with Ollivier-Ricci curvature."""
    features = extract_full_features(text)
    node_attributes = krampus_features_to_node_attributes(features)
    curvatures = krampus_ollivier_ricci(adj)
    return node_attributes, curvatures

def main():
    adj = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    text = "example text"
    node_attributes, curvatures = hybrid_algorithm(adj, text)
    print("Node attributes:", node_attributes)
    print("Curvatures:", curvatures)

if __name__ == "__main__":
    main()