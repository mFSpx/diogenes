# DARWIN HAMMER — match 15, survivor 2
# gen: 2
# parent_a: bayes_update.py (gen0)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# born: 2026-05-29T23:22:43Z

"""Hybrid Bayesian-Ollivier Ricci Module

Parents:
- bayes_update.py (Algorithm A): Provides Bayesian marginalization and update formulas.
- hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (Algorithm B): Provides feature extraction, graph construction, and Ollivier‑Ricci curvature computation.

Mathematical Bridge:
Feature values are interpreted as prior probabilities on graph nodes. For each ordered pair (i, j) we compute a Bayesian posterior
    w_{ij} = P(H_i | E_j) = prior_i * likelihood_j / marginal_{ij},
where
    marginal_{ij} = likelihood_j * prior_i + false_positive_i * (1 - prior_i)
and false_positive_i = 1 - prior_i.
These posteriors become edge weights that define the adjacency of a graph. The resulting graph is fed into the Ollivier‑Ricci pipeline, thus fusing Bayesian evidence updating with curvature analysis.
"""

from __future__ import annotations

import random
import math
import sys
import pathlib
import numpy as np
from typing import Dict, Tuple, List, Set

# ----------------------------------------------------------------------
# Algorithm A – Bayesian primitives
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Return the marginal probability P(E) for a single hypothesis."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Return the posterior P(H|E) using Bayes' rule."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def bayes_edge_weight(prior: float, likelihood: float) -> float:
    """Convenient wrapper returning the posterior weight for an edge."""
    false_positive = 1.0 - prior
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)


# ----------------------------------------------------------------------
# Algorithm B – Feature extraction and graph utilities
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Generate a synthetic high‑dimensional feature dict."""
    features: Dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def extract_master_vector(text: str) -> Dict[str, float]:
    """Collapse the full feature set to a compact master vector."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "bureaucratic_weaponization_index": f.get("resilience_bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": f.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": f.get("resilience_swarm_orchestration_density", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


def lazy_rw_distribution(adj: Dict[str, List[str]], node: str, alpha: float = 0.5) -> Dict[str, float]:
    """Return the lazy random‑walk distribution from a node."""
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def bfs_distances(adj: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    """All‑pairs shortest‑path distances via BFS (unweighted)."""
    node_ids = sorted(adj.keys())
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    for src in node_ids:
        si = idx[src]
        visited: Set[str] = {src}
        queue: List[Tuple[str, int]] = [(nb, 1) for nb in adj.get(src, [])]
        while queue:
            node, dist = queue.pop(0)
            if D[si, idx[node]] > dist:
                D[si, idx[node]] = dist
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
    return D, node_ids


def wasserstein1_graph(mu: Dict[str, float], nu: Dict[str, float],
                       D: np.ndarray, node_ids: List[str]) -> float:
    """Compute the 1‑Wasserstein distance between two discrete measures on a graph."""
    n = len(node_ids)
    idx = {v: i for i, v in enumerate(node_ids)}

    supply = np.array([mu.get(v, 0.0) for v in node_ids], dtype=np.float64)
    demand = np.array([nu.get(v, 0.0) for v in node_ids], dtype=np.float64)

    s_sum = supply.sum()
    d_sum = demand.sum()
    if s_sum > 0:
        supply /= s_sum
    if d_sum > 0:
        demand /= d_sum

    # Create sorted list of (cost, i, j)
    costs = [(D[i, j], i, j) for i in range(n) for j in range(n)]
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


def ollivier_ricci(adj: Dict[str, List[str]], alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
    """Compute Ollivier‑Ricci curvature for every edge of the (unweighted) graph."""
    D, node_ids = bfs_distances(adj)
    curvatures: Dict[Tuple[str, str], float] = {}

    seen: Set[Tuple[str, str]] = set()
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


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_bayesian_graph(master: Dict[str, float], threshold: float = 0.1) -> Dict[str, List[str]]:
    """
    Construct an undirected adjacency list where an edge (i, j) exists
    iff the Bayesian posterior weight w_{ij} exceeds *threshold*.
    The weight itself is not stored (the curvature routine treats the graph as unweighted),
    but the existence of the edge encodes Bayesian evidence.
    """
    nodes = list(master.keys())
    adj: Dict[str, List[str]] = {node: [] for node in nodes}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            w = bayes_edge_weight(prior=master[i], likelihood=master[j])
            if w > threshold:
                adj[i].append(j)
    return adj


def hybrid_curvature(master: Dict[str, float],
                     edge_threshold: float = 0.1,
                     alpha: float = 0.5) -> Dict[Tuple[str, str], float]:
    """
    End‑to‑end hybrid operation:
    1. Build a Bayesian‑derived graph from *master*.
    2. Compute Ollivier‑Ricci curvature on that graph.
    Returns a mapping from edge tuples to curvature values.
    """
    adj = build_bayesian_graph(master, threshold=edge_threshold)
    return ollivier_ricci(adj, alpha=alpha)


def hybrid_summary(master: Dict[str, float],
                   edge_threshold: float = 0.1,
                   alpha: float = 0.5) -> Tuple[int, float, float]:
    """
    Return a quick statistical summary of the hybrid structure:
    - number of nodes,
    - number of edges,
    - average curvature over all edges (0 if no edges).
    """
    adj = build_bayesian_graph(master, threshold=edge_threshold)
    curv = ollivier_ricci(adj, alpha=alpha)
    num_nodes = len(adj)
    num_edges = len(curv)
    avg_curv = sum(curv.values()) / num_edges if num_edges > 0 else 0.0
    return num_nodes, num_edges, avg_curv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy_text = "placeholder input for feature extraction"
    master_vec = extract_master_vector(dummy_text)

    # Run hybrid curvature computation
    curvatures = hybrid_curvature(master_vec, edge_threshold=0.15, alpha=0.6)

    # Print a few results
    print(f"Computed curvature on {len(curvatures)} edges.")
    for edge, kappa in list(curvatures.items())[:5]:
        print(f"Edge {edge}: curvature = {kappa:.4f}")

    # Summary
    nodes, edges, avg = hybrid_summary(master_vec, edge_threshold=0.15, alpha=0.6)
    print(f"\nSummary -> Nodes: {nodes}, Edges: {edges}, Avg curvature: {avg:.4f}")