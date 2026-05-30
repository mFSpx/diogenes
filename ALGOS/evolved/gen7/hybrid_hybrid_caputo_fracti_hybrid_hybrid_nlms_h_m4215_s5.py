# DARWIN HAMMER — match 4215, survivor 5
# gen: 7
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s5.py (gen1)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s0.py (gen6)
# born: 2026-05-29T23:54:23Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Lanczos Gamma and Caputo fractional kernel
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real z>0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        a = _LANCZOS_C[0]
        for i in range(1, len(_LANCZOS_C)):
            a += _LANCZOS_C[i] / (z + i - 1)
        t = z + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * a


def caputo_fractional_weight(distance: float, alpha: float) -> float:
    """
    Power‑law memory weight derived from the Caputo kernel.
    Implements  d^{alpha} / Gamma(alpha+1) .
    """
    if distance < 0:
        raise ValueError("distance must be non‑negative")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must lie in (0,1]")
    return (distance ** alpha) / gamma_lanczos(alpha + 1)


# ----------------------------------------------------------------------
# Parent B – RBF kernel and regret machinery
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action descriptor used by the regret component."""
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_kernel_matrix(features: Dict[str, List[float]],
                      epsilon: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build the symmetric RBF kernel matrix K_{ij}=exp(-(ε‖x_i−x_j‖)^2).

    Returns
    -------
    K : np.ndarray
        (n,n) kernel matrix.
    nodes : List[str]
        Ordering of node identifiers corresponding to rows/columns of K.
    """
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes


def compute_regret(actions: List[MathAction]) -> Dict[str, float]:
    """
    Simple regret definition: expected_value − cost − risk.
    """
    return {a.id: a.expected_value - a.cost - a.risk for a in actions}


# ----------------------------------------------------------------------
# Hybrid core – tree handling and composite kernel construction
# ----------------------------------------------------------------------
def _build_adjacency(edges: List[Tuple[Any, Any, float]]) -> Dict[Any, List[Tuple[Any, float]]]:
    """Utility to convert edge list to adjacency list."""
    adj: Dict[Any, List[Tuple[Any, float]]] = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    return adj


def _dijkstra(adj: Dict[Any, List[Tuple[Any, float]]],
              source: Any) -> Dict[Any, float]:
    """Single‑source shortest‑path distances using Dijkstra (non‑negative weights)."""
    import heapq
    dist: Dict[Any, float] = {node: math.inf for node in adj}
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def fractional_tree_distance_matrix(edges: List[Tuple[Any, Any, float]],
                                   nodes: List[Any],
                                   alpha: float) -> np.ndarray:
    """
    Compute the pairwise fractional distance matrix D^{(α)} for a tree.

    Parameters
    ----------
    edges : list of (u, v, weight)
        Undirected tree edges.
    nodes : list of node identifiers (order defines matrix rows/cols)
    alpha : fractional order (0 < alpha ≤ 1)

    Returns
    -------
    D : np.ndarray, shape (n,n)
        Fractional distance matrix where
        D_{ij} = caputo_fractional_weight(shortest_path_length(i,j), alpha)
    """
    adj = _build_adjacency(edges)
    n = len(nodes)
    D = np.empty((n, n), dtype=np.float64)

    for i, src in enumerate(nodes):
        dists = _dijkstra(adj, src)
        for j, dst in enumerate(nodes):
            D[i, j] = caputo_fractional_weight(dists[dst], alpha)
    return D


def composite_kernel(features: Dict[str, List[float]],
                     edges: List[Tuple[Any, Any, float]],
                     alpha: float,
                     epsilon: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """
    Build the hybrid kernel C = K ⊙ D^{(α)} (element‑wise product).

    Returns
    -------
    C : np.ndarray
        Composite kernel matrix.
    nodes : List[str]
        Node ordering (identical to the ordering used for the RBF kernel).
    """
    K, nodes = rbf_kernel_matrix(features, epsilon)
    D = fractional_tree_distance_matrix(edges, nodes, alpha)
    C = np.multiply(K, D)  # element‑wise multiplication
    return C, nodes


# ----------------------------------------------------------------------
# Hybrid action selection using the composite kernel
# ----------------------------------------------------------------------
def hybrid_regret_weights(actions: List[MathAction],
                         composite_kernel: np.ndarray,
                         node_index: Dict[str, int]) -> Dict[str, float]:
    """
    Modulate raw regrets by the average similarity of the action's node
    to all other nodes, as expressed by the composite kernel.

    For each action a associated with node id = a.id 
    """
    n = len(actions)
    modulated_regrets = {}

    for action in actions:
        node_id = action.id
        node_idx = node_index[node_id]
        avg_similarity = np.mean(composite_kernel[node_idx])
        raw_regret = compute_regret([action])[node_id]
        modulated_regret = raw_regret * avg_similarity
        modulated_regrets[node_id] = modulated_regret

    return modulated_regrets


def stable_hybrid_regret_decision(actions: List[MathAction],
                                  composite_kernel: np.ndarray,
                                  node_index: Dict[str, int],
                                  epsilon: float = 1e-6) -> str:
    """
    Select action with maximum modulated regret.

    To avoid non‑unique choices due to numerical issues, 
    select with a stable tolerance.
    """
    modulated_regrets = hybrid_regret_weights(actions, composite_kernel, node_index)
    best_action_id = max(modulated_regrets, key=modulated_regrets.get)
    best_regret = modulated_regrets[best_action_id]

    # Select with stable tolerance to avoid ties
    stable_best_actions = [a_id for a_id, regret in modulated_regrets.items() 
                          if abs(regret - best_regret) < epsilon]
    return random.choice(stable_best_actions)