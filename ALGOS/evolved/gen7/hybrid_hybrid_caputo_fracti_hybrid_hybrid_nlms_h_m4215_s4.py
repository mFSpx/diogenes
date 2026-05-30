# DARWIN HAMMER — match 4215, survivor 4
# gen: 7
# parent_a: hybrid_caputo_fractional_minimum_cost_tree_m35_s5.py (gen1)
# parent_b: hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s0.py (gen6)
# born: 2026-05-29T23:54:23Z

"""
Hybrid Fractional‑Tree / Regret‑RBF Algorithm
================================================

Parents
-------
* **Parent A** – *hybrid_caputo_fractional_minimum_cost_tree_m35_s5.py*  
  Provides a Caputo fractional memory kernel (power‑law weighting) and
  minimum‑cost tree distance calculations.

* **Parent B** – *hybrid_hybrid_nlms_hybrid_h_hybrid_hybrid_hybrid_m1247_s0.py*  
  Supplies an RBF similarity kernel over node feature vectors and a
  regret‑based action selection mechanism.

Mathematical Bridge
-------------------
Both parents rely on **weighted sums over pairwise relationships**:

* In Parent A the pairwise relationship is a *fractional distance*  
  \(d_{ij}^{(\alpha)} = \frac{d_{ij}^{\alpha}}{\Gamma(\alpha+1)}\) where
  \(d_{ij}\) is the ordinary shortest‑path distance on a tree and
  \(\alpha\in(0,1]\) is the Caputo order.

* In Parent B the pairwise relationship is an *RBF kernel*  
  \(K_{ij}=e^{-(\varepsilon\|x_i-x_j\|)^2}\) built from node feature vectors.

The hybrid algorithm forms a **composite kernel**
\[
C_{ij}=K_{ij}\; d_{ij}^{(\alpha)},
\]
which simultaneously encodes feature similarity (RBF) and fractional
memory (Caputo).  Regret values of actions are then modulated by the
composite kernel to bias the selection toward actions whose associated
nodes are both similar in feature space and close in the fractional‑tree
metric.

The module implements this fusion and provides three public functions
demonstrating the hybrid operation.
"""

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
    C = K * D  # element‑wise multiplication
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

    For each action a associated with node id = a.id (assume ids match node keys),
    compute:
        w_a = regret_a * mean_j C_{ij}
    """
    raw_regrets = compute_regret(actions)
    weighted = {}
    for a in actions:
        idx = node_index.get(a.id)
        if idx is None:
            # If the action does not correspond to a node, fall back to raw regret
            weighted[a.id] = raw_regrets[a.id]
            continue
        similarity = composite_kernel[idx, :].mean()
        weighted[a.id] = raw_regrets[a.id] * similarity
    return weighted


def select_hybrid_action(actions: List[MathAction],
                         composite_kernel: np.ndarray,
                         node_order: List[str]) -> MathAction:
    """
    Choose the action with the highest kernel‑weighted regret.
    """
    node_index = {node: i for i, node in enumerate(node_order)}
    weighted_regrets = hybrid_regret_weights(actions, composite_kernel, node_index)
    best_id = max(weighted_regrets, key=weighted_regrets.get)
    # Retrieve the original MathAction object
    return next(a for a in actions if a.id == best_id)


def hybrid_operation(features: Dict[str, List[float]],
                    actions: List[MathAction],
                    edges: List[Tuple[Any, Any, float]],
                    alpha: float = 0.7,
                    epsilon: float = 1.0) -> MathAction:
    """
    End‑to‑end hybrid routine:
        1. Build composite kernel C = K ⊙ D^{(α)}.
        2. Compute regret‑weighted scores.
        3. Return the best action.

    Parameters
    ----------
    features : dict
        Mapping node_id → feature vector.
    actions : list[MathAction]
        Candidate actions; each action's ``id`` must correspond to a node id.
    edges : list of (u, v, weight)
        Tree describing the underlying topology.
    alpha : float, optional
        Fractional order for the Caputo kernel (default 0.7).
    epsilon : float, optional
        Width parameter for the RBF kernel (default 1.0).

    Returns
    -------
    MathAction
        The selected action.
    """
    C, node_order = composite_kernel(features, edges, alpha, epsilon)
    return select_hybrid_action(actions, C, node_order)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree (4 nodes)
    tree_edges = [
        ("A", "B", 1.0),
        ("B", "C", 2.0),
        ("B", "D", 1.5)
    ]

    # Feature vectors for each node (2‑D)
    node_features = {
        "A": [0.0, 0.0],
        "B": [1.0, 0.0],
        "C": [1.0, 2.0],
        "D": [2.0, 0.0]
    }

    # Define actions – assume each action is tied to a node id
    actions = [
        MathAction(id="A", tokens=("move", "A"), expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", tokens=("move", "B"), expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="C", tokens=("move", "C"), expected_value=12.0, cost=5.0, risk=2.0),
        MathAction(id="D", tokens=("move", "D"), expected_value=7.0, cost=1.0, risk=0.2)
    ]

    selected = hybrid_operation(node_features, actions, tree_edges,
                                alpha=0.6, epsilon=1.2)

    print(f"Selected action: {selected.id} with expected value {selected.expected_value}")