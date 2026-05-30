# DARWIN HAMMER — match 5386, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1398_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2471_s0.py (gen5)
# born: 2026-05-30T00:01:41Z

import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple, Iterable


def stylometry_feature_vector(text: str) -> np.ndarray:
    """
    Produce a 3‑dimensional stylometry feature vector for the whole text.
    The three dimensions count first‑person, second‑person and third‑person
    pronouns respectively, normalised by the total number of words.
    """
    pronouns = {
        0: {"i", "me", "my", "mine", "myself"},
        1: {"you", "your", "yours", "yourself"},
        2: {"he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves"},
    }
    words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
    counts = np.zeros(3, dtype=float)

    for w in words:
        for idx, group in pronouns.items():
            if w in group:
                counts[idx] += 1
                break

    total = max(len(words), 1)
    return counts / total


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build an undirected adjacency list, compute Euclidean edge lengths for
    both directions, and compute the distance from *root* to every node
    (sum of edge lengths along the unique simple path).

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping ordered edge (u, v) → length
    root_dist : dict mapping node → distance from *root*
    """
    # adjacency
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        d = euclidean(nodes[u], nodes[v])
        edge_len[(u, v)] = d
        edge_len[(v, u)] = d  # store both orientations for easy lookup

    # BFS to compute root distances and parent pointers
    root_dist: Dict[str, float] = {root: 0.0}
    parent: Dict[str, str] = {root: ""}

    q = deque([root])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in root_dist:
                root_dist[nb] = root_dist[cur] + edge_len[(cur, nb)]
                parent[nb] = cur
                q.append(nb)

    # store parent map for later path reconstruction (optional)
    # we expose it via a closure in `path_to_root` below
    def path_to_root(node: str) -> List[str]:
        path = []
        while node != root:
            path.append(node)
            node = parent[node]
        path.append(root)
        path.reverse()
        return path

    # attach helper to the returned adjacency for callers that need it
    adj["_path_to_root"] = path_to_root  # type: ignore

    return adj, edge_len, root_dist


def _extract_path(adj: Dict[str, List[str]], node: str, root: str) -> List[str]:
    """
    Retrieve the stored helper from `tree_metrics` and use it.
    """
    helper = adj.get("_path_to_root")  # type: ignore
    if callable(helper):
        return helper(node)  # type: ignore
    # Fallback (should never happen)
    raise RuntimeError("Path helper missing from adjacency structure.")


def allocate_workshare_ssim(
    x: np.ndarray,
    y: np.ndarray,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    *,
    total_units: float,
) -> Dict[str, float]:
    """
    Distribute `total_units` across the nodes according to the structural
    similarity (SSIM‑like metric) between vectors `x` and `y`.

    The share for a node is proportional to the sum of edge lengths on the
    unique path from the node to the root, scaled by the similarity measure.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    # a simple similarity measure (the original code used a SSIM‑like formula)
    sim = np.sum(np.abs(x - y)) / (np.sum(np.abs(x)) + np.sum(np.abs(y)) + 1e-12)

    adj, edge_len, _ = tree_metrics(nodes, edges, root)

    shares: Dict[str, float] = {n: 0.0 for n in nodes}
    for node in nodes:
        if node == root:
            continue
        path = _extract_path(adj, node, root)
        path_len = sum(edge_len[(path[i], path[i + 1])] for i in range(len(path) - 1))
        shares[node] = path_len * sim

    # Normalise to `total_units`
    total_share = sum(shares.values()) + 1e-12
    for n in shares:
        shares[n] = shares[n] / total_share * total_units

    return shares


def hybrid_conductance_update(
    conductance: np.ndarray,
    feature_vector: np.ndarray,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> np.ndarray:
    """
    Update a conductance vector by adding a tree‑based contribution
    (edge‑length weighted by a similarity factor) and the stylometry
    feature vector. The dimensions of `conductance` and `feature_vector`
    must match.
    """
    if conductance.shape != feature_vector.shape:
        raise ValueError("conductance and feature_vector must have the same shape")

    # Re‑use the similarity metric from `allocate_workshare_ssim`
    # Here we simply take the mean of conductance as a proxy for similarity.
    sim = np.mean(conductance) / (np.max(conductance) + 1e-12)

    adj, edge_len, _ = tree_metrics(nodes, edges, root)

    tree_contrib = np.zeros_like(conductance, dtype=float)

    # Map each node to an index in the conductance vector.
    # For demonstration we order nodes alphabetically.
    node_list = sorted(nodes.keys())
    node_index = {n: i for i, n in enumerate(node_list)}

    for node in nodes:
        if node == root:
            continue
        path = _extract_path(adj, node, root)
        path_len = sum(edge_len[(path[i], path[i + 1])] for i in range(len(path) - 1))
        idx = node_index[node] % conductance.size  # wrap if vector shorter than node count
        tree_contrib[idx] += path_len * sim

    updated = conductance + tree_contrib + feature_vector
    return np.maximum(updated, 0.0)


if __name__ == "__main__":
    # Example graph
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"

    # Sample vectors
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # Stylometry feature vector (3‑dimensional)
    fv = stylometry_feature_vector("I love you and he loves them.")

    # Allocate workshare
    workshare = allocate_workshare_ssim(
        x, y, nodes, edges, root, total_units=10.0
    )
    print("Workshare per node:", workshare)

    # Conductance update
    conductance = np.array([1.0, 2.0, 3.0])
    updated_conductance = hybrid_conductance_update(
        conductance, fv, nodes, edges, root
    )
    print("Updated conductance:", updated_conductance)