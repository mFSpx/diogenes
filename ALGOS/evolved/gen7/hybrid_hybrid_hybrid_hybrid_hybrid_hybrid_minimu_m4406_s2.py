# DARWIN HAMMER — match 4406, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_krampu_m1451_s1.py (gen6)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m1333_s2.py (gen5)
# born: 2026-05-29T23:55:32Z

import numpy as np
import math
from typing import Tuple, Dict, List, Iterable, Optional

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Gini coefficient
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient using the definition based on pairwise
    absolute differences.

    G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))

    Returns 0.0 for empty or constant inputs.
    """
    if values.size == 0:
        return 0.0

    values = values.astype(float)
    if np.allclose(values, values[0]):          # all values equal -> no inequality
        return 0.0

    n = values.size
    mean = np.mean(values)
    if np.isclose(mean, 0.0):
        # When the mean is (near) zero the classic definition blows up.
        # In this degenerate case we fall back to the rank‑based formulation.
        sorted_vals = np.sort(values)
        cumvals = np.cumsum(sorted_vals)
        gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
        return float(gini)

    # Pairwise absolute differences – vectorised for speed
    diff_sum = np.abs(values[:, None] - values).sum()
    gini = diff_sum / (2 * n * n * mean)
    return float(gini)

# ----------------------------------------------------------------------
# Tree cost utilities
# ----------------------------------------------------------------------
def _build_adjacency_list(
    nodes: Dict[str, Point],
    edges: List[Edge]
) -> Dict[str, List[str]]:
    """Create an undirected adjacency list from the edge list."""
    adj: Dict[str, List[str]] = {k: [] for k in nodes}
    for a, b in edges:
        if a not in nodes or b not in nodes:
            raise KeyError(f"Edge ({a}, {b}) references undefined node.")
        adj[a].append(b)
        adj[b].append(a)
    return adj

def _bfs_distances(
    adj: Dict[str, List[str]],
    nodes: Dict[str, Point],
    root: str
) -> Dict[str, float]:
    """Breadth‑first search yielding cumulative Euclidean distance from root."""
    if root not in nodes:
        raise KeyError(f"Root node '{root}' not present in node dictionary.")

    dist: Dict[str, float] = {root: 0.0}
    queue: List[str] = [root]

    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + euclidean_distance(nodes[cur], nodes[nxt])
                queue.append(nxt)

    # Verify connectivity – all nodes must be reachable in a valid tree
    if len(dist) != len(nodes):
        missing = set(nodes) - set(dist)
        raise ValueError(f"Graph is not fully connected; unreachable nodes: {missing}")

    return dist

def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2
) -> float:
    """
    Compute the total material cost (sum of edge lengths) plus a weighted
    path‑length penalty from the root to every node.

    Parameters
    ----------
    nodes : dict
        Mapping from node identifier to its (x, y) coordinates.
    edges : list of tuple
        Undirected edges connecting node identifiers.
    root : str
        Identifier of the root node.
    path_weight : float, optional
        Multiplicative factor applied to the sum of root‑to‑node distances.

    Returns
    -------
    float
        The combined cost.
    """
    adj = _build_adjacency_list(nodes, edges)

    # Material cost – sum of all unique edge lengths
    material = sum(euclidean_distance(nodes[a], nodes[b]) for a, b in edges)

    # Path‑length penalty
    distances = _bfs_distances(adj, nodes, root)
    path_penalty = path_weight * sum(distances.values())

    return material + path_penalty

# ----------------------------------------------------------------------
# Bayesian update
# ----------------------------------------------------------------------
def bayes_posterior(
    prior: float,
    likelihood: float,
    false_positive: float
) -> float:
    """
    Compute the posterior probability using Bayes' theorem:

        posterior = (likelihood * prior) /
                    (likelihood * prior + false_positive * (1 - prior))

    The function validates that all inputs lie in [0, 1] and guards against
    division by zero (e.g., when both numerator and denominator are zero).
    """
    for name, val in zip(("prior", "likelihood", "false_positive"), (prior, likelihood, false_positive)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]; got {val}")

    numerator = likelihood * prior
    denominator = numerator + false_positive * (1.0 - prior)

    if np.isclose(denominator, 0.0):
        # Degenerate case – return prior as a neutral estimate
        return prior

    return numerator / denominator

# ----------------------------------------------------------------------
# Hybrid scoring
# ----------------------------------------------------------------------
def hybrid_score(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior: float,
    likelihood: float,
    false_positive: float,
    alpha: float = 0.5,
    beta: float = 0.5
) -> float:
    """
    Compute a richer hybrid score that blends three orthogonal aspects:

    1. **Structural cost** – `tree_cost`.
    2. **Uncertainty** – Gini coefficient of edge lengths.
    3. **Evidence** – Bayesian posterior probability.

    The formula is:

        score = (α * tree_cost + β * tree_cost * gini) * posterior

    where α + β = 1 (default α = β = 0.5).  This keeps the score
    proportional to the underlying cost while allowing the Gini term
    to modulate it based on inequality of edge lengths.

    Parameters
    ----------
    alpha, beta : float, optional
        Weighting coefficients; they must sum to 1.  Adjust to favour
        pure cost (α → 1) or cost amplified by inequality (β → 1).

    Returns
    -------
    float
        The final hybrid score.
    """
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("alpha and beta must sum to 1.")

    # 1️⃣ Structural cost
    cost = tree_cost(nodes, edges, root)

    # 2️⃣ Edge‑length inequality
    edge_lengths = np.array([euclidean_distance(nodes[a], nodes[b]) for a, b in edges])
    gini = gini_coefficient(edge_lengths)

    # 3️⃣ Bayesian evidence
    posterior = bayes_posterior(prior, likelihood, false_positive)

    # Blend the components
    blended_cost = alpha * cost + beta * cost * gini
    return blended_cost * posterior

# ----------------------------------------------------------------------
# Graph construction helpers
# ----------------------------------------------------------------------
def build_adjacency(
    matrix: np.ndarray,
    threshold: float = 1.0
) -> List[Tuple[int, int]]:
    """
    Build an undirected adjacency list from a point matrix.
    An edge (i, j) is added when the Euclidean distance between rows i and j
    is strictly less than `threshold`.

    Parameters
    ----------
    matrix : (n, d) ndarray
        Coordinate matrix (each row is a point in ℝᵈ).
    threshold : float, optional
        Distance cutoff for edge creation.

    Returns
    -------
    list of tuple
        Edge list with integer node identifiers.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    n = matrix.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(matrix[i] - matrix[j]) < threshold:
                edges.append((i, j))
    return edges

def node_curvature(
    adj_list: List[Tuple[int, int]],
    matrix: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute a simple curvature proxy for each node: the average inverse
    distance to its immediate neighbours.  Nodes with many close neighbours
    obtain higher curvature values.

    Parameters
    ----------
    adj_list : list of tuple
        Edge list produced by `build_adjacency`.
    matrix : (n, d) ndarray
        Coordinate matrix.
    eps : float, optional
        Small constant to avoid division by zero when two points coincide.

    Returns
    -------
    ndarray
        Curvature values (shape = (n,)).
    """
    n = matrix.shape[0]
    curvature = np.zeros(n, dtype=float)

    # Build neighbour sets for quick lookup
    neighbours: List[List[int]] = [[] for _ in range(n)]
    for i, j in adj_list:
        neighbours[i].append(j)
        neighbours[j].append(i)

    for i in range(n):
        nb = neighbours[i]
        if not nb:
            curvature[i] = 0.0
            continue

        inv_dists = []
        for j in nb:
            dist = np.linalg.norm(matrix[i] - matrix[j])
            inv_dists.append(1.0 / (dist + eps))
        curvature[i] = sum(inv_dists) / len(nb)

    return curvature

# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple square graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0)
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    root = "A"

    # Bayesian parameters
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2

    score = hybrid_score(
        nodes,
        edges,
        root,
        prior,
        likelihood,
        false_positive,
        alpha=0.6,
        beta=0.4
    )
    print(f"Hybrid score: {score:.6f}")

    # Geometry‑based adjacency & curvature demo
    matrix = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    adj = build_adjacency(matrix, threshold=1.5)
    curv = node_curvature(adj, matrix)
    print("Adjacency list:", adj)
    print("Node curvature:", curv)