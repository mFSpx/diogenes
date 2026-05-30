# DARWIN HAMMER — match 4266, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s2.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s0.py (gen3)
# born: 2026-05-29T23:54:33Z

"""hybrid_hybrid_sketch_tree_bayes_fusion.py
Hybrid algorithm merging the Count‑Min sketch / MinHash LSH dimensionality reduction
from *Parent A* with the matrix‑vector Bayesian edge‑prior update and tree‑cost
computation from *Parent B*.

Mathematical bridge
-------------------
1. Each node *v* is represented by a multiset of tokens.  A Count‑Min sketch
   **Sᵥ** (width *w*, depth *d*) compresses this multiset to a small integer
   matrix.
2. For any edge *e = (u, v)* an *evidence factor*  eₑ  is derived from the
   similarity of the two sketches, e.g. the inner product of their flattened
   count vectors.  This scalar quantifies how much the data at the two endpoints
   support the existence of the edge.
3. All edges share a common *likelihood* ℓ and *false‑positive* rate α.
   Using the evidence vector **E** (ordered like the non‑zero entries of the
   length matrix **L**) the posterior edge‑prior vector **P′** is obtained by
   the element‑wise Bayesian update (Parent B):

        P′ = (P * ℓ * E) / (P * ℓ * E + (1‑P) * α)

4. The total cost of a rooted tree is the sum of edge‑lengths weighted by the
   posterior probabilities plus a path‑weight term proportional to the root‑to‑
   node distances (γ).  This mirrors the cost formulation of Parent B while
   now the evidence **E** originates from the sketch‑based similarity of
   Parent A.

The three core functions below implement:
* sketch construction,
* evidence extraction from sketches,
* vectorised Bayesian update and cost evaluation.

All operations are NumPy‑based and rely only on the Python standard library."""

import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[int, int]          # indices of endpoint nodes
Sketch = List[List[int]]        # Count‑Min sketch (depth × width)

# ----------------------------------------------------------------------
# Sketch utilities (from Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> Sketch:
    """Return a Count‑Min sketch for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table

def flatten_sketch(sketch: Sketch) -> np.ndarray:
    """Flatten a 2‑D sketch into a 1‑D NumPy array (depth·width)."""
    return np.array(sketch, dtype=np.int32).ravel()

def minhash_lsh_index(docs: Dict[int, List[str]]) -> Dict[str, List[int]]:
    """Very light MinHash LSH: bucket docs by the minimum hash of their shingles."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs.items():
        if not shingles:
            key = "empty"
        else:
            key = min(hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles)
        buckets[key].append(doc_id)
    return dict(buckets)

# ----------------------------------------------------------------------
# Graph utilities (shared)
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_length_matrix(points: List[Point], edges: List[Edge]) -> Tuple[np.ndarray, List[Edge]]:
    """
    Build a symmetric |V|×|V| matrix **L** where L[i, j] = distance if (i, j) ∈ edges,
    otherwise 0.  Returns the matrix and the list of edges in the order they appear
    in the flattened upper‑triangle (used for vectorised operations).
    """
    n = len(points)
    L = np.zeros((n, n), dtype=np.float64)
    ordered_edges = []
    for (i, j) in edges:
        dist = euclidean_length(points[i], points[j])
        L[i, j] = L[j, i] = dist
        ordered_edges.append((i, j))
    return L, ordered_edges

def root_to_node_distances(points: List[Point], root: int, adjacency: List[Edge]) -> np.ndarray:
    """
    Compute the shortest‑path distance from *root* to every node using DFS
    on an undirected tree (assumes adjacency forms a tree).
    Returns a vector *d* of length |V| where d[root] = 0.
    """
    n = len(points)
    neigh = [[] for _ in range(n)]
    for u, v in adjacency:
        neigh[u].append(v)
        neigh[v].append(u)

    distances = np.full(n, np.inf, dtype=np.float64)
    distances[root] = 0.0
    stack = [(root, -1)]

    while stack:
        node, parent = stack.pop()
        for nb in neigh[node]:
            if nb == parent:
                continue
            distances[nb] = distances[node] + euclidean_length(points[node], points[nb])
            stack.append((nb, node))

    return distances

# ----------------------------------------------------------------------
# Evidence extraction (bridge between A and B)
# ----------------------------------------------------------------------
def edge_evidence_from_sketches(
    sketches: Dict[int, Sketch],
    ordered_edges: List[Edge]
) -> np.ndarray:
    """
    Compute an evidence factor *eₑ* for each edge using the inner product of the
    flattened Count‑Min sketches of its endpoints.  The result is normalized
    to lie in (0, 1] to serve as a multiplicative factor in the Bayesian update.
    """
    # Pre‑flatten all sketches for fast vectorised access
    flat = {node: flatten_sketch(sk) for node, sk in sketches.items()}
    evidences = []
    for u, v in ordered_edges:
        prod = np.dot(flat[u], flat[v])
        # Normalisation: add 1 to avoid division by zero, then map to (0,1]
        norm = prod + 1.0
        evidences.append(1.0 / norm)
    return np.array(evidences, dtype=np.float64)

# ----------------------------------------------------------------------
# Vectorised Bayesian edge‑prior update (Parent B)
# ----------------------------------------------------------------------
def bayesian_edge_update(
    priors: np.ndarray,
    likelihood: float,
    false_positive: float,
    evidence: np.ndarray
) -> np.ndarray:
    """
    Perform element‑wise Bayesian update for all edges:

        p' = (p * ℓ * e) / (p * ℓ * e + (1‑p) * α)

    Parameters
    ----------
    priors : np.ndarray
        Prior probabilities for each edge (shape (m,)).
    likelihood : float
        Global likelihood ℓ ∈ [0,1].
    false_positive : float
        Global false‑positive rate α ∈ [0,1].
    evidence : np.ndarray
        Evidence factors eₑ (shape (m,)), typically ∈ (0,1].

    Returns
    -------
    np.ndarray
        Posterior probabilities p′ with the same shape as *priors*.
    """
    if not (0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("likelihood and false_positive must be in [0,1]")
    if priors.shape != evidence.shape:
        raise ValueError("priors and evidence must have the same shape")
    numerator = priors * likelihood * evidence
    denominator = numerator + (1.0 - priors) * false_positive
    # Avoid division by zero – denominator is strictly positive because α>0 or
    # numerator>0; clamp to a tiny epsilon otherwise.
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    return numerator / denominator

# ----------------------------------------------------------------------
# Total cost computation (Parent B)
# ----------------------------------------------------------------------
def total_tree_cost(
    length_matrix: np.ndarray,
    posterior: np.ndarray,
    ordered_edges: List[Edge],
    root_distances: np.ndarray,
    gamma: float = 1.0
) -> float:
    """
    Compute the hybrid tree cost:

        C = Σₑ Lₑ · p′ₑ  +  γ· Σᵥ d_root(v)

    where Lₑ is the Euclidean length of edge *e*, p′ₑ its posterior,
    and d_root(v) the distance from the root to node *v*.

    Parameters
    ----------
    length_matrix : np.ndarray
        Full |V|×|V| matrix of Euclidean edge lengths.
    posterior : np.ndarray
        Posterior edge probabilities (ordered like *ordered_edges*).
    ordered_edges : List[Edge]
        Edge list matching the ordering of *posterior*.
    root_distances : np.ndarray
        Vector of distances from the root to each node.
    gamma : float
        Weight of the path‑weight term.

    Returns
    -------
    float
        The scalar total cost.
    """
    edge_cost = sum(
        length_matrix[u, v] * p_prime
        for (u, v), p_prime in zip(ordered_edges, posterior)
    )
    path_cost = gamma * float(root_distances.sum())
    return edge_cost + path_cost

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_pipeline(
    points: List[Point],
    edges: List[Edge],
    node_tokens: Dict[int, List[str]],
    prior_edge_prob: float = 0.5,
    likelihood: float = 0.8,
    false_positive: float = 0.1,
    gamma: float = 0.5,
    sketch_width: int = 64,
    sketch_depth: int = 4
) -> Tuple[float, np.ndarray]:
    """
    Execute the full hybrid algorithm:

    1. Build Count‑Min sketches for each node from *node_tokens*.
    2. Construct the Euclidean length matrix **L**.
    3. Derive evidence factors *eₑ* from sketch similarity.
    4. Vectorised Bayesian update of edge priors.
    5. Compute root‑to‑node distances (root = 0) and total cost.

    Returns
    -------
    total_cost : float
        The hybrid cost value.
    posterior : np.ndarray
        Posterior edge probabilities (shape (|E|,)).
    """
    # 1. Sketches
    sketches = {
        node: count_min_sketch(tokens, width=sketch_width, depth=sketch_depth)
        for node, tokens in node_tokens.items()
    }

    # 2. Length matrix and ordered edge list
    L, ordered_edges = build_length_matrix(points, edges)

    # 3. Evidence vector from sketches
    evidence = edge_evidence_from_sketches(sketches, ordered_edges)

    # 4. Prior vector (uniform for simplicity)
    priors = np.full(len(ordered_edges), prior_edge_prob, dtype=np.float64)

    posterior = bayesian_edge_update(priors, likelihood, false_positive, evidence)

    # 5. Root distances (choose node 0 as root)
    root_dist = root_to_node_distances(points, root=0, adjacency=ordered_edges)

    total_cost = total_tree_cost(L, posterior, ordered_edges, root_dist, gamma=gamma)

    return total_cost, posterior

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a small random geometric graph
    random.seed(42)
    np.random.seed(42)

    N = 6
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(N)]

    # Randomly connect nodes to form a tree (ensure connectivity)
    edges: List[Edge] = []
    available = set(range(N))
    available.remove(0)
    connected = {0}
    while available:
        u = random.choice(list(connected))
        v = random.choice(list(available))
        edges.append((u, v))
        connected.add(v)
        available.remove(v)

    # Add a few extra edges to make the graph non‑tree (optional)
    extra_edges = [(1, 3), (2, 5)]
    edges.extend(extra_edges)

    # Create dummy token sets per node
    vocab = [f"tok{i}" for i in range(20)]
    node_tokens = {
        i: random.choices(vocab, k=random.randint(5, 15))
        for i in range(N)
    }

    cost, post = hybrid_pipeline(
        points,
        edges,
        node_tokens,
        prior_edge_prob=0.6,
        likelihood=0.85,
        false_positive=0.05,
        gamma=0.3,
        sketch_width=32,
        sketch_depth=3,
    )

    print(f"Hybrid total cost: {cost:.4f}")
    print(f"Posterior edge probabilities: {post}")