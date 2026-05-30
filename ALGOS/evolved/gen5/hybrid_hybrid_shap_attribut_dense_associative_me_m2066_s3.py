# DARWIN HAMMER — match 2066, survivor 3
# gen: 5
# parent_a: hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s0.py (gen4)
# parent_b: dense_associative_memory.py (gen0)
# born: 2026-05-29T23:40:37Z

"""Hybrid SHAP-Hopfield Attribution and Retrieval

This module fuses two parent algorithms:

* **Parent A** – SHAP‑based graph leader election with perceptual hashing.
* **Parent B** – Dense Associative Memory (modern Hopfield network) with
  softmax‑based energy dynamics.

**Mathematical bridge**

We treat the SHAP attribution scores of graph nodes as a *query vector*
`ξ`.  The graph structure is turned into a memory matrix `M` where each
row encodes a node’s neighbourhood feature pattern (a sparse vector of the
node’s own value and the values of its immediate neighbours).  The Hopfield
energy function


E(ξ) = -β⁻¹ log Σ_i exp(β M_i·ξ) + ½ ‖ξ‖²


provides a principled way to attract the query toward a cluster of nodes
with similar SHAP scores.  The softmax attention `softmax(β M ξ)` yields a
distribution over rows (i.e. over graph nodes); we use this distribution to
perform a *probabilistic leader election*: nodes with high attention weight
are elected as leaders, while the softmax itself implements the “pheromone”
signal of the original hybrid algorithm.

Thus the SHAP attribution supplies the query, the graph supplies the memory,
and the Hopfield dynamics supply the unified clustering / leader selection
mechanism.

The module implements three core functions demonstrating this hybrid
operation and a small smoke test.
"""

import sys
import math
import random
from pathlib import Path
from itertools import combinations
import numpy as np

Node = int
Graph = dict[int, set[int]]
Model = dict[int, float]


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Exact Shapley kernel weight for a given subset size."""
    return (
        math.factorial(subset_size)
        * math.factorial(feature_count - subset_size - 1)
        / math.factorial(feature_count)
    )


def approximate_shap_values(
    graph: Graph, model: Model, feature_count: int
) -> dict[Node, float]:
    """
    Approximate SHAP values for each node using its own value and the values
    of its neighbours.  The exact combinatorial definition is O(2^n); we use
    a simple linear surrogate that respects the kernel weighting.

    Returns
    -------
    dict mapping node → SHAP attribution (float)
    """
    shap = {}
    for node, neighbours in graph.items():
        # Local feature vector: own value + neighbour values
        local_vals = [model.get(node, 0.0)] + [model.get(nb, 0.0) for nb in neighbours]
        # Weighted sum where weights follow the Shapley kernel (approx)
        weight = shapley_kernel_weight(len(local_vals) - 1, feature_count)
        shap[node] = weight * sum(local_vals) / max(1, len(local_vals))
    return shap


def build_memory_matrix(
    graph: Graph, model: Model, dim: int | None = None
) -> np.ndarray:
    """
    Construct the memory matrix M (N × d) for the Hopfield network.

    Each row corresponds to a node.  The row vector is sparse: entry `j`
    contains the model value of node `j` if `j` is the node itself or a
    neighbour, otherwise zero.  This encodes the graph topology into a set
    of pattern vectors.

    Parameters
    ----------
    graph : Graph
        Adjacency representation.
    model : Model
        Mapping node → scalar feature.
    dim : int, optional
        Dimensionality of the pattern vectors.  If omitted, the number of
        distinct nodes is used.

    Returns
    -------
    M : np.ndarray of shape (N, d)
    """
    nodes = sorted(graph.keys())
    if dim is None:
        dim = len(nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    M = np.zeros((len(nodes), dim), dtype=float)

    for n in nodes:
        row_idx = node_index[n]
        # own contribution
        col_idx = node_index[n]
        M[row_idx, col_idx] = model.get(n, 0.0)
        # neighbours
        for nb in graph[n]:
            if nb in node_index:
                M[row_idx, node_index[nb]] = model.get(nb, 0.0)
    return M


def hopfield_update(query: np.ndarray, M: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    One Hopfield (modern dense associative memory) update step.

    ξ_new = Mᵀ softmax(β M ξ)

    Parameters
    ----------
    query : np.ndarray shape (d,)
        Current state vector (here: SHAP attribution vector).
    M : np.ndarray shape (N, d)
        Memory matrix.
    beta : float
        Inverse temperature controlling sharpness.

    Returns
    -------
    np.ndarray shape (d,)
        Updated state vector.
    """
    scores = beta * (M @ query)               # shape (N,)
    # numerically stable softmax
    max_score = scores.max()
    exp_scores = np.exp(scores - max_score)
    softmax = exp_scores / exp_scores.sum()   # shape (N,)

    return M.T @ softmax                      # shape (d,)


def hybrid_leader_election(
    graph: Graph,
    model: Model,
    beta: float = 1.0,
    top_k: int = 3,
    seed: int | None = None,
) -> set[Node]:
    """
    Perform leader election by:

    1. Computing approximate SHAP values → query vector ξ.
    2. Building memory matrix M from the graph topology.
    3. Running a single Hopfield update to obtain attention weights.
    4. Selecting the `top_k` nodes with highest attention probability as leaders.

    The softmax weights act as pheromone signals, while the SHAP values act as
    the energy‑driving query.

    Parameters
    ----------
    graph : Graph
        Node adjacency.
    model : Model
        Node → scalar feature.
    beta : float
        Inverse temperature for Hopfield dynamics.
    top_k : int
        Number of leaders to elect.
    seed : int | None
        Random seed (unused here but kept for API compatibility).

    Returns
    -------
    set[Node]
        Selected leader nodes.
    """
    rng = random.Random(seed)

    feature_count = len(graph)  # treat each node as a feature
    shap_vals = approximate_shap_values(graph, model, feature_count)

    # Query vector ξ ordered by node index
    nodes = sorted(graph.keys())
    query = np.array([shap_vals[n] for n in nodes], dtype=float)

    M = build_memory_matrix(graph, model)

    updated = hopfield_update(query, M, beta=beta)

    # Convert updated vector back to probabilities (softmax again for robustness)
    probs = np.exp(updated - updated.max())
    probs /= probs.sum()

    # Choose top_k nodes by probability
    leader_indices = np.argpartition(-probs, range(top_k))[:top_k]
    leaders = {nodes[i] for i in leader_indices}
    return leaders


def retrieve_pattern(query: np.ndarray, M: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Retrieve the nearest stored pattern from memory using the Hopfield energy
    minimisation.  The returned pattern is the row of `M` with the highest
    attention weight after a single update.

    Parameters
    ----------
    query : np.ndarray shape (d,)
        Input query (e.g., SHAP vector).
    M : np.ndarray shape (N, d)
        Memory matrix.
    beta : float
        Inverse temperature.

    Returns
    -------
    np.ndarray shape (d,)
        Retrieved pattern (row of M).
    """
    scores = beta * (M @ query)
    max_idx = int(np.argmax(scores))
    return M[max_idx]


if __name__ == "__main__":
    # Smoke test: create a small random graph, random model values,
    # run hybrid leader election and pattern retrieval.
    rng = random.Random(42)
    num_nodes = 7
    # Random undirected graph
    graph: Graph = {i: set() for i in range(num_nodes)}
    for i, j in combinations(range(num_nodes), 2):
        if rng.random() < 0.3:
            graph[i].add(j)
            graph[j].add(i)

    # Random model values in [0,1]
    model: Model = {i: rng.random() for i in range(num_nodes)}

    leaders = hybrid_leader_election(graph, model, beta=2.0, top_k=2, seed=123)
    print("Leaders:", sorted(leaders))

    # Build memory and retrieve pattern for the SHAP query
    shap_vals = approximate_shap_values(graph, model, feature_count=num_nodes)
    query_vec = np.array([shap_vals[i] for i in range(num_nodes)], dtype=float)
    M = build_memory_matrix(graph, model)
    retrieved = retrieve_pattern(query_vec, M, beta=2.0)
    print("Retrieved pattern (non‑zero entries):", [(i, v) for i, v in enumerate(retrieved) if v != 0.0])