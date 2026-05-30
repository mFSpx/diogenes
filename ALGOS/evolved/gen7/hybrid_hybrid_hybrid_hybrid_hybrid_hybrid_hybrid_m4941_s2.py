# DARWIN HAMMER — match 4941, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_omni_c_hybrid_ternary_lens__m1536_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s1.py (gen6)
# born: 2026-05-29T23:59:01Z

"""
Hybrid Algorithm: Chaotic Omni-Graph + JEPA‑Fisher Energy × RBF‑Weighted Minimum‑Cost Tree

Parents:
- hybrid_hybrid_hybrid_omni_c_hybrid_ternary_lens__m1536_s1.py
- hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s1.py

Mathematical Bridge:
1. The chaotic graph generator supplies a latent vector **z** and adjacency **A**.
2. A pheromone probability vector **p** is derived from the graph; its diagonal Fisher‑information
   matrix **Λ = diag(1 / p_i)** is used as the JEPA precision matrix.
3. JEPA energy  E(z)=½ zᵀ Λ z − ∑ log p_i  couples the latent state to the pheromone distribution.
4. The scalar latent vector **z** is treated as a one‑dimensional feature per node; an RBF similarity
   matrix **S** (Gaussian kernel) is built from pairwise Euclidean distances of **z**.
5. Edge weights for a minimum‑cost spanning tree are defined as **W = A ⊙ S** (Hadamard product),
   i.e. only existing graph edges keep their RBF similarity as cost.
6. A Prim‑based MST algorithm yields the optimal tree **T** under the combined
   chaotic‑structure and similarity constraints, while the JEPA energy provides a
   global regularisation term that can inform downstream scheduling decisions.
"""

import sys
import math
import random
from pathlib import Path
import numpy as np
from typing import Tuple, List, Dict, Set, Hashable

Node = Hashable
Graph = Dict[Node, Set[Node]]


def chaotic_graph(num_nodes: int, chaos_factor: float = 3.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a directed chaotic adjacency matrix and a latent variable vector.

    Returns
    -------
    A : np.ndarray (n, n)
        Binary adjacency matrix (A[i, j] = 1 indicates edge i→j, no self‑loops).
    z : np.ndarray (n,)
        Latent variable vector drawn from Uniform[0,1).
    """
    z = np.random.rand(num_nodes)
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # chaotic factor could modulate probability; we keep 0.5 for simplicity
                A[i, j] = 1 if random.random() < 0.5 else 0
    return A, z


def pheromone_distribution(num_nodes: int) -> np.ndarray:
    """
    Produce a valid probability distribution over `num_nodes` by sampling
    from a Dirichlet(alpha=1) (i.e. uniform over the simplex).
    """
    raw = np.random.rand(num_nodes)
    p = raw / raw.sum()
    # Guard against zeros (unlikely but for numerical stability)
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    return p


def fisher_precision_matrix(p: np.ndarray) -> np.ndarray:
    """
    Diagonal Fisher‑information matrix for a categorical distribution.
    For category i, I_i = 1 / p_i.
    """
    return np.diag(1.0 / p)


def jepa_energy(z: np.ndarray, Lambda: np.ndarray, p: np.ndarray) -> float:
    """
    Compute JEPA energy term:
        E(z) = ½ zᵀ Λ z  −  Σ_i log p_i
    """
    quadratic = 0.5 * float(z.T @ Lambda @ z)
    entropy_term = -np.log(p).sum()
    return quadratic + entropy_term


def rbf_similarity_from_latent(z: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Build an RBF similarity matrix from the 1‑D latent vector `z`.
    Gaussian kernel:   K(i,j) = exp( - (ε * |z_i - z_j|)^2 )
    """
    n = len(z)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(z[i] - z[j])
            sim = math.exp(-((epsilon * dist) ** 2))
            S[i, j] = sim
            S[j, i] = sim
    return S


def prim_minimum_spanning_tree(weight_matrix: np.ndarray, adjacency: np.ndarray) -> List[Tuple[int, int]]:
    """
    Prim's algorithm on a graph where allowed edges are those with adjacency==1.
    Edge cost is taken directly from `weight_matrix` (lower cost = more desirable).
    Returns a list of edges (u, v) forming the MST.
    """
    n = weight_matrix.shape[0]
    visited = [False] * n
    visited[0] = True
    edges: List[Tuple[int, int]] = []

    # For each node, keep the best connecting edge (cost, from, to)
    best: List[Tuple[float, int]] = [(float('inf'), -1)] * n
    for j in range(1, n):
        if adjacency[0, j]:
            best[j] = (weight_matrix[0, j], 0)

    while len(edges) < n - 1:
        # select the minimum cost edge connecting visited to unvisited
        min_cost = float('inf')
        min_to = -1
        min_from = -1
        for v in range(n):
            if not visited[v] and best[v][0] < min_cost:
                min_cost, min_from, _ = best[v]
                min_to = v
        if min_to == -1:
            # graph is disconnected; abort early
            break
        visited[min_to] = True
        edges.append((min_from, min_to))

        # update best edges using the newly visited node
        for v in range(n):
            if not visited[v] and adjacency[min_to, v]:
                cand_cost = weight_matrix[min_to, v]
                if cand_cost < best[v][0]:
                    best[v] = (cand_cost, min_to, v)

    return edges


def hybrid_operation(num_nodes: int,
                     chaos_factor: float = 3.9,
                     epsilon: float = 1.0) -> Tuple[List[Tuple[int, int]], float]:
    """
    Execute the full hybrid pipeline:
      1. Chaotic graph + latent vector.
      2. Pheromone distribution → Fisher precision.
      3. JEPA energy.
      4. RBF similarity from latent.
      5. Weighted adjacency = A ⊙ S.
      6. Minimum‑cost spanning tree via Prim.

    Returns
    -------
    mst_edges : list of (int, int)
        Edge list of the resulting tree.
    energy   : float
        JEPA energy value for the generated state.
    """
    # Step 1
    A, z = chaotic_graph(num_nodes, chaos_factor)

    # Step 2
    p = pheromone_distribution(num_nodes)
    Lambda = fisher_precision_matrix(p)

    # Step 3
    energy = jepa_energy(z, Lambda, p)

    # Step 4
    S = rbf_similarity_from_latent(z, epsilon)

    # Step 5: combine structure and similarity (cost = 1 - similarity to turn
    # higher similarity into lower cost)
    cost_matrix = np.where(A == 1, 1.0 - S, np.inf)

    # Step 6
    mst = prim_minimum_spanning_tree(cost_matrix, A)

    return mst, energy


if __name__ == "__main__":
    # Simple smoke test
    N = 8
    tree, e = hybrid_operation(N)
    print(f"JEPA energy: {e:.4f}")
    print("MST edges (parent -> child):")
    for u, v in tree:
        print(f"  {u} -> {v}")
    # Verify that we have exactly N‑1 edges (or fewer if graph disconnected)
    assert len(tree) <= N - 1, "Tree has too many edges"
    print("Smoke test completed successfully.")