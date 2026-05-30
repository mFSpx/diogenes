# DARWIN HAMMER — match 3668, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m995_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1204_s1.py (gen6)
# born: 2026-05-29T23:51:13Z

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def compute_phash(values: List[float]) -> int:
    """
    Perceptual hash: 1 bit per element indicating >= average (max 64 bits).
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


def build_similarity_graph(
    data: List[List[float]],
    phash_bits: int = 64,
) -> np.ndarray:
    """
    Build a weighted adjacency matrix where the weight between two nodes is
    ``1 - normalized Hamming distance`` of their perceptual hashes.
    """
    n = len(data)
    hashes = [compute_phash(row[:phash_bits]) for row in data]
    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming_distance(hashes[i], hashes[j])
            weight = 1.0 - dist / phash_bits
            A[i, j] = A[j, i] = weight
    np.fill_diagonal(A, 1.0)  # self‑similarity
    return A


# ----------------------------------------------------------------------
# Sheaf Laplacian
# ----------------------------------------------------------------------


def compute_sheaf_laplacian(
    node_dims: Dict[int, int],
    edge_list: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute a block‑diagonal sheaf Laplacian.
    For each node ``i`` we place a ``d_i × d_i`` identity block on the diagonal.
    Off‑diagonal blocks are ``-I`` for each edge (u, v) respecting the smaller
    dimension. This yields a symmetric positive‑semidefinite matrix.
    """
    # total dimension = sum of node dimensions
    total_dim = sum(node_dims.values())
    L = np.zeros((total_dim, total_dim), dtype=np.float64)

    # map node index -> slice range in the global matrix
    offsets: Dict[int, Tuple[int, int]] = {}
    cur = 0
    for node, dim in sorted(node_dims.items()):
        offsets[node] = (cur, cur + dim)
        L[cur : cur + dim, cur : cur + dim] = np.eye(dim)
        cur += dim

    for u, v in edge_list:
        u_start, u_end = offsets[u]
        v_start, v_end = offsets[v]
        # use the smaller block size to keep the matrix square
        dim = min(u_end - u_start, v_end - v_start)
        L[u_start : u_start + dim, v_start : v_start + dim] = -np.eye(dim)
        L[v_start : v_start + dim, u_start : u_start + dim] = -np.eye(dim)

    # Degree matrix = sum of absolute off‑diagonal blocks per row
    degree = np.diag(np.abs(L).sum(axis=1))
    return degree - L  # standard Laplacian L = D - A_sheaf


# ----------------------------------------------------------------------
# Tropical max‑plus modulation
# ----------------------------------------------------------------------


def tropical_maxplus_mod(L: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """
    Apply tropical max‑plus algebra to ``L``.
    In the max‑plus semiring:
        a ⊕ b = max(a, b)
        a ⊗ b = a + b
    We interpret the modulation as ``L̄ = max(L + offset, 0)``.
    """
    return np.maximum(L + offset, 0.0)


# ----------------------------------------------------------------------
# Pheromone dynamics
# ----------------------------------------------------------------------


def decay_pheromones(pheromones: np.ndarray, decay_rate: float) -> np.ndarray:
    """Exponential decay of pheromones (0 < decay_rate ≤ 1)."""
    if not (0.0 < decay_rate <= 1.0):
        raise ValueError("decay_rate must be in (0, 1].")
    return pheromones * decay_rate


def reinforce_pheromones(
    pheromones: np.ndarray,
    leader_idx: int,
    reinforcement: float = 0.1,
) -> np.ndarray:
    """
    Increase pheromone levels on edges incident to the elected leader.
    """
    if not (0.0 < reinforcement):
        raise ValueError("reinforcement must be positive.")
    new_pheromones = pheromones.copy()
    new_pheromones[leader_idx, :] += reinforcement
    new_pheromones[:, leader_idx] += reinforcement
    return np.clip(new_pheromones, 0.0, None)


# ----------------------------------------------------------------------
# Fusion of similarity graph and sheaf Laplacian
# ----------------------------------------------------------------------


def pheromone_weighted_graph(
    A: np.ndarray,
    L_mod: np.ndarray,
    pheromones: np.ndarray,
) -> np.ndarray:
    """
    Fuse the similarity graph ``A`` with the tropical‑modulated Laplacian
    ``L_mod`` and the current pheromone matrix.
    The operation is element‑wise multiplication, followed by a symmetric
    symmetrisation to preserve undirectedness.
    """
    W = A * L_mod * pheromones
    # Ensure symmetry (important for later spectral operations)
    W = (W + W.T) / 2.0
    return W


# ----------------------------------------------------------------------
# VRAM‑aware leader election
# ----------------------------------------------------------------------


def estimate_vram_usage(matrix: np.ndarray) -> int:
    """Very rough estimate of VRAM usage in bytes for a dense matrix."""
    return matrix.nbytes


def leader_election(
    pheromone_biased_degrees: np.ndarray,
    vram_budget_bytes: int = 10 * 1024 * 1024,  # 10 MiB default
) -> int:
    """
    Select a leader node based on pheromone‑biased degrees while respecting a
    VRAM budget. The function trims the candidate set to the largest prefix
    that fits into the budget.
    """
    # Simulate the memory needed to materialise the candidate sub‑graph:
    # each candidate requires a row vector of float64 values.
    row_bytes = pheromone_biased_degrees.itemsize * pheromone_biased_degrees.shape[0]
    max_candidates = max(1, vram_budget_bytes // row_bytes)

    candidates = pheromone_biased_degrees[:max_candidates]
    return int(np.argmax(candidates))


# ----------------------------------------------------------------------
# Main simulation loop (example usage)
# ----------------------------------------------------------------------


def main() -> None:
    # Example data: three nodes with 8‑dimensional feature vectors
    data = [
        [random.random() for _ in range(8)],
        [random.random() for _ in range(8)],
        [random.random() for _ in range(8)],
    ]

    # Build similarity graph from perceptual hashes
    A = build_similarity_graph(data, phash_bits=8)

    # Sheaf description
    node_dims = {0: 3, 1: 4, 2: 5}
    edge_list = [(0, 1), (1, 2), (2, 0)]

    # Compute Laplacian and apply tropical modulation
    L = compute_sheaf_laplacian(node_dims, edge_list)
    L_mod = tropical_maxplus_mod(L, offset=0.5)

    # Initialise pheromones
    pheromones = np.random.rand(A.shape[0], A.shape[0])

    # Fuse structures
    W = pheromone_weighted_graph(A, L_mod, pheromones)

    # Compute pheromone‑biased degrees
    degrees = np.sum(W, axis=1)

    # Leader election respecting a 5 MiB VRAM budget
    leader = leader_election(degrees, vram_budget_bytes=5 * 1024 * 1024)

    # Update pheromones
    pheromones = decay_pheromones(pheromones, decay_rate=0.95)
    pheromones = reinforce_pheromones(pheromones, leader_idx=leader, reinforcement=0.2)

    # Output for inspection
    print("Adjacency matrix (A):")
    print(A)
    print("\nTropical‑modulated Laplacian (L̄):")
    print(L_mod)
    print("\nPheromone‑weighted graph (W):")
    print(W)
    print("\nLeader node:", leader)
    print("\nUpdated pheromones:")
    print(pheromones)


if __name__ == "__main__":
    main()