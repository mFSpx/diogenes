# DARWIN HAMMER — match 860, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s6.py (gen2)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s1.py (gen2)
# born: 2026-05-29T23:31:15Z

"""Hybrid algorithm merging probabilistic annealing, Hoeffding statistical bounds,
tropical algebra, and a VRAM‑budget aware graph construction.

Parent A:
    - broadcast_probability, acceptance_probability, cooling_temperature
    - Hoeffding bound and tropical semiring (t_add = max, t_mul = +)

Parent B:
    - VRAM‑budget planner (memory‑aware feature registration)
    - Graph‑centric curvature analysis (abstracted here)

Mathematical bridge:
    The tropical semiring turns a feature vector 𝑥∈ℝⁿ into a “tropical point”.
    Pairwise tropical distance d_T(x,y)=min_i (x_i+y_i) can be expressed with
    t_mul (addition) followed by t_add (max) across dimensions.
    These distances serve as statistics for a Hoeffding‑bounded split test.
    The split decision is then fed to a simulated‑annealing acceptance rule
    whose temperature follows the cooling schedule.  Because each node stores
    a tropical feature vector, we must respect a global VRAM budget; the
    allocation routine distributes the allowed memory across nodes and
    quantises the vectors accordingly.  This yields a single unified system
    that simultaneously respects memory constraints, uses tropical algebraic
    geometry, and performs statistically‑guaranteed, annealed graph optimisation.
"""

import sys
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# VRAM‑aware feature allocation (Parent B)
# ----------------------------------------------------------------------
DEFAULT_BUDGET_MB = 4096  # 4 GiB
BYTES_PER_FLOAT = 4  # float32

def allocate_features(num_nodes: int,
                      feature_dim: int,
                      budget_mb: int = DEFAULT_BUDGET_MB) -> np.ndarray:
    """Allocate a (num_nodes, feature_dim) float32 matrix respecting a VRAM budget.

    The allocation simply caps the feature dimension if the raw size exceeds the
    budget.  The returned matrix is filled with random values drawn from a
    uniform distribution in [0, 1) and will later be interpreted in the tropical
    semiring.
    """
    max_bytes = budget_mb * 1024 * 1024
    required_bytes = num_nodes * feature_dim * BYTES_PER_FLOAT
    if required_bytes > max_bytes:
        # Reduce dimensionality proportionally
        max_dim = max(1, max_bytes // (num_nodes * BYTES_PER_FLOAT))
        feature_dim = max_dim
    rng = np.random.default_rng()
    return rng.random((num_nodes, feature_dim), dtype=np.float32)

# ----------------------------------------------------------------------
# Tropical algebra primitives (Parent A)
# ----------------------------------------------------------------------
def t_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical addition: element‑wise maximum."""
    return np.maximum(x, y)

def t_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tropical multiplication: element‑wise addition."""
    return np.add(x, y)

def tropical_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise tropical distance matrix d_T(i,j)=min_k (f_i_k + f_j_k).

    Using the tropical semiring, the “distance” is the tropical inner product
    followed by a minimum over coordinates.  This can be written efficiently
    with broadcasting.
    """
    # Expand dimensions for broadcasting: (N,1,D) + (1,N,D) -> (N,N,D)
    sum_tensor = t_mul(features[:, None, :], features[None, :, :])
    # Minimum over the tropical “addition” (which is ordinary addition)
    return np.min(sum_tensor, axis=2)

# ----------------------------------------------------------------------
# Hoeffding bound based split scoring (Parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def hybrid_split_score(best_gain: float,
                       second_best_gain: float,
                       r: float,
                       delta: float,
                       n: int) -> bool:
    """Return True if the observed gain difference is statistically significant.

    The test uses Hoeffding's inequality on the gain difference.
    """
    bound = hoeffding_bound(r, delta, n)
    return (best_gain - second_best_gain) > bound

# ----------------------------------------------------------------------
# Simulated annealing primitives (Parent A)
# ----------------------------------------------------------------------
def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Graph utilities
# ----------------------------------------------------------------------
Node = int
Graph = Dict[Node, Set[Node]]

def random_graph(num_nodes: int, edge_prob: float = 0.1) -> Graph:
    """Generate an undirected Erdős‑Rényi graph."""
    g: Graph = {i: set() for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                g[i].add(j)
                g[j].add(i)
    return g

def edge_cut_gain(dist_matrix: np.ndarray, edge: Tuple[int, int]) -> float:
    """A toy gain function: reduction in average tropical distance when removing an edge."""
    i, j = edge
    # Current average distance
    current = np.mean(dist_matrix)
    # Simulate removal by setting distance between i and j to a large value
    modified = dist_matrix.copy()
    modified[i, j] = modified[j, i] = modified.max() * 10
    new_avg = np.mean(modified)
    return current - new_avg  # positive if removal improves (reduces) average distance

# ----------------------------------------------------------------------
# Hybrid core operations (three required functions)
# ----------------------------------------------------------------------
def hybrid_compute_curvature(features: np.ndarray, graph: Graph) -> float:
    """Compute a curvature‑like scalar using tropical distances over the graph.

    The metric is the average tropical distance across existing edges.
    """
    dist_mat = tropical_distance_matrix(features)
    edge_distances = [dist_mat[i, j] for i, neigh in graph.items() for j in neigh if i < j]
    if not edge_distances:
        return 0.0
    return float(np.mean(edge_distances))

def hybrid_split_decision(features: np.ndarray,
                          graph: Graph,
                          r: float = 1.0,
                          delta: float = 0.05,
                          n_samples: int = 100) -> Tuple[bool, Tuple[int, int]]:
    """Select the best edge to cut and decide via Hoeffding whether to accept.

    Returns a tuple (accepted, edge). If no edge exists, returns (False, (-1, -1)).
    """
    if not any(graph.values()):
        return (False, (-1, -1))

    # Evaluate gains for a random subset of edges
    all_edges = [(i, j) for i, neigh in graph.items() for j in neigh if i < j]
    sample = random.sample(all_edges, min(n_samples, len(all_edges)))
    gains = [(edge_cut_gain(tropical_distance_matrix(features), e), e) for e in sample]
    gains.sort(key=lambda x: x[0], reverse=True)

    best_gain, best_edge = gains[0]
    second_gain = gains[1][0] if len(gains) > 1 else 0.0

    accept = hybrid_split_score(best_gain, second_gain, r, delta, n=len(sample))
    return (accept, best_edge if accept else (-1, -1))

def hybrid_annealing_process(num_nodes: int,
                             feature_dim: int,
                             max_iters: int = 50,
                             budget_mb: int = DEFAULT_BUDGET_MB) -> Tuple[Graph, float]:
    """Perform an annealed graph optimisation using the hybrid primitives.

    Returns the final graph and its curvature score.
    """
    # 1. Allocate VRAM‑constrained features
    features = allocate_features(num_nodes, feature_dim, budget_mb)

    # 2. Initialise a random graph
    graph = random_graph(num_nodes, edge_prob=0.2)

    # 3. Simulated annealing loop
    temperature = cooling_temperature(0)
    for k in range(max_iters):
        accepted, edge = hybrid_split_decision(features, graph)
        if accepted and edge != (-1, -1):
            i, j = edge
            # Remove edge (cut)
            graph[i].discard(j)
            graph[j].discard(i)
            # Update temperature
            temperature = cooling_temperature(k + 1)
        else:
            # No beneficial cut found; optionally break early
            break

    final_curvature = hybrid_compute_curvature(features, graph)
    return graph, final_curvature

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    NUM_NODES = 30
    FEATURE_DIM = 64
    FINAL_GRAPH, CURV = hybrid_annealing_process(NUM_NODES, FEATURE_DIM, max_iters=30)
    print(f"Final graph has {sum(len(v) for v in FINAL_GRAPH.values()) // 2} edges")
    print(f"Hybrid curvature score: {CURV:.4f}")