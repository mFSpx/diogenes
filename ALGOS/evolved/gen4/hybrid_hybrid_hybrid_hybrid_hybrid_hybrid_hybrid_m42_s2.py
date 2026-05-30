# DARWIN HAMMER — match 42, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s0.py (gen3)
# born: 2026-05-29T23:26:30Z

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from dataclasses import asdict, dataclass

"""
Module docstring:
This module fuses the mathematical structures of 
'hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py' and 
'hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s0.py'. 
The bridge between the two parents lies in the application of 
information entropy to guide the sheaf section assignments 
in the dense associative memory, and the use of Bayesian update 
to inform the advisory VRAM preemption planner. 
The governing equations of both parents are integrated through 
the use of Bayesian update to inform the planning of VRAM 
allocation and the sheaf-aware dense associative memory.
"""

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a linear restriction map
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and
      ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

def hybrid_energy(sheaf: Sheaf, memory_matrix: np.ndarray) -> float:
    """
    Compute the hybrid energy function.

    This function integrates the sheaf sections with the memory matrix 
    of the dense associative memory, using the energy function to guide 
    the sheaf's section assignments.

    Parameters:
    sheaf (Sheaf): The cellular sheaf.
    memory_matrix (np.ndarray): The memory matrix.

    Returns:
    float: The hybrid energy.
    """
    energy = 0.0
    for node, section in sheaf._sections.items():
        energy += np.dot(section.T, np.dot(memory_matrix, section))
    return energy

def hybrid_update_rule(sheaf: Sheaf, memory_matrix: np.ndarray) -> None:
    """
    Update the sheaf sections using the hybrid update rule.

    This function uses Bayesian update to inform the planning of VRAM 
    allocation and the sheaf-aware dense associative memory.

    Parameters:
    sheaf (Sheaf): The cellular sheaf.
    memory_matrix (np.ndarray): The memory matrix.
    """
    for edge, (src_map, dst_map) in sheaf._restrictions.items():
        u, v = edge
        src_section = sheaf._sections[u]
        dst_section = sheaf._sections[v]
        # Apply Bayesian update
        posterior = np.dot(src_map.T, np.dot(memory_matrix, src_section)) / np.dot(dst_map.T, np.dot(memory_matrix, dst_section))
        sheaf.set_section(v, posterior)

def hybrid_retrieve(sheaf: Sheaf, query_vector: np.ndarray) -> np.ndarray:
    """
    Retrieve the section associated with the query vector.

    This function uses the sheaf-aware dense associative memory to 
    retrieve the section.

    Parameters:
    sheaf (Sheaf): The cellular sheaf.
    query_vector (np.ndarray): The query vector.

    Returns:
    np.ndarray: The retrieved section.
    """
    # Compute the similarity between the query vector and the sheaf sections
    similarities = []
    for node, section in sheaf._sections.items():
        similarity = np.dot(query_vector.T, section) / np.linalg.norm(query_vector) / np.linalg.norm(section)
        similarities.append(similarity)
    # Retrieve the section with the highest similarity
    max_similarity_idx = np.argmax(similarities)
    return list(sheaf._sections.values())[max_similarity_idx]

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)
    return adj, edge_len, dist

if __name__ == "__main__":
    # Create a sample sheaf
    sheaf = Sheaf({0: 2, 1: 2}, [(0, 1)])
    sheaf.set_restriction((0, 1), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_section(0, np.array([1, 0]))
    sheaf.set_section(1, np.array([0, 1]))

    # Create a sample memory matrix
    memory_matrix = np.array([[1, 0], [0, 1]])

    # Compute the hybrid energy
    energy = hybrid_energy(sheaf, memory_matrix)
    print("Hybrid Energy:", energy)

    # Update the sheaf sections
    hybrid_update_rule(sheaf, memory_matrix)

    # Retrieve a section
    query_vector = np.array([1, 0])
    retrieved_section = hybrid_retrieve(sheaf, query_vector)
    print("Retrieved Section:", retrieved_section)

    # Compute tree metrics
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("A", "C")]
    adj, edge_len, dist = tree_metrics(nodes, edges, "A")
    print("Adjacency:", adj)
    print("Edge Lengths:", edge_len)
    print("Distances:", dist)