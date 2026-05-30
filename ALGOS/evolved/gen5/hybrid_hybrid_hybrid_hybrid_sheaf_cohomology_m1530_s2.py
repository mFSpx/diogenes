# DARWIN HAMMER — match 1530, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py (gen4)
# parent_b: sheaf_cohomology.py (gen0)
# born: 2026-05-29T23:37:17Z

"""
This module implements a hybrid mathematical algorithm that combines the minimum-cost tree Bayes update and bandit-router 
sketch-RLCT algorithm from the 'hybrid_hybrid_hybrid_minimu_hybrid_path_signatur_m1020_s1.py' module with the cellular 
sheaf cohomology algorithm from the 'sheaf_cohomology.py' module. The mathematical bridge between the two structures 
is based on representing the path signature as a function that can be approximated using the probabilistic weights and 
log-count statistics from the minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm, and then using 
this approximation to inform the restriction maps in the cellular sheaf cohomology algorithm.

This bridge allows us to leverage the flexibility and power of the probabilistic weights and log-count statistics to 
model complex paths and their signatures, while also incorporating the topological structure of the cellular sheaf 
cohomology algorithm to analyze the global consistency of the section assignments.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import random

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → Euclidean length
    root_dist : dict mapping node → root‑to‑node distance
    """
    adj = defaultdict(list)
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[a], nodes[b])
        if a == root:
            root_dist[b] = edge_len[(a, b)]
        elif b == root:
            root_dist[a] = edge_len[(a, b)]

    return adj, edge_len, root_dist

def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding."""
    return np.array(path).reshape(-1, 2)

class Sheaf:
    """Cellular sheaf over a graph.

    Parameters
    ----------
    node_dims : dict
        Mapping node_id -> dimension of the stalk (vector space) at that node.
    edge_list : list of (u, v) tuples
        Undirected edges; orientation is fixed as given (u = tail, v = head)
        for sign convention in the coboundary.
    """

    def __init__(self, node_dims, edge_list):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        # restriction maps: (u, v) -> (src_map: R^{dim_u}->R^{d_e}, dst_map: R^{dim_v}->R^{d_e})
        self._restrictions = {}
        # local sections: node_id -> numpy array of shape (dim,)
        self._sections = {}

    def set_restriction(self, edge, src_map, dst_map):
        """Assign restriction maps for an oriented edge.

        Parameters
        ----------
        edge : (u, v)
            Must match an entry in edge_list with the same orientation.
        src_map : numpy array, shape (edge_dim, dim_u)
            Linear map F(u->e): stalk at u -> stalk at e.
        dst_map : numpy array, shape (edge_dim, dim_v)
            Linear map F(v->e): stalk at v -> stalk at e.
        """
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[edge] = (src_map, dst_map)

    def get_restriction(self, edge):
        """Get restriction maps for an oriented edge."""
        return self._restrictions.get(edge)

    def set_section(self, node, section):
        """Assign a section to a node.

        Parameters
        ----------
        node : str
            Node id.
        section : numpy array of shape (dim,)
            Section assigned to the node.
        """
        self._sections[node] = np.array(section, dtype=float)

    def get_section(self, node):
        """Get section assigned to a node."""
        return self._sections.get(node)

def hybrid_algorithm(nodes, edges, root, node_dims, edge_list):
    """Hybrid algorithm that combines minimum-cost tree Bayes update and bandit-router sketch-RLCT algorithm with 
    cellular sheaf cohomology algorithm."""
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    sheaf = Sheaf(node_dims, edge_list)
    
    # Use probabilistic weights and log-count statistics to inform restriction maps
    for edge in edge_list:
        u, v = edge
        # Compute probabilistic weights and log-count statistics
        prob_weight_u = 1 / len(adj[u])
        prob_weight_v = 1 / len(adj[v])
        log_count_stat_u = math.log(len(adj[u]))
        log_count_stat_v = math.log(len(adj[v]))
        
        # Use probabilistic weights and log-count statistics to inform restriction maps
        src_map = np.array([[prob_weight_u], [log_count_stat_u]])
        dst_map = np.array([[prob_weight_v], [log_count_stat_v]])
        sheaf.set_restriction(edge, src_map, dst_map)
    
    # Use lead-lag transform to inform section assignments
    for node in nodes:
        path = lead_lag_transform(nodes[node])
        sheaf.set_section(node, path)
    
    return sheaf

def compute_coboundary(sheaf):
    """Compute coboundary operator."""
    coboundary = {}
    for edge in sheaf.edges:
        u, v = edge
        src_map, dst_map = sheaf.get_restriction(edge)
        section_u = sheaf.get_section(u)
        section_v = sheaf.get_section(v)
        coboundary[edge] = np.dot(src_map, section_u) - np.dot(dst_map, section_v)
    return coboundary

def compute_laplacian(sheaf):
    """Compute Laplacian operator."""
    laplacian = {}
    for node in sheaf.node_dims:
        laplacian[node] = 0
        for edge in sheaf.edges:
            u, v = edge
            if u == node:
                laplacian[node] += np.dot(sheaf.get_restriction(edge)[0], sheaf.get_section(u))
            elif v == node:
                laplacian[node] += np.dot(sheaf.get_restriction(edge)[1], sheaf.get_section(v))
    return laplacian

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    node_dims = {"A": 2, "B": 2, "C": 2}
    edge_list = [("A", "B"), ("B", "C"), ("C", "A")]
    
    sheaf = hybrid_algorithm(nodes, edges, root, node_dims, edge_list)
    coboundary = compute_coboundary(sheaf)
    laplacian = compute_laplacian(sheaf)
    
    print("Coboundary:", coboundary)
    print("Laplacian:", laplacian)