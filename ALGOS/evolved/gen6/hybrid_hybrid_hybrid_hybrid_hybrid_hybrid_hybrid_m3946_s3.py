# DARWIN HAMMER — match 3946, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1509_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2338_s0.py (gen5)
# born: 2026-05-29T23:52:42Z

"""Hybrid Fusion of Sheaf Cohomology and Chaotic Omni‑Front Synthesis

Parents:
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1509_s1.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2338_s0.py

Mathematical Bridge:
The sections of a cellular sheaf provide a vector at each node.  
We first turn the pairwise differences of neighbouring sections into
Gaussian edge‑weights  

    w_{uv}=exp(-||s_u-s_v||²/(2σ²))

These weights are normalised per‑node to obtain a probability
distribution over incident edges.  The Shannon entropy of this
distribution quantifies the topological uncertainty around each node.
The entropy values are then used as multiplicative scaling factors for
the chaotic omni‑front synthesis core, thereby modulating the
randomly generated candidate solutions with the sheaf’s topological
information.
"""

import sys
import random
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Core data structure: fused Sheaf
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims, edge_list):
        """
        node_dims: mapping node -> dimension (int)
        edge_list: iterable of (u, v) tuples, undirected edges
        """
        self.node_dims = dict(node_dims)
        self.edges = [tuple(e) for e in edge_list]
        self._restrictions = {}          # (u,v) -> (src_map, dst_map)
        self._sections = {}              # node -> np.ndarray
        self._gaussian_weights = {}      # (u,v) -> float
        self._entropy = {}               # node -> float

    # ----- restriction maps ------------------------------------------------
    def set_restriction(self, edge, src_map, dst_map):
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must share row dimension")
        self._restrictions[(u, v)] = (src_map, dst_map)

    # ----- sections --------------------------------------------------------
    def set_section(self, node, value):
        val = np.asarray(value, dtype=float)
        if val.shape[0] != self.node_dims[node]:
            raise ValueError("section dimension must match node dimension")
        self._sections[node] = val

    # ----- Gaussian weights ------------------------------------------------
    def set_gaussian_weight(self, edge, weight):
        self._gaussian_weights[tuple(edge)] = float(weight)

    # ----- Entropy ---------------------------------------------------------
    def set_entropy(self, node, entropy):
        self._entropy[node] = float(entropy)

# ----------------------------------------------------------------------
# Helper mathematical functions
# ----------------------------------------------------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def shannon_entropy(probabilities):
    """Return Shannon entropy (base‑2) of a probability list."""
    entropy = 0.0
    for p in probabilities:
        if p > 0.0:
            entropy -= p * math.log(p, 2)
    return entropy

# ----------------------------------------------------------------------
# Fusion functions
# ----------------------------------------------------------------------
def init_fused_sheaf(node_dims, edges, sections, restrictions):
    """
    Initialise a Sheaf with given dimensions, edges, sections and
    restriction maps.
    sections: dict node -> iterable of floats
    restrictions: dict (u,v) -> (src_map, dst_map)
    """
    sheaf = Sheaf(node_dims, edges)
    for n, sec in sections.items():
        sheaf.set_section(n, sec)
    for e, (src, dst) in restrictions.items():
        sheaf.set_restriction(e, src, dst)
    return sheaf

def compute_gaussian_weights(sheaf, sigma=1.0):
    """
    For each edge (u,v) compute a Gaussian weight from the Euclidean
    distance of the sections at its endpoints.
    """
    for (u, v) in sheaf.edges:
        s_u = sheaf._sections[u]
        s_v = sheaf._sections[v]
        dist_sq = euclidean_distance(s_u, s_v) ** 2
        weight = math.exp(-dist_sq / (2.0 * sigma * sigma))
        sheaf.set_gaussian_weight((u, v), weight)

def compute_node_entropies(sheaf):
    """
    For each node, normalize incident Gaussian weights to a probability
    distribution and compute its Shannon entropy.  The result is stored
    in sheaf._entropy.
    """
    incident = defaultdict(list)   # node -> list of weights
    for (u, v), w in sheaf._gaussian_weights.items():
        incident[u].append(w)
        incident[v].append(w)

    for node, wlist in incident.items():
        total = sum(wlist)
        if total == 0:
            probs = [0.0 for _ in wlist]
        else:
            probs = [w / total for w in wlist]
        ent = shannon_entropy(probs)
        sheaf.set_entropy(node, ent)

def chaotic_omni_front_synthesis(sheaf, n_samples=10):
    """
    Generate candidate solution vectors.  For each sample we pick a random
    node, draw a random Gaussian perturbation around its section and scale
    the perturbation by the node's entropy (higher entropy -> larger
    exploration).  The function returns a list of np.ndarray objects.
    """
    solutions = []
    nodes = list(sheaf.node_dims.keys())
    max_dim = max(sheaf.node_dims.values())
    for _ in range(n_samples):
        node = random.choice(nodes)
        base = sheaf._sections[node]
        dim = sheaf.node_dims[node]
        entropy = sheaf._entropy.get(node, 0.0)
        scale = 1.0 + entropy   # entropy‑driven scaling
        perturb = np.random.normal(loc=0.0, scale=scale, size=dim)
        sol = base + perturb
        solutions.append(sol)
    return solutions

def evaluate_fused_cost(sheaf, solutions):
    """
    Simple cost: sum of squared distances between each solution and the
    nearest node section, weighted by the node entropy.
    """
    total_cost = 0.0
    for sol in solutions:
        # find nearest node by Euclidean distance
        best = None
        best_dist = None
        best_node = None
        for node, sect in sheaf._sections.items():
            d = euclidean_distance(sol, sect)
            if best_dist is None or d < best_dist:
                best_dist = d
                best = sect
                best_node = node
        entropy = sheaf._entropy.get(best_node, 0.0)
        total_cost += (best_dist ** 2) * (1.0 + entropy)
    return total_cost

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    node_dims = {0: 2, 1: 2, 2: 2}
    edges = [(0, 1), (1, 2), (2, 0)]

    # Random sections for each node
    sections = {
        0: [0.0, 0.0],
        1: [1.0, 0.0],
        2: [0.5, 0.866]   # roughly an equilateral triangle
    }

    # Identity restriction maps (for simplicity)
    restrictions = {
        (0, 1): (np.eye(2), np.eye(2)),
        (1, 2): (np.eye(2), np.eye(2)),
        (2, 0): (np.eye(2), np.eye(2))
    }

    # Initialise fused sheaf
    sheaf = init_fused_sheaf(node_dims, edges, sections, restrictions)

    # Compute Gaussian edge weights from sections
    compute_gaussian_weights(sheaf, sigma=0.5)

    # Derive entropy per node
    compute_node_entropies(sheaf)

    # Generate chaotic solutions modulated by entropy
    sols = chaotic_omni_front_synthesis(sheaf, n_samples=20)

    # Evaluate cost
    cost = evaluate_fused_cost(sheaf, sols)

    # Simple output
    print("Gaussian edge weights:")
    for e, w in sheaf._gaussian_weights.items():
        print(f"  {e}: {w:.4f}")

    print("\nNode entropies:")
    for n, e in sheaf._entropy.items():
        print(f"  Node {n}: {e:.4f}")

    print(f"\nGenerated {len(sols)} solutions, total fused cost = {cost:.4f}")

    sys.exit(0)