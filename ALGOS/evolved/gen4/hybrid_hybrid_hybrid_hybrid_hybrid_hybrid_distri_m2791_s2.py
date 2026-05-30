# DARWIN HAMMER — match 2791, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_bandit_m269_s0.py (gen3)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2.py (gen2)
# born: 2026-05-29T23:45:55Z

"""Hybrid Algorithm: Sheaf‑Laplacian Guided Perceptual‑Graph Workshare Allocation

Parents:
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_bandit_m269_s0.py (Sheaf Laplacian + sketch)
- hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s2.py (Perceptual‑hash graph + kinetic strike score)

Mathematical Bridge:
The perceptual‑hash similarity graph G provides a discrete Laplacian L_G = D – A
where A is the adjacency matrix.  Independently, a cellular sheaf 𝓕 over the same
node set yields the sheaf Laplacian L_𝓕 = δᵀδ (δ – restriction coboundary).  We fuse the
two by using the diagonal energy e_i = (L_𝓕)_{ii}² of the sheaf Laplacian to
modulate the edge weights of G:

    w_{ij} = a_{ij}·(1 + α·e_i)·(1 + α·e_j),

with α a tunable coupling constant.  The resulting weighted graph encodes both
topological similarity (hash distance) and sheaf‑induced “energy”.  A kinetic
score κ_i derived from the element vector (interpreted as a force‑time series)
biases the final work‑share allocation:

    s_i ∝ κ_i·deg_w(i),

where deg_w(i)=∑_j w_{ij}.  The allocation vector s is normalised to sum to one.
Thus the hybrid system simultaneously respects sheaf cohomology structure,
perceptual similarity, and physics‑driven importance.

The module implements the full pipeline and provides three core functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Set, Hashable

import numpy as np

# ---------------------------------------------------------------------------
# Sheaf utilities (Parent A core)
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf over a graph.

    Nodes carry vector spaces of prescribed dimensions; edges carry restriction
    maps from the incident node spaces to a common edge space.  For the purpose
    of this hybrid algorithm we only need the combinatorial Laplacian L = δᵀδ.
    """

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        """
        Parameters
        ----------
        node_dims: dict mapping node index → dimension (int)
        edge_list: list of directed edges (u, v)
        """
        self.node_dims = dict(node_dims)          # node_id -> int
        self.edges = list(edge_list)              # list of (u, v) tuples, orientation matters

    def compute_laplacian(self) -> np.ndarray:
        """Return the (combinatorial) sheaf Laplacian L = δᵀδ.

        The implementation follows the toy version from the parent: each edge
        contributes -1 to (u,v) and +1 to (v,u).  The resulting matrix is square
        with size = number of nodes.
        """
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            L[u, v] -= 1.0
            L[v, u] += 1.0
        return L

def sheaf_laplacian_energy(sheaf: Sheaf) -> Dict[int, float]:
    """Compute a scalar “energy” per node from the sheaf Laplacian.

    Energy is defined as the squared diagonal entry:
        e_i = (L_ii)²
    Returns a dict node → energy.
    """
    L = sheaf.compute_laplacian()
    energies = {i: float(L[i, i] ** 2) for i in range(L.shape[0])}
    return energies

# ---------------------------------------------------------------------------
# Perceptual‑hash graph utilities (Parent B core)
# ---------------------------------------------------------------------------

def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash based on average comparison."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit strings."""
    return (a ^ b).bit_count()

def build_phash_graph(elements: List[List[float]], max_hamming: int = 4) -> Tuple[Dict[int, Set[int]], List[int]]:
    """Construct an undirected similarity graph from perceptual hashes.

    Nodes i and j are connected iff their hashes differ by at most `max_hamming`
    bits.  Returns adjacency dict and the ordered list of node ids.
    """
    n = len(elements)
    hashes = [compute_phash(el) for el in elements]
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= max_hamming:
                graph[i].add(j)
                graph[j].add(i)
    return graph, list(range(n))

# ---------------------------------------------------------------------------
# Kinetic (strike) score utilities (Parent B physics side)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrikeState:
    """Simple kinetic descriptor for a vector of forces."""
    velocity: float
    distance: float
    peak: float

def kinetic_score_from_vector(vec: List[float]) -> StrikeState:
    """Interpret `vec` as a time‑series of forces and return a kinetic descriptor.

    - velocity ≈ sum of absolute differences (discrete derivative)
    - distance  ≈ sum of absolute values (L1 norm)
    - peak      ≈ max absolute value
    """
    if not vec:
        return StrikeState(0.0, 0.0, 0.0)
    diffs = [abs(vec[i] - vec[i - 1]) for i in range(1, len(vec))]
    velocity = sum(diffs)
    distance = sum(abs(v) for v in vec)
    peak = max(abs(v) for v in vec)
    return StrikeState(velocity, distance, peak)

def scalar_kinetic_metric(state: StrikeState, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    """Combine the three components of a StrikeState into a single scalar."""
    w_vel, w_dist, w_peak = weights
    return w_vel * state.velocity + w_dist * state.distance + w_peak * state.peak

# ---------------------------------------------------------------------------
# Hybrid core functions (the new contribution)
# ---------------------------------------------------------------------------

def build_hybrid_weighted_graph(
    elements: List[List[float]],
    alpha: float = 0.5
) -> Tuple[np.ndarray, List[int]]:
    """Create a weighted adjacency matrix that fuses sheaf Laplacian energy with
    the perceptual‑hash similarity graph.

    Steps
    -----
    1. Build the unweighted hash graph G = (V, A) using `build_phash_graph`.
    2. Construct a sheaf 𝓕 over the same node set:
       - node dimensions are set to the length of each element vector.
       - edges are taken directly from G (undirected → two directed edges).
    3. Compute node energies e_i from the sheaf Laplacian.
    4. For each edge (i, j) set weight:
           w_{ij} = a_{ij}·(1 + α·e_i)·(1 + α·e_j)
       where a_{ij}=1 if the edge exists, else 0.
    5. Return the symmetric weighted adjacency matrix W and the node ordering.
    """
    # 1. Perceptual‑hash graph
    unweighted_adj, nodes = build_phash_graph(elements)
    n = len(nodes)

    # 2. Sheaf definition (node_dims = vector length, directed edges from adjacency)
    node_dims = {i: len(elements[i]) for i in nodes}
    directed_edges = []
    for i in nodes:
        for j in unweighted_adj[i]:
            directed_edges.append((i, j))   # keep orientation as given

    sheaf = Sheaf(node_dims=node_dims, edge_list=directed_edges)

    # 3. Energy per node
    energies = sheaf_laplacian_energy(sheaf)

    # 4. Build weighted matrix
    W = np.zeros((n, n), dtype=float)
    for i in nodes:
        for j in unweighted_adj[i]:
            if i < j:  # ensure each undirected edge processed once
                e_i = energies[i]
                e_j = energies[j]
                weight = (1.0 + alpha * e_i) * (1.0 + alpha * e_j)
                W[i, j] = weight
                W[j, i] = weight
    return W, nodes

def hybrid_node_scores(
    elements: List[List[float]],
    alpha: float = 0.5,
    kinetic_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> Dict[int, float]:
    """Compute a hybrid importance score for each node.

    The score combines:
      • the weighted degree from the hybrid graph (graph topology + sheaf energy)
      • a scalar kinetic metric derived from the element vector.

    Returns a dict node → raw (non‑normalised) score.
    """
    W, nodes = build_hybrid_weighted_graph(elements, alpha=alpha)

    # Weighted degree per node
    weighted_degrees = {i: float(W[i].sum()) for i in nodes}

    # Kinetic metric per node
    kinetic_metrics = {}
    for i, vec in enumerate(elements):
        state = kinetic_score_from_vector(vec)
        kinetic_metrics[i] = scalar_kinetic_metric(state, weights=kinetic_weights)

    # Hybrid score = degree * kinetic
    scores = {
        i: weighted_degrees[i] * kinetic_metrics[i]
        for i in nodes
    }
    return scores

def allocate_workshare(
    elements: List[List[float]],
    alpha: float = 0.5,
    kinetic_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> Dict[int, float]:
    """Produce a normalized work‑share allocation vector based on hybrid scores.

    The allocation a_i for node i satisfies ∑_i a_i = 1 and reflects both
    topological (sheaf + hash) and physical (kinetic) considerations.
    """
    raw_scores = hybrid_node_scores(elements, alpha=alpha, kinetic_weights=kinetic_weights)
    total = sum(raw_scores.values())
    if total == 0.0:
        # fallback to uniform allocation
        n = len(elements)
        return {i: 1.0 / n for i in range(n)}
    allocation = {i: score / total for i, score in raw_scores.items()}
    return allocation

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate a modest synthetic dataset: 10 elements, each a random float vector
    random.seed(42)
    num_elements = 10
    dim = 8
    elements = [[random.uniform(-5, 5) for _ in range(dim)] for _ in range(num_elements)]

    # Run the hybrid allocation
    alloc = allocate_workshare(elements, alpha=0.3)

    # Print results in a readable form
    print("Hybrid work‑share allocation (node → proportion):")
    for node_id in sorted(alloc.keys()):
        print(f"  Node {node_id:2d}: {alloc[node_id]:.4f}")

    # Verify that allocations sum to 1 (within floating‑point tolerance)
    total_alloc = sum(alloc.values())
    print(f"\nTotal allocation sum: {total_alloc:.12f} (should be 1.0)")