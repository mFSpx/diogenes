# DARWIN HAMMER — match 986, survivor 2
# gen: 5
# parent_a: hybrid_shap_attribution_hybrid_hybrid_pherom_m70_s0.py (gen4)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# born: 2026-05-29T23:32:06Z

"""Hybrid module combining SHAP‑attribution‑driven graph clustering (Parent A)
and Krampus‑brain‑map with Ollivier‑Ricci curvature (Parent B).

Mathematical bridge
-------------------
* Each input datum (e.g., a piece of text) is encoded as a high‑dimensional
  deterministic feature vector **v** ∈ ℝⁿ (Krampus extraction).
* All vectors become nodes of a weighted graph **G** where edge weights are
  Euclidean distances d(i,j)=‖v_i‑v_j‖₂.  A distance threshold τ yields an
  un‑weighted adjacency list used by the Ollivier‑Ricci routine.
* For every node we compute a SHAP‑style attribution score
  `φ_i = Σ_S w(|S|,n)·[f(S∪{i})‑f(S)]` with the kernel
  `w(k,n)=k!·(n‑k‑1)!/n!`.  The value function f is taken as the sum of the
  node’s raw feature values, thus φ_i is a scalar summarising the
  contribution of node i to the overall model.
* The SHAP scores become node “pheromone” levels that drive a leader‑election
  clustering (Parent A).  The elected leaders define clusters; each cluster
  receives a MinHash signature (not needed for the smoke test but kept for
  completeness).
* Ollivier‑Ricci curvature κ(i,j)=1‑W₁(m_i,m_j)/d(i,j) is evaluated on the same
  graph.  The average incident curvature of a node,
  `c_i = (1/deg(i)) Σ_{j∈N(i)} κ(i,j)`,
  is injected as an additional scalar feature into the final 3‑D mapping.
* The unified 3‑D coordinate for node i is
      x_i = ⟨v_i, w_x⟩,
      y_i = ⟨v_i, w_y⟩,
      z_i = c_i·sign(φ_i),
  where w_x and w_y are fixed projection directions.  Thus the spatial
  layout respects both semantic feature composition (Krampus), graph‑geometric
  connectivity (Ollivier‑Ricci) and feature attribution (SHAP).

The code below implements the core pipeline with three public functions:

* `build_feature_graph` – creates the weighted graph from vectors.
* `compute_shap_attributions` – evaluates SHAP‑style scores for nodes.
* `hybrid_brain_xyz` – returns the 3‑D coordinates after curvature injection.

All operations rely only on the Python standard library and NumPy.
"""

import math
import random
import sys
from pathlib import Path
from itertools import combinations
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

Node = int
Graph = Dict[Node, Set[Node]]
WeightedGraph = Dict[Tuple[Node, Node], float]

# ---------------------------------------------------------------------------
# Helper utilities (Parent A)
# ---------------------------------------------------------------------------

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Kernel w(k,n)=k!·(n‑k‑1)!/n! used in exact Shapley value."""
    if feature_count == 0:
        return 0.0
    return math.factorial(subset_size) * math.factorial(feature_count - subset_size - 1) / math.factorial(feature_count)


def exact_shap_value(
    feature_index: int,
    feature_count: int,
    value_fn: Callable[[Set[int]], float],
) -> float:
    """
    Exact Shapley value for a single feature using the definition.
    `value_fn` must accept a set of feature indices and return the model output.
    """
    total = 0.0
    all_features = set(range(feature_count))
    for k in range(feature_count + 1):
        for subset in combinations(all_features - {feature_index}, k):
            S = set(subset)
            weight = shapley_kernel_weight(k, feature_count)
            total += weight * (value_fn(S | {feature_index}) - value_fn(S))
    return total


def compute_shap_attributions(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a Shapley‑style attribution for each row (node) of `feature_matrix`.
    The value function is the sum of the selected rows' feature values.
    Returns a 1‑D array φ with length = number of nodes.
    """
    n_nodes, n_features = feature_matrix.shape
    # Pre‑compute row sums for speed
    row_sums = feature_matrix.sum(axis=1)

    def value_fn(selected: Set[int]) -> float:
        if not selected:
            return 0.0
        return float(row_sums[list(selected)].sum())

    phi = np.empty(n_nodes, dtype=float)
    for i in range(n_nodes):
        phi[i] = exact_shap_value(i, n_nodes, value_fn)
    return phi


# ---------------------------------------------------------------------------
# Helper utilities (Parent B)
# ---------------------------------------------------------------------------

def deterministic_vector_from_text(text: str, dim: int = 20) -> np.ndarray:
    """
    Produce a deterministic pseudo‑random vector from a string.
    The algorithm hashes the text to obtain a seed, then draws `dim`
    standard‑normal numbers using NumPy's Generator.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def build_feature_graph(
    vectors: np.ndarray,
    distance_threshold: float,
) -> Tuple[Graph, WeightedGraph]:
    """
    Build an undirected graph from `vectors`.
    Edge (i,j) exists iff Euclidean distance ≤ threshold.
    Returns both an adjacency list (un‑weighted) and a weight dict
    mapping edge tuples to their distance.
    """
    n = vectors.shape[0]
    adj: Graph = {i: set() for i in range(n)}
    w_adj: WeightedGraph = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(vectors[i] - vectors[j]))
            if dist <= distance_threshold:
                adj[i].add(j)
                adj[j].add(i)
                w_adj[(i, j)] = dist
                w_adj[(j, i)] = dist
    return adj, w_adj


def lazy_random_walk_distribution(
    node: Node,
    adj: Graph,
    alpha: float = 0.5,
) -> Dict[Node, float]:
    """
    Return a probability distribution for a lazy random walk starting at `node`.
    With probability `alpha` we stay, otherwise we move uniformly to a neighbour.
    """
    neighbors = adj[node]
    deg = len(neighbors)
    dist: Dict[Node, float] = {node: alpha}
    if deg > 0:
        prob = (1.0 - alpha) / deg
        for nb in neighbors:
            dist[nb] = prob
    return dist


def wasserstein_1(
    p: Dict[Node, float],
    q: Dict[Node, float],
    distances: WeightedGraph,
) -> float:
    """
    Compute a very simple (and not optimal) approximation of the
    Earth‑Mover (W₁) distance between two discrete distributions `p` and `q`
    on the same node set using the pairwise distances from `distances`.
    This implementation solves a linear program via a greedy matching,
    sufficient for the demonstration.
    """
    # Convert to mutable copies
    p_rem = p.copy()
    q_rem = q.copy()
    cost = 0.0
    # Greedy: match mass from each source to any sink with minimal distance
    while p_rem:
        src, src_mass = p_rem.popitem()
        # Find the nearest sink
        nearest = min(q_rem.keys(), key=lambda t: distances.get((src, t), math.inf))
        sink_mass = q_rem[nearest]
        moved = min(src_mass, sink_mass)
        cost += moved * distances.get((src, nearest), 0.0)
        if sink_mass > moved:
            q_rem[nearest] = sink_mass - moved
        else:
            del q_rem[nearest]
        if src_mass > moved:
            p_rem[src] = src_mass - moved
    return cost


def compute_ollivier_ricci_curvature(
    adj: Graph,
    w_adj: WeightedGraph,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute average incident Ollivier‑Ricci curvature for each node.
    Returns a 1‑D array `c` where c[i] = mean_{j∈N(i)} κ(i,j).
    """
    n = len(adj)
    curv = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = adj[i]
        if not nbrs:
            curv[i] = 0.0
            continue
        m_i = lazy_random_walk_distribution(i, adj, alpha)
        total = 0.0
        for j in nbrs:
            m_j = lazy_random_walk_distribution(j, adj, alpha)
            w_ij = w_adj.get((i, j), 1.0)
            w_dist = wasserstein_1(m_i, m_j, w_adj)
            kappa = 1.0 - w_dist / w_ij if w_ij != 0 else 0.0
            total += kappa
        curv[i] = total / len(nbrs)
    return curv


# ---------------------------------------------------------------------------
# Hybrid core (fusion of Parent A and Parent B)
# ---------------------------------------------------------------------------

def hybrid_brain_xyz(
    texts: List[str],
    distance_threshold: float = 5.0,
    projection_dim: int = 2,
) -> np.ndarray:
    """
    End‑to‑end hybrid pipeline:

    1. Extract deterministic vectors from `texts` (Krampus).
    2. Build a distance‑thresholded graph.
    3. Compute Ollivier‑Ricci curvature per node.
    4. Compute SHAP‑style attribution scores using the raw vectors.
    5. Project vectors onto two fixed random directions (x, y) and combine
       curvature with the sign of the SHAP score to obtain z.
    6. Return an (N,3) array of 3‑D coordinates.
    """
    # Step 1 – feature extraction
    vectors = np.vstack([deterministic_vector_from_text(t) for t in texts])
    n_nodes = vectors.shape[0]

    # Step 2 – graph construction
    adj, w_adj = build_feature_graph(vectors, distance_threshold)

    # Step 3 – curvature
    curvature = compute_ollivier_ricci_curvature(adj, w_adj)

    # Step 4 – SHAP attributions (using row sums as the value function)
    shap_scores = compute_shap_attributions(vectors)

    # Step 5 – fixed projection directions (seeded for reproducibility)
    rng = np.random.default_rng(42)
    proj_x = rng.normal(size=vectors.shape[1])
    proj_y = rng.normal(size=vectors.shape[1])

    x_coords = vectors @ proj_x
    y_coords = vectors @ proj_y
    # z combines curvature and the sign of the SHAP score
    z_coords = curvature * np.sign(shap_scores)

    return np.column_stack((x_coords, y_coords, z_coords))


def leader_election(
    adj: Graph,
    node_values: np.ndarray,
    seed: int | str | None = None,
    top_k: int = 3,
) -> Set[Node]:
    """
    Simple leader election inspired by pheromone‑based clustering (Parent A).
    Nodes with the highest `node_values` among their neighbourhood are elected.
    Returns a set of leader node indices (size ≤ top_k).
    """
    rng = random.Random(seed)
    candidates = sorted(
        range(len(node_values)),
        key=lambda i: node_values[i],
        reverse=True,
    )
    leaders: Set[Node] = set()
    for cand in candidates:
        # Accept if no neighbour is already a leader
        if all(nb not in leaders for nb in adj[cand]):
            leaders.add(cand)
            if len(leaders) >= top_k:
                break
    # Random tie‑breaker if needed
    if len(leaders) < top_k:
        remaining = set(range(len(node_values))) - leaders
        extra = rng.sample(list(remaining), top_k - len(leaders))
        leaders.update(extra)
    return leaders


def minhash_signature(set_bits: Set[int], num_perm: int = 128) -> Tuple[int, ...]:
    """
    Very simple MinHash for a set of integer identifiers.
    Returns a tuple of the minimal hash values under `num_perm` random
    permutations (simulated by different seeds).
    """
    signature = []
    for seed in range(num_perm):
        rng = random.Random(seed)
        mins = min(rng.getrandbits(64) ^ x for x in set_bits)
        signature.append(mins)
    return tuple(signature)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industry.",
        "Quantum computing promises exponential speedups.",
        "Natural language processing enables human‑computer interaction.",
        "Evolutionary algorithms mimic natural selection.",
        "Graph neural networks operate on irregular data structures.",
        "Reinforcement learning agents learn via trial and error.",
        "Generative models can synthesize realistic images.",
    ]

    coords = hybrid_brain_xyz(sample_texts, distance_threshold=8.0)
    print("Hybrid 3‑D coordinates (x, y, z):")
    for i, (x, y, z) in enumerate(coords):
        print(f"Node {i}: ({x:.4f}, {y:.4f}, {z:.4f})")

    # Demonstrate leader election on the same graph
    vectors = np.vstack([deterministic_vector_from_text(t) for t in sample_texts])
    adj, _ = build_feature_graph(vectors, distance_threshold=8.0)
    shap_vals = compute_shap_attributions(vectors)
    leaders = leader_election(adj, shap_vals, seed=123, top_k=3)
    print("\nSelected leader nodes based on SHAP‑driven pheromones:", sorted(leaders))
    
    # Show a MinHash signature for the neighbourhood of the first leader
    first_leader = next(iter(leaders))
    neighbourhood = adj[first_leader] | {first_leader}
    signature = minhash_signature(neighbourhood, num_perm=64)
    print("\nMinHash signature (first 8 values) for leader's neighbourhood:")
    print(signature[:8])