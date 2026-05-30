# DARWIN HAMMER — match 5276, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m1140_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_shanno_m1769_s0.py (gen6)
# born: 2026-05-30T00:01:01Z

"""Hybrid Physarum‑Voronoi‑WTA Algorithm
====================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_hybrid_physar_hybrid_hybrid_ternar_m1140_s0.py``  
  Implements a Physarum‑inspired network where edge conductances are updated
  from fluxes driven by node “pressures”.  Pressures are obtained from Fisher
  information scores computed on text data.

* **Parent B** – ``hybrid_hybrid_hybrid_vorono_hybrid_hybrid_shanno_m1769_s0.py``  
  Provides a Voronoi partition of points, a Sparse Winner‑Take‑All (WTA)
  transformation of similarity vectors, and a Shannon‑entropy measure of the
  sparse binary tags.

Mathematical Bridge
-------------------
The bridge is built on the *spatial* representation of the Physarum network:

* Nodes of the Physarum graph are the **Voronoi seeds** produced by Parent B.
* Edge lengths are the Euclidean distances between seed points.
* Edge adjacency is derived from a distance‑based neighbourhood (a proxy for
  the Delaunay graph of the Voronoi diagram).

The **pressures** driving flux are the Fisher‑information scores of textual
data attached to each node.  The **gain** of the conductance update is
modulated by the **Shannon entropy** of the Sparse‑WTA output that measures
how “informative” the similarity pattern of a data point is with respect to
the seeds.  In this way the Voronoi‑WTA side supplies a data‑driven scaling
factor for the Physarum dynamics.

The resulting hybrid step consists of:

1. Build a Voronoi partition → adjacency & length matrices.
2. Compute node pressures from Fisher information.
3. Compute similarity vectors (inverse distances) → Sparse‑WTA → entropy.
4. Update edge conductances using Physarum flux, with gain multiplied by
   ``1 + entropy`` (entropy in ``[0, 1]`` after normalisation).

The code below implements this unified system with a minimal public API
and a smoke‑test that runs without external data."""

import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Any, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures (kept compatible with Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Parent A core functions (physarum)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """One‑step conductance update."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def fisher_information(text: str, feature_regex: re.Pattern) -> float:
    """Simple Fisher information: count of regex matches."""
    matches = feature_regex.findall(text)
    return float(len(matches))

# ----------------------------------------------------------------------
# Parent B core functions (Voronoi + Sparse‑WTA + Entropy)
# ----------------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    if not seeds.size:
        raise ValueError('seeds required')
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Binary region matrix: rows = seeds, columns = points."""
    regions = np.zeros((seeds.shape[0], points.shape[0]), dtype=int)
    for i, p in enumerate(points):
        regions[nearest(p, seeds), i] = 1
    return regions


def sparse_wta(similarity_vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """Sparse Winner‑Take‑All: keep top‑k entries per row, zero elsewhere."""
    num_vectors, num_features = similarity_vectors.shape
    wta = np.zeros_like(similarity_vectors, dtype=int)
    for i in range(num_vectors):
        top_k = np.argsort(-similarity_vectors[i])[:k]
        wta[i, top_k] = 1
    return wta


def shannon_entropy(binary_vector: np.ndarray) -> float:
    """Entropy of a binary vector (p = proportion of ones)."""
    p = binary_vector.mean()
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def build_length_matrix(nodes: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Euclidean distance matrix between all node positions."""
    ids = list(nodes.keys())
    coords = np.array([nodes[i] for i in ids])
    diff = coords[:, None, :] - coords[None, :, :]
    lengths = np.linalg.norm(diff, axis=2)
    return lengths


def build_adjacency_matrix(lengths: np.ndarray,
                           max_dist: float) -> np.ndarray:
    """Binary adjacency: 1 if distance ≤ max_dist and not self‑loop."""
    adj = (lengths <= max_dist).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def compute_pressures(node_texts: Dict[str, str],
                      feature_regex: re.Pattern) -> np.ndarray:
    """Pressure per node = Fisher information of its associated text."""
    pressures = np.array([fisher_information(node_texts[n], feature_regex)
                          for n in sorted(node_texts.keys())],
                         dtype=float)
    return pressures


def similarity_vectors(points: np.ndarray,
                       seeds: np.ndarray) -> np.ndarray:
    """
    Similarity = 1 / (1 + Euclidean distance) between each point and each seed.
    Returns matrix of shape (num_points, num_seeds).
    """
    dists = np.linalg.norm(points[:, None, :] - seeds[None, :, :], axis=2)
    return 1.0 / (1.0 + dists)


def sparse_wta_entropy(sim_vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Apply Sparse‑WTA to similarity vectors and compute normalized Shannon entropy
    for each row. Normalisation divides by the maximal possible entropy (log2(k)).
    """
    wta = sparse_wta(sim_vectors, k)
    ent = np.apply_along_axis(shannon_entropy, 1, wta)
    max_ent = math.log(k, 2) if k > 1 else 1.0
    return ent / max_ent  # values in [0, 1]


def hybrid_conductance_update(conductances: np.ndarray,
                              pressures: np.ndarray,
                              lengths: np.ndarray,
                              adjacency: np.ndarray,
                              entropy: np.ndarray,
                              dt: float = 1.0,
                              base_gain: float = 1.0,
                              decay: float = 0.05) -> np.ndarray:
    """
    Single hybrid Physarum‑Voronoi step.

    * ``conductances`` – symmetric matrix (N×N) of current edge conductances.
    * ``pressures``    – vector (N,) of node pressures.
    * ``lengths``      – matrix (N×N) of Euclidean edge lengths.
    * ``adjacency``    – binary matrix (N×N) indicating existing edges.
    * ``entropy``      – vector (N,) where each entry is the average entropy
                         of points belonging to the corresponding Voronoi cell.
                         Used to modulate the gain per node.

    The gain for edge (i, j) is ``base_gain * (1 + 0.5*(e_i + e_j))``.
    """
    N = pressures.shape[0]
    new_cond = conductances.copy()
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i, j] == 0:
                continue
            q = flux(conductances[i, j], lengths[i, j],
                     pressures[i], pressures[j])
            # node‑specific entropy scaling
            gain = base_gain * (1.0 + 0.5 * (entropy[i] + entropy[j]))
            new_cond[i, j] = update_conductance(conductances[i, j], q,
                                                dt=dt, gain=gain, decay=decay)
            new_cond[j, i] = new_cond[i, j]  # keep symmetry
    return new_cond


def hybrid_step(points: np.ndarray,
                seeds: np.ndarray,
                node_texts: Dict[str, str],
                feature_regex: re.Pattern,
                conductances: np.ndarray,
                max_adj_dist: float = 2.0,
                wta_k: int = 5,
                dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute one full hybrid iteration.

    Returns:
        new_conductances – updated conductance matrix.
        entropy_per_node – entropy vector used for the update.
    """
    # 1. Geometry
    node_ids = sorted(node_texts.keys())
    length_mat = build_length_matrix({nid: seeds[i] for i, nid in enumerate(node_ids)})
    adj_mat = build_adjacency_matrix(length_mat, max_adj_dist)

    # 2. Pressures from Fisher information
    pressures = compute_pressures(node_texts, feature_regex)

    # 3. Similarity → Sparse‑WTA → entropy per point
    sim_vecs = similarity_vectors(points, seeds)
    ent_per_point = sparse_wta_entropy(sim_vecs, k=wta_k)

    # 4. Aggregate point entropy to node (Voronoi) entropy
    region_mat = assign(points, seeds)                # seeds×points binary
    # avoid division by zero
    counts = region_mat.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1
    entropy_per_node = (region_mat @ ent_per_point) / counts.squeeze()

    # 5. Conductance update
    new_cond = hybrid_conductance_update(conductances,
                                         pressures,
                                         length_mat,
                                         adj_mat,
                                         entropy_per_node,
                                         dt=dt)

    return new_cond, entropy_per_node


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic 2‑D points and seed locations
    num_points = 100
    num_seeds = 6
    points = np.random.rand(num_points, 2) * 10.0
    seeds = np.random.rand(num_seeds, 2) * 10.0

    # Dummy texts for each seed (node).  Use a simple regex that matches the
    # letter “a” – the count will serve as a pressure proxy.
    node_texts = {f"node_{i}": "a " * random.randint(0, 20) for i in range(num_seeds)}
    feature_regex = re.compile(r"a")

    # Initialise conductances uniformly (positive values)
    init_cond = np.full((num_seeds, num_seeds), 0.5)
    np.fill_diagonal(init_cond, 0.0)   # no self‑loops

    # Run a few hybrid iterations
    conductances = init_cond
    for step in range(3):
        conductances, entropy = hybrid_step(points,
                                            seeds,
                                            node_texts,
                                            feature_regex,
                                            conductances,
                                            max_adj_dist=5.0,
                                            wta_k=5,
                                            dt=1.0)
        print(f"Step {step+1}: mean conductance = {conductances.mean():.4f}, "
              f"mean entropy = {entropy.mean():.4f}")

    sys.exit(0)