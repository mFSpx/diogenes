# DARWIN HAMMER — match 4842, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_krampu_nlms_m2094_s1.py (gen5)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# born: 2026-05-29T23:58:20Z

"""Hybrid Krampus‑NLMS‑Curvature Module
Parent A: hybrid_hybrid_hybrid_krampu_nlms_m2094_s1.py
Parent B: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py

Mathematical Bridge
-------------------
Both parents manipulate high‑dimensional feature vectors and linear
operations:

* **Parent A** builds a *sheaf* whose node sections are scalar signals
  derived from raw features and updates a Normalised LMS (NLMS) weight
  vector **w** using the classic rule  
  **w ← w + μ · e · x / (‖x‖²+ε)**.

* **Parent B** treats each master feature vector **vᵢ** as a graph node,
  computes an Ollivier‑Ricci curvature κᵢ for every node and injects the
  scalar curvature score into the Krampus brain‑map projection.

The fusion therefore identifies the *signal vector* **x** used by NLMS
with the *augmented feature vector* composed of the original sheaf
sections **plus** the curvature score.  The curvature becomes an
additional dimension in the NLMS adaptation, allowing the adaptive
filter to learn a mapping that respects both local pheromone‑type
signals and global graph‑geometric structure.

The module implements:
1. Construction of the curvature‑augmented graph (Parent B).
2. Generation of pheromone signals and assembly of a HybridSheaf
   (Parent A).
3. NLMS adaptation on the concatenated signal‑curvature vector.
"""

import math
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Pheromone / Sheaf / NLMS primitives
# ---------------------------------------------------------------------------

class PheromoneEntry:
    def __init__(self, feature, value, half_life):
        self.feature = feature          # hashable identifier
        self.value = value              # raw data (list, scalar, …)
        self.half_life = half_life
        self.signal = value             # for this toy version signal == value

class HybridSheaf:
    """A lightweight sheaf where each node holds a 1‑D section."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]],
                 width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)          # node → dimension (here always 1)
        self.edges = list(edge_list)
        self._restrictions = {}                  # (u,v) → (src_map, dst_map)
        self._sections = {}                      # node → np.ndarray
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: Tuple[int, int], src_map, dst_map):
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value):
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node):
        return self._sections[node]

    def _edge_dim(self, u, v):
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction for edge ({u},{v})")

class NLMS:
    """Normalised Least‑Mean‑Squares adaptive filter."""
    def __init__(self, weights: np.ndarray, mu: float = 0.5, eps: float = 1e-9):
        self.weights = np.array(weights, dtype=float)
        self.mu = mu
        self.eps = eps

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x))

    def update(self, x: np.ndarray, target: float) -> Tuple[np.ndarray, float]:
        if self.weights.shape != x.shape:
            raise ValueError('weights and input must have equal length')
        y = self.predict(x)
        error = target - y
        power = float(np.dot(x, x) + self.eps)
        self.weights = self.weights + self.mu * error * x / power
        return self.weights, error

def krampus_sticker_to_signals(feature_vector: Tuple[List[float], float, List[float]]) -> List[PheromoneEntry]:
    """Transform a (tokens, entropy, link_counts) tuple into pheromone entries."""
    tokens, entropy, link_counts = feature_vector
    signals = []
    for feature in (tokens, entropy, link_counts):
        half_life = math.exp(entropy)               # monotonic function of entropy
        # signal is reciprocal of length to keep magnitude bounded
        signal = 1.0 / (len(feature) if hasattr(feature, '__len__') else 1.0)
        signals.append(PheromoneEntry(feature, signal, half_life))
    return signals

def aggregate_signals(signals: List[PheromoneEntry],
                      node_dims: Dict[int, int],
                      edge_list: List[Tuple[int, int]]) -> HybridSheaf:
    """Create a sheaf whose node sections are the scalar signals."""
    sheaf = HybridSheaf(node_dims, edge_list)
    for idx, signal in enumerate(signals):
        node_id = idx  # simple 0‑based mapping
        sheaf.set_section(node_id, [signal.signal])
    return sheaf

# ---------------------------------------------------------------------------
# Parent B – Graph construction and Ollivier‑Ricci curvature (simplified)
# ---------------------------------------------------------------------------

def hybrid_build_adj(master_vectors: List[np.ndarray],
                    distance_thresh: float = 1.5) -> List[Tuple[int, int]]:
    """Build an undirected adjacency list by thresholding Euclidean distances."""
    n = len(master_vectors)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(master_vectors[i] - master_vectors[j])
            if dist <= distance_thresh:
                edges.append((i, j))
    return edges

def _lazy_random_walk_distribution(node: int,
                                   adjacency: Dict[int, List[int]]) -> Dict[int, float]:
    """Uniform distribution over the node and its neighbours (lazy walk)."""
    neighbours = adjacency.get(node, [])
    support = [node] + neighbours
    prob = 1.0 / len(support)
    return {v: prob for v in support}

def _wasserstein_1(dist_i: Dict[int, float],
                  dist_j: Dict[int, float],
                  edge_len: float) -> float:
    """Simplified 1‑Wasserstein distance for two discrete uniform measures.
    Because the support sets are tiny we approximate by the total variation
    multiplied by the edge length."""
    all_keys = set(dist_i) | set(dist_j)
    tv = 0.0
    for k in all_keys:
        tv += abs(dist_i.get(k, 0.0) - dist_j.get(k, 0.0))
    return 0.5 * tv * edge_len

def hybrid_node_curvature(master_vectors: List[np.ndarray],
                          edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute average Ollivier‑Ricci curvature per node."""
    # Build adjacency dict
    adjacency: Dict[int, List[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    curvatures = np.zeros(len(master_vectors))
    for u, v in edges:
        d_uv = np.linalg.norm(master_vectors[u] - master_vectors[v]) + 1e-12
        mu_u = _lazy_random_walk_distribution(u, adjacency)
        mu_v = _lazy_random_walk_distribution(v, adjacency)
        W1 = _wasserstein_1(mu_u, mu_v, d_uv)
        kappa = 1.0 - W1 / d_uv
        curvatures[u] += kappa
        curvatures[v] += kappa

    # Average over incident edges
    degree = np.array([len(adjacency.get(i, [])) for i in range(len(master_vectors))], dtype=float)
    # Avoid division by zero
    degree[degree == 0] = 1.0
    curvatures = curvatures / degree
    return curvatures

def hybrid_brain_xyz(master_vectors: List[np.ndarray],
                    curvature_scores: np.ndarray) -> np.ndarray:
    """Project each master vector to 3‑D, augmenting with curvature as the third axis."""
    # Simple linear projection onto first two axes
    proj = np.column_stack((master_vectors[:, 0], master_vectors[:, 1]))
    # Normalise curvature to comparable scale
    curv_norm = (curvature_scores - curvature_scores.min()) / (curvature_scores.ptp() + 1e-12)
    xyz = np.column_stack((proj, curv_norm))
    return xyz

# ---------------------------------------------------------------------------
# Hybrid Operations – Fusion of the two pipelines
# ---------------------------------------------------------------------------

def build_hybrid_sheaf_from_text(text: str,
                                 master_vectors: List[np.ndarray],
                                 edge_list: List[Tuple[int, int]]) -> Tuple[HybridSheaf, np.ndarray]:
    """
    1. Extract dummy Krampus features from the raw text.
    2. Convert them to pheromone signals.
    3. Build a sheaf whose sections are those signals.
    4. Compute curvature scores for the graph defined by master_vectors.
    5. Return the sheaf and the curvature vector (aligned by node index).
    """
    # ---- Step 1: dummy feature extraction ---------------------------------
    # For reproducibility we turn the text into a deterministic pseudo‑feature.
    random.seed(hash(text))
    tokens = [random.random() for _ in range(5)]          # 5 token‑like numbers
    entropy = -sum(t * math.log(t + 1e-12) for t in tokens)  # Shannon‑like entropy
    link_counts = [random.randint(0, 3) for _ in range(3)]

    # ---- Step 2: pheromone signals ----------------------------------------
    signals = krampus_sticker_to_signals((tokens, entropy, link_counts))

    # ---- Step 3: sheaf assembly --------------------------------------------
    node_dims = {i: 1 for i in range(len(signals))}
    sheaf = aggregate_signals(signals, node_dims, edge_list)

    # ---- Step 4: curvature computation --------------------------------------
    curvature = hybrid_node_curvature(master_vectors, edge_list)

    return sheaf, curvature

def nlms_adapt_on_hybrid(sheaf: HybridSheaf,
                         curvature: np.ndarray,
                         nlms: NLMS,
                         target: float) -> Tuple[np.ndarray, float]:
    """
    Build the NLMS input vector by concatenating:
      * the scalar sheaf sections (ordered by node index)
      * the corresponding curvature score for each node
    Perform one NLMS update step toward the supplied target.
    """
    # Extract ordered sections
    n_nodes = len(sheaf._sections)
    section_vec = np.empty(n_nodes, dtype=float)
    for i in range(n_nodes):
        section_vec[i] = sheaf.get_section(i)[0]

    # Concatenate curvature (same ordering)
    x = np.concatenate([section_vec, curvature[:n_nodes]])
    # Ensure NLMS weight dimension matches
    if nlms.weights.shape != x.shape:
        raise ValueError("NLMS weight dimension does not match hybrid input dimension")
    updated_weights, error = nlms.update(x, target)
    return updated_weights, error

def hybrid_predict_coordinates(text: str,
                               master_vectors: List[np.ndarray],
                               nlms: NLMS) -> np.ndarray:
    """
    End‑to‑end hybrid prediction:
      1. Build adjacency from master vectors.
      2. Assemble sheaf + curvature from the raw text.
      3. Run one NLMS adaptation step with a dummy target (here we use 0.0).
      4. Return the 3‑D brain‑map coordinates that include curvature.
    """
    edges = hybrid_build_adj(master_vectors)
    sheaf, curvature = build_hybrid_sheaf_from_text(text, master_vectors, edges)

    # Dummy target – in a real system this would be a supervised label.
    target = 0.0
    nlms_adapt_on_hybrid(sheaf, curvature, nlms, target)

    # Final coordinates using the curvature‑augmented brain map
    coords = hybrid_brain_xyz(np.array(master_vectors), curvature)
    return coords

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a small synthetic dataset of master vectors (dimension 4)
    np.random.seed(42)
    master_vecs = [np.random.randn(4) for _ in range(6)]

    # Build graph adjacency once (used by all calls)
    adj_edges = hybrid_build_adj(master_vecs)

    # Initialise NLMS with a weight vector matching the hybrid input size:
    # number of sheaf nodes == number of signals == 3 (tokens, entropy, link_counts)
    # plus curvature per node (same count).  We'll use the length of adj_edges
    # to decide node count; for the test we take the first 3 nodes.
    n_nodes = 3
    input_dim = n_nodes + n_nodes   # sections + curvature
    initial_weights = np.zeros(input_dim)

    nlms_filter = NLMS(initial_weights, mu=0.3)

    # Run the hybrid pipeline on a sample text
    sample_text = "The quick brown fox jumps over the lazy dog."
    coords = hybrid_predict_coordinates(sample_text, master_vecs, nlms_filter)

    print("Hybrid 3‑D coordinates (first 5 nodes):")
    print(coords[:5])
    print("Final NLMS weights:")
    print(nlms_filter.weights)