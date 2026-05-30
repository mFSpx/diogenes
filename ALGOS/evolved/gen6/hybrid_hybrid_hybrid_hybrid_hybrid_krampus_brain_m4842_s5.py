# DARWIN HAMMER — match 4842, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_krampu_nlms_m2094_s1.py (gen5)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# born: 2026-05-29T23:58:20Z

"""Hybrid Krampus‑NLMS‑Sheaf‑Ricci Module

This module fuses the two parent algorithms:

* **Parent A** – a sheaf‑based representation (`HybridSheaf`) together with a
  Normalised LMS adaptive filter (`NLMS`).  Signals are produced by
  `krampus_sticker_to_signals` as `PheromoneEntry` objects, each carrying a
  feature, a signal value and a half‑life derived from entropy.

* **Parent B** – a Krampus feature extractor combined with an Ollivier‑Ricci
  curvature pipeline.  Texts become high‑dimensional master vectors,
  a graph is built from Euclidean distances, lazy random‑walk
  distributions are formed on each node and a curvature
  `κ(i,j)=1‑W₁(m_i,m_j)/d(i,j)` is computed.  The average incident
  curvature of a node is injected as an extra scalar feature.

**Mathematical bridge**

Both parents operate on a *graph‑like* topology:

* In Parent A the sheaf defines a set of nodes `V` with sections
  `σ(v)∈ℝ^{d_v}` and linear restrictions on edges `E`.
* In Parent B a graph `G=(V,E)` is constructed from the same node set,
  edge weights are Euclidean distances between node sections, and a
  curvature scalar `c(v)` is derived per node.

The fusion maps the sheaf sections to node vectors of `G`,
computes curvature `c(v)`, and **augments** each sheaf section with this
scalar.  The resulting augmented section vector is then fed to the NLMS
update, allowing the adaptive filter to learn from both local signal
strength (via the pheromone entry) and global geometric connectivity
(via curvature).

The code below implements this pipeline with three public functions:

* `build_hybrid_sheaf(text, node_dims, edge_list)` – extracts Krampus
  features from raw text, converts them to `PheromoneEntry` signals and
  assembles a `HybridSheaf`.
* `compute_node_curvature(sheaf, distance_thresh=1.5)` – builds a graph
  from the sheaf, computes lazy‑walk distributions, evaluates an
  approximate Wasserstein‑1 distance and returns a dict
  `{node: average_curvature}`.
* `nlms_hybrid_update(sheaf, nlms, target)` – concatenates each node’s
  augmented section (original signal + curvature) into a single input
  vector `x`, runs the NLMS prediction, and updates the weight vector.

A small smoke test demonstrates end‑to‑end execution. """

import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Parent A – Sheaf & NLMS
# ---------------------------------------------------------------------------

class PheromoneEntry:
    def __init__(self, feature, value, half_life):
        self.feature = feature          # arbitrary hashable identifier
        self.value = value              # raw signal magnitude
        self.half_life = half_life      # decay parameter (unused in this demo)
        self.signal = value             # the scalar emitted to the sheaf


class HybridSheaf:
    """A simple sheaf with node sections and linear edge restrictions."""
    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]],
                 width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)          # dim per node name
        self.edges = list(edge_list)              # list of (src, dst)
        self._restrictions = {}                   # (src,dst) -> (A,B)
        self._sections = {}                       # node -> np.ndarray
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: Tuple[str, str],
                        src_map: List[float], dst_map: List[float]) -> None:
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: str, value: List[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def section(self, node: str) -> np.ndarray:
        return self._sections[node]

    def nodes(self) -> List[str]:
        return list(self._sections.keys())

    def edges_iter(self):
        return iter(self.edges)


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
        power = np.dot(x, x) + self.eps
        self.weights = self.weights + self.mu * error * x / power
        return self.weights, error


def krampus_sticker_to_signals(feature_vector: Tuple[List[float], float, List[float]]) -> List[PheromoneEntry]:
    """Convert a Krampus feature tuple into pheromone signals."""
    tokens, entropy, link_counts = feature_vector
    signals = []
    for feature in (tokens, [entropy], link_counts):
        # half‑life grows exponentially with entropy – a monotone map
        half_life = math.exp(entropy) if isinstance(entropy, (int, float)) else math.exp(entropy[0])
        # signal strength inversely proportional to length (avoid division by zero)
        signal_val = 1.0 / max(len(feature), 1)
        signals.append(PheromoneEntry(feature, signal_val, half_life))
    return signals


def aggregate_signals(signals: List[PheromoneEntry],
                      node_dims: Dict[str, int],
                      edge_list: List[Tuple[str, str]]) -> HybridSheaf:
    """Create a sheaf where each signal populates a distinct node."""
    sheaf = HybridSheaf(node_dims, edge_list)
    for idx, signal in enumerate(signals):
        node_name = f"node_{idx}"
        # Ensure the node dimension matches the signal dimensionality (here 1)
        sheaf.set_section(node_name, [signal.signal])
    return sheaf


# ---------------------------------------------------------------------------
# Parent B – Krampus feature extraction + Ollivier‑Ricci curvature
# ---------------------------------------------------------------------------

def extract_full_features(text: str) -> Tuple[List[float], float, List[float]]:
    """
    Deterministic stub that returns:
    * tokens      – a list of token “weights” (hash of word length)
    * entropy     – a scalar measuring text complexity
    * link_counts – a list of outgoing link counts (simulated)
    """
    words = text.split()
    tokens = [float(len(w) % 5 + 1) for w in words]          # pseudo‑token values
    entropy = -sum((len(w) / len(text)) * math.log(len(w) / len(text) + 1e-12) for w in words)
    link_counts = [float(random.randint(0, 3)) for _ in words]
    return tokens, entropy, link_counts


def build_adjacency_from_sheaf(sheaf: HybridSheaf, thresh: float) -> Dict[str, List[str]]:
    """
    Construct an un‑weighted adjacency list from the sheaf.
    Edge weight = Euclidean distance between node sections.
    An edge is kept iff distance ≤ thresh.
    """
    nodes = sheaf.nodes()
    adj: Dict[str, List[str]] = defaultdict(list)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            duv = np.linalg.norm(sheaf.section(u) - sheaf.section(v))
            if duv <= thresh:
                adj[u].append(v)
                adj[v].append(u)
    return dict(adj)


def lazy_random_walk_distribution(adj: Dict[str, List[str]], node: str) -> Dict[str, float]:
    """
    Return a probability distribution for a lazy random walk:
    stay with 0.5 probability, otherwise move uniformly to a neighbour.
    """
    neighbours = adj.get(node, [])
    n = len(neighbours)
    dist: Dict[str, float] = defaultdict(float)
    dist[node] += 0.5
    if n > 0:
        share = 0.5 / n
        for nb in neighbours:
            dist[nb] += share
    return dict(dist)


def wasserstein_1(dist_p: Dict[str, float],
                  dist_q: Dict[str, float],
                  adj: Dict[str, List[str]]) -> float:
    """
    Approximate W₁ distance between two lazy‑walk distributions by
    treating the graph as a metric space where the distance between
    adjacent nodes is 1 and using the L1 norm of the difference of the
    probability vectors as an upper bound.
    """
    # Gather all nodes appearing in either distribution
    all_nodes = set(dist_p) | set(dist_q)
    # Build probability vectors aligned to a common ordering
    vec_p = np.array([dist_p.get(node, 0.0) for node in all_nodes])
    vec_q = np.array([dist_q.get(node, 0.0) for node in all_nodes])
    # In a unit‑edge graph the earth mover distance reduces to L1/2
    return 0.5 * np.sum(np.abs(vec_p - vec_q))


def node_average_curvature(sheaf: HybridSheaf, adj: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute κ̄(v) = average_{u∈N(v)} (1 - W₁(m_v,m_u) / d(v,u))
    where d(v,u) is the Euclidean distance between sections.
    """
    curvature: Dict[str, float] = {}
    for v in sheaf.nodes():
        neigh = adj.get(v, [])
        if not neigh:
            curvature[v] = 0.0
            continue
        curv_sum = 0.0
        for u in neigh:
            d_uv = np.linalg.norm(sheaf.section(v) - sheaf.section(u))
            if d_uv == 0:
                continue
            m_v = lazy_random_walk_distribution(adj, v)
            m_u = lazy_random_walk_distribution(adj, u)
            w1 = wasserstein_1(m_v, m_u, adj)
            curv_sum += 1.0 - w1 / d_uv
        curvature[v] = curv_sum / len(neigh)
    return curvature


# ---------------------------------------------------------------------------
# Hybrid pipeline
# ---------------------------------------------------------------------------

def build_hybrid_sheaf(text: str,
                       node_dims: Dict[str, int],
                       edge_list: List[Tuple[str, str]]) -> HybridSheaf:
    """
    End‑to‑end construction:
    1. Extract Krampus features from raw text.
    2. Convert them to pheromone signals.
    3. Assemble a HybridSheaf where each signal occupies its own node.
    """
    feat_vec = extract_full_features(text)
    signals = krampus_sticker_to_signals(feat_vec)
    sheaf = aggregate_signals(signals, node_dims, edge_list)
    return sheaf


def compute_node_curvature(sheaf: HybridSheaf, distance_thresh: float = 1.5) -> Dict[str, float]:
    """
    Build a graph from the sheaf (thresholded Euclidean distances),
    compute lazy‑walk distributions, and return the per‑node average
    Ollivier‑Ricci curvature.
    """
    adj = build_adjacency_from_sheaf(sheaf, distance_thresh)
    curvature = node_average_curvature(sheaf, adj)
    # Store curvature as an extra sheaf section for downstream use
    for node, curv in curvature.items():
        # Append curvature as a second component to the existing 1‑D section
        current = sheaf.section(node)
        sheaf.set_section(node, np.append(current, curv).tolist())
    return curvature


def nlms_hybrid_update(sheaf: HybridSheaf,
                       nlms: NLMS,
                       target: float) -> Tuple[np.ndarray, float]:
    """
    Concatenate all node sections (now [signal, curvature]) into a single
    input vector `x`, run NLMS prediction, and update the weight vector.
    Returns the updated weights and the prediction error.
    """
    # Assemble input vector respecting the original node ordering
    x_parts = [sheaf.section(node) for node in sorted(sheaf.nodes())]
    x = np.concatenate(x_parts)
    # Resize NLMS weights if dimensions have changed (first call)
    if nlms.weights.shape != x.shape:
        nlms.weights = np.zeros_like(x)
    return nlms.update(x, target)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Dummy node/edge topology – five nodes, fully connected for illustration
    node_dimensions = {f"node_{i}": 1 for i in range(5)}
    edges = [(f"node_{i}", f"node_{j}") for i in range(5) for j in range(i + 1, 5)]

    # Sample text
    sample_text = "the quick brown fox jumps over the lazy dog"

    # 1️⃣ Build sheaf from text
    sheaf = build_hybrid_sheaf(sample_text, node_dimensions, edges)

    # 2️⃣ Compute curvature and augment sheaf sections
    curvature_map = compute_node_curvature(sheaf, distance_thresh=2.0)
    print("Node curvatures:", curvature_map)

    # 3️⃣ Initialise NLMS with zero weights of appropriate size
    dummy_target = 0.42  # arbitrary regression target
    nlms_model = NLMS(weights=np.zeros(0))  # will be resized on first update

    # 4️⃣ Perform hybrid NLMS update
    updated_weights, error = nlms_hybrid_update(sheaf, nlms_model, dummy_target)
    print("Updated NLMS weights:", updated_weights)
    print("Prediction error:", error)

    # Verify that the pipeline runs without exception
    sys.exit(0)