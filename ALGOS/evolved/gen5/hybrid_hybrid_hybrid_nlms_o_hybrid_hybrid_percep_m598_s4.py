# DARWIN HAMMER — match 598, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s4.py (gen2)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2.py (gen4)
# born: 2026-05-29T23:29:59Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Node:
    id: int
    weight: float

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def rbf_activation(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF activations for a single input vector x."""
    dists = np.linalg.norm(centers - x, axis=1)
    return np.exp(- (dists ** 2) / (2 * sigma ** 2))

class RBFNLMS:
    """RBF network whose output weights are adapted with the Normalised LMS rule."""
    def __init__(self, centers: np.ndarray, sigma: float = 1.0, mu: float = 0.5, eps: float = 1e-9):
        self.centers = centers                     # shape (M, D)
        self.sigma = sigma
        self.mu = mu
        self.eps = eps
        self.weights = np.random.rand(centers.shape[0])  # one weight per RBF centre

    def predict(self, x: np.ndarray) -> float:
        phi = rbf_activation(x, self.centers, self.sigma)   # shape (M,)
        return np.dot(self.weights, phi)

    def adapt(self, x: np.ndarray, target: float) -> float:
        """Perform one NLMS update and return the instantaneous error."""
        phi = rbf_activation(x, self.centers, self.sigma)
        y = np.dot(self.weights, phi)
        error = target - y
        power = np.dot(phi, phi) + self.eps
        self.weights += (self.mu * error / power) * phi
        return error

def construct_similarity_graph(weights: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """Build a fully‑connected graph where edge weights are similarity scores derived from
    the learned RBF output weights."""
    n = len(weights)
    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            # similarity in [0,1] – larger when weights are close
            sim = 1.0 - abs(weights[i] - weights[j]) / (1.0 + abs(weights[i] - weights[j]))
            graph[i].append((j, sim))
            graph[j].append((i, sim))
    return graph

def prim_mst(graph: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    """Return the Minimum Spanning Tree using Prim's algorithm.
    Each edge is (src, dst, similarity)."""
    n = len(graph)
    if n == 0:
        return []
    visited = {0}
    edges: List[Tuple[int, int, float]] = []
    candidate_edges: List[Tuple[float, int, int]] = []  # (‑sim, src, dst) for max‑similarity MST
    for dst, sim in graph[0]:
        candidate_edges.append((-sim, 0, dst))
    import heapq
    heapq.heapify(candidate_edges)

    while len(visited) < n:
        neg_sim, src, dst = heapq.heappop(candidate_edges)
        if dst in visited:
            continue
        visited.add(dst)
        edges.append((src, dst, -neg_sim))
        for nxt, sim in graph[dst]:
            if nxt not in visited:
                heapq.heappush(candidate_edges, (-sim, dst, nxt))
    return edges

def compute_phash(values: np.ndarray) -> int:
    """Perceptual hash based on the mean of the full vector."""
    if values.size == 0:
        return 0
    avg = np.mean(values)
    bits = 0
    for v in values:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

def similarity_matrix(features: Dict[int, np.ndarray]) -> np.ndarray:
    """Similarity matrix derived from perceptual hashes of feature vectors."""
    ids = list(features.keys())
    n = len(ids)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(features[i]) for i in ids]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                S[i, j] = 1.0
            else:
                ham = hamming_distance(hashes[i], hashes[j])
                sim = 1.0 - ham / max(1, 64)   # normalise to [0,1]
                S[i, j] = S[j, i] = sim
    return S

def hybrid_operation(features: Dict[int, np.ndarray], target: float,
                    sigma: float = 1.0, mu: float = 0.5,
                    epochs: int = 30) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Train the RBF‑NLMS model on the supplied node features and return
    the learned weights together with a minimum‑cost spanning tree."""
    # Stack feature vectors to form centre matrix
    centre_matrix = np.stack([features[i] for i in sorted(features.keys())])
    rbf_nlms = RBFNLMS(centers=centre_matrix, sigma=sigma, mu=mu)

    # Simple online training: each node contributes one sample per epoch
    node_ids = sorted(features.keys())
    for _ in range(epochs):
        for nid in node_ids:
            rbf_nlms.adapt(features[nid], target)

    # Build similarity graph from the final weights
    graph = construct_similarity_graph(rbf_nlms.weights)
    mst = prim_mst(graph)
    return rbf_nlms.weights, mst

if __name__ == "__main__":
    # Example usage with 10 nodes, each having a 10‑dimensional feature vector
    np.random.seed(42)
    features = {i: np.random.rand(10) for i in range(10)}
    target_value = 1.0

    learned_weights, mst_edges = hybrid_operation(features, target_value)

    print("Learned RBF output weights:")
    print(learned_weights)
    print("\nMinimum‑Cost Spanning Tree (src, dst, similarity):")
    for edge in mst_edges:
        print(edge)