# DARWIN HAMMER — match 634, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s3.py (gen4)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# born: 2026-05-29T23:30:10Z

"""Hybrid Perceptual‑Graph‑RBF Algorithm
Parents:
- hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s3.py
- hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py

Mathematical Bridge
-------------------
Parent A clusters data by *perceptual hash* and builds a Gaussian RBF kernel from
Euclidean distances between hyper‑dimensional (HD) vectors that represent each
cluster.  
Parent B treats samples as nodes of a weighted graph, attaches the *Krampus*
feature vector as a node attribute and computes Ollivier‑Ricci curvature from
lazy‑random‑walk distributions on that graph.

The fusion interprets each perceptual cluster as a graph node.  The HD vector of
the cluster provides the geometric edge weight (Euclidean distance) while the
aggregated Krampus features supply the node attribute used to build the lazy
random‑walk distribution.  Ollivier‑Ricci curvature κ₍ᵢⱼ₎ between two nodes is
then used to *modulate* the Gaussian RBF kernel:

    K₍ᵢⱼ₎ =  g(‖vᵢ‑vⱼ‖) · (1 + κ₍ᵢⱼ₎) ,  g(r)=exp(−(ε r)²)

Thus the surrogate model simultaneously respects perceptual similarity,
hyper‑dimensional geometry and graph‑theoretic curvature.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Dict, Tuple

import numpy as np

Vector = np.ndarray


# ----------------------------------------------------------------------
# Parent A utilities (perceptual hashing, HD vectors, RBF kernel)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def compute_phash(v: Vector) -> int:
    """Very lightweight perceptual hash: SHA‑256 of the raw bytes,
    truncated to the lowest 8 bits."""
    h = hashlib.sha256(v.tobytes()).digest()
    return h[0]


def cluster_by_phash(vectors: List[Vector]) -> Dict[int, List[int]]:
    """Group indices of *vectors* by their perceptual hash."""
    clusters: Dict[int, List[int]] = {}
    for idx, vec in enumerate(vectors):
        h = compute_phash(vec)
        clusters.setdefault(h, []).append(idx)
    return clusters


def morphology_influenced_vector(cluster_vectors: List[Vector]) -> Vector:
    """Aggregate a cluster into a bipolar hyper‑dimensional vector.
    The mean is taken and then binarised to ±1."""
    if not cluster_vectors:
        raise ValueError("empty cluster")
    mean_vec = np.mean(cluster_vectors, axis=0)
    return np.where(mean_vec >= 0, 1.0, -1.0)


# ----------------------------------------------------------------------
# Parent B utilities (Krampus features, lazy RW, Ollivier‑Ricci)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Placeholder: returns a deterministic pseudo‑feature dict based on *text*."""
    # In a real setting this would parse *text*.  Here we simply hash the string
    # to obtain reproducible pseudo‑values.
    base = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    random.seed(base)
    keys = [
        "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
        "recursion_score", "directive_ratio", "target_density",
        "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
        "wrath_velocity", "bureaucratic_weaponization_index",
        "resource_exhaustion_metric", "swarm_orchestration_density",
        "logic_crucifixion_index", "conspiracy_grounding_ratio",
        "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
        "asset_structuring_weight", "pitch_formatting_ratio",
        "agent_symmetry_ratio", "protocol_discipline", "manic_velocity",
    ]
    return {k: random.random() for k in keys}


def lazy_rw_distribution(adj: Dict[int, List[int]], node: int, alpha: float = 0.5) -> Dict[int, float]:
    """Lazy random walk distribution centred at *node*."""
    neighbours = adj.get(node, [])
    deg = len(neighbours)
    dist: Dict[int, float] = {node: alpha}
    if deg > 0:
        spread = (1.0 - alpha) / deg
        for nb in neighbours:
            dist[nb] = dist.get(nb, 0.0) + spread
    return dist


def ollivier_ricci_curvature(
    adj: Dict[int, List[int]],
    edge_weights: Dict[Tuple[int, int], float],
    i: int,
    j: int,
    alpha: float = 0.5,
) -> float:
    """Compute a simplified Ollivier‑Ricci curvature κ₍ᵢⱼ₎.
    The Earth‑Mover distance is approximated by the L1 distance of the lazy‑RW
    distributions, weighted by the shortest‑path distance between nodes."""
    # Lazy‑RW distributions for the two nodes
    mu_i = lazy_rw_distribution(adj, i, alpha)
    mu_j = lazy_rw_distribution(adj, j, alpha)

    # Union of support
    support = set(mu_i) | set(mu_j)

    # Approximate transport cost: sum |p−q| * d(node, support_node)
    # where d(node, support_node) is the edge weight if directly connected,
    # otherwise the shortest‑path distance (here we fall back to Euclidean
    # distance stored in edge_weights via the minimal path of length 1).
    cost = 0.0
    for u in support:
        pi = mu_i.get(u, 0.0)
        pj = mu_j.get(u, 0.0)
        diff = abs(pi - pj)
        if u == i:
            d_iu = 0.0
        elif (i, u) in edge_weights:
            d_iu = edge_weights[(i, u)]
        else:
            d_iu = 1.0  # fallback
        if u == j:
            d_ju = 0.0
        elif (j, u) in edge_weights:
            d_ju = edge_weights[(j, u)]
        else:
            d_ju = 1.0
        # Use the average distance to the two sources as transport cost
        cost += diff * (d_iu + d_ju) / 2.0

    # Geodesic distance between i and j (direct edge weight if exists)
    d_ij = edge_weights.get((i, j), edge_weights.get((j, i), 1.0))

    if d_ij == 0:
        return 0.0
    return 1.0 - (cost / d_ij)


# ----------------------------------------------------------------------
# Hybrid core: graph construction, curvature‑modulated RBF, training/prediction
# ----------------------------------------------------------------------
def build_graph(
    cluster_vectors: List[Vector],
    distance_threshold: float = None,
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    """Create an undirected graph where each node is a cluster.
    Edge weight = Euclidean distance between cluster vectors.
    Edges are kept if the distance is ≤ *distance_threshold* (or median if None)."""
    n = len(cluster_vectors)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    edge_weights: Dict[Tuple[int, int], float] = {}

    # Compute all pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(cluster_vectors[i], cluster_vectors[j])
            dists.append(d)

    if distance_threshold is None:
        distance_threshold = float(np.median(dists)) if dists else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(cluster_vectors[i], cluster_vectors[j])
            if d <= distance_threshold:
                adj[i].append(j)
                adj[j].append(i)
                edge_weights[(i, j)] = d
                edge_weights[(j, i)] = d
    return adj, edge_weights


def hybrid_kernel_matrix(
    cluster_vectors: List[Vector],
    adj: Dict[int, List[int]],
    edge_weights: Dict[Tuple[int, int], float],
    epsilon: float = 1.0,
) -> np.ndarray:
    """Return K where K₍ᵢⱼ₎ = g(‖vᵢ‑vⱼ‖)·(1+κ₍ᵢⱼ₎)."""
    n = len(cluster_vectors)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = euclidean(cluster_vectors[i], cluster_vectors[j])
            g = gaussian(r, epsilon)
            kappa = ollivier_ricci_curvature(adj, edge_weights, i, j)
            K[i, j] = g * (1.0 + kappa)
    return K


def solve_linear(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve K·α = y for α (regularized with a tiny ridge term)."""
    ridge = 1e-8 * np.eye(K.shape[0])
    return np.linalg.solve(K + ridge, y)


def hybrid_predict(
    sample: Vector,
    cluster_vectors: List[Vector],
    alpha: np.ndarray,
    adj: Dict[int, List[int]],
    edge_weights: Dict[Tuple[int, int], float],
    epsilon: float = 1.0,
) -> float:
    """Predict a scalar for *sample* using the trained coefficients."""
    # Find the nearest cluster (by Euclidean distance)
    dists = [euclidean(sample, cv) for cv in cluster_vectors]
    nearest = int(np.argmin(dists))
    # Compute kernel between the sample and every cluster, modulated by curvature
    k_vec = np.zeros(len(cluster_vectors))
    for i, cv in enumerate(cluster_vectors):
        r = euclidean(sample, cv)
        g = gaussian(r, epsilon)
        kappa = ollivier_ricci_curvature(adj, edge_weights, nearest, i)
        k_vec[i] = g * (1.0 + kappa)
    return float(k_vec @ alpha)


# ----------------------------------------------------------------------
# Demonstration functions (three required)
# ----------------------------------------------------------------------
def generate_synthetic_data(num_samples: int = 50, dim: int = 128) -> List[Vector]:
    """Create random bipolar hyper‑dimensional vectors."""
    return [np.where(np.random.rand(dim) > 0.5, 1.0, -1.0) for _ in range(num_samples)]


def train_hybrid_model(vectors: List[Vector]) -> Tuple[List[Vector], np.ndarray, Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    """Full training pipeline returning cluster vectors, coefficients and graph."""
    # 1. Cluster by perceptual hash
    clusters = cluster_by_phash(vectors)

    # 2. Build a hyper‑dimensional representative per cluster
    cluster_vecs: List[Vector] = []
    for idxs in clusters.values():
        cluster_vecs.append(morphology_influenced_vector([vectors[i] for i in idxs]))

    # 3. Graph construction
    adj, edge_weights = build_graph(cluster_vecs)

    # 4. Kernel matrix with curvature modulation
    K = hybrid_kernel_matrix(cluster_vecs, adj, edge_weights)

    # 5. Dummy target (e.g., sum of first 10 components)
    y = np.array([float(np.sum(v[:10])) for v in cluster_vecs])

    # 6. Solve for coefficients
    alpha = solve_linear(K, y)

    return cluster_vecs, alpha, adj, edge_weights


def evaluate_hybrid_model(cluster_vecs: List[Vector], alpha: np.ndarray,
                         adj: Dict[int, List[int]], edge_weights: Dict[Tuple[int, int], float]) -> None:
    """Run a quick sanity check on a few random samples."""
    for _ in range(5):
        sample = np.where(np.random.rand(cluster_vecs[0].size) > 0.5, 1.0, -1.0)
        pred = hybrid_predict(sample, cluster_vecs, alpha, adj, edge_weights)
        print(f"Prediction: {pred:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic hyper‑dimensional data
    data = generate_synthetic_data(num_samples=120, dim=256)

    # Train the hybrid model
    cluster_vectors, coeffs, adjacency, weights = train_hybrid_model(data)

    # Evaluate on a few random queries
    evaluate_hybrid_model(cluster_vectors, coeffs, adjacency, weights)