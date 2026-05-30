# DARWIN HAMMER — match 5482, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1213_s7.py (gen6)
# parent_b: hybrid_hybrid_distributed_l_hybrid_hybrid_model__m99_s1.py (gen3)
# born: 2026-05-30T00:02:24Z

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
Node = int
Graph = Dict[Node, Set[Node]]
WeightMap = Dict[Tuple[Node, Node], float]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


# ----------------------------------------------------------------------
# Utility functions (kept for backward compatibility)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash based on average threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def compute_dhash(values: List[float]) -> int:
    """Simple difference hash."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Core mathematical integration
# ----------------------------------------------------------------------
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v1 - v2)


def build_graph(elements: List[Morphology]) -> Tuple[Graph, WeightMap]:
    """
    Construct an undirected weighted graph.
    Nodes are element indices.
    Edge weight w_ij = exp(-d_ij / sigma) where d_ij is Euclidean distance
    and sigma is the median of all pairwise distances.
    """
    n = len(elements)
    vectors = [e.as_vector() for e in elements]

    # Compute all pairwise distances
    dists = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                d = 0.0
            else:
                d = euclidean_distance(vectors[i], vectors[j])
            dists[i, j] = d
            dists[j, i] = d

    # Median distance as scale parameter (avoid zero)
    nonzero = dists[dists > 0]
    sigma = np.median(nonzero) if nonzero.size > 0 else 1.0
    sigma = max(sigma, 1e-8)

    graph: Graph = {i: set() for i in range(n)}
    weights: WeightMap = {}

    # Connect nodes whose distance is not larger than 2 * sigma
    threshold = 2.0 * sigma
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= threshold:
                w = math.exp(-dists[i, j] / sigma)
                graph[i].add(j)
                graph[j].add(i)
                weights[(i, j)] = w
                weights[(j, i)] = w

    return graph, weights


def ollivier_ricci_curvature(
    graph: Graph, weights: WeightMap
) -> Dict[Tuple[Node, Node], float]:
    """
    Approximate Ollivier‑Ricci curvature for each edge.
    For an edge (i, j) we define probability measures μ_i and μ_j
    over the closed neighbourhood N(i)∪{i} and N(j)∪{j}
    proportional to the edge weights (self‑loop weight = 1).
    The curvature κ_ij = 1 - W1(μ_i, μ_j) / d(i, j) with d(i, j)=1.
    Using the L1 metric on the discrete space gives
    W1 = 0.5 * Σ_k |μ_i(k) - μ_j(k)|.
    """
    curvature: Dict[Tuple[Node, Node], float] = {}

    # Pre‑compute neighbourhood probability vectors
    prob_vectors: Dict[Node, Dict[Node, float]] = {}
    for node, neigh in graph.items():
        total = 1.0  # self‑loop weight
        for nb in neigh:
            total += weights[(node, nb)]
        probs = {node: 1.0 / total}
        for nb in neigh:
            probs[nb] = weights[(node, nb)] / total
        prob_vectors[node] = probs

    for i, neigh in graph.items():
        for j in neigh:
            if (i, j) in curvature:
                continue  # already processed (undirected)
            mu_i = prob_vectors[i]
            mu_j = prob_vectors[j]

            # Union of support
            support = set(mu_i) | set(mu_j)
            diff = sum(abs(mu_i.get(k, 0.0) - mu_j.get(k, 0.0)) for k in support)
            w1 = 0.5 * diff
            kappa = 1.0 - w1  # d(i,j)=1
            curvature[(i, j)] = kappa
            curvature[(j, i)] = kappa

    return curvature


def connected_components(graph: Graph) -> List[Set[Node]]:
    """Return a list of node sets, each representing a connected component."""
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(graph[cur] - visited)
        components.append(comp)

    return components


def sphericity_index(cluster: List[Morphology]) -> float:
    """
    Classical sphericity: Φ = (π^{1/3} (6V)^{2/3}) / A
    where V = l·w·h and A = 2(lw + lh + wh).
    """
    if not cluster:
        return 0.0
    vols = np.array([m.length * m.width * m.height for m in cluster])
    surf = np.array(
        [
            2 * (m.length * m.width + m.length * m.height + m.width * m.height)
            for m in cluster
        ]
    )
    V = np.mean(vols)
    A = np.mean(surf)
    if A == 0:
        return 0.0
    phi = (math.pi ** (1.0 / 3.0) * (6 * V) ** (2.0 / 3.0)) / A
    return phi


def gaussian_log_likelihood(data: np.ndarray) -> float:
    """
    Log‑likelihood of a multivariate Gaussian with unknown mean and covariance,
    using the maximum‑likelihood estimators (sample mean & covariance).
    """
    n, d = data.shape
    if n == 0:
        return 0.0
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False, bias=False)
    # Regularise covariance to avoid singularity
    eps = 1e-8
    cov += eps * np.eye(d)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Numerical safeguard
        logdet = np.log(eps)

    # Mahalanobis term simplifies to d (because of MLE)
    ll = -0.5 * n * (d * np.log(2 * math.pi) + logdet + d)
    return ll


def compute_bic(cluster: List[Morphology]) -> float:
    """
    BIC = -2 * loglik + k * log(n)
    k = d (means) + d(d+1)/2 (covariance matrix) = d + d(d+1)/2
    """
    if not cluster:
        return float("inf")
    data = np.stack([m.as_vector() for m in cluster])
    n, d = data.shape
    loglik = gaussian_log_likelihood(data)
    k = d + d * (d + 1) // 2
    bic = -2.0 * loglik + k * math.log(n)
    return bic


def compute_rict(cluster: List[Morphology]) -> float:
    """
    Approximate RLCT using the same Gaussian log‑likelihood.
    RLCT ≈ (d/2) * log(n) - loglik
    """
    if not cluster:
        return float("inf")
    data = np.stack([m.as_vector() for m in cluster])
    n, d = data.shape
    loglik = gaussian_log_likelihood(data)
    rict = (d / 2.0) * math.log(n) - loglik
    return rict


def leader_election(
    component: Set[Node],
    graph: Graph,
    curvature: Dict[Tuple[Node, Node], float],
    elements: List[Morphology],
) -> Node:
    """
    Score each node by:
        score = (sum of incident curvatures) * (1 / (BIC + ε)) * sphericity
    The node with the highest score becomes the leader of the component.
    """
    eps = 1e-12
    best_node = None
    best_score = -math.inf

    # Pre‑compute cluster‑wide statistics once
    cluster = [elements[i] for i in component]
    spher = sphericity_index(cluster)
    bic = compute_bic(cluster) + eps
    base = (1.0 / bic) * spher

    for node in component:
        curv_sum = sum(curvature.get((node, nb), 0.0) for nb in graph[node] if nb in component)
        score = curv_sum * base
        if score > best_score:
            best_score = score
            best_node = node

    return best_node if best_node is not None else next(iter(component))


def hybrid_operation(elements: List[Morphology]) -> None:
    """
    Full pipeline:
        1. Build weighted graph.
        2. Compute Ollivier‑Ricci curvature.
        3. Find connected components.
        4. For each component:
            a) elect a leader,
            b) compute sphericity, BIC, RLCT,
            c) report results.
    """
    graph, weights = build_graph(elements)
    curvature = ollivier_ricci_curvature(graph, weights)
    components = connected_components(graph)

    for comp_id, comp in enumerate(components, start=1):
        leader = leader_election(comp, graph, curvature, elements)
        cluster = [elements[i] for i in comp]
        sph = sphericity_index(cluster)
        bic = compute_bic(cluster)
        rict = compute_rict(cluster)
        print(
            f"Component {comp_id} (size {len(comp)}): Leader Node {leader}\n"
            f"    Sphericity = {sph:.6f}, BIC = {bic:.3f}, RLCT = {rict:.3f}"
        )


# ----------------------------------------------------------------------
# Example usage (can be removed or replaced by unit tests)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_elements = [
        Morphology(length=10.0, width=20.0, height=30.0, mass=40.0),
        Morphology(length=15.0, width=25.0, height=35.0, mass=45.0),
        Morphology(length=20.0, width=30.0, height=40.0, mass=50.0),
        Morphology(length=25.0, width=35.0, height=45.0, mass=55.0),
        Morphology(length=30.0, width=40.0, height=50.0, mass=60.0),
    ]
    hybrid_operation(sample_elements)