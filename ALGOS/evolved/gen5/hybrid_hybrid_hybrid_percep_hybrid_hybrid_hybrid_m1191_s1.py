# DARWIN HAMMER — match 1191, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s1.py (gen4)
# born: 2026-05-29T23:33:19Z

"""Hybrid Perceptual‑Fisher RBF‑Associative Algorithm
===================================================

Parents
-------
* **hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s0.py** – uses
  radial basis functions (RBF) to model similarity, perceptual hashing to
  cluster data and a Hoeffding bound to guide tree splits.
* **hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s1.py** – introduces
  the Fisher information score derived from a Gaussian beam and a
  Dense Associative Memory (DAM) whose energy function is used for
  decision‑hygiene scoring.

Mathematical Bridge
-------------------
The bridge is the *similarity* measure.  
Both parents compute a scalar similarity between two feature vectors:

* **Parent A** – `gaussian(euclidean(a, b))` (RBF kernel).
* **Parent B** – `fisher_score(theta, centre, width)` where `theta` can be
  interpreted as a distance‑like quantity.

We fuse them by **multiplying** the RBF kernel with the Fisher score,
thereby obtaining a *Fisher‑weighted RBF* similarity:


S_fRBF(a, b) = 𝔾ε(‖a‑b‖₂) · 𝔽(‖a‑b‖₂ ; μ, σ)


where `𝔾ε` is the Gaussian RBF with scale `ε` and `𝔽` is the Fisher
information of the same Gaussian beam (centre `μ`, width `σ`).

The resulting similarity feeds two downstream components:

1. **Perceptual clustering** – vectors whose perceptual hash differs by
   ≤ k bits are grouped; the cluster representative is used as a pattern
   for the DAM.
2. **Dense Associative Memory** – the hybrid similarity is used as the
   energy term that drives the soft‑max retrieval in the DAM, while a
   Hoeffding bound decides whether a split (e.g. a tree node) should be
   performed based on observed similarity statistics.

The code below implements this fused pipeline with three core functions
demonstrating the hybrid operation."""

import math
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

Vector = List[float]
Hash = int


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel 𝔾ε(r) = exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance ‖a‑b‖₂."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_beam(theta: float, centre: float, width: float) -> float:
    """Gaussian beam used by the Fisher score."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - centre) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, centre: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information of a Gaussian beam (Parent B)."""
    intensity = max(gaussian_beam(theta, centre, width), eps)
    derivative = intensity * (-(theta - centre) / (width * width))
    return (derivative * derivative) / intensity


def hybrid_fisher_rbf_similarity(
    a: Vector,
    b: Vector,
    centre: float = 0.0,
    width: float = 1.0,
    epsilon: float = 1.0,
) -> float:
    """
    Fisher‑weighted RBF similarity S_fRBF(a, b).

    The Euclidean distance plays the role of both the RBF radius and the
    Gaussian‑beam argument `theta`.
    """
    dist = euclidean(a, b)
    rbf = gaussian_rbf(dist, epsilon)
    fisher = fisher_score(dist, centre, width)
    return rbf * fisher


def compute_phash(values: List[float]) -> Hash:
    """Perceptual hash (average hash) of a numeric vector."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits for a 64‑bit integer
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: Hash, b: Hash) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")


def cluster_by_phash(vectors: List[Vector], max_hamming: int = 3) -> Dict[Hash, List[Vector]]:
    """
    Group vectors whose perceptual hashes differ by at most `max_hamming` bits.

    Returns a dictionary mapping the representative hash to the list of
    vectors in that cluster.
    """
    clusters: Dict[Hash, List[Vector]] = {}
    for vec in vectors:
        h = compute_phash(vec)
        # Find an existing cluster within the Hamming budget
        assigned = False
        for rep in list(clusters.keys()):
            if hamming_distance(h, rep) <= max_hamming:
                clusters[rep].append(vec)
                assigned = True
                break
        if not assigned:
            clusters[h] = [vec]
    return clusters


class DenseAssociativeMemory:
    """
    Simple Dense Associative Memory (DAM) with an energy based on
    Fisher‑weighted RBF similarity.

    Patterns are stored as rows of a 2‑D NumPy array.
    Retrieval uses a soft‑max over negative energies.
    """

    def __init__(self, patterns: np.ndarray, centre: float = 0.0, width: float = 1.0, epsilon: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)
        self.centre = centre
        self.width = width
        self.epsilon = epsilon

    def _energy(self, query: Vector) -> np.ndarray:
        """Energy vector E_i = -S_fRBF(query, pattern_i)."""
        energies = []
        for pat in self.patterns:
            sim = hybrid_fisher_rbf_similarity(
                query, pat.tolist(), self.centre, self.width, self.epsilon
            )
            energies.append(-sim)  # negative because lower energy = higher similarity
        return np.array(energies)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable soft‑max."""
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def retrieve(self, query: Vector) -> Tuple[int, float]:
        """
        Retrieve the index of the most likely stored pattern and its
        similarity score.
        """
        energies = self._energy(query)
        probs = self._softmax(-energies)  # higher similarity → higher probability
        idx = int(np.argmax(probs))
        similarity = -energies[idx]
        return idx, similarity

    def update(self, new_pattern: Vector):
        """Append a new pattern to the memory."""
        self.patterns = np.vstack([self.patterns, new_pattern])


def hoeffding_bound(n: int, R: float = 1.0, delta: float = 0.05) -> float:
    """
    Hoeffding bound ε = sqrt(R² * ln(1/δ) / (2n)).
    Used to decide whether a split in a decision tree is statistically
    justified.
    """
    if n <= 0:
        raise ValueError("sample count n must be positive")
    return math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))


def should_split(similarities: List[float], delta: float = 0.05) -> bool:
    """
    Decide whether to split a node based on the variance of observed
    similarities and the Hoeffding bound.
    """
    n = len(similarities)
    if n < 2:
        return False
    mean = sum(similarities) / n
    var = sum((s - mean) ** 2 for s in similarities) / (n - 1)
    bound = hoeffding_bound(n, R=math.sqrt(var), delta=delta)
    # Simple heuristic: split if the bound is smaller than a fraction of the mean
    return bound < 0.1 * mean


# ----------------------------------------------------------------------
# Demonstration functions (the required three+ hybrid operations)
# ----------------------------------------------------------------------


def hybrid_cluster_and_memory(vectors: List[Vector]) -> DenseAssociativeMemory:
    """
    1. Cluster vectors by perceptual hash.
    2. For each cluster compute the centroid.
    3. Initialise a DAM with the centroids as patterns.
    """
    clusters = cluster_by_phash(vectors)
    centroids = []
    for members in clusters.values():
        arr = np.array(members)
        centroid = arr.mean(axis=0).tolist()
        centroids.append(centroid)
    dam = DenseAssociativeMemory(np.array(centroids))
    return dam


def route_query(
    query: Vector,
    dam: DenseAssociativeMemory,
    similarity_history: List[float],
) -> Tuple[int, float]:
    """
    2. Retrieve the most similar pattern from the DAM using the
       Fisher‑weighted RBF similarity.
    3. Record the similarity and optionally trigger a split decision.
    """
    idx, sim = dam.retrieve(query)
    similarity_history.append(sim)
    if should_split(similarity_history):
        # In a full system we would split the node; here we simply log.
        print("[INFO] Split condition met (sim history size =", len(similarity_history), ")")
    return idx, sim


def incremental_learning(
    queries: List[Vector],
    dam: DenseAssociativeMemory,
    max_history: int = 100,
) -> None:
    """
    Process a stream of queries, updating the DAM with novel patterns
    that are sufficiently dissimilar from existing ones.
    """
    history: List[float] = []
    for q in queries:
        idx, sim = route_query(q, dam, history)
        # If similarity is below a threshold, treat as a new pattern
        if sim < 0.1:  # arbitrary novelty threshold
            dam.update(q)
            print(f"[INFO] Added new pattern, total patterns: {len(dam.patterns)}")
        # Keep history bounded
        if len(history) > max_history:
            history.pop(0)


if __name__ == "__main__":
    # Smoke test ---------------------------------------------------------
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic 8‑dimensional vectors
    data = [np.random.rand(8).tolist() for _ in range(50)]

    # Build the hybrid DAM from clustered centroids
    dam = hybrid_cluster_and_memory(data)

    # Create a small stream of queries (some similar, some novel)
    query_stream = [np.random.rand(8).tolist() for _ in range(20)]

    # Run incremental learning; should execute without errors
    incremental_learning(query_stream, dam)

    print("[SUCCESS] Hybrid algorithm executed without errors.")