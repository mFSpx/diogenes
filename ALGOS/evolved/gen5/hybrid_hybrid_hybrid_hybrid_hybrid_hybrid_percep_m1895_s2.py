# DARWIN HAMMER — match 1895, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py (gen3)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2.py (gen4)
# born: 2026-05-29T23:39:31Z

"""
Hybrid VRAM Allocation & Perceptual RBF Fusion

Parents:
- hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s2.py (Algorithm A)
- hybrid_hybrid_perceptual_de_hybrid_hybrid_rbf_su_m45_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a Bayesian decision‑hygiene score for a VRAM allocation
tree (prior probability of successful allocation).  Algorithm B supplies a
radial‑basis‑function (RBF) similarity between high‑dimensional feature
descriptions of a candidate configuration (e.g., model‑type, embedding lane,
LoRA adapters) and a reference “ideal” configuration derived from perceptual
hashes.  

The hybrid system treats the hygiene score as a prior 𝑝 and the RBF‑derived
similarity as a likelihood 𝓁.  A Bayesian update

    posterior = (𝓁 * 𝑝) / (𝓁 * 𝑝 + (1‑𝓁)*(1‑𝑝))

yields a posterior probability of successful VRAM allocation that fuses both
topologies.  The posterior drives a simple Hoeffding‑style split decision:
configurations with posterior > τ are accepted, otherwise rejected.

The code below implements:
1. Tree construction & distance metrics (A).
2. Decision‑hygiene scoring based on root‑to‑node distances (A).
3. RBF similarity using Euclidean distance and Gaussian kernel (B).
4. Perceptual hashing utilities (B).
5. Bayesian fusion of the two scores (core hybrid operation).
"""

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple, Set, Hashable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = List[float]
Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Algorithm A – Tree & Decision Hygiene
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # BFS to compute root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in visited:
                dist[nb] = dist[cur] + edge_len[(cur, nb)]
                visited.add(nb)
                queue.append(nb)
    return adj, edge_len, dist

def decision_hygiene_score(distances: Dict[str, float], budget: float) -> float:
    """
    Compute a hygiene score (0‑1) for a VRAM allocation tree.

    The score penalises large root‑to‑node distances relative to a budget.
    A simple formulation:
        score = exp( - (mean_distance / budget) )
    """
    if not distances:
        return 0.0
    mean_dist = sum(distances.values()) / len(distances)
    return math.exp(- (mean_dist / (budget + 1e-9)))  # avoid div‑zero

# ----------------------------------------------------------------------
# Algorithm B – Perceptual Hashing & RBF Similarity
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_dhash(values: List[float]) -> int:
    """Differential hash – 1 bit per adjacent pair."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Perceptual hash – threshold against mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits."""
    return bin(a ^ b).count('1')

def rbf_similarity(vec1: Vector, vec2: Vector, epsilon: float = 1.0) -> float:
    """Gaussian RBF similarity based on Euclidean distance."""
    return gaussian(euclidean(vec1, vec2), epsilon)

def perceptual_similarity(
    feat1: FeatureVec,
    feat2: FeatureVec,
) -> float:
    """
    Convert feature vectors to perceptual hashes and turn Hamming distance
    into a similarity in [0,1].
    """
    h1 = compute_phash(list(feat1))
    h2 = compute_phash(list(feat2))
    max_bits = max(h1.bit_length(), h2.bit_length(), 1)
    ham = hamming_distance(h1, h2)
    return 1.0 - ham / max_bits

# ----------------------------------------------------------------------
# Hybrid Core – Bayesian Fusion of Hygiene & RBF Scores
# ----------------------------------------------------------------------
def bayesian_update(prior: float, likelihood: float) -> float:
    """
    Perform a Bayesian update where:
        prior      = P(success before seeing similarity)
        likelihood = P(similarity | success)

    Returns posterior P(success | similarity).
    """
    # Guard against extreme values
    prior = min(max(prior, 1e-12), 1 - 1e-12)
    likelihood = min(max(likelihood, 1e-12), 1 - 1e-12)

    numerator = likelihood * prior
    denominator = numerator + (1 - likelihood) * (1 - prior)
    return numerator / denominator

def hybrid_vram_allocation_score(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    budget: float,
    candidate_feat: FeatureVec,
    reference_feat: FeatureVec,
    epsilon: float = 1.0,
) -> float:
    """
    End‑to‑end hybrid score for a candidate VRAM configuration.

    Steps
    -----
    1. Build the allocation tree and obtain root‑to‑node distances.
    2. Compute a decision‑hygiene prior from the distances and budget.
    3. Compute an RBF similarity between the candidate feature vector and a
       reference (ideal) feature vector.
    4. Fuse via Bayesian update → posterior probability of successful allocation.
    """
    _, _, dist = tree_metrics(nodes, edges, root)
    prior = decision_hygiene_score(dist, budget)

    # RBF similarity captures geometric closeness in feature space
    rbf_likelihood = rbf_similarity(candidate_feat, reference_feat, epsilon)

    # Optional: blend perceptual similarity as a secondary likelihood factor
    percept_likelihood = perceptual_similarity(candidate_feat, reference_feat)
    # Combine the two likelihoods multiplicatively (both must be high)
    combined_likelihood = rbf_likelihood * percept_likelihood

    posterior = bayesian_update(prior, combined_likelihood)
    return posterior

# ----------------------------------------------------------------------
# Simple Hoeffding‑style decision based on posterior
# ----------------------------------------------------------------------
def accept_configuration(posterior: float, threshold: float = 0.6) -> bool:
    """
    Mimic a Hoeffding tree split decision:
    accept if posterior exceeds the threshold.
    """
    return posterior >= threshold

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy tree representing VRAM allocation slots
    nodes = {
        "root": (0.0, 0.0),
        "model_A": (1.0, 2.0),
        "embed_B": (2.5, 1.5),
        "lora_C": (3.0, 3.0),
    }
    edges = [("root", "model_A"), ("model_A", "embed_B"), ("embed_B", "lora_C")]
    root = "root"
    budget = 10.0  # arbitrary VRAM budget unit

    # Feature vectors (e.g., [model_id, embedding_id, lora_id, size, latency])
    candidate_feat = [0.2, 0.8, 0.5, 0.6, 0.4]
    reference_feat = [0.0, 1.0, 0.5, 0.5, 0.5]

    posterior = hybrid_vram_allocation_score(
        nodes,
        edges,
        root,
        budget,
        candidate_feat,
        reference_feat,
        epsilon=1.5,
    )
    decision = accept_configuration(posterior, threshold=0.55)

    print(f"Hybrid posterior probability: {posterior:.4f}")
    print(f"Configuration accepted? {'Yes' if decision else 'No'}")