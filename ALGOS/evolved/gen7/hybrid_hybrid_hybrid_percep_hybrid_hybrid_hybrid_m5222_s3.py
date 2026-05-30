# DARWIN HAMMER — match 5222, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_perceptual_de_hybrid_label_foundry_m1030_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s1.py (gen6)
# born: 2026-05-30T00:00:45Z

"""Hybrid Perceptual‑RBF / Probabilistic‑Pruning Algorithm
========================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_perceptual_de_hybrid_label_foundry_m1030_s1.py``  
  Provides perceptual hashing utilities, a simple RBF surrogate model and a
  weak‑supervision labeling scheme.

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s1.py``  
  Supplies simulated‑annealing acceptance, exponential cooling and a
  linguistic‑similarity measure that is used for adaptive pruning of
  model tiers.

Mathematical Bridge
-------------------
The bridge is built on the *hash representation* produced by Parent A.
Each data point is mapped to an integer hash (binary string).  Identical
or near‑identical hashes are clustered; a cluster is treated as a
*model tier* (the ``ModelTier`` dataclass from Parent B).  The textual
field of a tier is the binary string of the representative hash, enabling
the linguistic‑similarity function to operate directly on the hash
space.  The average labeling confidence inside a cluster becomes the
*trust weight* of the tier.  During pruning, the simulated‑annealing
acceptance probability is modulated by both the expected value of a tier
and its trust‑weighted similarity to the current best tier, thereby
fusing the governing equations of both parents into a single decision
process.

The module implements:
* perceptual hashing (difference‑hash and average‑hash),
* a lightweight Gaussian‑RBF predictor,
* aggregation of weak‑supervision labels,
* construction of ``ModelTier`` objects from hash clusters,
* a trust‑aware pruning schedule that combines acceptance probability,
  cooling, and hash‑based linguistic similarity.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Perceptual hashing and RBF utilities
# ----------------------------------------------------------------------
Vector = Sequence[float]

def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count('1')

def combined_hash(values: List[float]) -> int:
    """Combine dhash and phash into a single 128‑bit integer."""
    dh = compute_dhash(values)
    ph = compute_phash(values)
    return (dh << 64) | ph

# Simple Gaussian RBF kernel
def _gaussian_kernel(r: float, sigma: float = 1.0) -> float:
    return math.exp(- (r ** 2) / (2 * sigma ** 2))

def rbf_predict(
    support_vectors: np.ndarray,
    support_values: np.ndarray,
    query: np.ndarray,
    sigma: float = 1.0,
) -> float:
    """
    Predict a scalar value for *query* using a Gaussian RBF interpolant
    built from *support_vectors* and their associated *support_values*.
    The weights are obtained by solving K·w = y where K_ij = kernel(||x_i-x_j||).
    """
    if support_vectors.ndim != 2:
        raise ValueError("support_vectors must be 2‑D")
    n = support_vectors.shape[0]
    if n == 0:
        raise ValueError("no support vectors provided")
    # Build the kernel matrix
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        diff = support_vectors[i] - support_vectors
        dists = np.linalg.norm(diff, axis=1)
        K[i] = [_gaussian_kernel(r, sigma) for r in dists]
    # Solve for weights
    w = np.linalg.solve(K, support_values)
    # Compute kernel between query and supports
    dists_q = np.linalg.norm(support_vectors - query, axis=1)
    k_q = np.array([_gaussian_kernel(r, sigma) for r in dists_q])
    return float(k_q @ w)

# ----------------------------------------------------------------------
# Parent B – Probabilistic decision making and linguistic similarity
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """A cluster of hashed vectors treated as a model tier."""
    name: str                # e.g. "tier_0"
    hash_repr: int           # representative hash (integer)
    trust: float             # average labeling confidence in [0,1]
    expected_value: float    # surrogate model prediction for the tier
    text: str = ''           # binary string of hash_repr (for similarity)

    def __post_init__(self):
        object.__setattr__(self, 'text', format(self.hash_repr, 'b'))

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def linguistic_similarity(hash1: int, hash2: int) -> float:
    """Similarity based on normalized Hamming distance (1 - normalized distance)."""
    max_bits = max(hash1.bit_length(), hash2.bit_length(), 1)
    hd = hamming_distance(hash1, hash2)
    return 1.0 - hd / max_bits

# ----------------------------------------------------------------------
# Hybrid core: clustering, label aggregation, tier construction,
# and trust‑aware pruning.
# ----------------------------------------------------------------------
def cluster_by_hash(
    vectors: List[Vector],
    hashes: List[int],
    max_hamming: int = 4
) -> Dict[int, List[int]]:
    """
    Simple agglomerative clustering: each vector index is assigned to the
    first cluster whose representative hash is within *max_hamming* bits.
    Returns a mapping from representative hash to list of vector indices.
    """
    clusters: Dict[int, List[int]] = {}
    for idx, h in enumerate(hashes):
        placed = False
        for rep in clusters:
            if hamming_distance(rep, h) <= max_hamming:
                clusters[rep].append(idx)
                placed = True
                break
        if not placed:
            clusters[h] = [idx]
    return clusters

def aggregate_labels(
    labels: List[int],
    confidences: List[float]
) -> Tuple[int, float]:
    """
    Weighted majority vote.
    Returns (aggregated_label, aggregated_confidence).
    """
    if not labels:
        raise ValueError("no labels to aggregate")
    weighted_sum = sum(l * c for l, c in zip(labels, confidences))
    total_conf = sum(confidences)
    agg_label = int(weighted_sum >= total_conf / 2)
    agg_conf = total_conf / len(labels)  # average confidence
    return agg_label, agg_conf

def build_model_tiers(
    vectors: List[Vector],
    hashes: List[int],
    labeling_results: List[Tuple[int, float]]
) -> List[ModelTier]:
    """
    Constructs ModelTier objects:
    * clusters vectors by hash,
    * fits a tiny RBF surrogate on each cluster,
    * aggregates labeling confidence as trust,
    * stores the surrogate prediction as expected_value.
    """
    clusters = cluster_by_hash(vectors, hashes)
    tiers: List[ModelTier] = []
    for tier_id, (rep_hash, idxs) in enumerate(clusters.items()):
        cluster_vecs = np.array([vectors[i] for i in idxs])
        # Dummy target for RBF: use aggregated label (0/1) as surrogate target
        lbls, confs = zip(*[labeling_results[i] for i in idxs])
        agg_label, agg_conf = aggregate_labels(list(lbls), list(confs))
        # Use agg_label as constant target for all support points
        support_vals = np.full(len(idxs), agg_label, dtype=float)
        # Predict at the centroid of the cluster
        centroid = cluster_vecs.mean(axis=0)
        pred = rbf_predict(cluster_vecs, support_vals, centroid)
        tier = ModelTier(
            name=f"tier_{tier_id}",
            hash_repr=rep_hash,
            trust=agg_conf,
            expected_value=pred,
        )
        tiers.append(tier)
    return tiers

def trust_aware_pruning(
    tiers: List[ModelTier],
    max_iterations: int = 20,
    init_temp: float = 1.0,
    alpha: float = 0.93
) -> ModelTier:
    """
    Simulated‑annealing search over *tiers*.
    The energy of a tier is defined as - (expected_value * trust).
    Moves consist of jumping to a randomly chosen tier; acceptance is
    modulated by both the energy delta and the linguistic similarity
    between current and candidate hashes.
    """
    if not tiers:
        raise ValueError("no tiers to prune")
    current = random.choice(tiers)
    best = current
    temperature = init_temp

    for k in range(max_iterations):
        candidate = random.choice(tiers)
        delta_e = - (candidate.expected_value * candidate.trust) + (current.expected_value * current.trust)
        sim = linguistic_similarity(current.hash_repr, candidate.hash_repr)
        # Blend similarity into acceptance probability
        prob = acceptance_probability(delta_e, temperature) * (0.5 + 0.5 * sim)
        if random.random() < prob:
            current = candidate
            if (candidate.expected_value * candidate.trust) > (best.expected_value * best.trust):
                best = candidate
        temperature = cooling_temperature(k, t0=init_temp, alpha=alpha)
    return best

# ----------------------------------------------------------------------
# Demonstration functions (minimum three)
# ----------------------------------------------------------------------
def demo_hash_and_cluster(vectors: List[Vector]) -> Tuple[List[int], Dict[int, List[int]]]:
    """Compute hashes for *vectors* and return clusters."""
    hashes = [combined_hash(v) for v in vectors]
    clusters = cluster_by_hash(vectors, hashes)
    return hashes, clusters

def demo_label_aggregation(labels: List[int], confidences: List[float]) -> Tuple[int, float]:
    """Aggregate weak‑supervision labels."""
    return aggregate_labels(labels, confidences)

def demo_full_pipeline(vectors: List[Vector]) -> ModelTier:
    """
    End‑to‑end execution:
    1. Hash & cluster,
    2. Generate dummy labeling results,
    3. Build ModelTier objects,
    4. Run trust‑aware pruning and return the selected tier.
    """
    hashes = [combined_hash(v) for v in vectors]
    # Dummy labeling: random label with confidence proportional to vector norm
    labeling_results = []
    for v in vectors:
        norm = np.linalg.norm(v)
        conf = min(1.0, norm / 10.0)
        label = random.choice([0, 1])
        labeling_results.append((label, conf))
    tiers = build_model_tiers(vectors, hashes, labeling_results)
    chosen = trust_aware_pruning(tiers)
    return chosen

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    # Generate 30 random 8‑dimensional vectors
    vectors = [np.random.rand(8).tolist() for _ in range(30)]

    # Demo 1: hash & cluster
    hashes, clusters = demo_hash_and_cluster(vectors)
    print(f"Generated {len(hashes)} hashes, formed {len(clusters)} clusters.")

    # Demo 2: label aggregation on a random subset
    sample_labels = [random.choice([0, 1]) for _ in range(10)]
    sample_confs = [random.random() for _ in range(10)]
    agg_label, agg_conf = demo_label_aggregation(sample_labels, sample_confs)
    print(f"Aggregated label: {agg_label}, confidence: {agg_conf:.3f}")

    # Demo 3: full pipeline
    best_tier = demo_full_pipeline(vectors)
    print(f"Selected tier: {best_tier.name}")
    print(f"  Hash (binary): {best_tier.text}")
    print(f"  Trust: {best_tier.trust:.3f}")
    print(f"  Expected value: {best_tier.expected_value:.3f}")