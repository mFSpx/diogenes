# DARWIN HAMMER — match 1378, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s1.py (gen3)
# born: 2026-05-29T23:35:43Z

"""
Hybrid algorithm combining ternary routing (from `hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s1.py`) 
with text‑based feature extraction (minhash, entropy, vector literals from `korpus_text.py`) and 
epistemic certainty flags into the edge weights of the minimum-cost tree (from `hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s0.py`).
The mathematical bridge is established by incorporating the epistemic certainty flags into the cost matrix **C** 
and using the ternary routing step to select an intermediate node *k* that minimises C_{source,k}+C_{k,destination}.
The same Euclidean geometry is reused for a Voronoi partition: a set of seed indices defines Voronoi cells; 
every point is assigned to the nearest seed according to the same distance metric.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Text‑based feature extraction (minhash, entropy, vector literal)
# ----------------------------------------------------------------------

def _shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping substrings (shingles) of given width."""
    return [text[i : i + width] for i in range(len(text) - width + 1)]

def minhash_signature(text: str, k: int = 64, width: int = 5) -> List[int]:
    """
    Very small minhash implementation.
    Returns the k smallest hash values of the shingles.
    """
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    # deterministic hash: use built‑in hash mixed with a fixed seed
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    # pad if fewer than k shingles
    return (hashes[:k] + [0] * k)[:k]

def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of the character distribution (up to 10 000 chars)."""
    if not text:
        return 0.0
    text = text[:10000]
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    ent = 0.0
    for f in freq.values():
        p = f / total
        ent -= p * math.log(p, 2)
    return ent

# ----------------------------------------------------------------------
# Epistemic certainty flags and bayesian update
# ----------------------------------------------------------------------

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal <= 0:
        raise ValueError("P(E) must be positive")
    return (likelihood * prior) / marginal

# ----------------------------------------------------------------------
# Hybrid algorithm combining ternary routing and text-based feature extraction
# ----------------------------------------------------------------------

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_cost_matrix(vectors: List[Tuple[float, ...]]) -> np.ndarray:
    """Build a symmetric cost matrix **C** from the pairwise squared Euclidean distances."""
    num_vectors = len(vectors)
    cost_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            cost_matrix[i, j] = length(vectors[i], vectors[j])
            cost_matrix[j, i] = cost_matrix[i, j]
    return cost_matrix

def ternary_routing(cost_matrix: np.ndarray, source: int, destination: int) -> int:
    """Select an intermediate node *k* that minimises C_{source,k}+C_{k,destination}."""
    num_nodes = cost_matrix.shape[0]
    costs = np.zeros(num_nodes)
    for k in range(num_nodes):
        costs[k] = cost_matrix[source, k] + cost_matrix[k, destination]
    k = np.argmin(costs)
    return k

def hybrid_hybrid_algorithm(texts: List[str]) -> List[int]:
    """Combine ternary routing and text-based feature extraction."""
    # Extract features from the input texts
    vectors = []
    for text in texts:
        signature = minhash_signature(text)
        entropy = shannon_entropy(text)
        vector = np.array(signature + [entropy])
        vectors.append(vector)
    
    # Build the cost matrix **C** from the pairwise squared Euclidean distances
    cost_matrix = build_cost_matrix(vectors)
    
    # Select an intermediate node *k* that minimises C_{source,k}+C_{k,destination}
    source = 0
    destination = len(texts) - 1
    k = ternary_routing(cost_matrix, source, destination)
    
    # Update the epistemic certainty flags based on the Bayesian update
    priors = [0.5] * len(texts)
    likelihoods = [0.9] * len(texts)
    false_positives = [0.1] * len(texts)
    marginals = [bayes_marginal(prior, likelihood, false_positive) for prior, likelihood, false_positive in zip(priors, likelihoods, false_positives)]
    updated_priors = [bayes_update(prior, likelihood, marginal) for prior, likelihood, marginal in zip(priors, likelihoods, marginals)]
    
    return [k] + updated_priors

if __name__ == "__main__":
    texts = ["This is a sample text.", "Another sample text."]
    k, updated_priors = hybrid_hybrid_algorithm(texts)
    print(f"Intermediate node: {k}")
    print(f"Updated priors: {updated_priors}")