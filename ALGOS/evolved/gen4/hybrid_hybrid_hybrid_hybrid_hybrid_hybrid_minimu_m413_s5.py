# DARWIN HAMMER — match 413, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s0.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_infotaxis_min_m178_s0.py (gen2)
# born: 2026-05-29T23:28:54Z

"""
Hybrid Algorithm: Semantic‑Infotaxis MinHash Bayes (Hybrid_A × Hybrid_B)

This module fuses the two parent algorithms:

* **Parent A (semantic_neighbors + label scoring + Bayesian update)**
* **Parent B (minimum‑cost tree, MinHash signatures, entropy‑driven infotaxis + Bayesian update)**

Mathematical bridge
-------------------
1. **Likelihood source** – The Euclidean distance between a document and its
   semantic neighbours is turned into a likelihood 𝓁(d) = exp(‑α·dist) (α>0).
2. **Prior source** – A MinHash signature of the token set is interpreted as a
   discrete probability distribution p_i over hash buckets; this distribution
   serves as the Bayesian *prior* for each edge.
3. **Bayesian update** – For every edge (u,v) we compute the marginal
   `m = bayes_marginal(prior, likelihood, false_positive)` and then the
   posterior `post = bayes_update(prior, likelihood, m)`.  The posterior
   replaces the original edge weight in the minimum‑cost tree.
4. **Entropy‑driven decision (infotaxis)** – The Shannon entropy of the
   MinHash distribution H(p) is the uncertainty measure.  For a candidate
   action (adding a token) we recompute the signature, update the posterior
   edge weights and evaluate the expected entropy reduction ΔH.  The action
   with maximal ΔH is selected.

The three core functions below implement this fused pipeline.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np

Point = tuple[float, float]
Edge = tuple[str, str]

# ----------------------------------------------------------------------
# Shared geometric utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian primitives (identical in both parents)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = L·P + FP·(1‑P)"""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)"""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Semantic‑neighbor utilities (from Parent A)
# ----------------------------------------------------------------------
def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Mock implementation: returns ``k`` synthetic neighbours with random distances.
    In a real system this would query an embedding index.
    """
    neighbours = []
    for i in range(k):
        neighbour_id = f"{doc_id}_nbr_{i}"
        dist = random.random() * 2.0  # distance in [0,2)
        neighbours.append((neighbour_id, dist))
    return neighbours

def semantic_likelihood(distance: float, scale: float = 1.0) -> float:
    """
    Convert Euclidean distance to a likelihood in (0,1].
    L = exp(-scale * distance)
    """
    return math.exp(-scale * distance)

# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: list[str], seed: int = 42, n_hashes: int = 128) -> np.ndarray:
    """
    Compute a MinHash signature: the minimum hash value for each of ``n_hashes`` hash functions.
    Returns an array of shape (n_hashes,) with uint64 values.
    """
    signature = np.full(n_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        for i in range(n_hashes):
            h = _hash(seed + i, token)
            if h < signature[i]:
                signature[i] = h
    return signature

def signature_distribution(sig: np.ndarray) -> np.ndarray:
    """
    Treat the signature as a histogram over its bucket values.
    Normalise to obtain a probability distribution.
    """
    # Bucket by modulo to keep a manageable number of bins
    bins = 256
    hist, _ = np.histogram(sig % bins, bins=bins, range=(0, bins))
    prob = hist.astype(float) / hist.sum()
    # Guard against zero‑sum (should not happen)
    if prob.sum() == 0:
        prob = np.full_like(prob, 1.0 / bins)
    return prob

def shannon_entropy(p: np.ndarray) -> float:
    """Standard Shannon entropy (base‑2)."""
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_edge_posterior(
    prior: float,
    distance: float,
    false_positive: float = 0.01,
    scale: float = 1.0,
) -> float:
    """
    Compute the Bayesian posterior weight for an edge using the semantic
    distance as likelihood.
    """
    likelihood = semantic_likelihood(distance, scale)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)

def hybrid_tree_weight(
    a: Point,
    b: Point,
    doc_a: str,
    doc_b: str,
    tokens_a: list[str],
    tokens_b: list[str],
    seed: int = 42,
) -> float:
    """
    Combine geometric length, semantic Bayesian posterior, and MinHash‑derived
    prior into a single edge weight.
    """
    # 1. Geometric component
    geo = length(a, b)

    # 2. Prior from MinHash similarity (Jaccard‑like via signature overlap)
    sig_a = minhash_signature(tokens_a, seed)
    sig_b = minhash_signature(tokens_b, seed)
    overlap = np.mean(sig_a == sig_b)  # proportion of identical hash values
    prior = overlap  # prior ∈ [0,1]

    # 3. Likelihood from semantic neighbour distance (use average of two docs)
    # For simplicity we treat the distance between the two docs as the mean of
    # their nearest neighbour distances.
    neigh_a = semantic_neighbors(doc_a, k=1)[0][1]
    neigh_b = semantic_neighbors(doc_b, k=1)[0][1]
    avg_dist = (neigh_a + neigh_b) / 2.0

    posterior = hybrid_edge_posterior(prior, avg_dist)

    # 4. Combine: geometric cost scaled by (1‑posterior) so that higher posterior
    #    yields lower effective cost.
    weight = geo * (1.0 - posterior)
    return weight

def infotaxis_action_selection(
    candidate_tokens: list[str],
    current_tokens: list[str],
    seed: int = 42,
    n_hashes: int = 128,
) -> str:
    """
    Choose the token that maximises expected entropy reduction of the MinHash
    signature after a Bayesian update of edge weights.

    Returns the selected token.
    """
    # Current signature distribution and entropy
    cur_sig = minhash_signature(current_tokens, seed, n_hashes)
    cur_dist = signature_distribution(cur_sig)
    cur_entropy = shannon_entropy(cur_dist)

    best_token = None
    best_delta = -np.inf

    for token in candidate_tokens:
        # Simulate adding the token
        new_tokens = current_tokens + [token]
        new_sig = minhash_signature(new_tokens, seed, n_hashes)
        new_dist = signature_distribution(new_sig)
        new_entropy = shannon_entropy(new_dist)

        # Expected entropy reduction (ΔH)
        delta_h = cur_entropy - new_entropy

        # Incorporate a Bayesian factor: treat the token's semantic likelihood
        # as exp(-dist) where dist is a mock distance to a random neighbour.
        mock_dist = random.random() * 2.0
        likelihood = semantic_likelihood(mock_dist, scale=0.5)
        posterior = bayes_update(0.5, likelihood, bayes_marginal(0.5, likelihood, 0.01))

        # Weight the entropy gain by the posterior (higher confidence → more
        # valuable gain)
        weighted_delta = delta_h * posterior

        if weighted_delta > best_delta:
            best_delta = weighted_delta
            best_token = token

    return best_token if best_token is not None else ""

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple geometric points
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)  # distance = 5

    # Mock document identifiers
    doc1 = "doc_A"
    doc2 = "doc_B"

    # Token sets for each document
    tokens1 = ["alpha", "beta", "gamma"]
    tokens2 = ["delta", "epsilon", "zeta", "beta"]

    # Compute hybrid edge weight
    w = hybrid_tree_weight(p1, p2, doc1, doc2, tokens1, tokens2)
    print(f"Hybrid edge weight between {doc1} and {doc2}: {w:.4f}")

    # Infotaxis‑style action selection
    candidates = ["theta", "iota", "kappa", "lambda"]
    chosen = infotaxis_action_selection(candidates, tokens1)
    print(f"Selected token to add (infotaxis): {chosen}")

    # Verify that the selected token actually changes entropy
    before_entropy = shannon_entropy(signature_distribution(minhash_signature(tokens1)))
    after_entropy = shannon_entropy(
        signature_distribution(minhash_signature(tokens1 + [chosen]))
    )
    print(f"Entropy before: {before_entropy:.4f}, after adding '{chosen}': {after_entropy:.4f}")