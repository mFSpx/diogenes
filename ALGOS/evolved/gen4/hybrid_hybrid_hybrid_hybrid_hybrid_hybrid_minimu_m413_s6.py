# DARWIN HAMMER — match 413, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s0.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_infotaxis_min_m178_s0.py (gen2)
# born: 2026-05-29T23:28:54Z

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
# Bayesian primitives 
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
# Semantic‑neighbor utilities 
# ----------------------------------------------------------------------
def semantic_neighbors(doc_id: str, k: int = 5) -> list[tuple[str, float]]:
    neighbours = []
    for i in range(k):
        neighbour_id = f"{doc_id}_nbr_{i}"
        dist = random.random() * 2.0  
        neighbours.append((neighbour_id, dist))
    return neighbours

def semantic_likelihood(distance: float, scale: float = 1.0) -> float:
    return math.exp(-scale * distance)

# ----------------------------------------------------------------------
# MinHash utilities 
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: list[str], seed: int = 42, n_hashes: int = 128) -> np.ndarray:
    signature = np.full(n_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        for i in range(n_hashes):
            h = _hash(seed + i, token)
            if h < signature[i]:
                signature[i] = h
    return signature

def signature_distribution(sig: np.ndarray) -> np.ndarray:
    bins = 256
    hist, _ = np.histogram(sig % bins, bins=bins, range=(0, bins))
    prob = hist.astype(float) / hist.sum()
    if prob.sum() == 0:
        prob = np.full_like(prob, 1.0 / bins)
    return prob

def shannon_entropy(p: np.ndarray) -> float:
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
    geo = length(a, b)

    sig_a = minhash_signature(tokens_a, seed)
    sig_b = minhash_signature(tokens_b, seed)
    overlap = np.mean(sig_a == sig_b)  
    prior = overlap  

    neigh_a = semantic_neighbors(doc_a, k=1)[0][1]
    neigh_b = semantic_neighbors(doc_b, k=1)[0][1]
    avg_dist = (neigh_a + neigh_b) / 2.0

    posterior = hybrid_edge_posterior(prior, avg_dist)

    weight = geo * (1.0 - posterior)
    return weight

def infotaxis_action_selection(
    candidate_tokens: list[str],
    current_tokens: list[str],
    seed: int = 42,
    n_hashes: int = 128,
) -> str:
    cur_sig = minhash_signature(current_tokens, seed, n_hashes)
    cur_dist = signature_distribution(cur_sig)
    cur_entropy = shannon_entropy(cur_dist)

    best_token = None
    best_delta = -np.inf

    for token in candidate_tokens:
        new_tokens = current_tokens + [token]
        new_sig = minhash_signature(new_tokens, seed, n_hashes)
        new_dist = signature_distribution(new_sig)
        new_entropy = shannon_entropy(new_dist)

        delta_h = cur_entropy - new_entropy

        mock_dist = random.random() * 2.0 
        bayes_factor = math.exp(-mock_dist)

        delta_h_bayes = delta_h * bayes_factor

        if delta_h_bayes > best_delta:
            best_delta = delta_h_bayes
            best_token = token

    return best_token

def improved_infotaxis_action_selection(
    candidate_tokens: list[str],
    current_tokens: list[str],
    doc_id: str,
    seed: int = 42,
    n_hashes: int = 128,
) -> str:
    cur_sig = minhash_signature(current_tokens, seed, n_hashes)
    cur_dist = signature_distribution(cur_sig)
    cur_entropy = shannon_entropy(cur_dist)

    best_token = None
    best_delta = -np.inf

    for token in candidate_tokens:
        new_tokens = current_tokens + [token]
        new_sig = minhash_signature(new_tokens, seed, n_hashes)
        new_dist = signature_distribution(new_sig)
        new_entropy = shannon_entropy(new_dist)

        delta_h = cur_entropy - new_entropy

        neigh_dist = semantic_neighbors(doc_id, k=1)[0][1]
        likelihood = semantic_likelihood(neigh_dist)
        prior = 0.5  # Use a non-informative prior

        marginal = bayes_marginal(prior, likelihood, 0.01)
        posterior = bayes_update(prior, likelihood, marginal)

        delta_h_bayes = delta_h * posterior

        if delta_h_bayes > best_delta:
            best_delta = delta_h_bayes
            best_token = token

    return best_token