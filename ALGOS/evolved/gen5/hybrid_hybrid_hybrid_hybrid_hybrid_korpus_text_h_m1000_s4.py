# DARWIN HAMMER — match 1000, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s0.py (gen4)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s6.py (gen4)
# born: 2026-05-29T23:32:20Z

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable
import numpy as np

def _shingles(text: str, width: int = 5) -> List[str]:
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i : i + width] for i in range(len(cleaned) - width + 1)]

def _hash_token(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    import hashlib
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], k: int = 64, seed: int = 0) -> List[int]:
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    signature = [2 ** 64 - 1] * k
    for i in range(k):
        for token in token_set:
            h = _hash_token(seed + i, token)
            if h < signature[i]:
                signature[i] = h
    return signature

def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def shannon_entropy(probs: List[float]) -> float:
    eps = np.finfo(float).eps
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)  
    return -np.sum(probs * np.log2(probs))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def linucb_confidence(A_inv: np.ndarray, x: np.ndarray, alpha: float) -> float:
    if A_inv.shape[0] != x.shape[0]:
        raise ValueError("Dimension mismatch between A_inv and feature vector x")
    variance = float(x.T @ A_inv @ x)
    return alpha * math.sqrt(variance)

def compute_pheromone_distribution(raw_signals: List[float]) -> List[float]:
    if not raw_signals:
        raise ValueError("Signal list cannot be empty")
    total = sum(raw_signals)
    if total == 0:
        n = len(raw_signals)
        return [1.0 / n] * n
    return [s / total for s in raw_signals]

def bayesian_posterior_with_entropy(prior: List[float], likelihood: List[float]) -> Tuple[List[float], float]:
    if len(prior) != len(likelihood):
        raise ValueError("Prior and likelihood must have the same length")
    unnorm = np.multiply(prior, likelihood)
    evidence = np.sum(unnorm)
    if evidence == 0:
        n = len(prior)
        posterior = [1.0 / n] * n
    else:
        posterior = (unnorm / evidence).tolist()
    entropy = shannon_entropy(posterior)
    return posterior, entropy

def hybrid_action_score(
    pheromone_signals: List[float],
    text: str,
    reference_signature: List[int],
    raw_regret: float,
    dance_signal: float,
    A_inv: np.ndarray,
    feature_vec: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    prior = compute_pheromone_distribution(pheromone_signals)
    tokens = _shingles(text, width=5)
    sig_i = minhash_signature(tokens, k=len(reference_signature), seed=42)
    sim_i = jaccard_similarity(sig_i, reference_signature) 
    likelihood = np.array([sim_i]) / (1 + sim_i)
    posterior, entropy = bayesian_posterior_with_entropy(prior, likelihood.tolist())
    regret_weight = sigmoid(-raw_regret)  
    conf = linucb_confidence(A_inv, feature_vec, alpha)
    score = regret_weight * (1 + sim_i) * dance_signal * (1 + beta * conf) * np.exp(-gamma * entropy)
    return score