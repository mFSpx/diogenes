# DARWIN HAMMER — match 1000, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s0.py (gen4)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s6.py (gen4)
# born: 2026-05-29T23:32:20Z

"""Hybrid Pheromone‑Bayes‑MinHash Regret Engine
================================================

This module fuses the two parent DARWIN HAMMER algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s0.py*  
  Provides pheromone‑based probability distributions, a Bayesian posterior update,
  and a Shannon‑entropy analysis of decision‑hygiene scores.

* **Parent B** – *hybrid_korpus_text_hybrid_hybrid_regret_m21_s6.py*  
  Supplies MinHash text signatures, regret‑weighted scalar values, a LinUCB‑style
  confidence bound, and a bounded “dance” control signal.

**Mathematical bridge**  
Both parents expose a *probability‑like* vector:

* Parent A’s pheromone probabilities `p_i` (sum = 1) are a discrete distribution over
  recent surface signals.
* Parent B’s MinHash signature similarity `sim(sig_i, sig_ref)` is a normalized
  Jaccard‑like overlap (also in `[0,1]`).

We treat the pheromone distribution as a *prior* over hypotheses and the MinHash
similarity as a *likelihood* of the current textual evidence.  The Bayesian update
produces a posterior `π_i`.  Its Shannon entropy `H(π)` quantifies the uncertainty
of the combined belief.  The final hybrid action score blends:


S_i = σ(R_i) · (1 + sim_i) · dance · (1 + β·conf_i) · exp(-γ·H(π))


where  

* `σ` – sigmoid‑scaled regret weight,  
* `R_i` – raw regret value,  
* `sim_i` – MinHash similarity,  
* `dance` – bounded control signal ∈[0,1],  
* `conf_i` – LinUCB confidence term,  
* `β, γ` – tunable scalars.

The code below implements the full pipeline with three public functions that
demonstrate the hybrid operation.  All dependencies are limited to the Python
standard library and NumPy as required."""


import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core utilities (shared by both parents)
# ----------------------------------------------------------------------


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i : i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # Use blake2b via hashlib (available in stdlib)
    import hashlib

    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64, seed: int = 0) -> List[int]:
    """
    Compute a MinHash signature of length *k* for the given *tokens*.
    The algorithm follows the classic MinHash scheme: for each of *k* hash
    functions we keep the minimum hash value over the token set.
    """
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
    """
    Approximate Jaccard similarity for MinHash signatures.
    It is the fraction of positions with equal hash values.
    """
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy H = -Σ p·log₂(p) for a probability vector."""
    eps = np.finfo(float).eps
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, eps, 1.0)  # avoid log(0)
    return -np.sum(probs * np.log2(probs))


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def linucb_confidence(A_inv: np.ndarray, x: np.ndarray, alpha: float) -> float:
    """
    LinUCB confidence term: α·√(xᵀ·A⁻¹·x)
    A_inv must be a symmetric positive‑definite matrix (inverse of A).
    """
    if A_inv.shape[0] != x.shape[0]:
        raise ValueError("Dimension mismatch between A_inv and feature vector x")
    variance = float(x.T @ A_inv @ x)
    return alpha * math.sqrt(variance)


# ----------------------------------------------------------------------
# Hybrid functions (the three required public entry points)
# ----------------------------------------------------------------------


def compute_pheromone_distribution(
    raw_signals: List[float],
) -> List[float]:
    """
    Convert a list of raw pheromone signal strengths into a normalized probability
    distribution.  This mirrors the DB‑based version of Parent A but works with an
    in‑memory list for testability.
    """
    if not raw_signals:
        raise ValueError("Signal list cannot be empty")
    total = sum(raw_signals)
    if total == 0:
        # Uniform fallback if all signals are zero
        n = len(raw_signals)
        return [1.0 / n] * n
    return [s / total for s in raw_signals]


def bayesian_posterior_with_entropy(
    prior: List[float],
    likelihood: List[float],
) -> Tuple[List[float], float]:
    """
    Perform a Bayesian update given a *prior* distribution and a *likelihood*
    vector (both must sum to 1).  Returns the posterior distribution and its
    Shannon entropy.
    """
    if len(prior) != len(likelihood):
        raise ValueError("Prior and likelihood must have the same length")
    unnorm = np.multiply(prior, likelihood)
    evidence = np.sum(unnorm)
    if evidence == 0:
        # Avoid division by zero – revert to uniform posterior
        n = len(prior)
        posterior = [1.0 / n] * n
    else:
        posterior = (unnorm / evidence).tolist()
    entropy = shannon_entropy(posterior)
    return posterior, entropy


def hybrid_action_score(
    # Pheromone / Bayesian side
    pheromone_signals: List[float],
    # Textual side
    text: str,
    reference_signature: List[int],
    # Regret / bandit side
    raw_regret: float,
    dance_signal: float,
    A_inv: np.ndarray,
    feature_vec: np.ndarray,
    # Tunable scalars
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> float:
    """
    Compute the final hybrid score for a single action.

    Steps:
    1. Normalize pheromone signals → prior `p_i`.
    2. Build MinHash signature of *text* → `sig_i`.
    3. Compute similarity `sim_i` with *reference_signature*.
    4. Form a likelihood vector `L_i = (1 + sim_i) / Z` (Z normalises the vector).
    5. Bayesian update → posterior `π_i` and entropy `H`.
    6. Regret weighting `σ(R) = sigmoid(-R)` (negative regret → higher weight).
    7. LinUCB confidence `conf_i = linucb_confidence(A_inv, feature_vec, alpha)`.
    8. Combine all terms according to the bridge equation.

    Returns a scalar score `S_i`.
    """
    # 1. Prior
    prior = compute_pheromone_distribution(pheromone_signals)

    # 2. MinHash signature of the current text
    tokens = _shingles(text, width=5)
    sig_i = minhash_signature(tokens, k=len(reference_signature), seed=42)

    # 3. Similarity
    sim_i = jaccard_similarity(sig_i, reference_signature)  # ∈[0,1]

    # 4. Likelihood – we embed the similarity as a simple linear factor.
    #    Adding a small epsilon avoids a zero vector.
    eps = 1e-8
    raw_likelihood = np.array([1.0 + sim_i + eps])
    likelihood = (raw_likelihood / raw_likelihood.sum()).tolist()

    # 5. Posterior & entropy
    posterior, entropy = bayesian_posterior_with_entropy(prior, likelihood)

    # 6. Regret weighting
    regret_weight = sigmoid(-raw_regret)  # higher regret → lower weight

    # 7. Confidence term
    conf = linucb_confidence(A_inv, feature_vec, alpha)

    # 8. Final hybrid score
    score = (
        regret_weight
        * (1.0 + sim_i)
        * max(0.0, min(1.0, dance_signal))  # clamp dance to [0,1]
        * (1.0 + beta * conf)
        * math.exp(-gamma * entropy)
    )
    return float(score)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Dummy pheromone signals (e.g., recent surface intensities)
    pheromones = [random.expovariate(1.0) for _ in range(7)]

    # 2. Reference MinHash signature – pretend it comes from the best‑so‑far action
    reference_text = "The quick brown fox jumps over the lazy dog."
    ref_tokens = _shingles(reference_text, width=5)
    ref_sig = minhash_signature(ref_tokens, k=64, seed=99)

    # 3. Bandit parameters
    dim = 8
    A = np.eye(dim) * 1.0  # simple identity covariance
    A_inv = np.linalg.inv(A)
    x = np.random.randn(dim)  # feature vector for the current action

    # 4. Run hybrid scoring for a sample action
    sample_text = "A fast auburn rabbit leaps across the sleepy cat."
    raw_regret = random.uniform(-2.0, 2.0)  # negative = good outcome
    dance = random.random()  # control signal in [0,1]

    score = hybrid_action_score(
        pheromone_signals=pheromones,
        text=sample_text,
        reference_signature=ref_sig,
        raw_regret=raw_regret,
        dance_signal=dance,
        A_inv=A_inv,
        feature_vec=x,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
    )

    print("Hybrid action score:", score)
    # Ensure the function runs without raising.
    sys.exit(0)