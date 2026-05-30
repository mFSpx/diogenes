# DARWIN HAMMER — match 5617, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1838_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bandit_router_m1212_s0.py (gen3)
# born: 2026-05-30T00:03:26Z

"""Hybrid Algorithm: hybrid_sparse_pheromone_fisher.py
Combines:
- Parent A (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1838_s0): hash‑based sparse expansion (WTA) and top‑k masking.
- Parent B (hybrid_hybrid_hybrid_pherom_hybrid_bandit_router_m1212_s0): deterministic MinHash signatures, similarity, and entropy.

Mathematical bridge:
Both parents expose a *confidence scalar* that rescales stochastic components.
* In Parent A the signal‑to‑noise gap is interpreted as a confidence scalar `c` that rescales the random coefficient in the sparse expansion.
* In Parent B the temperature (or context) modulates pheromone routing; we compute an analogous scalar from the token‑distribution entropy.

The fusion uses a single scalar `conf` derived from the data vector:

conf = 1 / (1 + entropy(normalized |values|))

`conf ∈ (0,1]` and is used to:
1. Scale the contribution of each hashed bucket in the sparse expansion.
2. Weight the MinHash‑based similarity when routing decisions are made.

Thus the hybrid algorithm simultaneously performs a sparse projection of a numeric signal and a similarity‑based routing of a symbolic token set, each modulated by a common confidence measure. """

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Parent A utilities (sparse WTA)
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, confidence: float = 1.0, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`.

    Each hashed bucket receives `confidence * v * sign`.  The confidence
    scalar rescales the random coefficient that would otherwise be 1.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        if v == 0.0:
            continue
        for r in range(3):  # three independent hashes per entry
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += confidence * sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return indices of the `k` largest‑absolute‑value entries in `values`.

    If `k` exceeds the length of the list, all indices are returned.
    """
    if k <= 0:
        return []
    if k >= len(values):
        return list(range(len(values)))
    # argsort by absolute value descending
    abs_idx = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    return abs_idx[:k]


# ----------------------------------------------------------------------
# Parent B utilities (deterministic MinHash & entropy)
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit hash of `token` combined with `seed`."""
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    """Deterministic MinHash signature for a list of tokens."""
    signature = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity based on identical hash positions."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Fusion core: confidence scalar
# ----------------------------------------------------------------------
def confidence_from_signal(values: List[float]) -> float:
    """Derive a confidence scalar from the numeric signal.

    The scalar is `1 / (1 + entropy(normalized |values|))`,
    guaranteeing a value in (0, 1].
    """
    if not values:
        return 1.0
    abs_vals = np.abs(values).astype(float)
    total = float(np.sum(abs_vals))
    if total == 0.0:
        return 1.0
    probs = abs_vals / total
    ent = calculate_entropy(probs.tolist())
    conf = 1.0 / (1.0 + ent)
    return conf


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_sparse_projection(values: List[float], m: int, k: int, salt: str = "") -> np.ndarray:
    """Perform sparse expansion and keep only the top‑k components.

    The confidence scalar derived from `values` rescales the expansion.
    """
    conf = confidence_from_signal(values)
    expanded = expand(values, m, confidence=conf, salt=salt)
    mask_idx = top_k_mask(expanded, k)
    result = np.zeros(m)
    for idx in mask_idx:
        result[idx] = expanded[idx]
    return result


def hybrid_token_similarity(tokens_a: List[str], tokens_b: List[str],
                            num_hashes: int = 64) -> float:
    """Compute confidence‑weighted MinHash similarity between two token sets.

    The confidence scalar is obtained from the *concatenated* token
    frequencies (treated as a signal) and used to weight the raw similarity.
    """
    # Build a simple frequency‑based numeric signal from tokens
    all_tokens = list(set(tokens_a + tokens_b))
    freq = [tokens_a.count(t) + tokens_b.count(t) for t in all_tokens]
    conf = confidence_from_signal(freq)

    sig_a = minhash_signature(tokens_a, num_hashes)
    sig_b = minhash_signature(tokens_b, num_hashes)
    raw_sim = minhash_similarity(sig_a, sig_b)
    return conf * raw_sim


def hybrid_process(numeric_signal: List[float],
                   token_set_a: List[str],
                   token_set_b: List[str],
                   m: int = 1024,
                   k: int = 32,
                   num_hashes: int = 128,
                   salt: str = "hybrid") -> Dict[str, Any]:
    """End‑to‑end hybrid routine.

    Returns a dictionary containing:
    - `sparse_vector`: top‑k sparse projection (numpy array)
    - `token_similarity`: confidence‑weighted MinHash similarity
    - `combined_score`: product of L2 norm of the sparse vector and token similarity
    """
    sparse_vec = hybrid_sparse_projection(numeric_signal, m, k, salt=salt)
    token_sim = hybrid_token_similarity(token_set_a, token_set_b, num_hashes=num_hashes)

    # A simple way to fuse the modalities: multiply norm by similarity
    norm = float(np.linalg.norm(sparse_vec))
    combined = norm * token_sim

    return {
        "sparse_vector": sparse_vec,
        "token_similarity": token_sim,
        "combined_score": combined
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic numeric signal
    np.random.seed(0)
    signal = np.random.randn(50).tolist()

    # Synthetic token sets
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    tokens_a = random.choices(vocab, k=30)
    tokens_b = random.choices(vocab, k=30)

    result = hybrid_process(
        numeric_signal=signal,
        token_set_a=tokens_a,
        token_set_b=tokens_b,
        m=512,
        k=20,
        num_hashes=64,
        salt="test"
    )

    print("Sparse vector (non‑zero entries):")
    nz = np.where(result['sparse_vector'] != 0)[0]
    print(f"Indices: {nz.tolist()}")
    print(f"Values: {result['sparse_vector'][nz].tolist()}")
    print(f"Token similarity (confidence weighted): {result['token_similarity']:.4f}")
    print(f"Combined score: {result['combined_score']:.4f}")