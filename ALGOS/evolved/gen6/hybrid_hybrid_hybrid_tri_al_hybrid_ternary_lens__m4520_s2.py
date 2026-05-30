# DARWIN HAMMER — match 4520, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s3.py (gen5)
# parent_b: hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s0.py (gen2)
# born: 2026-05-29T23:56:19Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s3.py (byte‑entropy signal scores + deterministic MinHash similarity)
- hybrid_ternary_lens_router_hybrid_decision_hygi_m44_s0.py (ternary lens router + decision hygiene Shannon entropy)

Mathematical Bridge:
Both parents rely on Shannon entropy as a quantitative core.
Parent A measures entropy of raw byte streams (content confidence);
Parent B measures entropy of ternary vectors derived from the same data (structural confidence).
The fusion therefore treats:
    • byte_entropy → c₁ (content confidence)
    • ternary_entropy → c₂ (structural confidence)
    • MinHash similarity → s (set‑based alignment)
and combines them in a single loss
    L = -α·(c₁) + β·(1‑s) + γ·(c₂) + δ·‖h₁‑h₂‖₂
where h₁,h₂ are deterministic MinHash signatures.
The implementation below provides three public functions that demonstrate this hybrid operation:
    1. `signal_score`
    2. `ternary_vector` + `vector_entropy`
    3. `hybrid_evaluation`
"""

import math
import random
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Shannon entropy of a byte sample, normalized to [0,1]."""
    if not data:
        return 0.0
    chunk = data[:sample]
    freq = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
    probs = freq / freq.sum()
    # avoid log(0)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    # Normalization: max entropy for 256 symbols is log2(256)=8
    return entropy / 8.0


def signal_score(data: bytes) -> float:
    """Public wrapper for the byte‑entropy based signal score."""
    return _byte_entropy(data)


# ----------------------------------------------------------------------
# Core utilities from Parent B (ternary lens router + decision hygiene)
# ----------------------------------------------------------------------
TERNARY_DIMS = 12  # as defined in the original parent


def ternary_vector(data: bytes, dims: int = TERNARY_DIMS) -> np.ndarray:
    """
    Deterministic ternary vector generator.
    For each dimension i, a SHA‑256 hash of (data || i) is taken,
    the first 4 bytes are interpreted as a uint32, reduced modulo 3,
    and finally shifted to the set {-1, 0, 1}.
    """
    vec = np.empty(dims, dtype=int)
    for i in range(dims):
        hasher = hashlib.sha256()
        hasher.update(data)
        hasher.update(i.to_bytes(2, byteorder="little"))
        digest = hasher.digest()[:4]
        val = int.from_bytes(digest, byteorder="little")
        vec[i] = (val % 3) - 1  # maps 0→-1, 1→0, 2→1
    return vec


def vector_entropy(vec: np.ndarray) -> float:
    """
    Shannon entropy of the ternary vector distribution,
    normalized to [0,1] (max entropy = log2(3)).
    """
    if vec.size == 0:
        return 0.0
    # Count occurrences of -1, 0, 1
    counts = np.bincount(vec + 1, minlength=3)  # shift to 0..2
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy / math.log2(3)


# ----------------------------------------------------------------------
# Deterministic MinHash utilities (Parent A)
# ----------------------------------------------------------------------
def _minhash_token_hash(token: str, seed: int) -> int:
    """Hash a token with a seed using SHA‑1 and return a 64‑bit integer."""
    h = hashlib.sha1()
    h.update(seed.to_bytes(4, byteorder="little"))
    h.update(token.encode("utf-8"))
    digest = h.digest()[:8]
    return int.from_bytes(digest, byteorder="little")


def minhash_signature(tokens: List[str], num_perm: int = 128) -> List[int]:
    """
    Deterministic MinHash signature for a list of tokens.
    For each permutation (seed) the minimal token hash is kept.
    """
    if not tokens:
        return [0] * num_perm
    signature = []
    for seed in range(num_perm):
        min_hash = min(_minhash_token_hash(tok, seed) for tok in tokens)
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity estimated from two MinHash signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Hybrid fusion
# ----------------------------------------------------------------------
@dataclass
class HybridResult:
    signal_a: float
    signal_b: float
    ternary_entropy_a: float
    ternary_entropy_b: float
    minhash_sim: float
    euclidean_dist: float
    loss: float


def _tokens_from_ternary(vec: np.ndarray) -> List[str]:
    """
    Convert a ternary vector into a token list.
    Positions with non‑zero entries become tokens of the form
    "pos{i}_{sign}" where sign is '+' or '-'.
    """
    tokens = []
    for i, val in enumerate(vec):
        if val == 0:
            continue
        sign = "+" if val == 1 else "-"
        tokens.append(f"pos{i}{sign}")
    return tokens


def hybrid_evaluation(
    data_a: bytes,
    data_b: bytes,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
    delta: float = 0.2,
    num_minhash_perm: int = 64,
) -> HybridResult:
    """
    Compute the unified hybrid metric for two byte streams.

    Steps:
    1. Compute byte‑entropy signal scores (c₁).
    2. Generate deterministic ternary vectors and their entropies (c₂).
    3. Derive MinHash signatures from the ternary tokens.
    4. Estimate similarity s and Euclidean distance d between signatures.
    5. Combine everything into the loss L = -α·c₁ + β·(1‑s) + γ·c₂ + δ·d.
    """
    # 1. Byte‑entropy signal scores
    sig_a = signal_score(data_a)
    sig_b = signal_score(data_b)

    # 2. Ternary vectors and entropies
    vec_a = ternary_vector(data_a)
    vec_b = ternary_vector(data_b)
    ent_a = vector_entropy(vec_a)
    ent_b = vector_entropy(vec_b)

    # 3. MinHash signatures from ternary tokens
    tokens_a = _tokens_from_ternary(vec_a)
    tokens_b = _tokens_from_ternary(vec_b)
    mh_a = minhash_signature(tokens_a, num_perm=num_minhash_perm)
    mh_b = minhash_signature(tokens_b, num_perm=num_minhash_perm)

    # 4. Similarity & Euclidean distance
    sim = minhash_similarity(mh_a, mh_b)
    euclid = np.linalg.norm(np.array(mh_a, dtype=np.float64) - np.array(mh_b, dtype=np.float64))

    # 5. Loss aggregation (averaging the two signal/entropy terms)
    avg_signal = (sig_a + sig_b) / 2.0
    avg_entropy = (ent_a + ent_b) / 2.0
    loss = -alpha * avg_signal + beta * (1.0 - sim) + gamma * avg_entropy + delta * euclid

    return HybridResult(
        signal_a=sig_a,
        signal_b=sig_b,
        ternary_entropy_a=ent_a,
        ternary_entropy_b=ent_b,
        minhash_sim=sim,
        euclidean_dist=euclid,
        loss=loss,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic byte strings
    data1 = b"The quick brown fox jumps over the lazy dog."
    data2 = b"The quick brown fox leaps over the lazy dog!"

    result = hybrid_evaluation(data1, data2)
    print("Hybrid Evaluation Result:")
    print(f"  Signal A          : {result.signal_a:.4f}")
    print(f"  Signal B          : {result.signal_b:.4f}")
    print(f"  Ternary Entropy A : {result.ternary_entropy_a:.4f}")
    print(f"  Ternary Entropy B : {result.ternary_entropy_b:.4f}")
    print(f"  MinHash Similarity: {result.minhash_sim:.4f}")
    print(f"  Euclidean Dist    : {result.euclidean_dist:.2f}")
    print(f"  Combined Loss     : {result.loss:.4f}")