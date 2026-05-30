# DARWIN HAMMER — match 1626, survivor 3
# gen: 5
# parent_a: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m397_s2.py (gen4)
# born: 2026-05-29T23:37:51Z

"""Hybrid Algorithm: Signal Scores + MinHash Similarity Fusion

Parents:
- hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s1 (signal scoring, entropy of bytes)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m397_s2 (deterministic MinHash, entropy of probability vectors, RBF)

Mathematical Bridge:
Both parents employ Shannon entropy as a core quantitative measure.
Parent A computes entropy of raw byte streams to derive a *signal_score*.
Parent B computes entropy of arbitrary probability distributions (e.g., hit/miss states) and uses deterministic MinHash signatures to estimate Jaccard‑like similarity.

The fusion treats the byte‑entropy as a *content confidence* term and the MinHash similarity as a *structural alignment* term.
A combined objective (or “hybrid loss”) is defined as:

    L = -α·signal_score          # reward high signal
        + β·(1 - sim)            # penalize low structural similarity
        + γ·expected_entropy    # regularize with expected entropy of hit/miss states
        + δ·gaussian(dist)      # encourage close signatures in Euclidean space

α,β,γ,δ are tunable scalars. The resulting system yields a unified decision metric that
balances raw data quality, token‑level similarity, and probabilistic uncertainty.

The module implements this fusion with three public functions:
- `signal_scores`                – original byte‑entropy based scoring.
- `minhash_signature` / `minhash_similarity` – deterministic MinHash utilities.
- `hybrid_evaluation`            – computes the fused loss and returns a `ConduitDecision`.
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
# Parent A – Signal scoring utilities
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Shannon entropy of a byte sample, normalized to [0,1]."""
    if not data:
        return 0.0
    chunk = data[:sample]
    # Count byte frequencies
    freq = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
    probs = freq / freq.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    # Normalization: max entropy for a byte is 8 bits
    return entropy / 8.0


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """Return (signal_score, noise_score) based on content heuristics."""
    size = len(data)
    entropy = _byte_entropy(data)

    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.04)

    signal = entropy + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus
    signal = max(0.0, min(1.0, signal))  # clamp to [0,1]

    # Noise is defined as the complement of signal, perturbed by a small random term
    noise = 1.0 - signal + random.uniform(-0.02, 0.02)
    noise = max(0.0, min(1.0, noise))

    return signal, noise


# ----------------------------------------------------------------------
# Parent B – Deterministic MinHash and entropy utilities
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit hash of token+seed using SHA‑256."""
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
    """Jaccard‑like similarity based on identical positions in two signatures."""
    if not sig1 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted expected entropy for a hit/miss scenario."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function (Gaussian)."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Shared data structure
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_content_signature(data: bytes, num_hashes: int = 64) -> List[int]:
    """
    Tokenize raw bytes into UTF‑8 words (fallback to hex chunks) and compute a MinHash signature.
    """
    try:
        text = data.decode("utf-8", errors="ignore")
        tokens = text.split()
    except Exception:
        # Fallback: treat every two‑byte chunk as a token
        tokens = [data[i : i + 2].hex() for i in range(0, len(data), 2)]
    return minhash_signature(tokens, num_hashes)


def hybrid_loss(
    signal: float,
    similarity: float,
    dist: float,
    exp_entropy: float,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.3,
    delta: float = 0.5,
) -> float:
    """
    Combined loss that rewards high signal, high similarity, low distance,
    and low expected entropy.
    """
    return (
        -alpha * signal
        + beta * (1.0 - similarity)
        + gamma * exp_entropy
        + delta * (1.0 - gaussian(dist))
    )


def hybrid_evaluation(
    data: bytes,
    reference_signature: List[int],
    hit_state: List[float],
    miss_state: List[float],
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> ConduitDecision:
    """
    Produce a `ConduitDecision` by fusing signal scores with MinHash similarity
    and entropy‑based regularization.
    """
    # 1. Signal scores (Parent A)
    signal, noise = signal_scores(
        data,
        status_code=status_code,
        mime=mime,
        keyword_hits=keyword_hits,
        structural_links=structural_links,
    )

    # 2. MinHash similarity (Parent B)
    sig = compute_content_signature(data)
    sim = minhash_similarity(sig, reference_signature)

    # 3. Euclidean distance between signatures (treated as vectors of floats)
    # Convert 64‑bit ints to normalized floats in [0,1]
    vec_sig = [h / ((1 << 64) - 1) for h in sig]
    vec_ref = [h / ((1 << 64) - 1) for h in reference_signature]
    dist = euclidean(vec_sig, vec_ref)

    # 4. Expected entropy (Parent B)
    p_hit = signal  # interpret signal as probability of a "hit"
    exp_ent = expected_entropy(p_hit, hit_state, miss_state)

    # 5. Hybrid loss
    loss = hybrid_loss(signal, sim, dist, exp_ent)

    # Decision logic (simple heuristic)
    action = "accept" if loss < 0.5 else "reject"
    confidence_gap = max(0.0, signal - noise)
    epsilon = dist
    dormancy_probability = noise * (1.0 - sim)
    recovery_priority = 1.0 - loss

    reason = (
        f"loss={loss:.3f}, signal={signal:.3f}, sim={sim:.3f}, "
        f"dist={dist:.3f}, exp_ent={exp_ent:.3f}"
    )

    return ConduitDecision(
        action=action,
        confidence_gap=confidence_gap,
        epsilon=epsilon,
        signal_score=signal,
        noise_score=noise,
        dormancy_probability=dormancy_probability,
        recovery_priority=recovery_priority,
        reason=reason,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    sample_data = b"The quick brown fox jumps over the lazy dog. " * 10

    # Reference signature (could be from a known-good document)
    reference_sig = compute_content_signature(b"Reference document with similar vocabulary.", num_hashes=64)

    # Example hit/miss state distributions
    hit_state = [0.7, 0.2, 0.1]
    miss_state = [0.1, 0.3, 0.6]

    decision = hybrid_evaluation(
        data=sample_data,
        reference_signature=reference_sig,
        hit_state=hit_state,
        miss_state=miss_state,
        status_code=200,
        mime="text/plain",
        keyword_hits=3,
        structural_links=2,
    )

    print(decision)