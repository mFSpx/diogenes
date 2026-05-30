# DARWIN HAMMER — match 212, survivor 1
# gen: 3
# parent_a: minhash.py (gen0)
# parent_b: hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s1.py (gen2)
# born: 2026-05-29T23:27:42Z

"""Hybrid MinHash‑NLMS with RLCT‑adjusted learning.

Parent A: minhash.py – provides set‑based shingling and MinHash signatures that
approximate Jaccard similarity.

Parent B: hybrid_rlct_grokking… – provides a Normalised Least‑Mean‑Squares (NLMS)
adaptive filter whose learning rate μ is modulated by a Real Log Canonical
Threshold (RLCT) derived from model complexity.

Mathematical bridge
------------------
A MinHash signature `s ∈ ℤ^k` is a high‑dimensional sparse representation of a
document.  We treat the signature as a deterministic feature vector `x`
(normalised to [0,1]) for the NLMS predictor.  The RLCT is approximated by the
entropy of the signature’s hash distribution:

    H(s) = - Σ_i p_i log p_i ,   p_i = count(hash_i)/k

A larger entropy indicates a more “complex” representation; the RLCT is taken
as `λ = 1 / (1 + H(s))`.  The effective learning rate becomes

    μ_eff = μ_base * λ

Thus the NLMS weight update adapts to the intrinsic complexity of the MinHash
signature, fusing the two parent topologies into a single learning system.

The module implements:
* MinHash utilities (shingles, signature, similarity)
* NLMS predictor/update with RLCT‑adjusted μ
* A simple trainer that learns to map a pair of signatures to their true Jaccard
  similarity.

"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Parent A – MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def shingles(text: str, width: int = 5) -> Set[str]:
    """Return a set of width‑wide word shingles."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Compute a k‑length MinHash signature for the given tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent B – NLMS with RLCT‑adjusted learning rate
# ----------------------------------------------------------------------
def rlct_from_signature(sig: List[int]) -> float:
    """
    Approximate the Real Log Canonical Threshold (RLCT) from a MinHash signature.

    The signature is treated as a discrete distribution over its hash values.
    Entropy H is computed and mapped to λ = 1 / (1 + H) ∈ (0,1].

    Returns
    -------
    float
        RLCT factor λ.
    """
    if not sig:
        return 1.0
    # Frequency of each distinct hash value
    counts = {}
    for h in sig:
        counts[h] = counts.get(h, 0) + 1
    k = len(sig)
    entropy = 0.0
    for c in counts.values():
        p = c / k
        entropy -= p * math.log(p + 1e-12)  # add epsilon to avoid log(0)
    return 1.0 / (1.0 + entropy)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu_base: float = 0.5,
    eps: float = 1e-9,
    rlct: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update with RLCT‑scaled learning rate.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector.
    x : np.ndarray
        Input feature vector.
    target : float
        Desired output.
    mu_base : float, optional
        Base learning rate (default 0.5).
    eps : float, optional
        Small constant to avoid division by zero.
    rlct : float, optional
        RLCT factor λ ∈ (0,1]; μ_eff = μ_base * λ.

    Returns
    -------
    tuple (new_weights, error)
        Updated weight vector and instantaneous error (target‑prediction).
    """
    y = nlms_predict(weights, x)
    e = target - y
    mu_eff = mu_base * rlct
    norm = float(x @ x) + eps
    new_weights = weights + (mu_eff / norm) * e * x
    return new_weights, e


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def signature_vector(text: str, k: int = 128, width: int = 5) -> np.ndarray:
    """
    Produce a normalised float feature vector from a document.

    The MinHash signature is scaled to the interval [0, 1] by dividing by MAX64.
    """
    sh = shingles(text, width)
    sig = signature(sh, k)
    return np.array(sig, dtype=np.float64) / MAX64


def train_nlms_on_pairs(
    pairs: List[Tuple[str, str]],
    epochs: int = 10,
    mu_base: float = 0.5,
    k: int = 128,
    width: int = 5,
) -> np.ndarray:
    """
    Train NLMS weights to predict true Jaccard similarity from two documents.

    The input vector for a pair (a, b) is the concatenation of their
    normalised MinHash signatures.  The target is the exact Jaccard similarity
    computed on the shingle sets.

    Returns
    -------
    np.ndarray
        Learned weight vector of length 2*k.
    """
    dim = 2 * k
    weights = np.zeros(dim, dtype=np.float64)

    for epoch in range(epochs):
        random.shuffle(pairs)
        total_err = 0.0
        for a, b in pairs:
            # Feature vector
            x_a = signature_vector(a, k, width)
            x_b = signature_vector(b, k, width)
            x = np.concatenate([x_a, x_b])

            # True Jaccard similarity on shingle sets
            set_a = shingles(a, width)
            set_b = shingles(b, width)
            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            target = inter / union if union else 0.0

            # RLCT factor from the *combined* signature
            combined_sig = signature(shingles(a, width) | shingles(b, width), k)
            rlct = rlct_from_signature(combined_sig)

            weights, err = nlms_update(weights, x, target, mu_base=mu_base, rlct=rlct)
            total_err += err ** 2
        # Simple progress print (can be silenced)
        print(f"Epoch {epoch+1}/{epochs}, MSE={total_err/len(pairs):.6f}")
    return weights


def predict_similarity(
    text_a: str,
    text_b: str,
    weights: np.ndarray,
    k: int = 128,
    width: int = 5,
) -> float:
    """
    Predict Jaccard similarity for a pair of texts using learned NLMS weights.
    """
    x_a = signature_vector(text_a, k, width)
    x_b = signature_vector(text_b, k, width)
    x = np.concatenate([x_a, x_b])
    return nlms_predict(weights, x)


def hybrid_similarity(
    text_a: str,
    text_b: str,
    k: int = 128,
    width: int = 5,
) -> float:
    """
    One‑shot similarity estimate that blends MinHash approximation with an
    RLCT‑scaled NLMS correction.

    The procedure:
    1. Compute raw MinHash similarity s₀.
    2. Build a temporary weight vector w₀ = s₀ * 1 (scalar) and treat it as a
       single‑parameter NLMS model.
    3. Adjust the prediction using the RLCT factor derived from the joint
       signature.

    This demonstrates the hybrid operation without a training phase.
    """
    # Raw MinHash similarity
    sig_a = signature(shingles(text_a, width), k)
    sig_b = signature(shingles(text_b, width), k)
    s0 = similarity(sig_a, sig_b)

    # RLCT factor from the union of shingles
    joint_sig = signature(shingles(text_a, width) | shingles(text_b, width), k)
    rlct = rlct_from_signature(joint_sig)

    # Simple NLMS‑style correction (scalar weight update)
    # Treat prediction as w * x where w = s0 and x = 1
    target = s0  # ideal target is the same as raw estimate for this demo
    mu_base = 0.3
    mu_eff = mu_base * rlct
    error = target - s0
    corrected = s0 + mu_eff * error  # = s0 + μ_eff*(0) = s0, but shows the flow
    # In practice the correction would come from a trained model; we return
    # the RLCT‑scaled raw estimate.
    return corrected * rlct  # blend raw estimate with complexity scaling


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic corpus
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do",
    ]

    # Prepare training pairs (including identical and different documents)
    training_pairs = [(docs[i], docs[j]) for i in range(len(docs)) for j in range(i, len(docs))]

    # Train NLMS on the pairs
    learned_weights = train_nlms_on_pairs(training_pairs, epochs=5, mu_base=0.6)

    # Predict on a few examples
    for a, b in [
        (docs[0], docs[1]),
        (docs[2], docs[3]),
        (docs[0], docs[2]),
    ]:
        pred = predict_similarity(a, b, learned_weights)
        raw = similarity(
            signature(shingles(a), k=128), signature(shingles(b), k=128)
        )
        print("\nPair:")
        print(f"  A: {a}")
        print(f"  B: {b}")
        print(f"  Raw MinHash similarity : {raw:.4f}")
        print(f"  NLMS‑predicted similarity: {pred:.4f}")

    # Demonstrate the one‑shot hybrid similarity
    print("\nOne‑shot hybrid similarity examples:")
    for a, b in [(docs[0], docs[1]), (docs[0], docs[2])]:
        print(f"{a[:30]}...  vs  {b[:30]}...  -> {hybrid_similarity(a, b):.4f}")