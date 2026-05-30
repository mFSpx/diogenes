# DARWIN HAMMER — match 3270, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m1272_s0.py (gen6)
# parent_b: hybrid_hoeffding_tree_tropical_maxplus_m18_s1.py (gen1)
# born: 2026-05-29T23:48:50Z

"""Hybrid Minimum-Cost Hoeffding–Tropical Algorithm
===================================================

This module fuses two DARWIN HAMMER ancestors:

* **Parent A** – *hybrid_hybrid_minimum_cost* – provides a
  distance‑modulated confidence term, MinHash signatures and a
  pheromone‑entropy component.  Its core equation is  

  ``S_i = σ(d_i) · (1 + sim(sig_i, sig_ref)) · (1 + β·conf_i) · Σ(p·log p)``

* **Parent B** – *hybrid_hoeffding_tree_tropical_maxplus* – supplies
  Hoeffding‑bound driven split decisions together with tropical
  (max‑plus) algebra used to evaluate ReLU‑style decision boundaries.

The **mathematical bridge** is the observation that the hybrid score
``S_i`` is a non‑negative scalar that can be interpreted as a
*tropical coefficient*.  By feeding ``S_i`` as the leading coefficient
of a tropical polynomial we obtain a *tropical gain* that can be
compared with the gain of alternative splits.  The Hoeffding bound then
governs whether the observed tropical gain gap justifies a split.

The three public functions illustrate this integration:

1. ``compute_hybrid_score`` – implements the Parent A formula.
2. ``tropical_gain_from_score`` – builds a tropical polynomial from
   the score and evaluates it at a feature vector.
3. ``hybrid_split_decision`` – uses the tropical gains of the best and
   second‑best candidate splits together with the Hoeffding bound from
   Parent B to produce a ``SplitDecision``.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (originating from Parent A)
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i: i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    hashes: List[int] = []
    for seed in range(k):
        min_hash = INT16_MAX
        for token in token_set:
            h = _hash_token(seed, token)
            if h < min_hash:
                min_hash = h
        hashes.append(min_hash)
    return hashes


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def jaccard_like_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Similarity = proportion of equal entries in two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have the same length")
    equal = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return equal / len(sig1)


def pheromone_entropy(probs: Iterable[float]) -> float:
    """Σ p·log(p) (negative entropy, matches Parent A)."""
    total = 0.0
    for p in probs:
        if p > 0:
            total += p * math.log(p)
    return total


# ----------------------------------------------------------------------
# Tropical algebra utilities (originating from Parent B)
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition = max."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication = +."""
    return np.add(x, y)


def t_matmul(A, B):
    """Tropical matrix multiplication."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial at point(s) x.

    coeffs: list/array of coefficients c_0 … c_{n-1}
    The tropical monomial for exponent e is c_e + e·x.
    The polynomial value is max_e (c_e + e·x).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)


def relu_layer_to_tropical(W, b):
    """Identity conversion – kept for API compatibility."""
    return np.asarray(W, dtype=float), np.asarray(b, dtype=float)


def tropical_network_eval(x, layers):
    """Forward pass through a tropical network (max‑plus analogue of ReLU)."""
    h = np.asarray(x, dtype=float).ravel()
    for W, b in layers:
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        # Tropical affine transform: y = max_j (W_ij + h_j) + b_i
        h = np.max(W + h, axis=1) + b
    return h


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_hybrid_score(
    point: Tuple[float, float],
    ref_point: Tuple[float, float],
    tokens: Iterable[str],
    ref_signature: List[int],
    pheromone_probs: Iterable[float],
    beta: float = 0.5,
) -> float:
    """
    Implements Parent A's hybrid score.

    * ``point`` and ``ref_point`` – 2‑D coordinates used to compute Euclidean distance.
    * ``tokens`` – iterable of tokens (e.g., shingles) from which a MinHash signature is built.
    * ``ref_signature`` – MinHash signature of a reference document.
    * ``pheromone_probs`` – probability distribution over pheromone trails.
    * ``beta`` – tunable weight for the confidence term.
    """
    # Euclidean distance
    dx = point[0] - ref_point[0]
    dy = point[1] - ref_point[1]
    d = math.hypot(dx, dy)

    # Confidence term (inverse distance, bounded)
    conf = 1.0 / (1.0 + d)

    # MinHash similarity
    sig = minhash_signature(tokens)
    sim = jaccard_like_similarity(sig, ref_signature)

    # Pheromone entropy term
    entropy = pheromone_entropy(pheromone_probs)

    # Hybrid score
    score = sigmoid(d) * (1.0 + sim) * (1.0 + beta * conf) * entropy
    return score


def tropical_gain_from_score(score: float, feature_vec: np.ndarray, degree: int = 3) -> float:
    """
    Build a tropical polynomial whose leading coefficient is the hybrid score
    and evaluate it at ``feature_vec``.

    The coefficient vector is ``[score, 0, 0, …]`` (length = degree).
    ``feature_vec`` can be a scalar or a vector; the polynomial is evaluated
    element‑wise and the maximum value is returned as the tropical gain.
    """
    coeffs = np.zeros(degree, dtype=float)
    coeffs[0] = score  # c_0 = score, higher coefficients stay zero
    # Evaluate tropical polynomial (max over c_e + e·x)
    gain = t_polyval(coeffs, feature_vec)
    # Reduce to a single scalar – we take the maximum gain over the vector
    return float(np.max(gain))


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Re‑implemented from Parent B (kept for clarity)."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a split‑evaluation step."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def hybrid_split_decision(
    best_gain: float,
    second_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Combines the Hoeffding bound (Parent B) with tropical gains derived from
    hybrid scores (Parent A).

    Returns a ``SplitDecision`` indicating whether to split a node in a
    streaming decision tree.
    """
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_gain
    split = gap > eps or eps < tie_threshold
    if gap > eps:
        reason = "gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "tie_threshold"
    else:
        reason = "wait"
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate synthetic data
    random.seed(42)
    point = (random.uniform(0, 10), random.uniform(0, 10))
    ref_point = (5.0, 5.0)

    # Textual component – simple shingle extraction
    sample_text = "The quick brown fox jumps over the lazy dog."
    tokens = _shingles(sample_text, width=5)

    # Reference signature (pretend we have a corpus)
    ref_tokens = _shingles("A different reference document for similarity.", width=5)
    ref_sig = minhash_signature(ref_tokens)

    # Pheromone probabilities (must sum to 1)
    pheromone_probs = np.random.dirichlet(np.ones(5), size=1).flatten()

    # 2. Compute hybrid score
    score = compute_hybrid_score(
        point=point,
        ref_point=ref_point,
        tokens=tokens,
        ref_signature=ref_sig,
        pheromone_probs=pheromone_probs,
        beta=0.7,
    )
    print(f"Hybrid score S_i = {score:.6f}")

    # 3. Convert score into a tropical gain
    feature_vector = np.array([random.random() for _ in range(4)])
    best_gain = tropical_gain_from_score(score, feature_vector, degree=4)

    # Simulate a second candidate (slightly worse)
    second_score = score * 0.9
    second_gain = tropical_gain_from_score(second_score, feature_vector, degree=4)

    print(f"Tropical gain (best)   = {best_gain:.6f}")
    print(f"Tropical gain (second) = {second_gain:.6f}")

    # 4. Apply Hoeffding‑bound split decision
    decision = hybrid_split_decision(
        best_gain=best_gain,
        second_gain=second_gain,
        r=1.0,          # range of gain (assumed)
        delta=0.05,
        n=500,          # number of observed examples
    )
    print(f"Split decision: {decision.should_split} (reason: {decision.reason})")
    print(f"Epsilon = {decision.epsilon:.6f}, Gain gap = {decision.gain_gap:.6f}")