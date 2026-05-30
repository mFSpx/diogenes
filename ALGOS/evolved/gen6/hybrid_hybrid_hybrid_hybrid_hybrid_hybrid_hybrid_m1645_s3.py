# DARWIN HAMMER — match 1645, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m807_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1126_s0.py (gen5)
# born: 2026-05-29T23:38:00Z

"""Hybrid Algorithm: Weekday-Weighted Curvature Krampus Projection (WWCKP)

Parents:
- **Parent A**: weekday_weight_vector from a work‑share calendar allocator, used as
  restriction maps in a sheaf‑cohomology‑style linear transformation.
- **Parent B**: curvature computation on a similarity graph, Krampus linear projection,
  MinHash signatures, regret‑weighted probability, sign quantisation and entropy.

Mathematical Bridge:
The weekday weight vector **w**∈ℝⁿ (n = number of groups) is interpreted as a
restriction‑map matrix **R = diag(w)** that scales node‑feature vectors before they
enter the Krampus projection.  Curvature κᵢ computed for each document node is
treated as an additional feature and concatenated with the MinHash signature.
The combined feature vector **fᵢ = [σᵢ ; κᵢ]** (σᵢ – MinHash signature) is first
transformed by **R**, then linearly projected by a Krampus matrix **K** ∈ℝ^{3×m}
(m = len(fᵢ)).  The projected 3‑D coordinates are quantised to a ternary alphabet,
weighted by a regret‑derived probability distribution, and finally evaluated with
Shannon entropy.  This unified pipeline respects both the weekday‑dependent
restriction maps and the curvature‑aware Krampus projection.

The implementation below provides three core functions that realise this hybrid
operation and a smoke‑test under ``if __name__ == "__main__"``.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from collections import defaultdict
import hashlib

import numpy as np

# ----------------------------------------------------------------------
# Constants & Helper Types
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
NUM_MINHASH = 64          # number of hash functions for MinHash
KRAMPUS_DIM = 3           # target dimension of Krampus projection
SEED = 42                 # reproducibility

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------
# Parent A – Calendar / Weight Vector
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: tuple, dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Parent B – MinHash, Curvature, Krampus Projection
# ----------------------------------------------------------------------
def _hash_fn(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash mixing a seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(4, "little"))
    h.update(token.encode("utf-8"))
    return int.from_bytes(h.digest(), "little") & MAX64


def minhash_signature(tokens: set, num_hashes: int = NUM_MINHASH) -> np.ndarray:
    """
    Compute a MinHash signature for a token set.
    Returns an array of length ``num_hashes`` with unsigned 64‑bit integers.
    """
    if not tokens:
        return np.full(num_hashes, MAX64, dtype=np.uint64)

    sig = np.full(num_hashes, MAX64, dtype=np.uint64)
    for i in range(num_hashes):
        # each i defines a distinct hash function via a different seed
        seed = i + 1
        mins = min(_hash_fn(seed, t) for t in tokens)
        sig[i] = mins
    return sig


def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return np.mean(sig1 == sig2)


def build_similarity_graph(docs: list[set], threshold: float = 0.3) -> dict[int, set[int]]:
    """
    Build an undirected graph where an edge (i, j) exists iff the estimated Jaccard
    similarity between document i and j exceeds ``threshold``.
    Returns an adjacency dict mapping node index to a set of neighbour indices.
    """
    n = len(docs)
    sigs = [minhash_signature(d) for d in docs]
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_estimate(sigs[i], sigs[j])
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def approximate_curvature(adj: dict[int, set[int]]) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature κᵢ for each node i.
    A simple proxy: κᵢ = 1 - (average neighbour degree / max degree).
    Returns an array κ of shape (n,).
    """
    n = max(adj.keys()) + 1 if adj else 0
    degrees = np.zeros(n, dtype=np.float64)
    for i, neigh in adj.items():
        degrees[i] = len(neigh)

    max_deg = degrees.max() if degrees.size else 1.0
    κ = np.zeros(n, dtype=np.float64)
    for i, neigh in adj.items():
        if not neigh:
            κ[i] = 0.0
            continue
        avg_neigh_deg = np.mean([degrees[j] for j in neigh])
        κ[i] = 1.0 - (avg_neigh_deg / max_deg)
    return κ


def krampus_projection(features: np.ndarray, weight_vec: np.ndarray) -> np.ndarray:
    """
    Apply the weekday‑dependent restriction map (diagonal scaling by ``weight_vec``)
    followed by a random linear projection to ℝ³ (the Krampus matrix).
    *features* shape: (m,) where m = len(weight_vec) + 1 (curvature appended).
    Returns a 3‑D coordinate vector.
    """
    # Restriction map R = diag(weight_vec) applied to the first |weight_vec| entries
    m = features.shape[0]
    w_len = weight_vec.shape[0]
    assert m == w_len + 1, "Feature length must be weight vector length plus curvature"

    restricted = np.empty_like(features)
    restricted[:w_len] = features[:w_len] * weight_vec
    restricted[-1] = features[-1]  # curvature passes unchanged

    # Krampus projection matrix K ∈ ℝ^{3×m}
    K = np.random.randn(KRAMPUS_DIM, m)
    projected = K @ restricted
    return projected


def sign_quantisation(vec: np.ndarray) -> np.ndarray:
    """Quantise a real vector to ternary values {-1, 0, 1}."""
    return np.where(vec > 0.5, 1, np.where(vec < -0.5, -1, 0))


def regret_weights(actions: list[tuple[str, float]]) -> np.ndarray:
    """
    Given a list of (action_id, expected_value) pairs, compute a regret‑based
    probability distribution p where p_i ∝ max(E) - E_i.
    Returns a normalized probability vector.
    """
    values = np.array([ev for _, ev in actions], dtype=np.float64)
    max_ev = values.max()
    regrets = max_ev - values
    # Avoid division by zero when all regrets are zero
    if regrets.sum() == 0:
        probs = np.full_like(regrets, 1.0 / len(regrets))
    else:
        probs = regrets / regrets.sum()
    return probs


def shannon_entropy(p: np.ndarray) -> float:
    """Standard Shannon entropy (base 2)."""
    p = p[p > 0]  # filter zero probabilities to avoid log(0)
    return -float(np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def compute_hybrid_features(docs: list[set]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute for each document:
    1. MinHash signature (flattened to float64).
    2. Approximate curvature κᵢ from the similarity graph.
    Returns a tuple (signatures, curvature_array) where:
      - signatures.shape = (n_docs, NUM_MINHASH)
      - curvature_array.shape = (n_docs,)
    """
    # 1. Signatures
    signatures = np.vstack([minhash_signature(d).astype(np.float64) for d in docs])

    # 2. Curvature via graph
    adj = build_similarity_graph(docs)
    curvature = approximate_curvature(adj)

    return signatures, curvature


def hybrid_projection(docs: list[set], date_today: date = None) -> np.ndarray:
    """
    End‑to‑end hybrid projection:
    * Compute MinHash signatures and curvature.
    * Build a weekday weight vector from *date_today* (defaults to today).
    * For each document, concatenate its signature (mean‑scaled) with curvature,
      apply the restriction map and Krampus projection, then sign‑quantise.
    Returns an array of ternary vectors of shape (n_docs, 3).
    """
    if date_today is None:
        date_today = date.today()
    dow = doomsday(date_today.year, date_today.month, date_today.day)
    w_vec = weekday_weight_vector(GROUPS, dow)          # shape (|GROUPS|,)

    sigs, curv = compute_hybrid_features(docs)         # sigs: (n, NUM_MINHASH)

    # Reduce the high‑dimensional signature to the same length as the weight vector
    # via a simple linear projection (mean over blocks).
    block = NUM_MINHASH // len(GROUPS)
    reduced = np.mean(sigs.reshape(sigs.shape[0], len(GROUPS), block), axis=2)

    # Normalise reduced signatures to unit sum (probability‑like)
    reduced = reduced / (reduced.sum(axis=1, keepdims=True) + 1e-12)

    # Assemble feature vectors f_i = [reduced_i ; κ_i]
    features = np.hstack([reduced, curv[:, None]])      # shape (n, |GROUPS|+1)

    # Apply Krampus projection per document
    projected = np.vstack([krampus_projection(f, w_vec) for f in features])

    # Sign quantisation to ternary alphabet
    ternary = sign_quantisation(projected)
    return ternary.astype(np.int8)


def hybrid_entropy_score(ternary_vectors: np.ndarray) -> float:
    """
    Compute a regret‑weighted Shannon entropy over the distribution of ternary vectors.
    The probability of each distinct ternary pattern is weighted by a regret‑derived
    distribution over synthetic actions (one action per pattern).
    """
    # Count occurrences of each unique pattern
    patterns, counts = np.unique(ternary_vectors, axis=0, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()

    # Create synthetic actions with expected values inversely proportional to pattern frequency
    expected_vals = 1.0 / (probs + 1e-12)                # rarer patterns have higher value
    actions = [(f"pat_{i}", ev) for i, ev in enumerate(expected_vals)]

    # Regret‑based weighting
    regret_p = regret_weights(actions)

    # Combine regret weights with pattern probabilities
    weighted_p = probs * regret_p
    weighted_p /= weighted_p.sum()                      # renormalise

    return shannon_entropy(weighted_p)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example corpus: three tiny documents
    docs = [
        {"the", "quick", "brown", "fox"},
        {"jumps", "over", "the", "lazy", "dog"},
        {"the", "quick", "blue", "hare"},
    ]

    ternary = hybrid_projection(docs)
    print("Ternary projection (shape {}):".format(ternary.shape))
    print(ternary)

    entropy = hybrid_entropy_score(ternary)
    print("\nHybrid entropy score:", _pct(entropy))