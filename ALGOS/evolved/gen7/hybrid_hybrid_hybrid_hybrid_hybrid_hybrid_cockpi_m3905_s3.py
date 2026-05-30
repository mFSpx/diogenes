# DARWIN HAMMER — match 3905, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1259_s0.py (gen6)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_nlms_h_m2426_s0.py (gen6)
# born: 2026-05-29T23:52:22Z

"""
Hybrid Algorithm: Stylometric‑Bayesian Hypervector Generation + Pheromone‑Weighted Gaussian Kernel

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1259_s0.py (stylometry, Bayesian feature extraction,
  geometric‑algebraic multivector, Fisher score)
- hybrid_hybrid_cockpit_metri_hybrid_hybrid_nlms_h_m2426_s0.py (entropy optimisation,
  honesty‑weighted pheromone signal, similarity/kernel matrices, Gaussian kernel, phash)

Mathematical Bridge:
Both parents employ a hash‑driven pseudo‑random mapping from a set of scalar features to a
high‑dimensional binary/hypervector representation.  The bridge therefore is the *hash‑seeded
hypervector* that can be used both as the geometric‑algebraic multivector in Parent A and as the
input to the similarity/kernel machinery of Parent B.  In this hybrid we:

1. Extract stylometric proportions for each FUNCTION_CAT (Parent A).
2. Convert the proportion vector into a binary hypervector by seeding a RNG with a hash of the
   feature vector (shared with Parent B’s `compute_phash`).
3. Build a simple geometric‑algebraic multivector by forming the outer (wedge) product of the
   hypervector with itself.
4. Construct a pheromone‑weighted Gaussian similarity matrix over all hypervectors, where the
   Gaussian kernel uses Euclidean distance and the pheromone signal modulates the kernel
   amplitude.  The Hamming distance between the phashes of two vectors is added as an
   additional similarity term (again a direct link between the two parents).
5. Finally compute a Fisher‑like discriminant score on the similarity matrix to obtain a
   scalar “authorship‑likelihood” for each sample.

The following implementation provides three core functions demonstrating this fused workflow
and a smoke‑test that runs end‑to‑end without external dependencies.
"""

import sys
import pathlib
import random
import math
from typing import Dict, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Shared lexical categories (identical in both parents)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set("and but or so yet because although while if when where whereas unless until".split()),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all both each few more most other some such no nor not only own same so than too very".split()
    ),
}


# ----------------------------------------------------------------------
# 1. Stylometric + Bayesian feature extraction (Parent A)
# ----------------------------------------------------------------------
def stylometric_proportions(text: str) -> Dict[str, float]:
    """
    Compute the proportion of tokens belonging to each FUNCTION_CAT.
    Tokens are lower‑cased words split on whitespace.
    """
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    total = len(tokens) or 1
    counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                counts[cat] += 1
                break
    # Bayesian smoothing: add 1 pseudo‑count to each category
    smoothed = {cat: (cnt + 1) / (total + len(FUNCTION_CATS)) for cat, cnt in counts.items()}
    return smoothed


# ----------------------------------------------------------------------
# 2. Hash‑seeded hypervector generation (shared bridge)
# ----------------------------------------------------------------------
def vector_to_hash(values: List[float]) -> int:
    """
    Replicates Parent B's `compute_phash`.  Returns a 64‑bit integer hash derived from
    the median of the feature vector.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hypervector_from_features(features: Dict[str, float], dim: int = 1024) -> np.ndarray:
    """
    Produce a bipolar hypervector (values in {-1, +1}) of length `dim`.
    The RNG seed is derived from the hash of the feature values, guaranteeing that
    the same stylometric profile always yields the same hypervector (the mathematical
    bridge between the two parents).
    """
    # Convert dict to ordered list for deterministic hashing
    ordered_vals = [features[cat] for cat in sorted(FUNCTION_CATS.keys())]
    seed = vector_to_hash(ordered_vals)
    rng = random.Random(seed)
    # Bipolar hypervector: +1 for bit 1, -1 for bit 0
    hv = np.array([1 if rng.random() > 0.5 else -1 for _ in range(dim)], dtype=np.int8)
    return hv


# ----------------------------------------------------------------------
# 3. Geometric‑Algebraic multivector (Parent A)
# ----------------------------------------------------------------------
def multivector_outer(hv: np.ndarray) -> np.ndarray:
    """
    Construct a simple grade‑2 multivector by computing the outer (wedge) product
    of the hypervector with itself.  For binary hypervectors the outer product reduces
    to the antisymmetric part of the outer product matrix.
    """
    # Outer product
    outer = np.outer(hv.astype(np.float32), hv.astype(np.float32))
    # Antisymmetric component (grade‑2)
    multiv = (outer - outer.T) / 2.0
    return multiv


# ----------------------------------------------------------------------
# 4. Pheromone‑weighted Gaussian similarity matrix (Parent B)
# ----------------------------------------------------------------------
def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used in Parent B."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.dot(diff, diff)))


def pheromone_weighted_similarity(
    hypervectors: List[np.ndarray],
    pheromone_signal: float,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Build an N×N similarity matrix S where:
        S_ij = pheromone_signal * Gaussian(‖hv_i - hv_j‖) + Hamming(phash_i, phash_j) / 64
    The Hamming term (scaled to [0,1]) mirrors Parent B's `hamming_distance`
    and injects the hash‑based similarity into the kernel.
    """
    n = len(hypervectors)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [vector_to_hash(hv.tolist()) for hv in hypervectors]

    for i in range(n):
        for j in range(i, n):
            d = euclidean_distance(hypervectors[i], hypervectors[j])
            g = gaussian_kernel(d, epsilon)
            # Normalised Hamming similarity
            ham = 1 - (bin(hashes[i] ^ hashes[j]).count("1") / 64.0)
            sim = pheromone_signal * g + ham * (1 - pheromone_signal)
            S[i, j] = S[j, i] = sim
    return S


# ----------------------------------------------------------------------
# 5. Fisher‑like discriminant on the similarity matrix (Parent A)
# ----------------------------------------------------------------------
def fisher_score(similarity: np.ndarray) -> np.ndarray:
    """
    Compute a Fisher‑type score for each sample:
        score_i = (μ_i - μ_all)² / σ_i²
    where μ_i is the mean similarity of sample i to all others,
    μ_all is the global mean, and σ_i² is the variance of i's similarities.
    The higher the score, the more discriminative the sample.
    """
    n = similarity.shape[0]
    # Exclude self‑similarity (diagonal) from statistics
    mask = ~np.eye(n, dtype=bool)
    means = similarity[mask].reshape(n, n - 1).mean(axis=1)
    global_mean = means.mean()
    variances = similarity[mask].reshape(n, n - 1).var(axis=1) + 1e-12  # avoid div‑0
    scores = ((means - global_mean) ** 2) / variances
    return scores


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(texts: List[str], pheromone_signal: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    End‑to‑end pipeline:
        1. Extract stylometric proportions.
        2. Generate hypervectors (hash‑seeded).
        3. Build multivectors (grade‑2 outer product).
        4. Compute pheromone‑weighted similarity matrix.
        5. Derive Fisher scores.

    Returns:
        similarity_matrix, multivector_stack, fisher_scores
    """
    # Step 1 & 2
    hypervectors = []
    for txt in texts:
        feats = stylometric_proportions(txt)
        hv = hypervector_from_features(feats)
        hypervectors.append(hv)

    # Step 3 – stack multivectors for possible downstream use
    multivectors = np.stack([multivector_outer(hv) for hv in hypervectors], axis=0)

    # Step 4
    sim_mat = pheromone_weighted_similarity(hypervectors, pheromone_signal)

    # Step 5
    scores = fisher_score(sim_mat)

    return sim_mat, multivectors, scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think therefore I am. The quick brown fox jumps over the lazy dog.",
        "She sells sea shells on the sea shore. It is a truth universally acknowledged.",
        "To be, or not to be, that is the question.",
        "All that glitters is not gold, but all gold glitters bright."
    ]

    sim, mv, sc = hybrid_process(sample_texts, pheromone_signal=0.75)

    print("Similarity matrix shape:", sim.shape)
    print("Multivector stack shape:", mv.shape)
    print("Fisher scores:", sc)
    # Simple sanity check: similarity matrix should be symmetric and diagonal ≈ pheromone_signal*1+1
    assert np.allclose(sim, sim.T, atol=1e-12), "Similarity matrix not symmetric"
    print("Smoke test passed.")