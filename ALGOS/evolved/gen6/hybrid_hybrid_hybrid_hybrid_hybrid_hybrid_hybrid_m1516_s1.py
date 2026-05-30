# DARWIN HAMMER — match 1516, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_shap_attribut_m1190_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_korpus_m402_s3.py (gen5)
# born: 2026-05-29T23:36:57Z

"""
Hybrid Stylometric–Hyperdimensional Model with RBF‑Shapley Fusion
================================================================

Parents:
- **Parent A**: stylometric extractor that builds a high‑dimensional frequency
  fingerprint and provides a Shapley attribution framework.
- **Parent B**: hyperdimensional representation based on min‑hash signatures and
  radial‑basis‑function (RBF) similarity kernels.

Mathematical Bridge
-------------------
A stylometric fingerprint is a point **x** ∈ ℝⁿ (n = number of linguistic
function‑categories).  By interpreting **x** as a hyperdimensional vector we can
project it onto a set of RBF centres **C** = {c₁,…,cₖ} using the Gaussian kernel

    φ_i(x) = exp(‑‖x‑c_i‖² / (2σ²))

which yields a new representation φ(x) ∈ ℝᵏ.  The φ‑vector is then compressed
with a min‑hash / dhash pipeline (Parent B) to obtain a compact binary
hypervector **h**.  Finally the Shapley value of each original stylometric
feature is estimated by marginalising the model output
    f(h) = Σ_i w_i φ_i(x)
over all feature permutations, using the same kernel‑weighted coalition values
as in Parent A.

The code below implements this fused pipeline with three core functions:
`stylometric_vector`, `rbf_hypervector`, and `shapley_attribution`.
"""

import re
import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (from Parent B)
# ----------------------------------------------------------------------
Vector = np.ndarray

def gaussian(r: float, sigma: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((r / sigma) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a - b))

def compute_dhash(values: list[float]) -> int:
    """Derivative hash – 1 bit per adjacent comparison."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: list[float]) -> int:
    """Perceptual hash – compare to mean of first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Simple min‑hash signature for a string (Parent B)."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = [sys.maxsize] * k
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h & ((1 << 31) - 1))
    return signature

# ----------------------------------------------------------------------
# Stylometric extractor (core of Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "they", "them", "their", "theirs", "themselves",
        "we", "us", "our", "ours", "ourselves"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at",
        "before", "behind", "below", "between", "by", "during",
        "for", "from", "in", "into", "of", "off", "on", "onto",
        "over", "through", "to", "under", "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could",
        "did", "do", "does", "had", "has", "have", "is",
        "may", "might", "must", "shall", "should", "was",
        "were", "will", "would"
    },
    "conjunction": {
        "and", "but", "or", "nor", "so", "yet", "because",
        "although", "while", "if", "when", "where", "unless",
        "until"
    },
}

def tokenize(text: str) -> list[str]:
    """Very simple whitespace / punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())

def stylometric_vector(text: str) -> Vector:
    """
    Compute a normalized frequency vector for each function category.
    Returns a NumPy array of shape (len(FUNCTION_CATS),).
    """
    tokens = tokenize(text)
    total = len(tokens) or 1
    cat_counts = []
    for cat in FUNCTION_CATS.values():
        count = sum(1 for t in tokens if t in cat)
        cat_counts.append(count / total)
    return np.array(cat_counts, dtype=float)

# ----------------------------------------------------------------------
# Hyperdimensional / RBF fusion (new hybrid core)
# ----------------------------------------------------------------------
def rbf_hypervector(
    vec: Vector,
    centres: np.ndarray,
    sigma: float = 1.0
) -> Vector:
    """
    Project a stylometric vector onto RBF centres.
    - `vec`   : original stylometric fingerprint (ℝⁿ).
    - `centres`: matrix of shape (k, n) containing k centre vectors.
    Returns a new vector φ ∈ ℝᵏ where φ_i = exp(-‖vec‑c_i‖² / (2σ²)).
    """
    diffs = centres - vec  # shape (k, n)
    dists = np.linalg.norm(diffs, axis=1)  # shape (k,)
    rbf_vals = np.vectorize(lambda r: gaussian(r, sigma))(dists)
    return rbf_vals

def compress_to_binary(rbf_vec: Vector) -> int:
    """
    Convert the continuous RBF vector into a binary hypervector using a
    perceptual hash (phash) on the raw float values.
    """
    # Normalise to [0,1] for stability
    min_v, max_v = rbf_vec.min(), rbf_vec.max()
    if max_v - min_v > 0:
        norm = (rbf_vec - min_v) / (max_v - min_v)
    else:
        norm = np.zeros_like(rbf_vec)
    return compute_phash(norm.tolist())

def shapley_attribution(
    vec: Vector,
    centres: np.ndarray,
    sigma: float = 1.0,
    n_perm: int = 120
) -> dict[int, float]:
    """
    Approximate Shapley values for each stylometric feature.
    The model output is defined as f(x) = Σ_i w_i * φ_i(x) where φ_i are the
    RBF‑encoded components.  We use a simple linear weighting w_i = 1 for
    demonstration.  The contribution of each original dimension is estimated
    by averaging marginal gains over random permutations (Monte‑Carlo Shapley).
    """
    n_features = vec.shape[0]
    shapley_vals = np.zeros(n_features)

    # Pre‑compute RBF of the full vector (used as baseline)
    full_phi = rbf_hypervector(vec, centres, sigma)
    full_output = full_phi.sum()  # linear weight = 1

    for _ in range(n_perm):
        perm = np.random.permutation(n_features)
        marginal = np.zeros(n_features)
        current = np.zeros_like(vec)

        prev_output = 0.0
        for idx in perm:
            current[idx] = vec[idx]
            phi = rbf_hypervector(current, centres, sigma)
            cur_output = phi.sum()
            marginal[idx] = cur_output - prev_output
            prev_output = cur_output

        shapley_vals += marginal

    shapley_vals /= n_perm
    return {i: float(v) for i, v in enumerate(shapley_vals)}

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_pipeline(
    text: str,
    n_centres: int = 16,
    sigma: float = 1.0,
    seed: int = 42
) -> dict[int, float]:
    """
    Complete workflow:
    1. Extract stylometric fingerprint.
    2. Randomly generate RBF centres (fixed by seed for reproducibility).
    3. Encode with RBF, compress to binary hypervector, and compute Shapley.
    Returns a dictionary mapping feature index → Shapley value.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Stylometry
    sv = stylometric_vector(text)                # shape (n,)

    # 2. RBF centres (sampled from a unit simplex)
    n_features = sv.shape[0]
    centres = np.random.rand(n_centres, n_features)

    # 3. RBF projection (continuous)
    rbf_vec = rbf_hypervector(sv, centres, sigma)

    # 4. Binary hypervector (optional – shown for completeness)
    binary_hv = compress_to_binary(rbf_vec)
    _ = binary_hv  # placeholder to illustrate usage

    # 5. Shapley attribution on original features
    shap_vals = shapley_attribution(sv, centres, sigma)

    return shap_vals

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I think that we should consider the consequences of our actions. "
        "Although it may be difficult, we must act with purpose and "
        "responsibility. The data shows that you and I are aligned."
    )
    result = hybrid_pipeline(sample_text)
    print("Shapley attribution per stylometric feature index:")
    for idx, val in sorted(result.items()):
        print(f"  Feature {idx} ({list(FUNCTION_CATS.keys())[idx]}): {val:.6f}")
    # Verify that binary hypervector generation does not raise
    sv = stylometric_vector(sample_text)
    centres = np.random.rand(16, sv.shape[0])
    rbf_vec = rbf_hypervector(sv, centres)
    bhv = compress_to_binary(rbf_vec)
    print(f"Binary hypervector (phash): {bhv:#018x}")
    # Simple distance sanity check
    dist = euclidean(sv, sv)
    print(f"Self‑distance (should be 0.0): {dist:.6f}")