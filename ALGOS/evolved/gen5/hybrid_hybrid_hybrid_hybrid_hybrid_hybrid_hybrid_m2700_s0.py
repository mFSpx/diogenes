# DARWIN HAMMER — match 2700, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hard_t_m625_s0.py (gen4)
# born: 2026-05-29T23:43:36Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s3.py
- Parent B: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hard_t_m625_s0.py

Mathematical Bridge
-------------------
Parent A provides a MinHash signature σ(toks) for a token set and a Jaccard‑like
similarity 𝑠∈[0,1].  Parent B creates a discrete audit classification vector
c∈{0,1}^K, extracts a lead‑lag transformed path 𝓁(p)∈ℝ^M and mixes a spline
basis B∈ℝ^{M×M} with learned weights w∈ℝ^M (the KAN step).

The fusion treats the MinHash similarity 𝑠 as a scalar weight that modulates
the KAN mixing.  Concretely, the audit one‑hot vector c is first combined with
the lead‑lag features 𝓁(p) by element‑wise multiplication, producing a weighted
feature vector f = s·(c⊙𝓁(p)).  The spline basis B mixes f to a final schedule
y = B · f, which is finally projected onto a morphology‑derived hyper‑dimensional
vector v (generated from the Morphology object).  The dot product ⟨y, v⟩ yields
the unified hybrid score.

Thus the core topologies of both parents are mathematically fused through:
    σ  →  s                (MinHash similarity)
    c, p → 𝓁(p)           (audit one‑hot & lead‑lag)
    s·(c⊙𝓁(p)) → f        (similarity‑scaled weighted features)
    B·f → y               (KAN spline mixing)
    ⟨y, v⟩ → score       (projection onto morphology space)

The implementation below follows this pipeline and provides three public
functions that demonstrate the hybrid operation.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


class Morphology:
    """Simple geometric description used to seed a hyper‑dimensional vector."""

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def _morphology_seed(m: Morphology) -> int:
    """Create a reproducible 64‑bit seed from morphology attributes."""
    raw = f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def morphology_vector(m: Morphology, dim: int = 128, seed: Any = None) -> np.ndarray:
    """
    Deterministic pseudo‑random vector representing the morphology.
    The vector is L2‑normalized to lie on the unit hypersphere.
    """
    rng_seed = seed if seed is not None else _morphology_seed(m)
    rng = random.Random(rng_seed)
    vec = np.array([rng.random() for _ in range(dim)], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------


CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "deprecated",
    "experimental",
}


def classification_one_hot(label: str) -> np.ndarray:
    """One‑hot encoding of an audit classification."""
    if label not in CLASSIFICATIONS:
        raise ValueError(f"Unknown classification: {label}")
    sorted_labels = sorted(CLASSIFICATIONS)
    vec = np.zeros(len(sorted_labels), dtype=np.float64)
    vec[sorted_labels.index(label)] = 1.0
    return vec


def lead_lag_transform(path: List[float]) -> np.ndarray:
    """
    Lead‑lag transform extracts linear (original), lagged (shift‑right by one)
    and quadratic (squared) components and concatenates them.
    """
    if not path:
        raise ValueError("path must contain at least one element")
    orig = np.asarray(path, dtype=np.float64)
    lag = np.concatenate(([orig[0]], orig[:-1]))
    quad = orig ** 2
    return np.concatenate([orig, lag, quad])


def spline_basis(num_features: int, scale: float = 0.5) -> np.ndarray:
    """
    Simple piecewise‑linear spline basis matrix B∈ℝ^{N×N}.
    B_{ij} = max(0, 1 - |i‑j|/ (scale·N) )
    """
    N = num_features
    B = np.zeros((N, N), dtype=np.float64)
    denom = scale * N if scale * N != 0 else 1.0
    for i in range(N):
        for j in range(N):
            B[i, j] = max(0.0, 1.0 - abs(i - j) / denom)
    return B


def kan_mix(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    KAN mixing step: y = B·w.
    """
    if basis.shape[1] != weights.shape[0]:
        raise ValueError("basis column size must match weight vector length")
    return basis @ weights


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------


def compute_hybrid_signature(
    tokens: List[str],
    morphology: Morphology,
    sig_dim: int = 128,
) -> Tuple[List[int], List[int]]:
    """
    Produce two MinHash signatures:
    - sig_tokens : from the raw token list.
    - sig_morph  : from a tokenisation of the morphology vector (thresholded).
    """
    sig_tokens = minhash_signature(tokens, k=sig_dim)

    # Tokenise morphology vector by thresholding at the median value.
    vec = morphology_vector(morphology, dim=sig_dim)
    median = np.median(vec)
    morph_tokens = [f"dim_{i}" for i, v in enumerate(vec) if v > median]
    sig_morph = minhash_signature(morph_tokens, k=sig_dim)

    return sig_tokens, sig_morph


def compute_weighted_features(
    classification: str,
    path: List[float],
    sim: float,
) -> np.ndarray:
    """
    Combine audit one‑hot, lead‑lag transformed path and MinHash similarity.
    Returns the weighted feature vector f = sim·(c ⊙ ℓ(p)).
    """
    c_vec = classification_one_hot(classification)
    l_vec = lead_lag_transform(path)

    # Align dimensions: repeat the one‑hot vector to match l_vec length.
    repeat = math.ceil(l_vec.size / c_vec.size)
    c_expanded = np.tile(c_vec, repeat)[: l_vec.size]

    weighted = sim * (c_expanded * l_vec)
    return weighted


def hybrid_score(
    morphology: Morphology,
    tokens: List[str],
    classification: str,
    path: List[float],
    sig_dim: int = 128,
) -> float:
    """
    Full hybrid scoring pipeline.
    1. Compute MinHash similarity between token signature and morphology‑derived signature.
    2. Build weighted feature vector from audit classification, path and similarity.
    3. Mix the features with a spline basis (KAN step).
    4. Project the mixed vector onto the morphology hyper‑dimensional vector.
    """
    # Step 1 – similarity
    sig_tokens, sig_morph = compute_hybrid_signature(tokens, morphology, sig_dim=sig_dim)
    sim = similarity(sig_tokens, sig_morph)

    # Step 2 – weighted features
    f = compute_weighted_features(classification, path, sim)

    # Step 3 – KAN mixing
    B = spline_basis(f.size, scale=0.5)
    y = kan_mix(B, f)

    # Step 4 – projection onto morphology space
    v = morphology_vector(morphology, dim=y.size)
    score = float(np.dot(y, v))

    return score


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo() -> None:
    # Sample morphology
    morph = Morphology(length=2.5, width=1.2, height=0.8, mass=3.4)

    # Sample token set (could be extracted from a document)
    tokens = [
        "quantum",
        "entanglement",
        "hyperdimensional",
        "signature",
        "minhash",
        "vector",
        "morphology",
    ]

    # Audit classification
    classification = "research_only"

    # Example path (e.g., time‑series of a sensor)
    path = [random.random() for _ in range(10)]

    # Compute hybrid score
    sc = hybrid_score(morph, tokens, classification, path)
    print(f"Hybrid score: {sc:.6f}")


if __name__ == "__main__":
    _demo()