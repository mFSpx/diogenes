# DARWIN HAMMER — match 93, survivor 4
# gen: 3
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s1.py (gen2)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# born: 2026-05-29T23:26:46Z

"""Hybrid RBF‑Stylometry Surrogate
================================

This module fuses the two parent algorithms:

* **Parent A** – ``rbf_surrogate.py`` – provides a radial‑basis‑function (RBF) surrogate
  that maps a feature vector *x* to a scalar prediction via
  ``prediction = Σ w_k·φ(||x‑c_k||)`` with Gaussian kernel ``φ(r)=exp(-(ε·r)²)``.

* **Parent B** – ``hard_truth_math_model_pool`` – extracts a 96‑dimensional stylometric
  fingerprint from a piece of text (function‑category frequencies) and builds a
  perceptual hash from that fingerprint.  Pairwise similarity is measured by the
  normalized Hamming distance of the hashes.

The **mathematical bridge** is the use of the stylometric fingerprint as the
feature vector for the RBF surrogate.  The surrogate predicts an “authenticity”
score for each document, while the hash similarity captures perceptual
information that is not linearly related to the fingerprint.  The hybrid
similarity between two documents *i* and *j* is defined as a convex combination


S_ij = α·S_hash(i,j) + (1‑α)·S_rbf(i,j)


where ``S_hash`` is the normalized hash similarity and ``S_rbf`` is a Gaussian
kernel applied to the difference of the surrogate predictions.  This yields a
single unified similarity matrix that leverages both the continuous RBF model
and the discrete perceptual hash.

The implementation below provides the core operations, a simple RBF fitting
routine (ridge‑regularised linear solve), and a smoke‑test that runs end‑to‑end.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, Hashable, Set

import numpy as np
import hashlib
import re
from collections import Counter

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 64‑bit based on mean threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()

# ----------------------------------------------------------------------
# Utilities from Parent B (stylometry)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    """Extract lowercase ASCII words (allowing internal apostrophes)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    """Normalized frequency of each FUNCTION_CAT in *text*."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a fixed‑size (dim) stylometric fingerprint.
    The first |FUNCTION_CATS| entries are the category frequencies;
    the remaining entries are filled with a deterministic hash‑derived pattern.
    """
    cat_vec = np.array(list(lsm_vector(text).values()), dtype=np.float64)
    # Pad / truncate to dim
    if dim < len(cat_vec):
        cat_vec = cat_vec[:dim]
    else:
        pad = np.zeros(dim - len(cat_vec), dtype=np.float64)
        cat_vec = np.concatenate([cat_vec, pad])
    # Add a deterministic pseudo‑random component based on stable hash
    h = stable_hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    noise = rng.random(dim) * 1e-6
    return cat_vec + noise

def stable_hash(text: str) -> int:
    """Deterministic 64‑bit hash used for seeding RNG."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)

# ----------------------------------------------------------------------
# RBF Surrogate dataclass (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """RBF surrogate prediction for a single input vector."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def fit_rbf_surrogate(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> RBFSurrogate:
    """
    Fit an RBF surrogate using the given *points* (as feature vectors) and
    target *values*.  Centers are taken to be the points themselves.
    The linear system (ΦᵀΦ + λI)w = Φᵀy is solved for the weights.
    """
    X = np.asarray(list(points), dtype=np.float64)
    y = np.asarray(list(values), dtype=np.float64).reshape(-1, 1)

    n = X.shape[0]
    # Build kernel matrix Φ (n×n)
    Phi = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            Phi[i, j] = gaussian(euclidean(X[i], X[j]), epsilon)

    # Regularised normal equations
    A = Phi.T @ Phi + ridge * np.eye(n)
    b = Phi.T @ y
    w = np.linalg.solve(A, b).flatten().tolist()
    centers = [tuple(row) for row in X]
    return RBFSurrogate(centers=centers, weights=w, epsilon=epsilon)

def hybrid_similarity(
    texts: List[str],
    surrogate: RBFSurrogate,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute a hybrid similarity matrix for *texts*.
    - ``S_hash`` uses perceptual hashes of the stylometric fingerprints.
    - ``S_rbf`` uses a Gaussian kernel on the surrogate predictions.
    The final similarity is ``α·S_hash + (1‑α)·S_rbf``.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    # 1️⃣ Stylometric fingerprints and hashes
    fingerprints = [stylometry_features(t) for t in texts]
    hashes = [compute_phash(fp.tolist()) for fp in fingerprints]

    n = len(texts)
    S_hash = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S_hash[i, j] = S_hash[j, i] = sim

    # 2️⃣ Surrogate predictions
    preds = np.array([surrogate.predict(fp) for fp in fingerprints], dtype=np.float64)

    # Gaussian kernel on absolute prediction differences
    sigma = np.std(preds) if np.std(preds) > 0 else 1.0
    S_rbf = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            diff = abs(preds[i] - preds[j])
            sim = math.exp(- (diff ** 2) / (2 * sigma ** 2))
            S_rbf[i, j] = S_rbf[j, i] = sim

    # 3️⃣ Convex combination
    return alpha * S_hash + (1.0 - alpha) * S_rbf

def predict_authenticity(text: str, surrogate: RBFSurrogate) -> float:
    """
    Convenience wrapper: compute the stylometric fingerprint of *text* and
    return the surrogate's prediction (an “authenticity” score).
    """
    fp = stylometry_features(text)
    return surrogate.predict(fp)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn animal leaped above a sleepy canine.",
        "In a distant galaxy, the starship engaged its warp drive.",
        "She sells seashells by the seashore, and they sparkle in sunlight.",
    ]

    # Extract stylometric vectors for fitting
    feats = [stylometry_features(t) for t in sample_texts]

    # Dummy target values (e.g., human‑rated quality)
    targets = np.linspace(0.2, 0.8, num=len(sample_texts))

    # Fit surrogate
    surrogate = fit_rbf_surrogate(feats, targets, epsilon=0.5, ridge=1e-6)

    # Compute hybrid similarity matrix
    S = hybrid_similarity(sample_texts, surrogate, alpha=0.6)

    print("Hybrid similarity matrix (shape {}):".format(S.shape))
    print(S)

    # Predict authenticity for a new sentence
    new_text = "Quantum entanglement allows particles to affect each other instantly."
    score = predict_authenticity(new_text, surrogate)
    print("\nAuthenticity score for new text: {:.4f}".format(score))