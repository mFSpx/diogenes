# DARWIN HAMMER — match 5165, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_bandit_router_m394_s0.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_korpus_text_h_m1775_s0.py (gen5)
# born: 2026-05-30T00:00:20Z

"""
Hybrid Algorithm: Krampus‑Bandit Meets MinHash‑Entropy

Parents:
- hybrid_hybrid_hybrid_geomet_hybrid_bandit_router_m394_s0.py (geometric‑product / Ollivier‑Ricci curvature + bandit confidence)
- hybrid_hybrid_decision_hygi_hybrid_korpus_text_h_m1775_s0.py (MinHash signatures + Shannon entropy)

Mathematical Bridge:
Both parents operate on a scalar “score” that is compared against a target.
In the geometric‑bandit parent the score is 𝑝 = W·x and the update uses the
curvature c = ‖p‑t‖² together with a confidence term 𝑢(S,Nₐ).
In the text‑entropy parent a compact feature vector 𝑥̂ is obtained from a
MinHash signature and its Shannon entropy H(𝑥̂) quantifies information richness.

The hybrid therefore:
1. Transforms raw text into a normalized MinHash feature vector 𝑥̂.
2. Computes a prediction 𝑝 = W·𝑥̂.
3. Evaluates curvature c = ‖p‑t‖² (TTT‑Linear rule).
4. Uses the entropy H(𝑥̂) as the “store value” S in the bandit confidence term.
5. Updates the weight matrix with a gradient scaled by both curvature and confidence.

Thus the linear‑bandit core is enriched by a data‑dependent entropy term,
creating a unified learning step that respects both parents’ mathematics.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Iterable, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑B utilities: MinHash and Shannon entropy
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1
_MAX_HASH = 2 ** 64 - 1  # 64‑bit hash space


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i: i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            np.uint8(hashlib.blake2b(data, digest_size=8).digest()), dtype=np.uint64
        ),
        "big",
    )


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k
    seeds = [random.randrange(0, 2 ** 32) for _ in range(k)]
    signature = []
    for seed in seeds:
        min_hash = min(_hash_token(seed, t) for t in token_set)
        signature.append(min_hash)
    return signature


def minhash_for_text(text: str, k: int = 64, width: int = 5) -> List[int]:
    """Convenience wrapper: MinHash signature of a raw text string."""
    return minhash_signature(_shingles(text or "", width=width), k=k)


def shannon_entropy(values: List[int]) -> float:
    """Shannon entropy of a discrete distribution defined by *values*."""
    if not values:
        return 0.0
    # Build frequency table
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    total = len(values)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


# ----------------------------------------------------------------------
# Parent‑A utilities: curvature, confidence and weight update
# ----------------------------------------------------------------------
def krampus_ollivier_ricci_curvature(W: np.ndarray, x: np.ndarray, target: float) -> float:
    """
    Curvature as the squared residual of the linear prediction.
    Equivalent to the TTT‑Linear model's update rule.
    """
    pred = float(W @ x)
    residual = pred - target
    return residual * residual  # ‖p‑t‖²


def confidence_term(S: float, N_a: int) -> float:
    """
    Bandit confidence term, modulated by a store value *S*.
    Here *S* will be the Shannon entropy of the MinHash signature.
    """
    return (1 + S / (S + 1.0)) / math.sqrt(1.0 + N_a)


def hybrid_weight_update(
    W: np.ndarray,
    x: np.ndarray,
    target: float,
    N_a: int,
    entropy: float,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Single hybrid update step.

    - Gradient is taken from the ordinary least‑squares loss: ∇ = (p‑t)·x
    - Curvature scales the step (larger curvature → smaller step).
    - Confidence term further rescales the learning rate using entropy.
    """
    pred = float(W @ x)
    residual = pred - target
    grad = residual * x  # shape matches W
    curvature = krampus_ollivier_ricci_curvature(W, x, target)
    conf = confidence_term(entropy, N_a)
    # Guard against division by zero
    if curvature == 0.0:
        curvature = 1e-12
    update = lr * conf * grad / curvature
    W -= update  # gradient descent direction (negative because residual = pred‑t)
    return W


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def text_to_feature_vector(text: str, k: int = 64) -> Tuple[np.ndarray, float]:
    """
    Convert *text* into a normalized float feature vector and its entropy.

    Returns
    -------
    x : np.ndarray
        Shape (k,) with values in [0,1].
    H : float
        Shannon entropy of the underlying integer signature.
    """
    signature = minhash_for_text(text, k=k)
    # Normalise to [0,1] for linear algebra stability
    x = np.array([h / _MAX_HASH for h in signature], dtype=np.float64)
    H = shannon_entropy(signature)
    return x, H


def hybrid_predict(W: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction using the current weight matrix."""
    return float(W @ x)


def hybrid_train(
    texts: List[str],
    targets: List[float],
    k: int = 64,
    epochs: int = 5,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Train a weight vector on a small dataset using the hybrid rule.

    Parameters
    ----------
    texts : list of raw strings
    targets : list of scalar targets (same length as *texts*)
    k : length of MinHash signature (feature dimension)
    epochs : number of passes over the data
    lr : base learning rate

    Returns
    -------
    W : np.ndarray of shape (k,)
        Learned weight vector.
    """
    if len(texts) != len(targets):
        raise ValueError("texts and targets must have the same length")
    # Initialise weights randomly
    W = np.random.randn(k) * 0.01
    # Simple per‑sample counter for the bandit term
    pull_counts = np.zeros(k, dtype=int)

    for epoch in range(epochs):
        for txt, t in zip(texts, targets):
            x, ent = text_to_feature_vector(txt, k=k)
            # Update pull counts for the indices that are “active”.
            # Here we treat every dimension as an arm.
            pull_counts += 1
            N_a = pull_counts.mean()  # a scalar proxy for total pulls
            W = hybrid_weight_update(W, x, t, int(N_a), ent, lr=lr)
    return W


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Data‑driven algorithms blend statistics with geometry.",
    ]
    # Dummy regression targets (e.g., sentiment scores)
    sample_targets = [0.2, -0.1, 0.5]

    learned_W = hybrid_train(sample_texts, sample_targets, k=64, epochs=10, lr=0.05)

    # Show predictions on the training set
    for txt, tgt in zip(sample_texts, sample_targets):
        x_vec, _ = text_to_feature_vector(txt, k=64)
        pred = hybrid_predict(learned_W, x_vec)
        print(f"Text: {txt[:40]:<45} | Target: {tgt:5.2f} | Pred: {pred:5.2f}")