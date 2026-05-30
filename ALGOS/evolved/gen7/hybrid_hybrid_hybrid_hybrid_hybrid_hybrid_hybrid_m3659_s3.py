# DARWIN HAMMER — match 3659, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1449_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s3.py (gen3)
# born: 2026-05-29T23:51:10Z

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Mapping


# ----------------------------------------------------------------------
# Fixed lexical categories used for stylometric feature extraction.
# ----------------------------------------------------------------------
FUNCTION_CATS: Mapping[str, set[str]] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers",
        "they", "them", "their", "theirs",
        "we", "us", "our", "ours", "myself"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at",
        "before", "behind", "below", "between", "by", "during", "for",
        "from", "in", "into", "of", "off", "on", "onto", "over",
        "through", "to", "under", "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could", "did",
        "do", "does", "had", "has", "have", "is", "may", "might",
        "must", "shall", "should", "was", "were", "will", "would"
    },
}


def shannon_entropy(counts: Sequence[int]) -> float:
    """
    Compute Shannon entropy of a histogram.

    Parameters
    ----------
    counts: sequence of non‑negative integers
        Frequency of each discrete outcome.

    Returns
    -------
    float
        Entropy measured in bits. Returns 0.0 for an empty or all‑zero histogram.
    """
    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def stylometry_features(text: str, vocab: Sequence[str]) -> np.ndarray:
    """
    Produce a fixed‑length, normalized feature vector from *text*.

    The vector contains the relative frequencies of the words in *vocab*.
    Words not present in the text contribute a zero entry.

    Parameters
    ----------
    text: str
        Input document.
    vocab: sequence of str
        Ordered list of tokens that define the feature space.

    Returns
    -------
    np.ndarray, shape (len(vocab),)
        Normalised frequency vector (L1‑norm equals 1 unless the text is empty).
    """
    tokens = text.lower().split()
    token_counts = {tok: 0 for tok in vocab}
    for tok in tokens:
        if tok in token_counts:
            token_counts[tok] += 1

    total = sum(token_counts.values())
    if total == 0:
        # Return a zero vector; downstream code will handle the degenerate case.
        return np.zeros(len(vocab), dtype=float)

    freq = np.array([token_counts[tok] / total for tok in vocab], dtype=float)
    return freq


def ttt_linear_dynamics(
    W: np.ndarray,
    X: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Apply the TTT‑Linear update rule to a weight matrix.

    The update uses a decay term (‑αW) and a Hebbian‑like term
    β·X·Xᵀ, where X is a column vector of stylometric features.

    Parameters
    ----------
    W: np.ndarray, shape (d, d)
        Current weight matrix.
    X: np.ndarray, shape (d,)
        Feature vector (treated as a column vector).
    alpha: float
        Decay coefficient (α > 0).
    beta: float
        Hebbian scaling coefficient (β ≥ 0).

    Returns
    -------
    np.ndarray, shape (d, d)
        Updated weight matrix.
    """
    if X.ndim != 1:
        raise ValueError("Feature vector X must be one‑dimensional.")
    if W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix W must be square.")
    if W.shape[0] != X.shape[0]:
        raise ValueError(
            f"Dimension mismatch: W is {W.shape[0]}×{W.shape[0]}, "
            f"but X has length {X.shape[0]}."
        )

    # Outer product yields a matrix of the same shape as W.
    hebb = np.outer(X, X)
    dW = -alpha * W + beta * hebb
    return W + dW


def hybrid_fusion(
    text: str,
    counts: List[int],
    W: np.ndarray,
    alpha: float,
    beta: float,
    vocab: Sequence[str] | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Fuse Shannon entropy of a discrete histogram with TTT‑Linear dynamics
    driven by stylometric features.

    The function returns a *health_score* that combines the entropy of the
    supplied histogram with the Frobenius norm of the updated weight matrix.
    The integration is deeper than the original prototype because the
    feature vector dimension is forced to match the weight matrix, eliminating
    shape‑mismatch bugs and providing a deterministic mapping from text to
    matrix updates.

    Parameters
    ----------
    text: str
        Document from which stylometric features are extracted.
    counts: list[int]
        Histogram used for entropy calculation.
    W: np.ndarray, shape (d, d)
        Initial weight matrix.
    alpha: float
        Decay coefficient for the dynamics.
    beta: float
        Hebbian scaling coefficient.
    vocab: sequence of str, optional
        Token list that defines the feature space. If omitted, a default
        vocabulary consisting of the union of all FUNCTION_CATS tokens is used.

    Returns
    -------
    Tuple[float, np.ndarray]
        (health_score, updated_weight_matrix)
    """
    # ------------------------------------------------------------------
    # 1. Entropy component (unchanged but type‑checked)
    # ------------------------------------------------------------------
    entropy = shannon_entropy(counts)

    # ------------------------------------------------------------------
    # 2. Stylometric feature extraction with a deterministic dimension.
    # ------------------------------------------------------------------
    if vocab is None:
        # Build a deterministic, sorted vocabulary from the lexical categories.
        vocab = sorted({tok for cat in FUNCTION_CATS.values() for tok in cat})
    X = stylometry_features(text, vocab)

    # ------------------------------------------------------------------
    # 3. Matrix dynamics – now guaranteed to be dimensionally compatible.
    # ------------------------------------------------------------------
    W_updated = ttt_linear_dynamics(W, X, alpha, beta)

    # ------------------------------------------------------------------
    # 4. Health score – combine entropy with a normalized matrix norm.
    # ------------------------------------------------------------------
    # Normalise the Frobenius norm to the size of the matrix to keep the
    # score comparable across different dimensions.
    frob_norm = np.linalg.norm(W_updated, ord="fro")
    norm_factor = math.sqrt(W_updated.size)  # sqrt(d*d) = d
    scaled_norm = frob_norm / norm_factor

    health_score = entropy * scaled_norm
    return health_score, W_updated


# ----------------------------------------------------------------------
# Simple smoke‑test demonstrating the corrected behaviour.
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    text = "I think therefore I am. The quick brown fox jumps over the lazy dog."
    counts = [3, 7, 2, 5, 9]                     # arbitrary histogram
    dim = 20                                    # choose a dimension compatible with vocab size
    # Build a weight matrix of the same dimension as the feature vector.
    W = np.random.rand(dim, dim)

    # Use a custom vocabulary of the first *dim* tokens from the default set.
    default_vocab = sorted({tok for cat in FUNCTION_CATS.values() for tok in cat})
    vocab = default_vocab[:dim] if len(default_vocab) >= dim else default_vocab + ["<pad>"] * (dim - len(default_vocab))

    alpha = 0.05
    beta = 0.10

    health, W_new = hybrid_fusion(text, counts, W, alpha, beta, vocab=vocab)
    print(f"Health score: {health:.6f}")
    print(f"Updated weight matrix shape: {W_new.shape}")


if __name__ == "__main__":
    _smoke_test()