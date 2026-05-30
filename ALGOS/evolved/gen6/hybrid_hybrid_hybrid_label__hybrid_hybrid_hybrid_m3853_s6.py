# DARWIN HAMMER — match 3853, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# born: 2026-05-29T23:52:03Z

"""
Hybrid Algorithm: Morphology‑Stylometry Fusion via Sinusoidal Linear Operator

Parents:
- hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (Labeling + Morphology)
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (Stylometry + Sinusoidal Linear Operator)

Mathematical Bridge:
The morphology of a document is distilled into a 4‑dimensional vector M
(length, width, height, mass).  Stylometry supplies a K‑dimensional normalized
frequency vector S over functional word categories.  We concatenate them
into a single feature vector X = [M, S]ᵀ.  X is transformed by a sinusoidal
linear operator L(X) = sin(W·X + b) where W and b are fixed matrices derived
from a deterministic pseudo‑random seed.  The sinusoid approximates the
non‑linear kernels of a Kolmogorov‑Arnold‑Network (KAN).  A final element‑wise
activation A(y) = exp(−y²) provides a smooth, universal‑approximation
layer.  The scalar output ŷ = 1ᵀ·A(L(X)) is interpreted as a confidence score
for the labeling function defined in the original Parent A.

The implementation below fuses these components into three public functions:
- `extract_morphology`
- `stylometry_features`
- `hybrid_predict`
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Labeling primitives
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Simple geometric description of a text document."""
    length: float   # total character count
    width: float    # average word length
    height: float   # number of sentences
    mass: float     # total token count


def sphericity_index(length: float, width: float, height: float) -> float:
    """Classic sphericity for a rectangular prism.
    Returns ( (length * width * height) ** (2/3) ) / ( (length**2 + width**2 + height**2) / 3 )**0.5
    """
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    volume = length * width * height
    surface = (length ** 2 + width ** 2 + height ** 2) / 3.0
    return (volume ** (2.0 / 3.0)) / math.sqrt(surface)


def extract_morphology(text: str) -> Morphology:
    """Derive a Morphology instance from raw text."""
    chars = len(text)
    tokens = text.split()
    token_cnt = len(tokens)
    avg_word_len = sum(len(t) for t in tokens) / token_cnt if token_cnt else 0.0
    sentences = max(text.count('.'), text.count('!'), text.count('?'))
    sentences = sentences if sentences else 1  # avoid division by zero
    return Morphology(
        length=float(chars),
        width=avg_word_len,
        height=float(sentences),
        mass=float(token_cnt),
    )


def morphology_vector(morph: Morphology) -> np.ndarray:
    """Normalized 4‑D vector derived from Morphology."""
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


# ----------------------------------------------------------------------
# Parent B – Stylometry & Sinusoidal Linear Operator
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
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
    "negation": set("no not never none neither cannot can't won't don't".split()),
}

CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)


def _tokenize(text: str) -> List[str]:
    """Very simple word tokenizer."""
    import re
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())


def stylometry_features(text: str) -> np.ndarray:
    """Normalized frequency vector over FUNCTION_CATS."""
    tokens = _tokenize(text)
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts.get(w, 0) for w in cat_words)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0.0:
        vec /= total
    return vec


def _base_sinusoid(pool_size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a deterministic sinusoidal weight matrix W and bias b.
    pool_size – dimensionality of the input feature vector.
    Returns (W, b) where:
        W shape = (pool_size, pool_size)
        b shape = (pool_size,)
    """
    rng = random.Random(seed)
    W = np.fromiter((rng.uniform(-1.0, 1.0) for _ in range(pool_size * pool_size)),
                    dtype=float).reshape(pool_size, pool_size)
    b = np.fromiter((rng.uniform(-math.pi, math.pi) for _ in range(pool_size)),
                    dtype=float)
    return W, b


def sinusoidal_linear_operator(x: np.ndarray) -> np.ndarray:
    """
    Apply a sinusoidal linear transformation:
        y = sin(W·x + b)
    W and b are generated once per module load.
    """
    if not hasattr(sinusoidal_linear_operator, "_W"):
        pool = x.shape[0]
        sinusoidal_linear_operator._W, sinusoidal_linear_operator._b = _base_sinusoid(pool)
    W = sinusoidal_linear_operator._W
    b = sinusoidal_linear_operator._b
    return np.sin(W @ x + b)


def kan_approximation(y: np.ndarray) -> np.ndarray:
    """
    Simple smooth activation mimicking a KAN node:
        a_i = exp( - (y_i)^2 )
    Guarantees bounded output in (0, 1].
    """
    return np.exp(-np.square(y))


# ----------------------------------------------------------------------
# Hybrid Core – Fusion of Morphology & Stylometry
# ----------------------------------------------------------------------


def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Build the concatenated feature vector X = [M, S].
    M – 4‑D normalized morphology vector.
    S – NUM_CATS‑D stylometry frequency vector.
    """
    morph = extract_morphology(text)
    m_vec = morphology_vector(morph)                # shape (4,)
    s_vec = stylometry_features(text)               # shape (NUM_CATS,)
    return np.concatenate([m_vec, s_vec])           # shape (4 + NUM_CATS,)


def hybrid_predict(text: str) -> float:
    """
    Produce a scalar confidence score for the document.
    Pipeline:
        X  -> sinusoidal_linear_operator -> kan_approximation -> sum
    The sum aggregates contributions of all transformed dimensions.
    """
    X = hybrid_feature_vector(text)
    y = sinusoidal_linear_operator(X)
    a = kan_approximation(y)
    return float(a.sum())


def label_document(text: str, threshold: float = 0.5) -> Tuple[int, float]:
    """
    Convert the hybrid confidence into a binary label.
    Returns (label, confidence) where label is 1 if confidence >= threshold.
    """
    conf = hybrid_predict(text)
    label = 1 if conf >= threshold else 0
    return label, conf


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "I can't believe that you haven't finished the report yet. "
        "The data, however, shows a clear upward trend, and we should "
        "consider publishing it soon. Nevertheless, there are still "
        "some issues that need to be addressed."
    )
    label, confidence = label_document(sample, threshold=0.5)
    print(f"Hybrid confidence: {confidence:.4f}")
    print(f"Assigned label: {label}")
    # Ensure the pipeline runs without raising.
    sys.exit(0)