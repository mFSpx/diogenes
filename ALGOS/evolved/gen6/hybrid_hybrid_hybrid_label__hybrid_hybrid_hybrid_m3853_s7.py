# DARWIN HAMMER — match 3853, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (gen5)
# born: 2026-05-29T23:52:03Z

"""Hybrid Algorithm: Morphology‑Stylometry Fusion via Sinusoidal Linear Operator

Parents:
- hybrid_hybrid_label_foundry_hybrid_hybrid_hard_t_m304_s2.py (Labeling & Morphology)
- hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m546_s3.py (Stylometry & Sinusoidal Operator)

Mathematical Bridge:
The morphology of an endpoint is described by a 4‑dimensional vector 
M = (length, width, height, mass).  Stylometry extracts a normalized 
frequency vector S ∈ ℝ^{NUM_CATS}.  We concatenate them into a single 
feature vector X = [M, S] ∈ ℝ^{4+NUM_CATS}.  The sinusoidal linear operator 
implements a single‑layer KAN‑like mapping:

    Y = X + A·sin(W·X + b)

where W ∈ ℝ^{d_out×d_in}, b ∈ ℝ^{d_out}, A ∈ ℝ^{d_out} are random 
parameters and sin is applied element‑wise.  This provides universal 
approximation while preserving the discrete counting of the labeling 
functions.  The final label is obtained by a linear read‑out 
L = softmax(C·Y) and taking argmax.

The module therefore fuses the discrete weak‑supervision labeling primitives 
with continuous stylometric feature extraction through a mathematically 
coherent sinusoidal KAN layer.
"""

import sys
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

# ----------------------------------------------------------------------
# Weak‑supervision labeling decorator (Parent A)
# ----------------------------------------------------------------------
def labeling_function(name: str | None = None):
    """Decorator that tags a function as a labeling function."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

# ----------------------------------------------------------------------
# Morphology based geometric primitive (Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a simple sphericity proxy: (surface area) / (volume)^{2/3}."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    surface = 2 * (length * width + width * height + height * length)
    volume = length * width * height
    return surface / (volume ** (2.0 / 3.0))

# ----------------------------------------------------------------------
# Stylometry feature extraction (Parent B)
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
    "negation": set("no not never none neither cannot cant wont dont".split()),
}
CATEGORY_ORDER = list(FUNCTION_CATS.keys())
NUM_CATS = len(CATEGORY_ORDER)

def _tokenize(text: str) -> List[str]:
    """Very simple word tokenizer."""
    import re
    return re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())

def stylometry_features(text: str) -> np.ndarray:
    """
    Normalized frequency vector over FUNCTION_CATS.
    Returns a (NUM_CATS,) float array that sums to 1 (or zeros if no matches).
    """
    tokens = _tokenize(text)
    counts = Counter(tokens)
    vec = np.zeros(NUM_CATS, dtype=float)
    for idx, cat in enumerate(CATEGORY_ORDER):
        cat_words = FUNCTION_CATS[cat]
        cat_count = sum(counts[w] for w in cat_words if w in counts)
        vec[idx] = cat_count
    total = vec.sum()
    if total > 0.0:
        vec /= total
    return vec

# ----------------------------------------------------------------------
# Sinusoidal linear operator – KAN‑like layer (Parent B)
# ----------------------------------------------------------------------
def _base_sinusoid(pool_size: int) -> np.ndarray:
    """
    Create a (pool_size, pool_size) matrix with sinusoidal basis:
    S[i, j] = sin(2π * i * j / pool_size)
    """
    i = np.arange(pool_size).reshape(-1, 1)
    j = np.arange(pool_size).reshape(1, -1)
    return np.sin(2.0 * math.pi * i * j / pool_size)

def sinusoidal_operator(
    x: np.ndarray,
    out_dim: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply a single‑layer sinusoidal operator:
        y = x + a * sin(W @ x + b)

    Parameters
    ----------
    x : np.ndarray
        Input feature vector (1‑D).
    out_dim : int
        Desired output dimensionality.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    y : np.ndarray
        Transformed vector of shape (out_dim,).
    """
    if seed is not None:
        np.random.seed(seed)
    d_in = x.shape[0]
    # Random weight matrix with values in [-1, 1]
    W = np.random.uniform(-1.0, 1.0, size=(out_dim, d_in))
    b = np.random.uniform(-math.pi, math.pi, size=out_dim)
    a = np.random.uniform(0.0, 1.0, size=out_dim)

    linear = W @ x + b
    y = x[:out_dim] + a * np.sin(linear)
    return y

# ----------------------------------------------------------------------
# Hybrid prediction pipeline
# ----------------------------------------------------------------------
def hybrid_feature_vector(
    morphology: Morphology,
    text: str,
) -> np.ndarray:
    """
    Build the concatenated feature vector [M, S] where:
        M = (length, width, height, mass, sphericity)
        S = stylometry_features(text)
    """
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    m_vec = np.array(
        [morphology.length, morphology.width, morphology.height, morphology.mass, sph],
        dtype=float,
    )
    s_vec = stylometry_features(text)
    return np.concatenate([m_vec, s_vec])

def hybrid_predict(
    morphology: Morphology,
    text: str,
    out_dim: int = 8,
    seed: int | None = 42,
) -> Tuple[int, np.ndarray]:
    """
    Produce a label prediction from morphology + text.

    Returns
    -------
    label : int
        Predicted class index (argmax of softmax probabilities).
    probs : np.ndarray
        Probability distribution over `out_dim` classes.
    """
    x = hybrid_feature_vector(morphology, text)
    y = sinusoidal_operator(x, out_dim=out_dim, seed=seed)

    # Simple linear read‑out followed by softmax
    np.random.seed(seed)
    C = np.random.uniform(-1.0, 1.0, size=(out_dim, out_dim))
    logits = C @ y
    exp_logits = np.exp(logits - np.max(logits))  # stability
    probs = exp_logits / exp_logits.sum()
    label = int(np.argmax(probs))
    return label, probs

# ----------------------------------------------------------------------
# Example weak‑supervision labeling functions (Parent A)
# ----------------------------------------------------------------------
@labeling_function(name="contains_negation")
def lf_contains_negation(doc: dict) -> int:
    """Label 1 if any negation word appears, else 0."""
    tokens = _tokenize(doc.get("text", ""))
    return int(any(tok in FUNCTION_CATS["negation"] for tok in tokens))

@labeling_function(name="high_sphericity")
def lf_high_sphericity(doc: dict) -> int:
    """Label 1 if sphericity index exceeds a threshold."""
    morph = doc.get("morphology")
    if not isinstance(morph, Morphology):
        return 0
    return int(sphericity_index(morph.length, morph.width, morph.height) > 1.5)

# ----------------------------------------------------------------------
# Fusion evaluator
# ----------------------------------------------------------------------
def fuse_labels(
    docs: List[Dict[str, Any]],
    out_dim: int = 8,
    seed: int | None = 42,
) -> List[ProbabilisticLabel]:
    """
    Run weak‑supervision labeling functions and the hybrid predictor,
    returning a probabilistic label for each document.
    """
    results: List[ProbabilisticLabel] = []
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        # weak supervision vote (majority)
        votes = [fn(doc) for fn in (lf_contains_negation, lf_high_sphericity)]
        weak_label = int(sum(votes) >= len(votes) / 2)

        # hybrid model prediction
        morph: Morphology = doc["morphology"]
        text: str = doc["text"]
        pred_label, probs = hybrid_predict(morph, text, out_dim=out_dim, seed=seed)

        # combine: confidence is weighted average (0.4 weak, 0.6 model)
        confidence = 0.4 * (weak_label == pred_label) + 0.6 * probs[pred_label]
        results.append(ProbabilisticLabel(doc_id=doc_id, label=pred_label, confidence=confidence))
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    docs = [
        {
            "doc_id": "doc1",
            "text": "I do not like this article because it is boring.",
            "morphology": Morphology(length=2.0, width=1.0, height=1.5, mass=0.8),
        },
        {
            "doc_id": "doc2",
            "text": "The quick brown fox jumps over the lazy dog.",
            "morphology": Morphology(length=3.0, width=2.0, height=2.5, mass=1.2),
        },
    ]

    probs = fuse_labels(docs)
    for pl in probs:
        print(f"Document {pl.doc_id}: label={pl.label}, confidence={pl.confidence:.3f}")

    # Verify that the core functions run without exception
    for d in docs:
        vec = hybrid_feature_vector(d["morphology"], d["text"])
        assert vec.shape[0] == 5 + NUM_CATS
        y = sinusoidal_operator(vec, out_dim=8)
        assert y.shape == (8,)
    print("Smoke test passed.")