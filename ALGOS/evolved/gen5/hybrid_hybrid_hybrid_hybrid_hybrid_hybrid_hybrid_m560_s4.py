# DARWIN HAMMER — match 560, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s2.py (gen4)
# born: 2026-05-29T23:29:46Z

import uuid
import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """Immutable representation of a text span."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """A lightweight pheromone record with exponential decay."""

    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = max(1, half_life_seconds)  # avoid division by zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since the last decay event."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Stylometric feature extraction
# ----------------------------------------------------------------------
_FUNCTION_CATS: Dict[str, Tuple[str, ...]] = {
    "pronoun": (
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "ours"
    ),
    "article": ("a", "an", "the"),
    "preposition": (
        "about", "above", "after", "against", "around", "as", "at", "before",
        "behind", "below", "between", "by", "during", "for", "from", "in",
        "into", "of", "off", "on", "onto", "over", "through", "to", "under",
        "with", "without"
    ),
    "auxiliary": (
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might"
    ),
}


def _tokenise(text: str) -> List[str]:
    """Very light tokeniser – lower‑case and split on non‑word characters."""
    return re.findall(r"\b\w+\b", text.lower())


def stylometric_feature_extraction(texts: List[str]) -> np.ndarray:
    """
    Return a (len(texts) × |categories|) matrix where each entry is the raw count
    of words belonging to a functional category.
    """
    n = len(texts)
    m = len(_FUNCTION_CATS)
    features = np.zeros((n, m), dtype=np.int32)

    for idx, txt in enumerate(texts):
        tokens = _tokenise(txt)
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        for cat_i, (cat_name, cat_words) in enumerate(_FUNCTION_CATS.items()):
            count = sum(token_counts.get(w, 0) for w in cat_words)
            features[idx, cat_i] = count

    return features.astype(np.float64)


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(dist_matrix: np.ndarray) -> np.ndarray:
    """
    A simple proxy for Ollivier‑Ricci curvature on a complete weighted graph.
    For an edge (i, j) we use  κ(i,j) = w_ij / (1 + w_ij)  where w_ij is the
    distance between i and j.  The curvature matrix is symmetric with zeros on
    the diagonal.
    """
    # Avoid division by zero – distances are non‑negative, diagonal is zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = dist_matrix / (1.0 + dist_matrix)
    np.fill_diagonal(curvature, 0.0)
    return curvature


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def _pairwise_euclidean(a: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix for rows of *a*.
    Uses broadcasting; O(n²) memory but far faster than Python loops.
    """
    diff = a[:, None, :] - a[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def hybrid_algorithm(
    texts: List[str],
    spans: List[Span],
    distance_threshold: float,
) -> List[Span]:
    """
    Fuse stylometric profiling with Ollivier‑Ricci curvature and filter spans.

    1. Extract a stylometric feature matrix *F*.
    2. Compute the Euclidean distance matrix *D* on *F*.
    3. Derive a curvature matrix *C* from *D*.
    4. Build an *effective* distance matrix
           E = D * (1 - C)      # edges with high curvature are effectively shorter
    5. For each span, drop it if any other span lies within *distance_threshold*
       under *E*.
    """
    if len(texts) != len(spans):
        raise ValueError("The number of texts must match the number of spans.")

    # 1. Stylometric features
    feature_matrix = stylometric_feature_extraction(texts)          # (n, k)

    # 2. Pairwise Euclidean distances
    distance_matrix = _pairwise_euclidean(feature_matrix)          # (n, n)

    # 3. Curvature from distances
    curvature_matrix = ollivier_ricci_curvature(distance_matrix)   # (n, n)

    # 4. Effective distance that respects curvature
    effective_distance = distance_matrix * (1.0 - curvature_matrix)

    # 5. Span filtering
    n = len(spans)
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue
        # Any other *kept* span within the threshold?
        close = (effective_distance[i] < distance_threshold) & keep_mask
        # Exclude self‑comparison
        close[i] = False
        if np.any(close):
            # Prefer the span with the higher score; discard the lower‑scored one(s)
            candidates = np.where(close)[0]
            for j in candidates:
                if spans[i].score >= spans[j].score:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break

    return [spans[i] for i in range(n) if keep_mask[i]]


# ----------------------------------------------------------------------
# Simple demo (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts = [
        "I think that this is a sample text.",
        "Another example, perhaps a bit longer than the first.",
        "A third sample text, with different pronouns and prepositions."
    ]

    demo_spans = [
        Span(0, 10, "sample text", "label_a", 0.82),
        Span(15, 30, "another example", "label_b", 0.76),
        Span(35, 50, "third sample", "label_c", 0.64)
    ]

    # A relatively small threshold to illustrate filtering
    thresh = 2.5
    filtered = hybrid_algorithm(demo_texts, demo_spans, thresh)

    for s in filtered:
        print(s)