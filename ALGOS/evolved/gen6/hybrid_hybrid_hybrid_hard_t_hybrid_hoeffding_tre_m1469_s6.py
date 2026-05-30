# DARWIN HAMMER — match 1469, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (gen2)
# parent_b: hybrid_hoeffding_tree_hybrid_gini_coeffici_m685_s0.py (gen5)
# born: 2026-05-29T23:36:38Z

"""Hybrid Stylometry‑Hoeffding Model
Parents:
- hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (stylometry & lexical statistics)
- hybrid_hoeffding_tree_hybrid_gini_coeffici_m685_s0.py (Hoeffding bound & Gini‑adjusted split)

Mathematical bridge:
The stylometry routine yields a dense numeric representation 𝑥∈ℝ^d of a text.
Treating the components of 𝑥 as a distribution over “feature categories” we compute its
Gini coefficient G(𝑥).  In the Hoeffding‑tree split test the range parameter r is scaled
by (1‑G) so that a more homogeneous stylometric profile (low G) tightens the bound,
while a highly diverse profile (high G) relaxes it.  This fuses the lexical‑statistical
topology of parent A with the statistical‑learning topology of parent B."""

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities
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
    """Return lower‑cased alphabetic tokens."""
    return [w for w in (text or "").lower().split() if w.isalpha()]


def lsm_vector(text: str) -> dict[str, float]:
    """Lexical‑style‑matrix: proportion of each functional category."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic 48‑bit integer hash."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a fixed‑size numeric fingerprint.
    The first 8 entries are handcrafted ratios,
    the remainder are zero‑padded LSM values (flattened).
    """
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    handcrafted = [
        total_words / 500.0,
        sum(len(w) for w in ws) / total_words / 12.0,
        (text.count("\n") + 1) / 200.0,
        sum(text.count(p) for p in "!?") / total_chars,
        sum(text.count(p) for p in ";:") / total_chars,
        sum(text.count(p) for p in "-—") / total_chars,
        sum(1 for ch in text if ch.isupper()) / total_chars,
        stable_hash(text) % 1.0,  # pseudo‑random but deterministic scalar
    ]

    lsm = lsm_vector(text)
    lsm_vals = np.array([lsm.get(cat, 0.0) for cat in sorted(FUNCTION_CATS.keys())])
    # Pad / truncate to reach dim‑len(handcrafted)
    needed = dim - len(handcrafted)
    if needed > 0:
        pad = np.zeros(needed)
        lsm_vals = np.concatenate([lsm_vals, pad])[:needed]
    else:
        lsm_vals = lsm_vals[:dim]

    return np.concatenate([np.array(handcrafted, dtype=float), lsm_vals])


# ----------------------------------------------------------------------
# Parent B – Hoeffding‑tree split utilities (Gini‑adjusted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def gini_coefficient(values: Iterable[float]) -> float:
    """Classic Gini impurity for a non‑negative distribution."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (r^2 * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r>0, 0<δ<1, n>0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = (
        "gap_exceeds_bound"
        if gap > eps
        else ("tie_threshold" if eps < tie_threshold else "wait")
    )
    return SplitDecision(split, eps, gap, reason)


def hybrid_split_decision(
    feature_vector: np.ndarray,
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """
    Gini‑adjusted Hoeffding split decision.
    The Gini of the stylometric vector modulates the effective range r.
    """
    gini = gini_coefficient(feature_vector)
    adjusted_r = r * (1.0 - gini)  # high Gini → looser bound
    return should_split(
        best_gain, second_best_gain, adjusted_r, delta, n, tie_threshold
    )


# ----------------------------------------------------------------------
# Hybrid API – three representative functions
# ----------------------------------------------------------------------
def text_to_feature_vector(text: str, dim: int = 96) -> np.ndarray:
    """Convenient wrapper: raw text → stylometry fingerprint."""
    return stylometry_features(text, dim)


def evaluate_split(
    text: str,
    best_gain: float,
    second_best_gain: float,
    r: float = 1.0,
    delta: float = 0.05,
    n: int = 100,
) -> SplitDecision:
    """
    End‑to‑end hybrid evaluation for a single text instance.
    """
    vec = text_to_feature_vector(text)
    return hybrid_split_decision(vec, best_gain, second_best_gain, r, delta, n)


def batch_similarity_matrix(texts: List[str], dim: int = 96) -> np.ndarray:
    """
    Build an (m×m) similarity matrix for a batch of texts using cosine similarity
    on the hybrid feature vectors.
    """
    vectors = np.stack([text_to_feature_vector(t, dim) for t in texts])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a galaxy far, far away, there lived a small planet.",
        "Mathematics is the language of the universe; it speaks in numbers."
    ]

    # Compute feature vectors
    feats = [text_to_feature_vector(t) for t in sample_texts]
    print("Feature vectors (first 5 elements each):")
    for i, f in enumerate(feats):
        print(f"  [{i}] {f[:5]}")

    # Batch similarity
    sim = batch_similarity_matrix(sample_texts)
    print("\nSimilarity matrix:")
    print(sim)

    # Simulate split decisions with arbitrary gains
    for txt in sample_texts:
        decision = evaluate_split(
            txt,
            best_gain=random.uniform(0.2, 0.5),
            second_best_gain=random.uniform(0.0, 0.19),
            r=1.0,
            delta=0.05,
            n=200,
        )
        print(f"\nDecision for \"{txt[:30]}...\":")
        print(f"  Should split: {decision.should_split}")
        print(f"  ε: {decision.epsilon:.5f}, gap: {decision.gain_gap:.5f}, reason: {decision.reason}")

    sys.exit(0)