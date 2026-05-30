# DARWIN HAMMER — match 2474, survivor 5
# gen: 4
# parent_a: hybrid_hard_truth_math_model_pool_m8_s3.py (gen1)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (gen3)
# born: 2026-05-29T23:42:29Z

"""Hybrid Stylometry‑Morphology Infotaxis Model
Parents:
- hybrid_hard_truth_math_model_pool_m8_s3.py (stylometry & LSM utilities)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s4.py (morphology‑driven recovery priority & infotaxis)

Mathematical Bridge:
The stylometry pipeline yields a high‑dimensional semantic vector **v** for a document.
Parent B defines a *recovery priority* **p∈[0,1]** from physical morphology.
We fuse them by weighting the cosine similarity **c = (v₁·v₂)/(|v₁||v₂|)** with **p**,
forming a hybrid affinity  

    h = c · p  

The set of affinities for candidate actions is interpreted as a probability‑like mass
and fed into the classic infotaxis entropy estimator  

    E = p_hit·H(hit) + (1−p_hit)·H(miss),

where **p_hit** is the normalized hybrid affinity for the “hit’’ outcome and  
**H(q) = −q·log(q) − (1−q)·log(1−q)**.  This unified system lets morphology modulate
semantic decision‑making in an information‑theoretic manner.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import re
import hashlib
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – stylometry / LSM utilities
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
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Extract lowercase alphabetic words (including simple apostrophes)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Low‑dimensional function‑category vector (LSM)."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic 48‑bit hash used by the original model."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a fixed‑size dense feature vector from raw text.
    The first few entries are hand‑crafted ratios; the remainder is filled
    with a deterministic pseudo‑random projection of the LSM dictionary.
    """
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    # Hand‑crafted scalar features (scaled to roughly [0,1])
    feats: List[float] = [
        total_words / 500.0,                                 # word count normalised
        sum(len(w) for w in ws) / total_words / 12.0,       # avg word length
        (text.count("\n") + 1) / 200.0,                      # line density
        sum(text.count(p) for p in "!?") / total_chars,     # exclamation/question density
        sum(text.count(p) for p in ";:") / total_chars,     # semicolon/colon density
        len(set(ws)) / total_words,                         # lexical richness
        len(re.findall(r"\d+", text)) / total_words,        # numeric token ratio
    ]

    # Append deterministic projection of LSM categories
    lsm = lsm_vector(text)
    rng = np.random.default_rng(stable_hash(text) % (2**32))
    lsm_vals = np.array([lsm.get(cat, 0.0) for cat in sorted(FUNCTION_CATS.keys())])
    proj = rng.standard_normal((len(lsm_vals), dim - len(feats)))
    proj_sum = (lsm_vals[:, None] * proj).sum(axis=0)
    feats.extend(proj_sum.tolist())

    arr = np.array(feats, dtype=np.float64)
    # Clip to [0,1] for safety
    return np.clip(arr, 0.0, 1.0)


# ----------------------------------------------------------------------
# Parent B – Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """Physical proxy for how long an object needs to right itself."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority p∈[0,1] derived from righting‑time index.
    Larger righting times give lower priority; the mapping is linear
    after clipping to [0, max_index].
    """
    rti = righting_time_index(m)
    clipped = max(0.0, min(rti, max_index))
    return 1.0 - clipped / max_index


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Robust cosine similarity in [−1,1] (handles zero‑vectors)."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def hybrid_affinity(text_a: str, text_b: str, morph: Morphology) -> float:
    """
    Core fusion:
        h = cos(v_a, v_b) * p(morph)
    Returns a scalar in [−1,1] scaled by the morphology priority.
    """
    vec_a = stylometry_features(text_a)
    vec_b = stylometry_features(text_b)
    c = cosine_similarity(vec_a, vec_b)          # semantic similarity
    p = recovery_priority(morph)                # morphology‑derived priority
    return c * p


def binary_entropy(p: float) -> float:
    """Shannon entropy for a binary outcome, p∈[0,1]."""
    eps = 1e-12
    p = max(eps, min(1.0 - eps, p))
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def infotaxis_expected_entropy(affinities: List[float]) -> float:
    """
    Given a list of hybrid affinities for a candidate action,
    treat the normalised positive part as p_hit and compute the
    expected entropy as described in Parent B.
    """
    if not affinities:
        return float("inf")
    # Clip negatives to zero because they do not represent a “hit’’ probability
    pos = [max(0.0, a) for a in affinities]
    total = sum(pos) + 1e-12
    p_hit = sum(pos) / total
    # Entropy for hit / miss binary states
    return p_hit * binary_entropy(1.0) + (1.0 - p_hit) * binary_entropy(0.0)


def rank_actions(
    texts: List[str],
    morphologies: List[Morphology],
) -> List[Tuple[int, float]]:
    """
    For each text (treated as a possible action) compute its hybrid affinity
    against every other text, aggregate via infotaxis entropy, and return a
    ranking (lower entropy = more informative / preferable).
    """
    n = len(texts)
    scores: List[Tuple[int, float]] = []
    for i in range(n):
        # Compute affinities of text i with all others using its own morphology
        affs = [
            hybrid_affinity(texts[i], texts[j], morphologies[i])
            for j in range(n) if j != i
        ]
        entropy = infotaxis_expected_entropy(affs)
        scores.append((i, entropy))
    # Sort by ascending entropy (more informative first)
    scores.sort(key=lambda x: x[1])
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn animal leaps above a sleepy canine.",
        "Quantum mechanics describes the behavior of particles at microscopic scales.",
    ]

    sample_morphologies = [
        Morphology(length=2.0, width=0.5, height=0.3, mass=1.2),
        Morphology(length=1.8, width=0.4, height=0.35, mass=1.0),
        Morphology(length=0.2, width=0.2, height=0.5, mass=0.05),
    ]

    # Demonstrate hybrid affinity between first two texts
    h12 = hybrid_affinity(sample_texts[0], sample_texts[1], sample_morphologies[0])
    print(f"Hybrid affinity (text0 vs text1) = {h12:.4f}")

    # Compute ranking of actions
    ranking = rank_actions(sample_texts, sample_morphologies)
    print("\nAction ranking (index, entropy):")
    for idx, ent in ranking:
        print(f"  Text {idx}: entropy = {ent:.6f}")

    # Verify that infotaxis_expected_entropy does not raise
    dummy_affs = [0.2, -0.1, 0.5]
    print("\nInfotaxis entropy on dummy affinities:", infotaxis_expected_entropy(dummy_affs))