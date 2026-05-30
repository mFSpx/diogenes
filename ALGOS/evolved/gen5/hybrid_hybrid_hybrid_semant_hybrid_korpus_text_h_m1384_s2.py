# DARWIN HAMMER — match 1384, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s2.py (gen3)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py (gen4)
# born: 2026-05-29T23:35:44Z

"""Hybrid Semantic‑Bayesian‑MinHash Curvature Algorithm
Parents:
- hybrid_hybrid_semantic_neig_hybrid_bayes_update__m1099_s2.py
- hybrid_korpus_text_hybrid_hybrid_regret_m21_s1.py

Mathematical bridge:
1. The morphology‑derived recovery priority `R(m)` is treated as a Bayesian prior.
2. Cosine similarity `S(a,b)` between document embeddings provides the likelihood.
3. The posterior `P = (S·R)/(S·R + (1−S)(1−R))` is interpreted as an Ollivier‑Ricci‑like curvature.
4. A regret‑weighted MinHash Jaccard similarity `J(t1,t2)` (from the second parent) modulates this curvature,
   entropy, and vector literal construction, acting as an adaptive scaling factor that reflects
   textual overlap.

The resulting hybrid functions simultaneously exploit geometric morphology, probabilistic
evidence integration, and set‑based textual similarity in a single unified system.
"""

import math
import random
import sys
import pathlib
import re
import hashlib
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Morphology & recovery priority (Parent A)
# ----------------------------------------------------------------------
class Morphology:
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized to [0,1] – acts as a prior probability."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Text utilities (Parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    # deterministic ordering by seed index
    return sorted([_hash(i, t) for i, t in enumerate(toks)])[:k]


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return signature(tokens, k=k)


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    set1, set2 = set(sig1), set(sig2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def regret_weighted_similarity(text1: str, text2: str, k: int = 64) -> float:
    return jaccard_similarity(minhash_for_text(text1, k), minhash_for_text(text2, k))


def entropy_for_text(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def raw_embedding(text: str) -> np.ndarray:
    """Simple ordinal embedding (Unicode code point)."""
    return np.fromiter((ord(c) for c in text), dtype=np.float64)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    den = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if den == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / den)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_curvature(
    morph: Morphology,
    text_a: str,
    text_b: str,
    k_minhash: int = 64,
) -> float:
    """
    Compute a curvature‑like value for the edge (text_a, text_b).

    Steps:
    1. Prior `R` from morphology.
    2. Likelihood `S` = cosine similarity of raw embeddings.
    3. Posterior `P` = (S·R)/(S·R + (1−S)(1−R)).
    4. Scale `P` by regret‑weighted MinHash similarity `J`.
    """
    R = recovery_priority(morph)
    vec_a = raw_embedding(text_a)
    vec_b = raw_embedding(text_b)
    S = cosine_similarity(vec_a, vec_b)

    # Bayesian posterior (Ollivier‑Ricci analogue)
    numerator = S * R
    denominator = numerator + (1.0 - S) * (1.0 - R)
    P = numerator / denominator if denominator != 0 else 0.0

    # Regret‑weighted MinHash factor
    J = regret_weighted_similarity(text_a, text_b, k=k_minhash)

    return P * J


def hybrid_entropy(
    morph: Morphology,
    text: str,
    reference_text: str,
    k_minhash: int = 64,
) -> float:
    """
    Entropy of `text` modulated by morphology and textual overlap.

    E_raw = entropy(text)
    R = recovery_priority(morph)
    J = regret_weighted_similarity(text, reference_text)

    Output = E_raw * R * J
    """
    E_raw = entropy_for_text(text)
    R = recovery_priority(morph)
    J = regret_weighted_similarity(text, reference_text, k=k_minhash)
    return E_raw * R * J


def hybrid_vector_literal(
    morph: Morphology,
    text: str,
    reference_text: str,
    k_minhash: int = 64,
) -> str:
    """
    Produce a normalized vector literal where each component is scaled by both
    the morphology prior and the MinHash similarity to a reference text.

    Steps:
    1. Compute base ordinal embedding.
    2. Scale each component by `scale = R * J`.
    3. Normalize by the maximum Unicode code point (65535) for readability.
    """
    R = recovery_priority(morph)
    J = regret_weighted_similarity(text, reference_text, k=k_minhash)
    scale = R * J

    embedding = raw_embedding(text) * scale
    # Guard against division by zero when text is empty
    normalized = embedding / 65535.0 if embedding.size > 0 else np.array([], dtype=np.float64)

    # Produce string literal in the same format as the original parent
    components = [f"{v:.8f}" for v in normalized]
    return "[" + ",".join(components) + "]"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphology (roughly human‑like proportions)
    sample_morph = Morphology(length=1.8, width=0.5, height=0.3, mass=70.0)

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast dark-colored fox leaped above a sleepy canine."
    reference = "Foxes are swift and dogs are often lazy."

    curv = hybrid_curvature(sample_morph, text1, text2)
    ent = hybrid_entropy(sample_morph, text1, reference)
    vec_lit = hybrid_vector_literal(sample_morph, text1, reference)

    print(f"Hybrid curvature: {curv:.6f}")
    print(f"Hybrid entropy   : {ent:.6f}")
    print(f"Hybrid vector    : {vec_lit}")