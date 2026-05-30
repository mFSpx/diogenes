# DARWIN HAMMER — match 4683, survivor 2
# gen: 7
# parent_a: hybrid_percyphon_hybrid_hybrid_hybrid_m1250_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1915_s0.py (gen5)
# born: 2026-05-29T23:57:26Z

"""Hybrid Percyphon‑Darwin Hammer & Cockpit‑Pheromone Fusion

This module fuses the mathematical cores of the two parent algorithms:

* **Parent A** – Percyphon.ai / Darwin Hammer hybrid:
  - MinHash signatures for token‑set similarity.
  - Perceptual hash (p‑hash) of numeric feature vectors.
* **Parent B** – Cockpit linguistic style + weighted feature extraction + pheromone entropy.

**Mathematical bridge**

Both parents expose a *signature* that can be compared with a Hamming‑type distance
(MinHash, p‑hash) and a *real‑valued feature vector* that can be compared with a
cosine similarity (linguistic style vector weighted by the positive/negative
weight arrays).  The bridge is therefore a linear combination of three similarity
measures:


S = α·S_minhash + β·S_phash + γ·S_style


where each `S_*` is a normalised similarity in `[0,1]`.  The pheromone entropy
is used as an additional weighting factor for the style similarity, providing
the “trust‑weighted” component described in Parent B.

The resulting hybrid system can compare two entities that carry:
- a token set (`tokens`),
- a numeric node‑feature list (`node_features`),
- a free‑form textual description (`text`).

The three signatures are fused into a single similarity score that can drive
leader election, procedural generation, or any downstream decision logic.
"""

import hashlib
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by Percyphon.ai / Darwin Hammer."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """Return a MinHash signature (uint64 vector) of length *k*."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return np.full(k, MAX64, dtype=np.uint64)

    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        sig[i] = min(_hash(i, t) for t in toks)
    return sig


def compute_phash(values: List[float]) -> int:
    """64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()


def signature_hamming(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Normalised Hamming similarity for two equal‑length uint64 signatures."""
    if sig_a.shape != sig_b.shape:
        raise ValueError("Signature shapes must match")
    total_bits = sig_a.size * 64
    diff = np.bitwise_xor(sig_a, sig_b).astype(np.uint64)
    differing_bits = np.unpackbits(diff.view(np.uint8)).sum()
    return 1.0 - differing_bits / total_bits  # similarity in [0,1]


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Regular expressions for each style category
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|"
    r"not now|before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|fri", re.I)  # truncated in parent, placeholder


_REGEX_MAP = {
    "evidence": _EVIDENCE_RE,
    "planning": _PLANNING_RE,
    "delay": _DELAY_RE,
    # placeholders for the remaining categories (use generic word boundaries)
    "support": re.compile(r"\bsupport\b", re.I),
    "boundary": re.compile(r"\bboundary\b", re.I),
    "outcome": re.compile(r"\boutcome\b", re.I),
    "impulsive": re.compile(r"\bimpulsive\b", re.I),
    "scarcity": re.compile(r"\bscarcity\b", re.I),
    "risk": re.compile(r"\brisk\b", re.I),
}


def lsm_vector(text: str) -> np.ndarray:
    """
    Linguistic Style Matching vector.

    For each feature in ``_FEATURE_ORDER`` we count regex matches, then normalise
    by the total number of matches.  The result is a float vector of length 9.
    """
    counts = []
    total = 0
    for feat in _FEATURE_ORDER:
        regex = _REGEX_MAP.get(feat, re.compile(r"\b" + re.escape(feat) + r"\b", re.I))
        c = len(regex.findall(text))
        counts.append(c)
        total += c
    if total == 0:
        return np.zeros(len(_FEATURE_ORDER), dtype=np.float64)
    return np.array([c / total for c in counts], dtype=np.float64)


def pheromone_signal(token_counts: Dict[str, int], half_life_seconds: float) -> float:
    """
    Entropy‑based pheromone signal with exponential decay.

    The raw signal is the Shannon entropy (base‑2) of the token frequency
    distribution.  It is then decayed by a factor ``2 ** (-t / half_life)`` where
    ``t`` is assumed equal to ``half_life_seconds`` for a single half‑life step.
    """
    total = sum(token_counts.values())
    if total == 0 or half_life_seconds <= 0:
        return 0.0
    probs = np.array(list(token_counts.values()), dtype=np.float64) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    decay_factor = 2 ** (-1)  # one half‑life elapsed
    return float(entropy * decay_factor)


def weighted_style_similarity(
    text_a: str,
    text_b: str,
    positive_weights: np.ndarray = _POSITIVE_WEIGHTS,
    negative_weights: np.ndarray = _NEGATIVE_WEIGHTS,
) -> float:
    """
    Cosine similarity between two LSM vectors after applying trust weights.

    The trust weight vector is ``positive_weights - negative_weights``.  The
    resulting weighted vectors are L2‑normalised before the cosine is computed.
    """
    w = positive_weights.astype(np.float64) - negative_weights.astype(np.float64)
    vec_a = lsm_vector(text_a) * w
    vec_b = lsm_vector(text_b) * w

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# Hybrid interface (the “bridge”)
# ----------------------------------------------------------------------
def hybrid_similarity(
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    node_features_a: List[float],
    node_features_b: List[float],
    text_a: str,
    text_b: str,
    token_counts_a: Dict[str, int],
    token_counts_b: Dict[str, int],
    k: int = 128,
    alpha: float = 0.33,
    beta: float = 0.33,
    gamma: float = 0.34,
    half_life_seconds: float = 60.0,
) -> float:
    """
    Compute a fused similarity score for two entities.

    Parameters
    ----------
    tokens_a, tokens_b : iterable of str
        Token sets used for MinHash.
    node_features_a, node_features_b : list of float
        Numeric feature vectors used for perceptual hashing.
    text_a, text_b : str
        Free‑form textual descriptions for linguistic style analysis.
    token_counts_a, token_counts_b : dict
        Frequency maps required for the pheromone entropy term.
    k : int
        Length of the MinHash signature (default 128).
    alpha, beta, gamma : float
        Linear blending coefficients that must sum to 1.
    half_life_seconds : float
        Decay parameter for the pheromone signal.

    Returns
    -------
    float
        Combined similarity in the range [0,1].
    """
    # --- MinHash similarity -------------------------------------------------
    sig_a = minhash_signature(tokens_a, k=k)
    sig_b = minhash_signature(tokens_b, k=k)
    s_minhash = signature_hamming(sig_a, sig_b)  # already normalised

    # --- Perceptual‑hash similarity -----------------------------------------
    ph_a = compute_phash(node_features_a)
    ph_b = compute_phash(node_features_b)
    phamming = hamming_distance(ph_a, ph_b) / 64.0
    s_phash = 1.0 - phamming  # similarity

    # --- Weighted linguistic style similarity --------------------------------
    raw_style_sim = weighted_style_similarity(text_a, text_b)

    # Apply pheromone trust factor (average of the two signals)
    pher_a = pheromone_signal(token_counts_a, half_life_seconds)
    pher_b = pheromone_signal(token_counts_b, half_life_seconds)
    pheromone_factor = (pher_a + pher_b) / 2.0
    # Normalise pheromone factor to [0,1] assuming max entropy for a uniform
    # distribution over 256 tokens (~8 bits).  This is a heuristic.
    pheromone_norm = min(pheromone_factor / 8.0, 1.0)

    s_style = raw_style_sim * pheromone_norm

    # --- Linear blend --------------------------------------------------------
    total = alpha + beta + gamma
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        # Normalise automatically if the user supplied non‑unit sum.
        alpha, beta, gamma = alpha / total, beta / total, gamma / total

    combined = alpha * s_minhash + beta * s_phash + gamma * s_style
    return float(combined)


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for a quick sanity check
    tokens1 = ["alpha", "beta", "gamma", "delta"]
    tokens2 = ["beta", "epsilon", "zeta", "eta"]

    node_feat1 = [random.random() for _ in range(20)]
    node_feat2 = [random.random() for _ in range(20)]

    text1 = (
        "The evidence confirms the plan, but we must delay the rollout due to "
        "resource constraints. Support from the team is essential."
    )
    text2 = (
        "Verified sources show that the schedule is on track. No delay is expected, "
        "and risk is low."
    )

    token_counts1 = {t: random.randint(1, 5) for t in tokens1}
    token_counts2 = {t: random.randint(1, 5) for t in tokens2}

    sim = hybrid_similarity(
        tokens_a=tokens1,
        tokens_b=tokens2,
        node_features_a=node_feat1,
        node_features_b=node_feat2,
        text_a=text1,
        text_b=text2,
        token_counts_a=token_counts1,
        token_counts_b=token_counts2,
        k=128,
        alpha=0.3,
        beta=0.3,
        gamma=0.4,
        half_life_seconds=120.0,
    )

    print(f"Hybrid similarity score: {sim:.4f}")