# DARWIN HAMMER — match 4413, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m2091_s2.py (gen4)
# parent_b: hybrid_krampus_stickers_hybrid_korpus_text_h_m1495_s1.py (gen5)
# born: 2026-05-29T23:55:36Z

"""
Hybrid Algorithm: Fusion of DARWIN HAMMER (match 2091) and DARWIN HAMMER (match 1495)

Mathematical Bridge
-------------------
The parent *Hybrid Workshare Allocator* (Algorithm A) produces a **weekday‑weight
vector** `w ∈ ℝ^G` (G = number of groups) that is a row‑stochastic sinusoidal
rotation. The parent *Krampus Sticker Math* (Algorithm B) yields a **MinHash
similarity** `s ∈ [0,1]` between a text and a reference signature, which is then
modulated by Shannon entropy `H` and a regret‑weighting term `r`.

The hybrid unifies these by **projecting the scalar similarity‑entropy product
onto the group simplex** using the weekday‑weight vector as a linear map:


score_g = w_g * (s * (1 – λ·H) * (1 – μ·r))


where `λ, μ ∈ [0,1]` are tunable damping coefficients.  
Thus the temporal pattern of the week influences how much each group “receives”
the text‑selection score, while the text‑specific information remains governed
by the MinHash/entropy/regret pipeline.

The module implements this fused computation together with three public
functions that expose the hybrid behaviour.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
from typing import List, Iterable, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Constants & Helpers (from Algorithm A)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# MinHash & Entropy utilities (from Algorithm B)
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # 64‑bit Blake2b hash, truncated to 8 bytes
    return int.from_bytes(
        np.frombuffer(
            np.frombuffer(hashlib.blake2b(data, digest_size=8).digest(), dtype=np.uint8),
            dtype=np.uint64,
        ),
        "big",
    )


def shingles(text: str, width: int = 5) -> List[str]:
    """Return all contiguous substrings of length ``width``."""
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    Compute a MinHash signature of length ``k`` from an iterable of tokens.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    hashes = []
    for seed in range(k):
        hash_values = [_hash(seed, t) for t in toks]
        hashes.append(min(hash_values))
    return hashes


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """
    if len(sig1) != len(sig2):
        raise ValueError("signatures must be of equal length")
    intersection = sum(1 for a, b in zip(sig1, sig2) if a == b)
    union = len(sig1)  # MinHash guarantees union = length
    return intersection / union if union else 0.0


def shannon_entropy(text: str) -> float:
    """
    Compute the Shannon entropy of the character distribution in ``text``.
    Spaces are ignored and case is folded.
    """
    cleaned = text.replace(" ", "").lower()
    if not cleaned:
        return 0.0
    probs = [cleaned.count(ch) / len(cleaned) for ch in set(cleaned)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ----------------------------------------------------------------------
# Regret weighting (light version of Algorithm B)
# ----------------------------------------------------------------------


class MathAction:
    """
    Simple container for regret‑related parameters.
    In the original parent this carried expected value, cost and risk.
    Here we expose only a single scalar ``regret`` that will be used as a
    multiplicative damping factor.
    """

    def __init__(self, regret: float):
        if not (0.0 <= regret <= 1.0):
            raise ValueError("regret must be between 0 and 1")
        self.regret = regret


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------


def compute_hybrid_score(
    text: str,
    reference_sig: List[int],
    groups: Tuple[str, ...] = GROUPS,
    today: date = date.today(),
    lambda_entropy: float = 0.3,
    mu_regret: float = 0.4,
    action: MathAction = MathAction(regret=0.2),
) -> Tuple[np.ndarray, float]:
    """
    Compute a *group‑wise* hybrid score for ``text``.

    Returns
    -------
    (group_scores, overall_score) where
        group_scores : np.ndarray shape (G,)
        overall_score: float = sum(group_scores)

    The computation follows the bridge formula described in the module docstring.
    """
    # 1. Weekday weight vector (Algorithm A)
    dow = doomsday(today.year, today.month, today.day)
    w = weekday_weight_vector(groups, dow)  # shape (G,)

    # 2. MinHash similarity (Algorithm B)
    txt_sig = signature(shingles(text))
    s = jaccard_similarity(txt_sig, reference_sig)  # scalar ∈ [0,1]

    # 3. Entropy term
    H = shannon_entropy(text) / math.log2(256)  # normalise to [0,1] assuming 8‑bit alphabet
    entropy_factor = 1.0 - lambda_entropy * H

    # 4. Regret term
    regret_factor = 1.0 - mu_regret * action.regret

    # 5. Fuse everything
    base = s * entropy_factor * regret_factor
    group_scores = w * base
    overall_score = float(group_scores.sum())
    return group_scores, overall_score


def aggregate_hybrid_scores(
    texts: List[str],
    reference_sig: List[int],
    groups: Tuple[str, ...] = GROUPS,
    today: date = date.today(),
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Process a collection of ``texts`` and return a mapping
    ``text -> group_score_vector`` using the hybrid scoring pipeline.
    """
    result: Dict[str, np.ndarray] = {}
    for txt in texts:
        grp_scores, _ = compute_hybrid_score(txt, reference_sig, groups, today, **kwargs)
        result[txt] = grp_scores
    return result


def select_texts_via_softmax(
    scores: Dict[str, np.ndarray],
    temperature: float = 1.0,
    rng: random.Random = random,
) -> List[Tuple[str, float]]:
    """
    Perform a softmax over the *overall* scores (sum across groups) and
    return a list of (text, probability) sorted descendingly.

    ``temperature`` controls the sharpness of the distribution.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    overall = np.array([v.sum() for v in scores.values()], dtype=np.float64)
    # Numerical stability
    max_val = overall.max()
    exp_vals = np.exp((overall - max_val) / temperature)
    probs = exp_vals / exp_vals.sum()
    sorted_idx = np.argsort(-probs)
    texts = list(scores.keys())
    return [(texts[i], float(_pct(probs[i]))) for i in sorted_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reference text (could be a policy document, etc.)
    reference = "The quick brown fox jumps over the lazy dog."
    ref_sig = signature(shingles(reference))

    sample_texts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How quickly daft jumping zebras vex.",
    ]

    # Compute hybrid scores
    group_score_map = aggregate_hybrid_scores(sample_texts, ref_sig)

    # Display per‑group scores
    for txt, vec in group_score_map.items():
        print(f"Text: {txt[:30]:<35} | Group scores: {vec}")

    # Choose texts according to softmax probabilities
    selection = select_texts_via_softmax(group_score_map, temperature=0.5)
    print("\nSelection probabilities (temperature=0.5):")
    for txt, prob in selection:
        print(f"{prob:6.2%} – {txt[:40]}")