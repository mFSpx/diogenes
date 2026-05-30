# DARWIN HAMMER — match 1495, survivor 2
# gen: 5
# parent_a: krampus_stickers.py (gen0)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s4.py (gen4)
# born: 2026-05-29T23:36:49Z

"""Hybrid Korpus-Stickers & MinHash Regret Engine

Parents:
- krampus_stickers.py (Korpus Text Math Helpers)
- hybrid_korpus_text_hybrid_hybrid_regret_m21_s4.py (MinHash‑based Regret Bandit)

Mathematical Bridge:
Both parents operate on token collections derived from raw text.  
Korpus provides deterministic tokenisation, whitespace normalisation,
token counting and Shannon entropy of the character stream.  
The Regret engine builds MinHash signatures from an *iterable of tokens*
and measures Jaccard similarity between signatures.

The hybrid algorithm therefore:
1. Uses Korpus utilities to obtain a clean token list and an entropy weight.
2. Feeds that token list into the MinHash pipeline to obtain a signature.
3. Combines the signature Jaccard similarity with the entropy weight and the
   action’s expected value / cost / risk to produce a unified “regret‑aware”
   score.

The following module implements this fusion and demonstrates three core
operations:
- `preprocess_text` – Korpus preprocessing + entropy.
- `minhash_signature` – MinHash generation from Korpus tokens.
- `hybrid_action_score` – Regret‑aware scoring using similarity and entropy.
"""

import re
import math
import random
import hashlib
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Dict, Any

import numpy as np
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Korpus Text Helpers (Parent A)
# ----------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    """Collapse all whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_list(text: str) -> List[str]:
    """Return a list of non‑whitespace tokens."""
    return re.findall(r"\S+", text or "")


def token_count(text: str) -> int:
    """Count tokens in the given text."""
    return len(token_list(text))


def shannon_entropy(seq: Iterable[Any]) -> float:
    """Compute Shannon entropy (base‑2) of an iterable of hashable items."""
    freq: Dict[Any, int] = {}
    total = 0
    for item in seq:
        freq[item] = freq.get(item, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def entropy_for_text(text: str) -> float:
    """Shannon entropy of the first 10 000 characters."""
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


# ----------------------------------------------------------------------
# MinHash & Regret Utilities (Parent B)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action descriptor used by the regret bandit."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping character shingles of given width."""
    if width <= 0:
        raise ValueError("width must be positive")
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Compute a MinHash signature from a set of tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return []
    hashes: List[int] = []
    for seed in range(k):
        hash_values = [_hash(seed, t) for t in toks]
        hashes.append(min(hash_values))
    return hashes


def jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    intersection = sum(1 for a, b in zip(sig1, sig2) if a == b)
    union = len(sig1)  # For MinHash, union ≈ signature length
    return intersection / union


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------


def preprocess_text(raw: str) -> Tuple[str, List[str], float]:
    """
    Apply Korpus preprocessing:
    - Normalise whitespace.
    - Tokenise.
    - Compute character‑level Shannon entropy.

    Returns a tuple of (clean_text, token_list, entropy).
    """
    clean = normalize_ws(raw)
    toks = token_list(clean)
    ent = entropy_for_text(clean)
    return clean, toks, ent


def minhash_signature_from_text(text: str, k: int = 128, shingle_width: int = 5) -> List[int]:
    """
    Build a MinHash signature directly from raw text.
    The text is first shingled (character level) and then each shingle
    is treated as a token for the MinHash algorithm.
    """
    shg = shingles(text, width=shingle_width)
    return signature(shg, k=k)


def hybrid_action_score(
    action: MathAction,
    reference_text: str,
    candidate_text: str,
    k: int = 128,
    shingle_width: int = 5,
    entropy_weight: float = 0.5,
) -> float:
    """
    Compute a regret‑aware score for `action` given a reference and a candidate text.

    Steps:
    1. Build MinHash signatures for both texts.
    2. Estimate Jaccard similarity.
    3. Retrieve entropy of the candidate (higher entropy → more informational content).
    4. Combine similarity, entropy, and the action's intrinsic values:

       score = (sim * (1 + entropy_weight * norm_entropy))
               + expected_value
               - cost
               - risk

    `norm_entropy` is the candidate entropy normalised to [0, 1] using a
    simple max‑entropy cap of 8 bits (typical for English characters).
    """
    # 1‑2: similarity via MinHash
    sig_ref = minhash_signature_from_text(reference_text, k=k, shingle_width=shingle_width)
    sig_cand = minhash_signature_from_text(candidate_text, k=k, shingle_width=shingle_width)
    sim = jaccard_similarity(sig_ref, sig_cand)

    # 3: entropy weighting
    _, _, cand_entropy = preprocess_text(candidate_text)
    max_entropy = 8.0  # log2(256) – upper bound for byte‑wise entropy
    norm_entropy = min(cand_entropy / max_entropy, 1.0)

    # 4: combine with action parameters
    score = (
        sim * (1.0 + entropy_weight * norm_entropy)
        + action.expected_value
        - action.cost
        - action.risk
    )
    return score


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def select_action_via_softmax(
    actions: List[MathAction],
    reference_text: str,
    candidate_texts: List[str],
    k: int = 128,
    shingle_width: int = 5,
) -> Tuple[MathAction, float]:
    """
    Score each action against its corresponding candidate text,
    apply a softmax over the scores, and randomly draw an action.

    Returns the chosen action and the probability assigned to it.
    """
    if len(actions) != len(candidate_texts):
        raise ValueError("actions and candidate_texts must have the same length")

    scores = np.array(
        [
            hybrid_action_score(
                act,
                reference_text,
                cand,
                k=k,
                shingle_width=shingle_width,
            )
            for act, cand in zip(actions, candidate_texts)
        ]
    )
    probs = softmax(scores)
    chosen_idx = np.random.choice(len(actions), p=probs)
    return actions[chosen_idx], float(probs[chosen_idx])


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal sanity check that runs without external data.
    ref = "The quick brown fox jumps over the lazy dog."
    candidates = [
        "A swift auburn animal leaps above a sleepy canine.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    actions = [
        MathAction(id="a1", expected_value=1.2, cost=0.1, risk=0.05),
        MathAction(id="a2", expected_value=0.8, cost=0.2, risk=0.10),
        MathAction(id="a3", expected_value=1.0, cost=0.05, risk=0.02),
    ]

    chosen, prob = select_action_via_softmax(actions, ref, candidates)
    print(f"Chosen action: {chosen.id} (probability {prob:.3f})")
    # Show individual scores for debugging
    for act, cand in zip(actions, candidates):
        sc = hybrid_action_score(act, ref, cand)
        print(f"Action {act.id} score: {sc:.4f}")
    sys.exit(0)