# DARWIN HAMMER — match 1940, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_decisi_m153_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m530_s1.py (gen5)
# born: 2026-05-29T23:39:52Z

"""Hybrid algorithm combining linguistic feature extraction (Parent A) with
minhash signature similarity and regret‑weighted scoring (Parent B).

Mathematical bridge:
- Parent A extracts a set of lexical tokens from a text and scores the presence
  of evidential keywords via a regex.  The token set is a discrete representation
  of the linguistic content.
- Parent B converts any iterable of tokens into a fixed‑size minhash signature
  (function `signature`) and measures similarity between two signatures
  (`similarity`).  This provides a quantitative, vector‑based distance that can
  be combined with other scalar scores.
- The hybrid algorithm maps the token set from A into B’s signature space,
  uses the signature similarity as the core similarity metric, and modulates
  it with the evidential weight from A.  The combined scalar is then passed
  through the regret‑weighted sigmoid function from B to obtain a final
  decision score.

The implementation below provides three core functions that demonstrate this
fusion and a simple smoke test.
"""

import math
import random
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

import numpy as np

# -------------------- Parent A components --------------------

FUNCTION_CATS = {
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

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

def _tokenize(text: str) -> List[str]:
    """Basic whitespace/token punctuation tokeniser used by Parent A."""
    # Lower‑case and split on non‑word characters
    return [t.lower() for t in re.findall(r"\b\w+\b", text)]

def extract_linguistic_tokens(text: str) -> Set[str]:
    """
    Extract a set of tokens that belong to any FUNCTION_CATS category.
    This mirrors Parent A's linguistic feature extraction.
    """
    tokens = _tokenize(text)
    selected = {tok for tok in tokens if any(tok in cat for cat in FUNCTION_CATS.values())}
    return selected

def evidential_weight(text: str) -> float:
    """
    Compute a scalar weight based on the number of evidential keywords
    found in the text (Parent A's regex‑based evidence scoring).
    """
    count = len(EVIDENCE_RE.findall(text))
    # Use a smooth increasing function; log(1+count) avoids zero division
    return math.log1p(count)


# -------------------- Parent B components --------------------

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(np.uint8(np.frombuffer(data, dtype=np.uint8)), dtype=np.uint8).tobytes(),
        "big",
    )  # Simplified deterministic hash for illustration


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """
    Min‑hash style signature of a token set (Parent B).
    Returns a list of k integers where each entry is the minimum hash
    across all tokens for a given seed.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Jaccard‑like similarity between two min‑hash signatures (Parent B).
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (Parent B)."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


@dataclass(frozen=True)
class MathAction:
    """Action descriptor used by Parent B."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# -------------------- Hybrid core functions --------------------

def text_signature(text: str, k: int = 128) -> List[int]:
    """
    Convert raw text into a min‑hash signature.
    The token extraction uses Parent A's linguistic filter, providing the
    mathematical bridge between the two parents.
    """
    tokens = extract_linguistic_tokens(text)
    return signature(tokens, k=k)


def hybrid_similarity(text_a: str, text_b: str, k: int = 128) -> float:
    """
    Compute similarity between two texts by:
    1. Mapping each text to a min‑hash signature (bridge).
    2. Measuring signature overlap (Parent B).
    """
    sig_a = text_signature(text_a, k=k)
    sig_b = text_signature(text_b, k=k)
    return similarity(sig_a, sig_b)


def regret_weighted_score(
    action: MathAction,
    sim: float,
    evid_weight: float,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """
    Combine similarity, evidential weight and action attributes into a
    regret‑weighted decision score.

    Formula (invented for the hybrid):
        raw = alpha * sim * evid_weight - beta * (action.cost + action.risk)
        score = sigmoid(raw) * action.expected_value

    The sigmoid is the regret‑weighted transformation from Parent B.
    """
    raw = alpha * sim * evid_weight - beta * (action.cost + action.risk)
    # sigmoid expects an array; we use a single‑element array for convenience
    prob = float(sigmoid(np.array([raw]))[0])
    return prob * action.expected_value


def hybrid_decision(text_a: str, text_b: str, action: MathAction) -> Tuple[float, float]:
    """
    End‑to‑end hybrid decision routine.
    Returns a tuple (similarity, final_score).
    """
    sim = hybrid_similarity(text_a, text_b)
    ev_weight = (evidential_weight(text_a) + evidential_weight(text_b)) / 2.0
    final = regret_weighted_score(action, sim, ev_weight)
    return sim, final


# -------------------- Smoke test --------------------
if __name__ == "__main__":
    sample_a = (
        "The report includes verified evidence from the source and a "
        "screenshot of the log file confirming the breach."
    )
    sample_b = (
        "We have a citation that confirms the fact, plus a SHA256 hash "
        "as proof of authenticity."
    )
    test_action = MathAction(id="audit", expected_value=0.85, cost=0.1, risk=0.05)

    similarity_score, decision_score = hybrid_decision(sample_a, sample_b, test_action)

    print(f"Hybrid similarity (signature‑based): {similarity_score:.4f}")
    print(f"Hybrid decision score: {decision_score:.4f}")