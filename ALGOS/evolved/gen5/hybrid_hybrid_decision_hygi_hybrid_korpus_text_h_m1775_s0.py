# DARWIN HAMMER — match 1775, survivor 0
# gen: 5
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s3.py (gen1)
# parent_b: hybrid_korpus_text_hybrid_hybrid_regret_m21_s7.py (gen4)
# born: 2026-05-29T23:38:43Z

"""
Hybrid Algorithm: Decision Hygiene meets MinHash and Shannon Entropy

This module fuses the governing equations of hybrid_decision_hygiene_shannon_entropy_m12_s3.py 
and hybrid_korpus_text_hybrid_hybrid_regret_m21_s7.py. The mathematical bridge between the two 
parents lies in the use of Shannon entropy to weigh the importance of features in the decision 
hygiene algorithm. The MinHash signature is used to generate a compact representation of text 
data, which is then used to compute the Shannon entropy.

Parents:
- hybrid_decision_hygiene_shannon_entropy_m12_s3.py
- hybrid_korpus_text_hybrid_hybrid_regret_m21_s7.py
"""

import math
import random
import sys
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterable

# Constants and utilities
INT16_MAX = 2 ** 15 - 1

def _shingles(text: str, width: int = 5) -> List[str]:
    """Return a list of overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i : i + width] for i in range(len(cleaned) - width + 1)]


def _hash_token(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of *token* mixed with *seed*."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Compute a MinHash signature of length *k* for the given *tokens*."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k
    # generate k independent seeds
    seeds = [random.randrange(0, 2 ** 32) for _ in range(k)]
    signature = []
    for seed in seeds:
        min_hash = min(_hash_token(seed, t) for t in token_set)
        signature.append(min_hash)
    return signature


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Parent‑A style helper: MinHash signature of a raw text string."""
    return minhash_signature(_shingles(text or ""), width=5, k=k)  # type: ignore[arg-type]


def shannon_entropy(chars: List[str]) -> float:
    """Simple Shannon entropy over a list of characters."""
    if not chars:
        return 0.0
    prob = {}
    for c in chars:
        prob[c] = prob.get(c, 0) + 1
    total = len(chars)
    return -sum((count / total) * math.log2(count / total) for count in prob.values())


def entropy_for_text(text: str) -> float:
    """Parent‑A helper: entropy of the first 10 000 characters."""
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


# Decision Hygiene features and weights
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

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


def compute_feature_entropy(text: str, feature: str) -> float:
    """Compute the Shannon entropy of a feature in the given text."""
    if feature == "evidence":
        matches = EVIDENCE_RE.findall(text)
    elif feature == "planning":
        matches = PLANNING_RE.findall(text)
    elif feature == "delay":
        matches = DELAY_RE.findall(text)
    elif feature == "support":
        matches = SUPPORT_RE.findall(text)
    elif feature == "boundary":
        matches = BOUNDARY_RE.findall(text)
    elif feature == "outcome":
        matches = OUTCOME_RE.findall(text)
    elif feature == "impulsive":
        matches = IMPULSIVE_RE.findall(text)
    elif feature == "scarcity":
        matches = SCARCITY_RE.findall(text)
    elif feature == "risk":
        matches = RISK_RE.findall(text)
    else:
        return 0.0
    return shannon_entropy(matches)


def hybrid_entropy(text: str) -> float:
    """Compute the hybrid entropy of the given text."""
    entropy = 0.0
    for i, feature in enumerate(_FEATURE_ORDER):
        if feature in ["evidence", "planning", "delay", "support", "boundary", "outcome"]:
            weight = _POSITIVE_WEIGHTS[i]
        else:
            weight = _NEGATIVE_WEIGHTS[i - 6] if i >= 6 else 0
        feature_entropy = compute_feature_entropy(text, feature)
        entropy += weight * feature_entropy
    return entropy


def hybrid_min_hash(text: str, k: int = 64) -> List[int]:
    """Compute the MinHash signature of the given text."""
    return minhash_for_text(text, k=k)


if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    print(hybrid_entropy(text))
    print(hybrid_min_hash(text))