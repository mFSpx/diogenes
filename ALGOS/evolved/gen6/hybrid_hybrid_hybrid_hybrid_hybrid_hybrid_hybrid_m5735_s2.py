# DARWIN HAMMER — match 5735, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m1019_s1.py (gen5)
# born: 2026-05-30T00:04:35Z

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (Parent A)
# ----------------------------------------------------------------------
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


def extract_regex_features(text: str) -> Dict[str, int]:
    """Count occurrences of each regex category in *text*."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
    }


# ----------------------------------------------------------------------
# Stylometry utilities (Parent B – simplified)
# ----------------------------------------------------------------------
def stylometry_features(text: str) -> Dict[str, float]:
    """
    Very lightweight stylometry: return a dict with five numeric descriptors.
    - total_words: number of whitespace‑separated tokens
    - unique_words: number of distinct tokens (case‑insensitive)
    - avg_word_len: average token length
    - sentence_count: number of sentence‑ending punctuation marks (., !, ?)
    - punct_count: total punctuation characters
    """
    words = re.findall(r"\b\w+\b", text)
    total_words = len(words)
    unique_words = len(set(w.lower() for w in words))
    avg_word_len = sum(len(w) for w in words) / total_words if total_words else 0.0
    sentence_count = len(re.findall(r"[.!?]", text))
    punct_count = len(re.findall(r"[^\w\s]", text))

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_word_len": avg_word_len,
        "sentence_count": sentence_count,
        "punct_count": punct_count,
    }


# ----------------------------------------------------------------------
# Data structure for certainty output
# ----------------------------------------------------------------------
@dataclass
class CertaintyFlag:
    confidence_bps: int  # basis points (0‑10000)
    label: str           # qualitative label


def _label_from_confidence(conf: int) -> str:
    """Map confidence to a qualitative label."""
    if conf >= 7500:
        return "high"
    if conf >= 5000:
        return "medium"
    if conf >= 2500:
        return "low"
    return "very low"


# ----------------------------------------------------------------------
# Core hybrid pipeline
# ----------------------------------------------------------------------
def _vector_from_features(regex_feat: Dict[str, int],
                          styl_feat: Dict[str, float]) -> np.ndarray:
    """Concatenate regex counts and stylometry numbers into a single vector."""
    regex_vec = np.array([regex_feat[k] for k in ("evidence", "planning",
                                                 "delay", "support", "boundary")],
                         dtype=float)
    styl_vec = np.array([styl_feat[k] for k in ("total_words", "unique_words",
                                                "avg_word_len", "sentence_count",
                                                "punct_count")],
                        dtype=float)
    return np.concatenate([regex_vec, styl_vec])


def _cosine_similarity_matrix(vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise cosine similarity matrix W and node strengths w.

    Returns:
        W – (n, n) symmetric matrix with cosine similarities (range [0,1]).
        w – (n,) array of node strengths (ℓ₁‑norm of each vector).
    """
    n = len(vectors)
    W = np.zeros((n, n), dtype=float)
    strengths = np.zeros(n, dtype=float)

    for i, vi in enumerate(vectors):
        strengths[i] = np.sum(np.abs(vi))
        norm_i = np.linalg.norm(vi)
        for j in range(i, n):
            vj = vectors[j]
            norm_j = np.linalg.norm(vj)
            if norm_i == 0.0 or norm_j == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(vi, vj) / (norm_i * norm_j))
            W[i, j] = sim
            W[j, i] = sim
    return W, strengths


def curvature_to_certainty(
    W: np.ndarray,
    strengths: np.ndarray,
    epsilon: float = 1e-12,
) -> Dict[Tuple[int, int], CertaintyFlag]:
    """
    Compute Ollivier‑Ricci curvature on each edge using the surrogate
    κ_{ij} = 1 - |w_i - w_j| / (w_i + w_j + ε) and map it to a CertaintyFlag.

    The result dictionary is keyed by (i, j) with i < j.
    """
    n = W.shape[0]
    certainty: Dict[Tuple[int, int], CertaintyFlag] = {}

    for i in range(n):
        wi = strengths[i]
        for j in range(i + 1, n):
            wj = strengths[j]
            curvature = 1.0 - abs(wi - wj) / (wi + wj + epsilon)
            # Clamp curvature to [-1, 1] to guard against numerical issues
            curvature = max(-1.0, min(1.0, curvature))
            confidence = int(((curvature + 1.0) / 2.0) * 10000)
            certainty[(i, j)] = CertaintyFlag(confidence, _label_from_confidence(confidence))

    return certainty


def hybrid_confidence(texts: List[str]) -> Dict[Tuple[int, int], CertaintyFlag]:
    """
    End‑to‑end hybrid operation:
    1. Extract regex and stylometry features for each document.
    2. Build concatenated feature vectors.
    3. Form cosine similarity matrix and node strengths.
    4. Compute curvature‑based certainty flags for every unordered
    document pair.
    """
    vectors = []
    for text in texts:
        regex_feat = extract_regex_features(text)
        styl_feat = stylometry_features(text)
        vectors.append(_vector_from_features(regex_feat, styl_feat))

    W, strengths = _cosine_similarity_matrix(vectors)
    return curvature_to_certainty(W, strengths)


def main():
    texts = ["This is a test.", "Another test.", "Is this a test?"]
    certainty_flags = hybrid_confidence(texts)
    for pair, flag in certainty_flags.items():
        print(f"Pair {pair}: Confidence {flag.confidence_bps} ({flag.label})")


if __name__ == "__main__":
    main()