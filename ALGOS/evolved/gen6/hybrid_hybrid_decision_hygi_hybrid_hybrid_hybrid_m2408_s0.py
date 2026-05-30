# DARWIN HAMMER — match 2408, survivor 0
# gen: 6
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s3.py (gen5)
# born: 2026-05-29T23:42:06Z

"""Hybrid Decision Hygiene and Hybrid RBF Surrogate Module.

This module fuses the *Decision Hygiene* algorithm with the *Hybrid RBF Surrogate* 
calculation. The mathematical bridge is the **feature-count vector** produced 
by the hygiene regexes, which is used to compute the Gaussian RBF similarity 
matrix. The hygiene score is then multiplied by a factor that depends on 
the Gini coefficient of the similarity matrix, thus rewarding decisions 
that are both well-scored and information-rich.

The final hybrid score combines the benefits of both parents, providing a more 
comprehensive evaluation of decision-making cues.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Tuple
import numpy as np
import random

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
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
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panic|panicked)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return np.linalg.norm(a - b)

def rbf_similarity_matrix(
    features: List[List[float]],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Vectorised computation of the Gaussian RBF similarity matrix.

    Returns
    -------
    S : ndarray, shape (n, n)
        S[i, j] = exp(-epsilon^2 * ||x_i - x_j||^2)
    """
    X = np.asarray(features, dtype=float)          # (n, d)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T           # (n, n)
    np.clip(dists_sq, 0, None, out=dists_sq)                  # numerical safety
    return np.exp(-epsilon ** 2 * dists_sq)

def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini coefficient for a non‑negative 1‑D iterable.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    cum = np.cumsum(xs_sorted)
    gini = (n + 1 - 2 * np.sum(cum) / xs_sorted.sum()) / n
    return float(gini)

def extract_features(text: str) -> List[float]:
    """
    Extract features from the input text using the hygiene regexes.
    """
    features = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
        len(OUTCOME_RE.findall(text)),
        len(IMPULSIVE_RE.findall(text)),
    ]
    return features

def compute_hygiene_score(features: List[float]) -> float:
    """
    Compute the hygiene score using the feature-count vector.
    """
    weights = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    score = sum(f * w for f, w in zip(features, weights))
    return score

def compute_hybrid_score(text: str) -> float:
    """
    Compute the hybrid score by combining the hygiene score and the Gini coefficient.
    """
    features = extract_features(text)
    hygiene_score = compute_hygiene_score(features)
    similarity_matrix = rbf_similarity_matrix([features])
    gini = gini_coefficient(similarity_matrix.flatten())
    hybrid_score = hygiene_score * (1 + gini)
    return hybrid_score

if __name__ == "__main__":
    text = "This is a test text with some evidence and planning."
    hybrid_score = compute_hybrid_score(text)
    print(hybrid_score)