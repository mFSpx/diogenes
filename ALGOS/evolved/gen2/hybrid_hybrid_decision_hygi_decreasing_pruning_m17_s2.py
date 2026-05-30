# DARWIN HAMMER — match 17, survivor 2
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:22:55Z

"""Hybrid Decision Hygiene & Entropy Pruning Module

Parents:
- hybrid_decision_hygiene_shannon_entropy_m12_s4.py (Decision Hygiene + Shannon Entropy)
- decreasing_pruning.py (Exponential pruning schedule)

Mathematical Bridge:
The decision‑hygiene algorithm produces a weighted linear score **S** from a
9‑dimensional feature count vector **v**.  The Shannon‑entropy routine treats the
same vector as a discrete distribution and yields an entropy **H(v)** (bits).
Both quantities are scalars derived from the same underlying observation vector.
We fuse them by letting the entropy modulate the pruning probability **p(t)**
from the decreasing‑pruning schedule:


p(t) = min(1, λ·exp(-α·t))
γ(v) = 1 + H(v) / H_max(v)          # entropy normalisation factor ≥ 1
p_hybrid(t, v) = p(t) / γ(v)        # higher entropy → lower effective prune prob.


The hybrid score **S_h** combines the original hygiene score with the
entropy‑adjusted pruning probability:


S_h = S * (1 - p_hybrid(t, v))


Thus, when the observed text is information‑rich (high entropy) the algorithm
prunes less aggressively and preserves more of the hygiene contribution;
conversely, low‑entropy (repetitive) inputs are pruned more heavily.

The module implements:
1. Feature extraction and vectorisation (parent A).
2. Shannon entropy calculation (parent A).
3. Exponential pruning probability (parent B).
4. Hybrid functions that intertwine the two mathematical structures.
"""

import re
import math
import random
import sys
from pathlib import Path
from collections import Counter
import numpy as np

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

def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using the parent‑A regexes."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def feature_vector(text: str) -> np.ndarray:
    """Return a 9‑element numpy array of counts ordered as _FEATURE_ORDER."""
    c = _raw_counts(text)
    return np.array(
        [
            c["evidence_count"],
            c["planning_count"],
            c["delay_count"],
            c["support_count"],
            c["boundary_count"],
            c["outcome_count"],
            c["impulsive_count"],
            c["scarcity_count"],
            c["risk_count"],
        ],
        dtype=np.int64,
    )

def hygiene_score(vector: np.ndarray) -> tuple[int, str]:
    """Original decision‑hygiene score and textual label."""
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))

    risk_present = any(vector[idx] > 0 for idx in (6, 7, 8))
    if risk_present and raw_score < 2500:
        label = "critical_risk_or_pain_signal"
    elif raw_score >= 7000:
        label = "high_decision_hygiene"
    elif raw_score >= 3000:
        label = "improving_decision_hygiene"
    elif raw_score <= -2500:
        label = "strained_decision_context"
    else:
        label = "neutral_or_unclear"
    return raw_score, label

def shannon_entropy(vector: np.ndarray) -> float:
    """Shannon entropy (bits) of the non‑zero entries of *vector*."""
    if vector.sum() == 0:
        return 0.0
    probs = vector[vector > 0].astype(float) / float(vector.sum())
    return -float(np.sum(probs * np.log2(probs)))

# ----------------------------------------------------------------------
# Parent B – exponential pruning schedule
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decay pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: list, t: float, lam: float = 1.0, alpha: float = 0.2,
                seed: int | str | None = None) -> list:
    """Return a subset of *edges* that survives stochastic pruning."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------
def entropy_normalisation_factor(vector: np.ndarray) -> float:
    """
    Compute γ(v) = 1 + H(v) / H_max(v), where H_max is the entropy of a uniform
    distribution over the *k* non‑zero categories of *v*.
    """
    k = np.count_nonzero(vector)
    if k <= 1:
        return 1.0  # entropy zero → factor 1
    h = shannon_entropy(vector)
    h_max = math.log2(k)
    return 1.0 + (h / h_max)

def hybrid_prune_probability(t: float, vector: np.ndarray,
                             lam: float = 1.0, alpha: float = 0.2) -> float:
    """
    Entropy‑adjusted pruning probability:
        p_hybrid = p(t) / γ(v)
    Higher entropy reduces the effective prune chance.
    """
    base_p = prune_probability(t, lam, alpha)
    gamma = entropy_normalisation_factor(vector)
    return min(1.0, base_p / gamma)

def hybrid_score(text: str, t: float,
                 lam: float = 1.0, alpha: float = 0.2) -> tuple[float, str]:
    """
    Compute a hybrid decision‑hygiene score that incorporates entropy‑aware
    pruning. Returns (S_h, label) where S_h = S * (1 - p_hybrid).
    """
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)
    p_hybrid = hybrid_prune_probability(t, vec, lam, alpha)
    hybrid = raw_score * (1.0 - p_hybrid)
    return hybrid, label

def prune_feature_vector(vec: np.ndarray, t: float,
                         lam: float = 1.0, alpha: float = 0.2,
                         seed: int | str | None = None) -> np.ndarray:
    """
    Treat each non‑zero feature count as an "edge" with weight equal to the count.
    Apply entropy‑aware pruning to possibly zero‑out some features.
    """
    indices = np.nonzero(vec)[0].tolist()
    # Convert to list of (index, count) tuples for deterministic handling
    edges = [(i, int(vec[i])) for i in indices]
    pruned = prune_edges(edges, t, lam, alpha, seed)
    new_vec = np.zeros_like(vec)
    for idx, cnt in pruned:
        new_vec[idx] = cnt
    return new_vec

def decision_hybrid(text: str, t: float,
                    lam: float = 1.0, alpha: float = 0.2,
                    seed: int | str | None = None) -> dict:
    """
    End‑to‑end hybrid pipeline:
    1. Extract raw feature vector.
    2. Apply entropy‑aware pruning.
    3. Compute hygiene score on the pruned vector.
    4. Return a dictionary with all intermediate values.
    """
    original_vec = feature_vector(text)
    pruned_vec = prune_feature_vector(original_vec, t, lam, alpha, seed)
    raw_score, raw_label = hygiene_score(pruned_vec)
    entropy = shannon_entropy(pruned_vec)
    prob = hybrid_prune_probability(t, pruned_vec, lam, alpha)
    return {
        "original_vector": original_vec,
        "pruned_vector": pruned_vec,
        "raw_score": raw_score,
        "raw_label": raw_label,
        "entropy_bits": entropy,
        "effective_prune_probability": prob,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have evidence that the plan is solid, but we need to verify the "
        "source and get a screenshot.  The deadline is tomorrow, so delay is "
        "not an option.  I will ask a friend for support.  No risk is present."
    )
    t_now = 2.5  # arbitrary time step

    # Hybrid score demonstration
    h_score, h_label = hybrid_score(sample_text, t_now)
    print(f"Hybrid score: {h_score:.2f}, label: {h_label}")

    # Full decision pipeline
    result = decision_hybrid(sample_text, t_now, seed=42)
    print("\nDecision pipeline output:")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        print(f"{k}: {v}")

    # Ensure the module runs without external files
    sys.exit(0)