# DARWIN HAMMER — match 17, survivor 5
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:22:55Z

import re
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union

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


def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
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
    total = vector.sum()
    if total == 0:
        return 0.0
    probs = vector[vector > 0].astype(float) / float(total)
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Parent B – exponential pruning schedule
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decay pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(
    edges: List,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    seed: Union[int, str, None] = None,
) -> List:
    """Return a subset of *edges* that survives stochastic pruning."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


# ----------------------------------------------------------------------
# Enhanced Fusion – deeper mathematical integration
# ----------------------------------------------------------------------
def entropy_normalisation_factor(vector: np.ndarray) -> float:
    """
    Compute γ(v) = 1 + H(v) / H_max(v), where H_max is the entropy of a uniform
    distribution over the *k* non‑zero categories of *v*.
    """
    k = np.count_nonzero(vector)
    if k <= 1:
        return 1.0
    h = shannon_entropy(vector)
    h_max = math.log2(k)
    return 1.0 + (h / h_max)


def entropy_deviation_factor(vector: np.ndarray) -> float:
    """
    KL‑divergence‑like deviation from uniformity.
    Returns a factor ≥ 1 that grows as the distribution becomes more peaked.
    """
    k = np.count_nonzero(vector)
    if k <= 1:
        return 1.0
    probs = vector[vector > 0].astype(float) / float(vector.sum())
    uniform = 1.0 / k
    # Using base‑2 log to keep units in bits
    kl = float(np.sum(probs * np.log2(probs / uniform)))
    return 1.0 + kl  # 1 when uniform, >1 otherwise


def hybrid_prune_probability(
    t: float,
    vector: np.ndarray,
    lam: float = 1.0,
    alpha: float = 0.2,
    beta: float = 0.5,
) -> float:
    """
    Entropy‑aware pruning probability.

    The base exponential decay is first modulated by a *entropy boost*:
        λ_eff = λ * (1 + β * H/H_max)

    Then the result is attenuated by the normalisation factor γ(v) to keep
    probabilities bounded.
    """
    base = prune_probability(t, lam, alpha)
    # Entropy boost – higher entropy lifts the effective λ, encouraging
    # preservation of information‑rich inputs.
    gamma = entropy_normalisation_factor(vector)
    h = shannon_entropy(vector)
    k = np.count_nonzero(vector)
    h_max = math.log2(k) if k > 1 else 1.0
    lam_eff = lam * (1.0 + beta * (h / h_max))
    boosted = min(1.0, lam_eff * math.exp(-alpha * t))
    return min(1.0, boosted / gamma)


def hybrid_score(
    text: str,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    beta: float = 0.5,
    delta: float = 0.3,
) -> Tuple[float, str]:
    """
    Compute a deeper hybrid decision‑hygiene score.

    Steps:
    1. Build the feature vector.
    2. Compute the raw hygiene score S and its label.
    3. Derive entropy‑based factors:
       • γ(v) – normalisation (used inside pruning)
       • η(v) – deviation from uniformity (KL‑like)
    4. Compute entropy‑adjusted prune probability p_h.
    5. Blend the raw score with both pruning and an entropy reward term:
          S_h = S * (1 - p_h) * (1 + δ * (H / H_max) * η(v))

    The extra multiplicative term rewards high‑entropy, uniformly‑distributed
    observations, making the fusion mathematically richer.
    """
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)

    # Entropy metrics
    h = shannon_entropy(vec)
    k = np.count_nonzero(vec)
    h_max = math.log2(k) if k > 1 else 1.0
    gamma = entropy_normalisation_factor(vec)
    eta = entropy_deviation_factor(vec)

    # Entropy‑aware prune probability
    p_h = hybrid_prune_probability(t, vec, lam, alpha, beta)

    # Entropy reward factor – bounded between 1 and 1+δ
    reward = 1.0 + delta * (h / h_max) * eta

    hybrid = raw_score * (1.0 - p_h) * reward
    return hybrid, label


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We need evidence and a solid plan. "
        "Please verify the source, record the log, and keep the documentation. "
        "Avoid impulsive actions and consider the risk."
    )
    score, lbl = hybrid_score(sample_text, t=3.0)
    print(f"Hybrid score: {score:.2f}, label: {lbl}")