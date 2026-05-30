# DARWIN HAMMER — match 2236, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_model__m1151_s2.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s2.py (gen3)
# born: 2026-05-29T23:41:26Z

"""Hybrid Decision-Hoeffding-Tropical Model
Parents:
- hybrid_hybrid_hoeffding_tre_hybrid_hybrid_model__m1151_s2.py (Hoeffding bound + tropical algebra)
- hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s2.py (Regex‑based decision hygiene + privacy/resource linear filter)

Mathematical bridge:
The regex‑derived hygiene scores form a real‑valued feature vector **x** for each
incoming instance.  This vector is interpreted as the tropical variable of a
tropical polynomial  p(x)=max_i (c_i + ⟨e_i, x⟩)  where the coefficients **c_i**
are learned/initialized weights.  The polynomial value acts as a “gain’’ measure
for a Hoeffding‑bound split test.  The split decision therefore couples the
statistical Hoeffding test (parent A) with the content‑driven hygiene features
(parent B).  After a split decision, a privacy‑aware linear budget filter (parent
B) selects a subset of entities using the same hygiene scores, completing the
fusion of both topologies.
"""

import math
import random
import sys
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Hoeffding bound & tropical algebra utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    """Tropical polynomial evaluation: max_i (coeff_i + i * x)."""
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    # Broadcast to match dimensions of x
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def relu_layer_to_tropical(W, b):
    """Identity conversion – kept for API compatibility."""
    return np.asarray(W, dtype=float), np.asarray(b, dtype=float)

# ----------------------------------------------------------------------
# Parent B – Decision hygiene regexes & privacy/resource filter
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
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|bankrupt|poverty)\b", re.I)

_REGEX_LIST = [
    EVIDENCE_RE,
    PLANNING_RE,
    DELAY_RE,
    SUPPORT_RE,
    BOUNDARY_RE,
    OUTCOME_RE,
    IMPULSIVE_RE,
    SCARCITY_RE,
]

def compute_hygiene_scores(texts):
    """Return an (N, 8) array of regex match counts for each input string."""
    scores = np.zeros((len(texts), len(_REGEX_LIST)), dtype=float)
    for i, txt in enumerate(texts):
        for j, regex in enumerate(_REGEX_LIST):
            scores[i, j] = len(regex.findall(txt))
    return scores

def privacy_resource_filter(entity_ids, scores, privacy_budget, resource_budget):
    """
    Greedy filter that selects a subset of entities whose summed scores
    respect two linear budgets:
        sum(scores[:,0]) <= resource_budget   (e.g., computational cost)
        sum(scores[:,1]) <= privacy_budget    (e.g., privacy‑sensitive signals)
    Returns the list of selected entity ids.
    """
    idx = np.argsort(-scores[:, 0])  # prioritize high‑resource‑score entities
    selected = []
    res_used = 0.0
    priv_used = 0.0
    for i in idx:
        r = scores[i, 0]
        p = scores[i, 1]
        if res_used + r <= resource_budget and priv_used + p <= privacy_budget:
            selected.append(entity_ids[i])
            res_used += r
            priv_used += p
    return selected

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def tropical_gain_from_hygiene(scores, coeff_matrix):
    """
    Evaluate a set of tropical polynomials (one per column of coeff_matrix)
    on the hygiene score vectors.
    - scores: (N, D) where D = number of hygiene features (8)
    - coeff_matrix: (K, D+1) where each row holds (c0, c1, ..., cD)
      interpreted as coefficients of a tropical polynomial:
          p(x) = max_j (c_j + j * x)  with j ranging over dimensions.
    Returns an (N, K) array of polynomial values.
    """
    N, D = scores.shape
    K = coeff_matrix.shape[0]
    # Broadcast to compute c_j + j * x for each dimension j
    exponents = np.arange(D, dtype=float)  # shape (D,)
    # coeff_matrix[:, 1:] corresponds to per‑dimension coefficients
    base = coeff_matrix[:, 0][:, np.newaxis]  # (K,1)
    per_dim = coeff_matrix[:, 1:]  # (K,D)
    # Compute term = base + per_dim * scores.T  (using tropical addition = max, multiplication = +)
    # We use t_matmul which implements max(a_i + b_j)
    term = t_matmul(per_dim, scores.T)  # shape (K, N)
    # Add the constant term (base) via tropical addition (max)
    gain = t_add(term, base)  # shape (K, N)
    return gain.T  # (N, K)

def hybrid_split_decision(hygiene_scores, r=0.1, delta=0.05, n_instances=100):
    """
    Perform a Hoeffding split test where the gain is derived from tropical
    evaluation of hygiene scores.
    Returns a SplitDecision object.
    """
    # Randomly initialise a small set of tropical coefficient rows
    K = 5  # number of candidate polynomials
    D = hygiene_scores.shape[1]
    coeff_matrix = np.random.uniform(-1, 1, size=(K, D + 1))
    gains = tropical_gain_from_hygiene(hygiene_scores, coeff_matrix)  # (N, K)
    # For each instance we take the best and second‑best gain
    best_gains = np.max(gains, axis=1)
    # Zero out the best column to find the second best
    mask = gains == best_gains[:, np.newaxis]
    gains_masked = np.where(mask, -np.inf, gains)
    second_best = np.max(gains_masked, axis=1)
    # Aggregate over the stream (mean gain) to feed Hoeffding test
    best_mean = float(np.mean(best_gains))
    second_mean = float(np.mean(second_best))
    return should_split(best_mean, second_mean, r, delta, n_instances)

def hybrid_process(text_stream, privacy_budget=10.0, resource_budget=15.0):
    """
    Full pipeline:
    1. Compute hygiene scores for each incoming text.
    2. Decide whether to split the stream using Hoeffding + tropical gain.
    3. If a split is decided, apply the privacy/resource filter to obtain
       a subset of texts that respect the budgets.
    Returns a tuple (split_decision, selected_texts).
    """
    scores = compute_hygiene_scores(text_stream)
    split_dec = hybrid_split_decision(scores, n_instances=len(text_stream))
    selected = []
    if split_dec.should_split:
        entity_ids = list(range(len(text_stream)))
        selected_ids = privacy_resource_filter(entity_ids, scores,
                                               privacy_budget, resource_budget)
        selected = [text_stream[i] for i in selected_ids]
    return split_dec, selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy_texts = [
        "The evidence was verified and the source is reliable.",
        "We need to plan the roadmap and schedule the next steps.",
        "Please wait for the report, it will be ready tomorrow.",
        "Contact the doctor for support and advice.",
        "Do not share this, privacy is important.",
        "The task is completed and shipped.",
        "I'm angry, I want to act right now!",
        "I'm broke and cannot afford the service."
    ]
    decision, kept = hybrid_process(dummy_texts, privacy_budget=5.0, resource_budget=12.0)
    print("Split decision:", decision)
    print("Selected texts after filter:")
    for t in kept:
        print("-", t)