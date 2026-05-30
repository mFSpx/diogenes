# DARWIN HAMMER — match 2259, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1136_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s0.py (gen5)
# born: 2026-05-29T23:41:42Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Match 1136 (Regret & Fisher) and Match 607 (Stylometry & NLMS)

This module mathematically bridges the two parent algorithms by using
*stylometry feature vectors* extracted from free‑form text to modulate the
*Fisher information* weighting inside the regret‑weighted strategy of parent A,
and by using *MinHash‑style signatures* from parent A to scale the learning
rate of an NLMS (Normalized Least‑Mean‑Squares) adaptive filter derived from
parent B.  The result is a unified system where linguistic patterns directly
influence both decision‑theoretic action selection and adaptive signal
processing.

The core mathematical interface consists of:
1. `stylometry_vector` → a normalized probability vector **s** ∈ ℝ^d.
2. `fisher_score(θ, center, width)` → Fisher information **I(θ)**.
3. A scaling factor α = 1 + ⟨s, w_f⟩ where **w_f** is a fixed weight vector,
   applied multiplicatively to the Fisher score.
4. `signature(tokens, k)` produces a MinHash sketch **σ**; its Jaccard‑like
   similarity `similarity(σ₁, σ₂)` yields a factor β ∈ [0,1] that modulates the
   NLMS step size.

The three public functions below showcase the hybrid operation:
* `extract_stylometry_features(text)` – builds the linguistic vector **s**.
* `hybrid_regret_weighted_strategy(actions, counterfactuals, text, ...)` –
   uses α·I(θ) as the regret weighting before a soft‑max.
* `adaptive_nlms_update(w, x, d, lr, ref_text, cur_text)` – updates NLMS weights
   using β from signature similarity between a reference and current text.

All components rely only on the Python standard library and NumPy. """


import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities from Parent A
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash‑style sketch of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash sketches."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)
    return exp_vals / exp_vals.sum()


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Stylometry utilities from Parent B
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
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
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _tokenize(text: str) -> List[str]:
    """Very light tokenizer: split on whitespace and strip punctuation."""
    tokens = []
    for raw in text.lower().split():
        token = raw.strip(PUNCT)
        if token:
            tokens.append(token)
    return tokens


def extract_stylometry_features(text: str) -> Dict[str, float]:
    """
    Compute normalized frequencies for each functional category defined in
    ``FUNCTION_CATS``.  The returned dict maps category name → frequency (sum to 1).
    """
    tokens = _tokenize(text)
    total = len(tokens) or 1
    cat_counts = {cat: 0 for cat in FUNCTION_CATS}
    for token in tokens:
        for cat, vocab in FUNCTION_CATS.items():
            if token in vocab:
                cat_counts[cat] += 1
    # Normalize to probabilities
    return {cat: cnt / total for cat, cnt in cat_counts.items()}


# ----------------------------------------------------------------------
# Data structures shared between both parents
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass
class TreeNode:
    name: str
    size: int
    prior_probability: float


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def hybrid_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    text: str,
    temperature: float = 1.0,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
) -> Dict[str, float]:
    """
    Compute a probability distribution over ``actions`` using regret weighting.
    The Fisher information for each action is scaled by a factor derived from
    stylometry features of ``text`` (the linguistic bridge).

    Returns:
        dict mapping action.id → probability (soft‑maxed).
    """
    # 1️⃣ Stylometry vector → scalar scaling α
    stylometry = extract_stylometry_features(text)
    # Fixed weights for categories (could be learned; here we use uniform)
    w_f = np.ones(len(stylometry))
    s_vec = np.array(list(stylometry.values()))
    alpha = 1.0 + float(np.dot(s_vec, w_f) / len(w_f))  # α ≥ 1

    # 2️⃣ Build a lookup for counterfactual expectations per action
    cf_by_action: Dict[str, List[MathCounterfactual]] = {}
    for cf in counterfactuals:
        cf_by_action.setdefault(cf.action_id, []).append(cf)

    raw_scores = []
    for act in actions:
        # Expected outcome under counterfactual distribution
        cfs = cf_by_action.get(act.id, [])
        if cfs:
            weighted_outcome = sum(cf.outcome_value * cf.probability for cf in cfs) / sum(
                cf.probability for cf in cfs
            )
        else:
            weighted_outcome = act.expected_value  # no alternative info → neutral

        regret = act.expected_value - weighted_outcome  # positive = missed opportunity
        # Fisher information with stylometry scaling
        fisher = fisher_score(act.expected_value, center=fisher_center, width=fisher_width)
        weighted_fisher = alpha * fisher
        raw_scores.append(regret * weighted_fisher)

    probs = _softmax(np.array(raw_scores), temperature=temperature)
    return {act.id: float(p) for act, p in zip(actions, probs)}


def adaptive_nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    base_lr: float,
    reference_text: str,
    current_text: str,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Perform one NLMS weight update where the learning rate is modulated by
    signature similarity between ``reference_text`` and ``current_text``.
    This creates a direct mathematical coupling between the textual domain
    (parent B) and the MinHash signatures (parent A).

    Args:
        w: current weight vector (shape (n,))
        x: input feature vector (shape (n,))
        d: desired scalar output
        base_lr: baseline learning rate (μ)
        reference_text: text used as a stable anchor (e.g., model prompt)
        current_text: newly observed text
        epsilon: small constant to avoid division by zero

    Returns:
        Updated weight vector.
    """
    # Compute similarity factor β ∈ (0, 1]
    sig_ref = signature(_tokenize(reference_text), k=64)
    sig_cur = signature(_tokenize(current_text), k=64)
    beta = similarity(sig_ref, sig_cur)  # if texts are identical → 1.0

    # NLMS step size = μ * β
    mu = base_lr * beta

    # NLMS update
    y = float(np.dot(w, x))
    e = d - y
    norm_x = float(np.dot(x, x) + epsilon)
    w_new = w + (mu * e / norm_x) * x
    return w_new


def compute_signature_similarity(text_a: str, text_b: str, k: int = 128) -> float:
    """
    Convenience wrapper that tokenizes two texts and returns their MinHash
    similarity.  This function can be used independently of the NLMS update.
    """
    sig_a = signature(_tokenize(text_a), k=k)
    sig_b = signature(_tokenize(text_b), k=k)
    return similarity(sig_a, sig_b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=1.2),
        MathAction(id="B", expected_value=0.8),
        MathAction(id="C", expected_value=1.5),
    ]

    # Counterfactual outcomes (simulated)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=1.0, probability=0.6),
        MathCounterfactual(action_id="A", outcome_value=0.9, probability=0.4),
        MathCounterfactual(action_id="B", outcome_value=0.7, probability=1.0),
        MathCounterfactual(action_id="C", outcome_value=1.6, probability=0.5),
        MathCounterfactual(action_id="C", outcome_value=1.4, probability=0.5),
    ]

    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog, "
        "but it might not be true because not all foxes are quick."
    )

    # Hybrid regret strategy
    probs = hybrid_regret_weighted_strategy(
        actions, counterfactuals, sample_text, temperature=0.5, fisher_center=0.0, fisher_width=1.0
    )
    print("Hybrid regret probabilities:", probs)

    # NLMS update demo
    np.random.seed(0)
    w = np.random.randn(5)
    x = np.random.randn(5)
    d = 0.3
    ref_txt = "The early bird catches the worm."
    cur_txt = sample_text
    w_new = adaptive_nlms_update(w, x, d, base_lr=0.1, reference_text=ref_txt, current_text=cur_txt)
    print("Weight update norm diff:", np.linalg.norm(w_new - w))

    # Signature similarity check
    sim = compute_signature_similarity(ref_txt, cur_txt)
    print("Signature similarity:", sim)