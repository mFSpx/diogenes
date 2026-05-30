# DARWIN HAMMER — match 1203, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m468_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s1.py (gen4)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m468_s1.py (Regret‑Weighted Hoeffding Tree + Bandit with Gini‑modulated confidence)
- Parent B: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s1.py (High‑dimensional text feature projection via bilinear form + ternary routing)

Mathematical Bridge
------------------
Parent B projects a high‑dimensional text feature vector **f** ∈ ℝⁿ onto a low‑dimensional
model space using a bilinear form **P** ∈ ℝⁿˣᵐ:

    v = f · P                     # v ∈ ℝᵐ

The resulting projected values **v** form a distribution whose inequality is measured
by the Gini coefficient **G(v)** (Parent A).  This Gini value modulates the
regret‑weighted confidence bound **ε** of each bandit arm:

    ε = ε₀ · (1 + λ_g · G(v))

A temperature‑dependent developmental rate ρ(T) (from Parent B) further scales the
split‑gain estimate used in the Hoeffding‑tree‑like decision:

    gain_gap = ρ(T) · (max_gain – ε)

The hybrid system therefore shares a common core:
1. Text → feature vector **f** (Parent B)
2. Projection **v = f·P** (Parent B)
3. Gini coefficient **G(v)** (Parent A)
4. Confidence modulation **ε** (Parent A)
5. Temperature scaling **ρ(T)** (Parent B)
6. Final bandit selection using the adjusted confidence.

The code below implements this fused pipeline with three public functions that
demonstrate the hybrid operation.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Parent‑A utilities (Gini, hash, etc.)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic hash used by the original algorithm."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini coefficient of a non‑negative value list."""
    arr = np.asarray(list(values), dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    if arr.size == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    # G = (2*Σi*i*yi)/(n*Σyi) - (n+1)/n
    g = (2.0 * np.sum((np.arange(1, n + 1) * sorted_arr))) / (n * sum_y) - (n + 1) / n
    return g


# ----------------------------------------------------------------------
# Parent‑B utilities (text features, bilinear projection, temperature)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def _tokenize(text: str) -> List[str]:
    """Very light tokenizer – split on whitespace and strip punctuation."""
    tokens = []
    for raw in text.lower().split():
        token = raw.strip("!?;:,.—-()[]{}\"'`/\\|@#$%^")
        if token:
            tokens.append(token)
    return tokens


def calculate_text_features(text: str) -> np.ndarray:
    """
    Produce a high‑dimensional feature vector based on lexical categories.
    The dimension equals the total number of distinct categories (9) plus a
    bag‑of‑words count for the 100 most frequent tokens (a simple fallback).
    """
    tokens = _tokenize(text)
    cat_counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for i, (_, cat_set) in enumerate(FUNCTION_CATS.items()):
        cat_counts[i] = sum(tok in cat_set for tok in tokens)

    # Simple bag‑of‑words fallback: count the first 100 unique tokens.
    vocab = {}
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
        if len(vocab) > 100:
            break
    bow = np.zeros(min(100, len(vocab)), dtype=float)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            bow[idx] += 1.0

    return np.concatenate([cat_counts, bow])


def temperature_schedule(step: int, k: float = 0.01, t0: int = 100) -> float:
    """
    Sigmoidal temperature schedule used as the developmental rate ρ(T).
    Returns a value in (0, 1) that grows with the training step.
    """
    return 1.0 / (1.0 + math.exp(-k * (step - t0)))


def bilinear_projection(features: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """
    Apply a bilinear form: v = f · P
    where f is 1‑D (1×n) and P is (n×m). Returns a 1‑D vector of length m.
    """
    if features.ndim != 1:
        raise ValueError("features must be a 1‑D vector")
    if projection.shape[0] != features.shape[0]:
        raise ValueError("projection matrix row size must match feature dimension")
    return features @ projection


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_adjusted_confidence(
    base_epsilon: float,
    lambda_g: float,
    projected: np.ndarray,
    temperature: float,
    max_gain: float,
) -> Tuple[float, float]:
    """
    Implements the fused equations:

        G = Gini(projected)
        ε = base_epsilon * (1 + lambda_g * G)
        gain_gap = temperature * (max_gain - ε)

    Returns (epsilon, gain_gap).
    """
    G = gini_coefficient(projected)
    epsilon = base_epsilon * (1.0 + lambda_g * G)
    gain_gap = temperature * (max_gain - epsilon)
    return epsilon, gain_gap


def hybrid_bandit_select(
    actions: List[BanditAction],
    projected: np.ndarray,
    base_epsilon: float = 0.1,
    lambda_g: float = 0.5,
    step: int = 0,
    max_gain: float = 1.0,
) -> BanditAction:
    """
    Adjust each action's confidence bound using the Gini‑modulated epsilon,
    then pick the arm with the highest adjusted bound.

    The temperature schedule depends on the training step.
    """
    temperature = temperature_schedule(step)
    epsilon, _ = compute_adjusted_confidence(
        base_epsilon, lambda_g, projected, temperature, max_gain
    )
    # Adjust confidence: original bound minus epsilon (more regret → lower bound)
    adjusted = [
        (a, a.confidence_bound - epsilon * (1.0 - a.propensity))
        for a in actions
    ]
    # Select the action with the maximal adjusted confidence
    selected_action, _ = max(adjusted, key=lambda pair: pair[1])
    return selected_action


def hybrid_split_gain_estimate(
    projected: np.ndarray,
    base_epsilon: float = 0.1,
    lambda_g: float = 0.5,
    step: int = 0,
    max_gain: float = 1.0,
) -> float:
    """
    Return the gain_gap value that would be used by a Hoeffding‑tree split
    decision after incorporating the hybrid modulation.
    """
    temperature = temperature_schedule(step)
    _, gain_gap = compute_adjusted_confidence(
        base_epsilon, lambda_g, projected, temperature, max_gain
    )
    return gain_gap


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text
    sample_text = (
        "The quick brown fox jumps over the lazy dog while I contemplate "
        "the meaning of existence in a quiet room."
    )

    # 1. Feature extraction & projection
    feats = calculate_text_features(sample_text)
    # Random but reproducible projection matrix (seeded)
    rng = np.random.default_rng(42)
    projection_matrix = rng.normal(
        loc=0.0, scale=1.0, size=(feats.shape[0], 16)
    )  # project to 16‑D
    proj = bilinear_projection(feats, projection_matrix)

    # 2. Prepare dummy bandit actions
    dummy_actions = [
        BanditAction(
            action_id=f"arm_{i}",
            propensity=random.uniform(0.1, 1.0),
            expected_reward=random.uniform(0.0, 1.0),
            confidence_bound=random.uniform(0.5, 1.5),
        )
        for i in range(5)
    ]

    # 3. Hybrid bandit selection
    chosen = hybrid_bandit_select(
        actions=dummy_actions,
        projected=proj,
        base_epsilon=0.05,
        lambda_g=0.8,
        step=150,
        max_gain=1.2,
    )
    print(f"Chosen action: {chosen.action_id}")

    # 4. Hybrid split‑gain estimate (just to exercise the function)
    gain = hybrid_split_gain_estimate(
        projected=proj,
        base_epsilon=0.05,
        lambda_g=0.8,
        step=150,
        max_gain=1.2,
    )
    print(f"Hybrid gain_gap estimate: {gain:.4f}")

    # Simple sanity checks
    assert isinstance(gain, float)
    assert isinstance(chosen, BanditAction)
    print("Hybrid algorithm smoke test completed successfully.")