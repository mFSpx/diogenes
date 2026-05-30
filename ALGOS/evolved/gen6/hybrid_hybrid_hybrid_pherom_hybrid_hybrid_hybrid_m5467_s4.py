# DARWIN HAMMER — match 5467, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1774_s3.py (gen5)
# born: 2026-05-30T00:02:06Z

"""Hybrid Algorithm integrating:
- Parent A: pheromone‑infotaxis surface usage + decision‑hygiene Shannon entropy.
- Parent B: EndpointCircuitBreaker health, morphology sphericity, hygiene scaling,
  normalized entropy, stylometry vectors, and minimum‑cost tree expectation.

Mathematical Bridge
------------------
Both parents expose scalar quality factors that weight downstream products:
    • Parent A produces a *pheromone probability*  p_i ∈ [0,1] for each action i,
      derived from a pheromone matrix τ and modulated by the Shannon entropy
      H of decision‑hygiene scores.
    • Parent B yields a *system health factor*  h_sys = h_cb·σ ∈ [0,1] (circuit‑breaker
      health times morphological sphericity) and a *text‑pair similarity* S
      (hygiene scaling·entropy·stylometry·tree‑cost).

The fusion multiplies the three scalars, giving a unified metric

        Score(i, a, b) = (h_cb·σ) · (H / H_max) · p_i · S(a, b)

where
    – H_max is the theoretical maximum entropy for the current number of actions,
    – p_i = τ_i / Σ_j τ_j,
    – S(a,b) = s_a·s_b·h_a·h_b·(f̂_a·f̂_b)·C(T).

The implementation below provides the building blocks and three public functions
that illustrate the hybrid operation."""


import math
import random
import sys
from pathlib import Path
import re
from collections import Counter
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict


# ----------------------------------------------------------------------
# Parent A components (pheromone + decision hygiene)
# ----------------------------------------------------------------------
def initialize_pheromone(actions: List[str], init_val: float = 1.0) -> np.ndarray:
    """Create a pheromone vector τ for a list of actions."""
    return np.full(len(actions), init_val, dtype=float)


def decay_pheromone(tau: np.ndarray, decay_rate: float = 0.1) -> np.ndarray:
    """Exponential decay of pheromone levels."""
    return tau * (1.0 - decay_rate)


def deposit_pheromone(tau: np.ndarray, idx: int, amount: float = 1.0) -> np.ndarray:
    """Add pheromone to the chosen action."""
    tau[idx] += amount
    return tau


def pheromone_probabilities(tau: np.ndarray) -> np.ndarray:
    """Normalize pheromone vector to a probability distribution."""
    total = np.sum(tau)
    if total == 0:
        return np.full_like(tau, 1.0 / len(tau))
    return tau / total


def hygiene_scores(texts: List[str]) -> np.ndarray:
    """Very light decision‑hygiene scoring based on regex keyword counts."""
    # Count occurrences of each regex category per text
    patterns = {
        "evidence": re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),
        "planning": re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I),
        "risk": re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I),
    }
    scores = []
    for txt in texts:
        count = sum(len(pat.findall(txt)) for pat in patterns.values())
        scores.append(count)
    # Convert to a positive float array
    return np.array(scores, dtype=float)


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    eps = np.finfo(float).eps
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log2(probs))


def normalized_entropy(scores: np.ndarray) -> float:
    """Normalize entropy by the maximum possible entropy for the given length."""
    if len(scores) == 0:
        return 0.0
    probs = scores / np.sum(scores) if np.sum(scores) > 0 else np.full_like(scores, 1.0 / len(scores))
    H = shannon_entropy(probs)
    H_max = math.log2(len(scores))
    return H / H_max if H_max > 0 else 0.0


# ----------------------------------------------------------------------
# Parent B components (circuit‑breaker, morphology, stylometry, tree cost)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    @property
    def health(self) -> float:
        """Health factor h_cb ∈ [0,1] (1 = healthy, 0 = open)."""
        return 0.0 if self.open else 1.0


def sphericity_index(morph: Morphology) -> float:
    """Simple sphericity σ = (volume)^(1/3) / max(dimensions). Returns (0,1]."""
    volume = morph.length * morph.width * morph.height
    if volume <= 0:
        return 0.0
    cubic_root = volume ** (1.0 / 3.0)
    max_dim = max(morph.length, morph.width, morph.height)
    return min(1.0, cubic_root / max_dim)


def stylometry_vector(text: str, vocab: List[str]) -> np.ndarray:
    """Create a normalized term‑frequency vector over a fixed vocabulary."""
    words = re.findall(r"\b\w+\b", text.lower())
    counter = Counter(words)
    vec = np.array([counter.get(tok, 0) for tok in vocab], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def expected_min_cost_tree(_texts: List[str]) -> float:
    """Placeholder for expected minimum‑cost tree C(T). Returns a constant >0."""
    return 1.0  # In a real system this would be computed from a graph.


def hygiene_scaling(scores: np.ndarray) -> np.ndarray:
    """Scale hygiene scores to [0,1] (s_i)."""
    if scores.size == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_similarity(
    text_a: str,
    text_b: str,
    morph_a: Morphology,
    morph_b: Morphology,
    breaker: EndpointCircuitBreaker,
    vocab: List[str],
) -> float:
    """
    Compute the Parent B similarity component S and multiply by the system health
    factor (h_cb·σ). Returns the product (h_cb·σ)·S.
    """
    # Hygiene scaling for each text (using the same scores function as Parent A)
    scores = hygiene_scores([text_a, text_b])
    s_i = hygiene_scaling(scores)

    # Normalized Shannon entropy for each text (as h_i)
    probs = scores / np.sum(scores) if np.sum(scores) > 0 else np.full_like(scores, 1.0 / 2)
    h_i = np.array([shannon_entropy(np.array([p, 1 - p])) for p in probs])

    # Stylometry vectors
    f_a = stylometry_vector(text_a, vocab)
    f_b = stylometry_vector(text_b, vocab)
    dot_product = float(np.dot(f_a, f_b))

    # Expected minimum‑cost tree (placeholder)
    C_T = expected_min_cost_tree([text_a, text_b])

    # Parent B raw similarity
    S = (
        s_i[0] * s_i[1] *
        h_i[0] * h_i[1] *
        dot_product *
        C_T
    )

    # System health factor from Parent B
    h_cb = breaker.health
    sigma = sphericity_index(morph_a) * sphericity_index(morph_b)

    return (h_cb * sigma) * S


def hybrid_action_score(
    actions: List[str],
    chosen_idx: int,
    tau: np.ndarray,
    texts: List[str],
    morphs: List[Morphology],
    breaker: EndpointCircuitBreaker,
    vocab: List[str],
) -> float:
    """
    Combine Parent A pheromone probability, normalized entropy of decision‑hygiene
    scores, and Parent B similarity/health factor into a single scalar score.
    """
    # Parent A: pheromone probability for the chosen action
    p_vec = pheromone_probabilities(tau)
    p_i = p_vec[chosen_idx]

    # Parent A: normalized entropy H/H_max from hygiene scores over all texts
    scores = hygiene_scores(texts)
    H_norm = normalized_entropy(scores)

    # Parent B: similarity between the text associated with the chosen action
    # and a reference (here we pick the first text as reference)
    ref_idx = 0
    sim = hybrid_similarity(
        texts[chosen_idx],
        texts[ref_idx],
        morphs[chosen_idx],
        morphs[ref_idx],
        breaker,
        vocab,
    )

    # Final fused score
    return H_norm * p_i * sim


def run_hybrid_cycle(
    actions: List[str],
    texts: List[str],
    morphs: List[Morphology],
    breaker: EndpointCircuitBreaker,
    vocab: List[str],
    iterations: int = 5,
) -> Tuple[np.ndarray, List[int]]:
    """
    Execute a simple pheromone‑based decision loop, updating pheromones after each
    selection and returning the final pheromone vector and the sequence of chosen
    action indices.
    """
    tau = initialize_pheromone(actions)
    chosen_history: List[int] = []

    for _ in range(iterations):
        # Choose action proportionally to pheromone probabilities
        probs = pheromone_probabilities(tau)
        chosen = int(np.random.choice(len(actions), p=probs))
        chosen_history.append(chosen)

        # Compute hybrid score (demonstration purpose only)
        _ = hybrid_action_score(
            actions, chosen, tau, texts, morphs, breaker, vocab
        )

        # Update pheromones: decay then deposit on chosen action
        tau = decay_pheromone(tau, decay_rate=0.05)
        tau = deposit_pheromone(tau, chosen, amount=1.0)

        # Simulate random success/failure for the circuit breaker
        if random.random() < 0.2:  # 20% chance of failure
            breaker.record_failure()
        else:
            breaker.record_success()

    return tau, chosen_history


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny problem space
    actions = ["email", "call", "log", "escalate"]
    texts = [
        "Please verify the source and attach the screenshot.",
        "We need to plan the next phase and update the roadmap.",
        "Risk of self‑harm detected, immediate help required.",
        "All tests passed, deployment is green and verified."
    ]
    morphs = [
        Morphology(1.0, 0.5, 0.2, 0.1),
        Morphology(1.2, 0.6, 0.3, 0.2),
        Morphology(0.8, 0.4, 0.2, 0.15),
        Morphology(1.1, 0.55, 0.25, 0.18)
    ]
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    vocab = ["verify", "source", "plan", "risk", "test", "green", "escalate", "call", "email", "log"]

    final_tau, history = run_hybrid_cycle(
        actions, texts, morphs, breaker, vocab, iterations=10
    )

    print("Final pheromone levels:", final_tau)
    print("Chosen action sequence:", [actions[i] for i in history])
    print("Circuit breaker health:", breaker.health)