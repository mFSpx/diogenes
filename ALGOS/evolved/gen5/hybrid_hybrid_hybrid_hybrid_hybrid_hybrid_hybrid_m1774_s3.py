# DARWIN HAMMER — match 1774, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s5.py (gen3)
# born: 2026-05-29T23:38:46Z

"""Hybrid Algorithm combining DARWIN HAMMER parents:
- Parent A: EndpointCircuitBreaker, ModelTier, sphericity_index, health‑weighted SHAP.
- Parent B: Hygiene regex scaling, normalized Shannon entropy, stylometry expected vector,
  and minimum‑cost tree expectation.

Mathematical Bridge
------------------
Both parents expose a *scalar quality factor* that can weight a downstream product:
    • Parent A provides a health score `h_cb ∈ [0,1]` derived from the circuit‑breaker
      state and a morphology‑driven sphericity `σ ∈ (0,1]`.
    • Parent B builds a hybrid similarity score `S` for a pair of texts:
          S = s₁·s₂·h₁·h₂·(f̂₁·f̂₂)·C(T)
      where `sᵢ` is the hygiene scaling, `hᵢ` the normalised entropy,
      `f̂ᵢ` the posterior‑weighted stylometry vector and `C(T)` the expected
      minimum‑cost of a tree.

The fusion multiplies the two scalar factors, yielding a unified metric

    Score = (h_cb·σ) · S

Thus the circuit‑breaker health and morphology directly modulate the text‑pair
similarity computed by the hygiene/entropy/stylometry pipeline. The implementation
below provides the necessary building blocks and three public functions that
illustrate the hybrid operation."""


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
# Parent A components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
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
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = "2026-05-29T23:25:31Z"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = "2026-05-29T23:26:00Z"
        if self.failures >= self.failure_threshold:
            self.open = True

    def health_score(self) -> float:
        """Return a scalar ∈[0,1] reflecting remaining health."""
        if self.open:
            return 0.0
        return max(0.0, 1.0 - self.failures / self.failure_threshold)


class ModelTier:
    """Manage tiered model priorities."""

    def __init__(self, tiers: List[str] = None):
        self.tiers = tiers if tiers else ["low", "medium", "high"]
        self.current = 0  # index into tiers

    @property
    def level(self) -> str:
        return self.tiers[self.current]

    def promote(self) -> None:
        if self.current < len(self.tiers) - 1:
            self.current += 1

    def demote(self) -> None:
        if self.current > 0:
            self.current -= 1


def sphericity_index(morph: Morphology) -> float:
    """σ = (geometric_mean(dimensions)) / max(dimension)."""
    dims = np.array([morph.length, morph.width, morph.height])
    geo_mean = np.prod(dims) ** (1.0 / len(dims))
    return geo_mean / max(dims)


def shap_attribution(features: np.ndarray, health: float) -> np.ndarray:
    """
    Very lightweight SHAP‑like attribution: each feature is scaled by the health score.
    In a real setting this would be replaced by a proper SHAP computation.
    """
    return features * health


# ----------------------------------------------------------------------
# Parent B components
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
    r"\b(?:pause|sleep|wait|tomorrow|later|delay|postpone|defer)\b",
    re.I,
)
# Additional dummy patterns to reach length 8
PATTERNS = [EVIDENCE_RE, PLANNING_RE, DELAY_RE,
            re.compile(r"\b(?:error|fail|fault|bug|exception)\b", re.I),
            re.compile(r"\b(?:success|ok|passed|green)\b", re.I),
            re.compile(r"\b(?:warning|caution|alert)\b", re.I),
            re.compile(r"\b(?:update|patch|upgrade)\b", re.I),
            re.compile(r"\b(?:rollback|revert)\b", re.I)]


def hygiene_scaling(text: str) -> float:
    """
    Compute s = (‖c‖₁ + 1) / (len(c) + 1) where c is the count vector of pattern matches.
    """
    counts = np.array([len(p.findall(text)) for p in PATTERNS], dtype=float)
    l1 = np.sum(np.abs(counts))
    return (l1 + 1.0) / (len(counts) + 1.0)


def normalized_entropy(text: str) -> float:
    """
    Shannon entropy of token distribution, normalised by the maximal entropy for the
    observed vocabulary size.
    """
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    total = len(tokens)
    freq = Counter(tokens)
    probs = np.array([c / total for c in freq.values()], dtype=float)
    H = -np.sum(probs * np.log2(probs))
    H_max = math.log2(len(freq)) if len(freq) > 1 else 0.0
    return H / H_max if H_max > 0 else 0.0


def stylometry_vector(text: str, dim: int = 10) -> np.ndarray:
    """
    Produce a deterministic pseudo‑stylometry vector from the text.
    The vector is based on character‑bigram frequencies, normalised to unit length.
    """
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    counter = Counter(bigrams)
    vec = np.zeros(dim, dtype=float)
    for i, (bg, cnt) in enumerate(counter.items()):
        idx = i % dim
        vec[idx] += cnt
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def expected_stylometry_vector(text: str, posterior: np.ndarray) -> np.ndarray:
    """
    f̂ = f ⊙ p  where ⊙ denotes element‑wise product.
    The posterior vector is assumed to have the same length as the stylometry vector.
    """
    f = stylometry_vector(text, dim=len(posterior))
    return f * posterior


def expected_tree_cost(posterior: np.ndarray, lengths: np.ndarray) -> float:
    """
    C(T) = Σₑ pₑ·ℓₑ  (expected length of each edge weighted by its posterior belief).
    """
    if posterior.shape != lengths.shape:
        raise ValueError("posterior and lengths must have the same shape")
    return float(np.sum(posterior * lengths))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_pair_score(
    text_a: str,
    text_b: str,
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    posterior: np.ndarray,
    edge_lengths: np.ndarray,
) -> float:
    """
    Compute the unified hybrid score for a pair of texts.
    Score = (health·σ) · s₁·s₂·h₁·h₂·(f̂₁·f̂₂)·C(T)
    """
    # Parent A scalar factor
    health = cb.health_score()
    sigma = sphericity_index(morph)
    factor_a = health * sigma

    # Parent B components
    s1 = hygiene_scaling(text_a)
    s2 = hygiene_scaling(text_b)
    h1 = normalized_entropy(text_a)
    h2 = normalized_entropy(text_b)
    fhat1 = expected_stylometry_vector(text_a, posterior)
    fhat2 = expected_stylometry_vector(text_b, posterior)
    dot_f = float(np.dot(fhat1, fhat2))
    cost = expected_tree_cost(posterior, edge_lengths)

    return factor_a * s1 * s2 * h1 * h2 * dot_f * cost


def circuit_breaker_demo() -> None:
    """Show health score evolution and its influence on a dummy SHAP vector."""
    cb = EndpointCircuitBreaker(failure_threshold=4)
    features = np.array([0.2, 0.5, 0.3])
    for i in range(6):
        if i % 2 == 0:
            cb.record_success()
        else:
            cb.record_failure()
        health = cb.health_score()
        shap_vals = shap_attribution(features, health)
        print(f"Step {i}: health={health:.2f}, SHAP={shap_vals}")


def hygiene_entropy_demo(text: str) -> Tuple[float, float]:
    """Return hygiene scaling and normalised entropy for a single text."""
    s = hygiene_scaling(text)
    h = normalized_entropy(text)
    print(f"Hygiene scaling={s:.3f}, Normalised entropy={h:.3f}")
    return s, h


def full_hybrid_demo() -> None:
    """Run the complete hybrid score on two example texts."""
    txt1 = "The audit confirmed the evidence and the plan was updated after a short pause."
    txt2 = "Verification failed, but the subsequent rollback succeeded without errors."
    # Circuit breaker state
    cb = EndpointCircuitBreaker()
    cb.record_failure()
    cb.record_failure()  # 2 failures out of threshold 3 → health = 1/3
    # Morphology (arbitrary dimensions)
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=0.8)
    # Posterior probabilities for a toy tree (length 10)
    posterior = np.linspace(0.1, 1.0, 10)
    posterior /= posterior.sum()  # normalise to sum 1
    edge_lengths = np.random.rand(10) * 5.0  # random lengths in [0,5)
    score = hybrid_pair_score(txt1, txt2, cb, morph, posterior, edge_lengths)
    print(f"Hybrid pair score = {score:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Circuit Breaker Demo ===")
    circuit_breaker_demo()
    print("\n=== Hygiene & Entropy Demo ===")
    hygiene_entropy_demo(
        "The system will pause for a short while before proceeding with the update."
    )
    print("\n=== Full Hybrid Demo ===")
    full_hybrid_demo()