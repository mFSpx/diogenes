# DARWIN HAMMER — match 607, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s3.py (gen3)
# born: 2026-05-29T23:29:59Z

"""Hybrid Stylometry‑NLMS Endpoint Engine
Parents:
- hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s2.py (stylometry & function‑category vectors)
- hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s3.py (endpoint health & NLMS adaptive filter)

Mathematical Bridge:
The stylometry feature vector 𝑥 ∈ ℝ^d (derived from textual analysis) is fed as the input
to a Normalized Least‑Mean‑Squares (NLMS) adaptive filter:
    Δw = μ·H·D(t)·e·x / (‖x‖²+ε)
where:
* w – filter weight vector,
* e = y_target – y_pred – instantaneous error,
* μ – base learning rate,
* H – composite health score of an endpoint (circuit‑breaker state, morphology and
  failure‑rate),
* D(t) – day‑of‑week modulation factor ∈ (0,1],
* ε – stability constant.

Thus the textual characteristics drive the adaptive learning dynamics, while the
endpoint’s health modulates the effective step‑size, fusing the two parent
topologies into a single unified system."""
import sys
import math
import random
from pathlib import Path
from datetime import date
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry utilities
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
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    """Lower‑case, whitespace‑split tokens that consist only of alphabetic characters."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def lsm_vector(text: str) -> Dict[str, float]:
    """Low‑dimensional semantic (LDS) vector based on FUNCTION_CATS."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """Return a fixed‑size float vector capturing basic stylometric statistics."""
    ws = words(text)
    total_words = max(1, len(ws))
    total_chars = max(1, len(text or ""))

    # Simple handcrafted scalar features (5 of them)
    scalars = np.array([
        total_words / 500.0,                                      # normalized length
        sum(len(w) for w in ws) / total_words / 12.0,            # avg token length
        (text.count("\n") + 1) / 200.0,                           # line density
        sum(text.count(p) for p in "!?") / total_chars,          # exclamation/question density
        sum(text.count(p) for p in ";:") / total_chars,          # punctuation density
    ])

    # Category‑level LSM vector (len = len(FUNCTION_CATS))
    cat_vec = np.array(list(lsm_vector(text).values()))

    # Concatenate and pad/truncate to the requested dimension
    raw = np.concatenate([scalars, cat_vec])
    if raw.size < dim:
        raw = np.pad(raw, (0, dim - raw.size))
    else:
        raw = raw[:dim]
    return raw.astype(float)

# ----------------------------------------------------------------------
# Parent B – Endpoint health & NLMS core
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Cyclic day index in [0,6] (0 = Monday, 6 = Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7

class EndpointCircuitBreaker:
    """Tracks consecutive failures and opens/closes the circuit."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (endpoint may receive work)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0,1]."""
        return min(self.failures / self.failure_threshold, 1.0)

class Morphology:
    """Simple geometric descriptor of an endpoint."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = max(length, 1e-6)
        self.width = max(width, 1e-6)
        self.height = max(height, 1e-6)
        self.mass = max(mass, 1e-6)

    @property
    def sphericity(self) -> float:
        """Ratio of the smallest to largest dimension (≈1 for a cube)."""
        dims = np.array([self.length, self.width, self.height])
        return float(dims.min() / dims.max())

    @property
    def flatness(self) -> float:
        """Flatness as (largest - smallest) / largest."""
        dims = np.array([self.length, self.width, self.height])
        return float((dims.max() - dims.min()) / dims.max())

def day_modulation(d: int) -> float:
    """Map weekday index d∈[0,6] to a smooth factor in (0.5,1]."""
    # Cosine‑based modulation: Monday (0) → 1.0, Sunday (6) → ~0.5
    return 0.5 * (1.0 + math.cos(math.pi * d / 6.0))

def composite_health(cb: EndpointCircuitBreaker, morph: Morphology, today: date) -> float:
    """
    Composite health H ∈ [0,1] = (1 - failure_rate) * sphericity * (1 - flatness) * D(t)
    where D(t) is the day‑of‑week modulation.
    """
    base = (1.0 - cb.failure_rate())
    geom = morph.sphericity * (1.0 - morph.flatness)
    dow = day_modulation(doomsday(today.year, today.month, today.day))
    return max(0.0, min(1.0, base * geom * dow))

def nlms_update(weights: np.ndarray, x: np.ndarray, error: float,
                mu: float, epsilon: float = 1e-8) -> np.ndarray:
    """
    NLMS weight update:
        w ← w + μ·e·x / (‖x‖² + ε)
    """
    norm_sq = float(np.dot(x, x))
    step = (mu * error) / (norm_sq + epsilon)
    return weights + step * x

# ----------------------------------------------------------------------
# Fusion – Hybrid Engine
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    """Container bundling circuit‑breaker, morphology and NLMS state."""
    cb: EndpointCircuitBreaker
    morph: Morphology
    weights: np.ndarray          # NLMS weight vector
    base_mu: float = 0.05        # base learning rate
    epsilon: float = 1e-8

    def health_factor(self, today: date) -> float:
        """Effective scaling factor μ·H·D(t) used in NLMS updates."""
        return self.base_mu * composite_health(self.cb, self.morph, today)

def initialize_endpoint(dim: int = 96,
                       failure_threshold: int = 3,
                       length: float = 1.0,
                       width: float = 1.0,
                       height: float = 1.0,
                       mass: float = 1.0,
                       base_mu: float = 0.05) -> Endpoint:
    """Factory that creates a fresh Endpoint with zero‑initialized NLMS weights."""
    cb = EndpointCircuitBreaker(failure_threshold=failure_threshold)
    morph = Morphology(length, width, height, mass)
    weights = np.zeros(dim, dtype=float)
    return Endpoint(cb=cb, morph=morph, weights=weights, base_mu=base_mu)

def hybrid_predict(endpoint: Endpoint, text: str) -> float:
    """
    Produce a scalar prediction y = w·x where x is the stylometry feature vector.
    """
    x = stylometry_features(text, dim=endpoint.weights.size)
    return float(np.dot(endpoint.weights, x))

def hybrid_train_step(endpoint: Endpoint, text: str, target: float, today: date) -> Tuple[float, float]:
    """
    One training iteration:
      1. Compute feature vector x.
      2. Predict ŷ = w·x.
      3. Compute error e = target - ŷ.
      4. Update weights via NLMS scaled by health factor.
      5. Update circuit‑breaker status based on error magnitude.
    Returns (prediction, error).
    """
    x = stylometry_features(text, dim=endpoint.weights.size)
    y_pred = float(np.dot(endpoint.weights, x))
    error = target - y_pred

    # Scale learning rate by health
    mu_eff = endpoint.health_factor(today)
    endpoint.weights = nlms_update(endpoint.weights, x, error, mu_eff, endpoint.epsilon)

    # Simple health feedback: large error triggers a failure, small error a success
    if abs(error) > 0.5:   # threshold chosen arbitrarily for demo
        endpoint.cb.record_failure()
    else:
        endpoint.cb.record_success()

    return y_pred, error

def batch_hybrid_train(endpoint: Endpoint,
                       data: List[Tuple[str, float]],
                       today: date) -> Dict[str, float]:
    """
    Train over a list of (text, target) pairs.
    Returns statistics: mean_error, success_rate, health.
    """
    errors = []
    successes = 0
    for txt, tgt in data:
        _, err = hybrid_train_step(endpoint, txt, tgt, today)
        errors.append(abs(err))
        if abs(err) <= 0.5:
            successes += 1
    mean_err = float(np.mean(errors)) if errors else 0.0
    success_rate = successes / max(1, len(data))
    health = composite_health(endpoint.cb, endpoint.morph, today)
    return {
        "mean_error": _pct(mean_err),
        "success_rate": _pct(success_rate),
        "health": _pct(health),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy endpoint
    ep = initialize_endpoint(dim=96,
                             failure_threshold=4,
                             length=2.0,
                             width=1.5,
                             height=1.0,
                             mass=3.0,
                             base_mu=0.1)

    # Sample data: short sentences with synthetic targets
    sample_data = [
        ("The quick brown fox jumps over the lazy dog.", 0.8),
        ("I think, therefore I am.", 0.2),
        ("Data science combines statistics, computer science, and domain knowledge.", 0.9),
        ("Hello world!", 0.1),
        ("Artificial intelligence is transforming many industries.", 0.85),
    ]

    today = date.today()
    stats = batch_hybrid_train(ep, sample_data, today)

    # Display results
    print("Training statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Run a prediction on a fresh sentence
    test_sentence = "Machine learning models improve with more data."
    pred = hybrid_predict(ep, test_sentence)
    print(f"\nPrediction for test sentence: {pred:.4f}")
    print(f"Endpoint health after training: {composite_health(ep.cb, ep.morph, today):.4f}")