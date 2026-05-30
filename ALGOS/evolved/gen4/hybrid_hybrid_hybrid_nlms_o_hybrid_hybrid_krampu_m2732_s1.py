# DARWIN HAMMER — match 2732, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_gliner_zero_s_m26_s7.py (gen2)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_rlct_g_m989_s0.py (gen3)
# born: 2026-05-29T23:45:10Z

"""Hybrid NLMS‑Bandit‑Pheromone System
Parent A: adaptive NLMS filter (linear prediction & weight update).
Parent B: contextual bandit with pheromone‑based signals and entropy‑driven exploration.
Mathematical bridge – the NLMS weight vector is treated as a policy parameter θ.
For each extracted text span we build a 26‑dim character‑frequency feature x.
The NLMS prediction ŷ = θ·x is interpreted as an *expected reward* (propensity).
A soft‑max over the propensities yields a probability distribution π over actions.
The Shannon entropy H(π) modulates the NLMS step‑size μ (high entropy → larger μ,
low entropy → smaller μ), while pheromone signals provide an additive bias to the
propensities. The combined update simultaneously:
* adapts θ via the classic NLMS rule using the observed reward,
* updates pheromone levels for the selected action,
* and adjusts μ based on H(π)."""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import re

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str = ""          # optional semantic label
    score: float = 0.0       # confidence from extractor
    backend: str = "hybrid"


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Parent A – NLMS core (prediction & adaptation)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Classic Normalised Least‑Mean‑Squares adaptation.

    Returns (new_weights, error) where error = target - ŷ.
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / power) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Parent B – Span extraction & pheromone system
# ----------------------------------------------------------------------
def _char_frequency_vector(text: str) -> np.ndarray:
    """26‑dim L2‑normalised frequency vector of a‑z."""
    vec = np.zeros(26, dtype=float)
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_spans(text: str) -> List[Span]:
    """Token‑wise zero‑shot extractor; assigns a random confidence."""
    spans: List[Span] = []
    for match in re.finditer(r'\S+', text):
        start, end = match.start(), match.end()
        token = match.group()
        score = random.uniform(0.5, 1.0)
        spans.append(Span(start, end, token, score=score, backend="zero_shot"))
    return spans


class PheromoneSystem:
    """Very light pheromone manager – exponential decay with half‑life."""

    def __init__(self, half_life_seconds: float = 60.0):
        self.half_life = half_life_seconds
        self.signals: Dict[str, float] = defaultdict(float)

    def get(self, key: str) -> float:
        """Current signal value after decay."""
        raw = self.signals[key]
        # In this sandbox we treat elapsed time as zero; real systems would timestamp.
        return raw * math.pow(0.5, 0 / self.half_life)

    def deposit(self, key: str, amount: float) -> None:
        """Add amount to the pheromone signal."""
        self.signals[key] += amount


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def softmax(vals: np.ndarray) -> np.ndarray:
    """Numerically stable soft‑max."""
    max_v = np.max(vals)
    exp_vals = np.exp(vals - max_v)
    return exp_vals / np.sum(exp_vals)


def shannon_entropy(probs: np.ndarray) -> float:
    """H(p) = -∑ p log p (base e). Zero probabilities are ignored."""
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log(probs[mask])))


# ----------------------------------------------------------------------
# Hybrid core functions (≥3)
# ----------------------------------------------------------------------
def hybrid_compute_propensities(
    weights: np.ndarray,
    spans: List[Span],
    pheromone: PheromoneSystem,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    For each span compute:
    1. Feature vector x (char‑frequency).
    2. NLMS prediction ŷ = w·x (raw propensity).
    3. Add pheromone bias specific to the span text.
    Returns the raw propensity array and the list of feature vectors.
    """
    xs: List[np.ndarray] = []
    raw = np.empty(len(spans), dtype=float)
    for i, span in enumerate(spans):
        x = _char_frequency_vector(span.text)
        xs.append(x)
        pred = nlms_predict(weights, x)
        pher_bias = pheromone.get(span.text)
        raw[i] = pred + pher_bias
    return raw, xs


def hybrid_select_action(
    propensities: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """
    Convert propensities to a probability distribution via soft‑max
    (temperature controls exploration) and sample an action index.
    Returns (selected_index, probability_vector).
    """
    scaled = propensities / max(1e-9, temperature)
    probs = softmax(scaled)
    idx = int(np.random.choice(len(probs), p=probs))
    return idx, probs


def hybrid_step(
    weights: np.ndarray,
    spans: List[Span],
    true_rewards: List[float],
    pheromone: PheromoneSystem,
    base_mu: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    One hybrid iteration:
    1. Compute propensities (NLMS + pheromone).
    2. Derive action probabilities, entropy and adapt μ.
    3. Sample an action, observe its true reward.
    4. NLMS weight update using the observed reward as target.
    5. Deposit pheromone proportional to reward.
    Returns (updated_weights, observed_reward).
    """
    # 1. Propensities & feature vectors
    raw_prop, feature_vectors = hybrid_compute_propensities(weights, spans, pheromone)

    # 2. Entropy‑driven learning‑rate modulation
    _, probs = hybrid_select_action(raw_prop)  # we need the full distribution
    entropy = shannon_entropy(probs)
    # Map entropy ∈ [0, log N] → μ ∈ [0.1·base_mu, base_mu]
    max_entropy = math.log(len(spans)) if len(spans) > 1 else 0.0
    mu = base_mu * (0.1 + 0.9 * (entropy / max_entropy) if max_entropy > 0 else 0.5)

    # 3. Sample action
    chosen_idx, _ = hybrid_select_action(raw_prop, temperature=1.0 / (mu + 1e-9))
    chosen_span = spans[chosen_idx]
    x_chosen = feature_vectors[chosen_idx]
    reward = true_rewards[chosen_idx]

    # 4. NLMS update (target = reward)
    new_weights, _ = nlms_update(weights, x_chosen, target=reward, mu=mu)

    # 5. Pheromone reinforcement
    pheromone.deposit(chosen_span.text, amount=reward * 0.01)

    return new_weights, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    sample_text = "The quick brown fox jumps over the lazy dog."
    spans = extract_spans(sample_text)

    # Fake ground‑truth rewards: higher for longer tokens
    true_rewards = [len(span.text) / 10.0 for span in spans]

    # Initialise NLMS weights (26‑dim) and pheromone system
    w = np.zeros(26, dtype=float)
    pher = PheromoneSystem(half_life_seconds=120.0)

    # Run a few hybrid steps
    for step in range(5):
        w, r = hybrid_step(w, spans, true_rewards, pheromone=pher, base_mu=0.6)
        print(f"Step {step+1}: observed reward={r:.3f}, weight norm={np.linalg.norm(w):.3f}")

    sys.exit(0)