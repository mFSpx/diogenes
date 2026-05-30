# DARWIN HAMMER — match 4077, survivor 4
# gen: 5
# parent_a: hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s1.py (gen4)
# born: 2026-05-29T23:53:26Z

"""Hybrid Krampus‑Pheromone‑Fisher‑SSIM Algorithm
Parents:
- hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s2.py (text feature → pheromone decay)
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1215_s1.py (Gaussian beam, Fisher score, SSIM, bandit)

Mathematical bridge:
Each scalar feature *fᵢ* extracted from text is interpreted as a pheromone signal
sᵢ(t) with exponential half‑life τᵢ that depends on the document entropy H:
 τᵢ = τ₀·(1 + H).  
The instantaneous signal magnitude is used as the “angle” θᵢ of a Gaussian beam.
From θᵢ we compute a Fisher information score 𝓕ᵢ = fisher_score(θᵢ, μ, σ) that quantifies the
sensitivity of the signal to perturbations.  A multi‑armed bandit uses the
pair (𝓕ᵢ, SSIM(prev, curr)) as its reward to decide which pheromone to
reinforce (reset τᵢ) at each step.  The final hybrid metric is the
decayed‑and‑aggregated pheromone vector weighted by its Fisher scores.

The implementation below follows this blueprint and provides three public
functions:
    extract_features(text) → dict
    update_pheromones(features, state, dt) → new_state
    hybrid_document_metric(state) → float
"""

import math
import random
import sys
import pathlib
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – text feature extraction and pheromone model
# ----------------------------------------------------------------------


def normalize_ws(text: str) -> str:
    """Collapse whitespace to a single space and strip."""
    return " ".join(text.split()) if text else ""


def token_count(text: str) -> int:
    """Count whitespace‑separated tokens."""
    return len(text.split())


def shannon_entropy(symbols: List[str]) -> float:
    """Shannon entropy H = -Σ p·log₂(p) for a list of symbols."""
    if not symbols:
        return 0.0
    total = len(symbols)
    freq = Counter(symbols)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def entropy_for_text(text: str, max_len: int = 10_000) -> float:
    """Entropy of the first `max_len` characters of `text`."""
    if not text:
        return 0.0
    snippet = list(text[:max_len])
    return shannon_entropy(snippet)


def extract_features(text: str) -> Dict[str, float]:
    """
    Simple feature extractor used by Parent A.
    Returns a dict mapping feature names to normalized scalar values.
    """
    txt = normalize_ws(text)
    if not txt:
        return {}

    # Basic scalar features
    length = len(txt)
    tokens = token_count(txt)
    entropy = entropy_for_text(txt)

    # Normalisation (avoid division by zero)
    max_len = 1_000.0
    max_tokens = 200.0
    max_entropy = 8.0  # bits, typical upper bound for ASCII

    feats = {
        "length": min(length / max_len, 1.0),
        "tokens": min(tokens / max_tokens, 1.0),
        "entropy": min(entropy / max_entropy, 1.0),
    }
    return feats


# ----------------------------------------------------------------------
# Pheromone entry & decay utilities
# ----------------------------------------------------------------------


class PheromoneEntry:
    """Container for a single pheromone signal."""

    def __init__(self, value: float, half_life: float, timestamp: float):
        self.value = float(value)          # current magnitude
        self.half_life = float(half_life)  # half‑life τ in seconds
        self.timestamp = float(timestamp)  # last update time (seconds)

    def decay(self, now: float) -> None:
        """Apply exponential decay from self.timestamp up to `now`."""
        dt = max(now - self.timestamp, 0.0)
        if dt <= 0.0 or self.half_life <= 0.0:
            return
        decay_factor = 0.5 ** (dt / self.half_life)
        self.value *= decay_factor
        self.timestamp = now

    def reinforce(self, new_half_life: float, now: float) -> None:
        """Reset half‑life and refresh timestamp (value stays as decayed)."""
        self.half_life = new_half_life
        self.timestamp = now


def init_pheromones(features: Dict[str, float], base_half_life: float = 30.0) -> Dict[str, PheromoneEntry]:
    """
    Map feature magnitudes to pheromone entries.
    Half‑life τᵢ = base_half_life·(1 + H) where H is the document entropy feature.
    """
    entropy = features.get("entropy", 0.0)
    tau_factor = 1.0 + entropy  # slower decay for high‑entropy texts
    pheros: Dict[str, PheromoneEntry] = {}
    now = datetime.now(tz=timezone.utc).timestamp()
    for name, mag in features.items():
        pheros[name] = PheromoneEntry(value=mag, half_life=base_half_life * tau_factor, timestamp=now)
    return pheros


def decay_pheromones(state: Dict[str, PheromoneEntry], dt: float) -> None:
    """Decay every entry by `dt` seconds."""
    now = datetime.now(tz=timezone.utc).timestamp()
    for entry in state.values():
        entry.decay(now)


# ----------------------------------------------------------------------
# Parent B – Gaussian beam, Fisher score, SSIM, simple bandit
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Unnormalised Gaussian."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def compute_ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0,
                 k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Simplified SSIM for 1‑D vectors.
    μx, μy – means; σx², σy² – variances; σxy – covariance.
    """
    if x.shape != y.shape:
        raise ValueError("vectors must have equal shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x2 = np.var(x)
    sigma_y2 = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Bandit (UCB1) to choose pheromone reinforcement
# ----------------------------------------------------------------------


class UCB1Bandit:
    """Very lightweight Upper‑Confidence‑Bound bandit."""

    def __init__(self, arms: List[str]):
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total = 0

    def select(self) -> str:
        """Select arm with highest UCB value; break ties randomly."""
        self.total += 1
        ucb_vals = {}
        for arm in self.counts:
            if self.counts[arm] == 0:
                return arm  # ensure each arm is tried at least once
            avg = self.values[arm] / self.counts[arm]
            bonus = math.sqrt(2 * math.log(self.total) / self.counts[arm])
            ucb_vals[arm] = avg + bonus
        max_val = max(ucb_vals.values())
        candidates = [arm for arm, val in ucb_vals.items() if val == max_val]
        return random.choice(candidates)

    def update(self, arm: str, reward: float) -> None:
        """Incorporate observed reward."""
        self.counts[arm] += 1
        self.values[arm] += reward


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------


def update_pheromones(features: Dict[str, float],
                     state: Dict[str, PheromoneEntry],
                     dt: float) -> Dict[str, PheromoneEntry]:
    """
    Core hybrid step:
    1. Decay existing pheromones by `dt`.
    2. Compute Fisher scores for each feature (θ = magnitude).
    3. Compute SSIM between previous and current pheromone vectors.
    4. Use a UCB1 bandit to pick a feature; reinforce its pheromone
       (reset half‑life based on current entropy).
    Returns the updated state dictionary.
    """
    # 1 – decay
    now = datetime.now(tz=timezone.utc).timestamp()
    for entry in state.values():
        entry.decay(now)

    # 2 – Fisher scores (use Gaussian centre=0.5, width=0.2 as heuristics)
    center, width = 0.5, 0.2
    fisher_vals = {}
    for name, mag in features.items():
        fisher_vals[name] = fisher_score(theta=mag, center=center, width=width)

    # 3 – SSIM between old and new magnitude vectors
    old_vec = np.array([e.value for e in state.values()], dtype=float)
    # simulate a "new" vector by adding a tiny random jitter to each entry
    jitter = np.random.normal(scale=0.001, size=old_vec.shape)
    new_vec = old_vec + jitter
    ssim_reward = compute_ssim(old_vec, new_vec)

    # 4 – Bandit decision
    bandit = UCB1Bandit(arms=list(state.keys()))
    # Seed bandit with prior counts/rewards derived from Fisher scores
    for arm in bandit.counts:
        bandit.update(arm, fisher_vals.get(arm, 0.0))

    chosen = bandit.select()
    # Combine Fisher reward with SSIM to form reinforcement signal
    reward = fisher_vals.get(chosen, 0.0) * (0.5 + 0.5 * ssim_reward)
    bandit.update(chosen, reward)

    # Reinforce chosen pheromone: new half‑life proportional to entropy
    entropy = features.get("entropy", 0.0)
    new_half_life = 30.0 * (1.0 + entropy)   # same rule as init
    state[chosen].reinforce(new_half_life, now)

    return state


def hybrid_document_metric(state: Dict[str, PheromoneEntry]) -> float:
    """
    Produce a single scalar metric for the document.
    The metric is the dot product of the decayed pheromone vector with its
    Fisher‑information weighting vector.
    """
    mags = np.array([e.value for e in state.values()], dtype=float)
    # Fisher weighting uses the same Gaussian parameters as before
    center, width = 0.5, 0.2
    weights = np.array([fisher_score(theta=v, center=center, width=width) for v in mags], dtype=float)
    if np.linalg.norm(weights) == 0:
        return 0.0
    weighted = mags * weights
    return float(np.sum(weighted) / np.linalg.norm(weights))


def hybrid_process(text: str, dt: float = 5.0, steps: int = 3) -> float:
    """
    Convenience wrapper that runs the full hybrid pipeline on `text`.
    - `dt` seconds between decay steps.
    - `steps` number of iterative updates.
    Returns the final hybrid metric.
    """
    feats = extract_features(text)
    state = init_pheromones(feats)

    for _ in range(steps):
        # Re‑extract features each step to allow entropy changes (simulated)
        feats = extract_features(text)
        state = update_pheromones(feats, state, dt)

    return hybrid_document_metric(state)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "In the quiet of the night, the algorithm whispered its secrets. "
        "Entropy rose as the words tangled, forming a lattice of pheromones "
        "that decayed and rebounded under a Gaussian gaze."
    )
    metric = hybrid_process(sample, dt=2.0, steps=5)
    print(f"Hybrid document metric: {metric:.6f}")