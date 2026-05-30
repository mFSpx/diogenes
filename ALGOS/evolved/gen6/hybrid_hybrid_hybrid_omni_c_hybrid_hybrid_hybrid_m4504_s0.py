# DARWIN HAMMER — match 4504, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m2407_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1774_s2.py (gen5)
# born: 2026-05-29T23:56:19Z

"""Hybrid Algorithm: Fisher‑Bandit‑Circuit‑Entropy Fusion

Parents:
- hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m2407_s3.py (graph‑bandit + Fisher information)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1774_s2.py (morphology‑circuit‑breaker + Shannon entropy & stylometry)

Mathematical bridge:
Both parents expose a scalar confidence/health factor that modulates learning.
Parent A supplies a *propensity* (bandit confidence) and a Fisher information term *F*.
Parent B supplies a *health_score* derived from the endpoint circuit‑breaker state and a
*sphericity* index *S* computed from the morphology.  

The hybrid defines a combined weight  

    w = propensity · health_score  

and a unified adaptive step size  

    η̂ = η · F · S  

The loss mixes the L2 prediction error with an entropy‑stylometry regulariser,
both scaled by *w*.  This fuses the graph‑latent prediction dynamics with the
morphology‑driven reliability assessment in a single training step.

The module implements the core equations from both parents and three
high‑level hybrid operations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable
import numpy as np
import re
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – Fisher core
# ----------------------------------------------------------------------
def compute_fisher_information(theta: float, mu: float, sigma: float, v: float) -> Tuple[float, float]:
    """
    Gaussian‑shaped intensity I and Fisher information F for a scalar angle model.
    Returns (intensity, fisher_information), both scaled by factor v.
    """
    I = np.exp(-((theta - mu) / sigma) ** 2)                     # intensity
    # Fisher information of a Gaussian w.r.t. the mean; avoid division by zero
    F = (2 * (theta - mu) / sigma ** 2) ** 2 / (I + 1e-12)
    return v * I, v * F

# ----------------------------------------------------------------------
# Parent A – Bandit core (minimal stub for propensity)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # confidence scalar ∈ (0,1]
    expected_reward: float
    confidence_bound: float
    algorithm: str

# ----------------------------------------------------------------------
# Parent B – Morphology & Circuit Breaker
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""
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
        if self.failures >= self.failure_threshold:
            self.open = True
        self.last_event_at = "2026-05-29T23:27:10Z"

    def health_score(self) -> float:
        """
        Returns a normalized health score ∈ [0,1], where 1 = fully healthy.
        """
        if self.open:
            return 0.0
        return max(0.0, 1.0 - self.failures / self.failure_threshold)

# ----------------------------------------------------------------------
# Parent B – Entropy & Stylometry utilities
# ----------------------------------------------------------------------
def shannon_entropy(tokens: List[str]) -> float:
    """Standard Shannon entropy of a token list."""
    n = len(tokens)
    if n == 0:
        return 0.0
    freq = Counter(tokens)
    probs = np.array(list(freq.values())) / n
    return -np.sum(probs * np.log2(probs + 1e-12))

def stylometry_vector(tokens: List[str]) -> np.ndarray:
    """
    Frequency vector (relative) of the most common 100 tokens.
    """
    most_common = Counter(tokens).most_common(100)
    total = sum(cnt for _, cnt in most_common) or 1
    return np.array([cnt / total for _, cnt in most_common])

def tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())

# ----------------------------------------------------------------------
# Hybrid utilities – mathematical bridge
# ----------------------------------------------------------------------
def compute_sphericity(morph: Morphology) -> float:
    """
    Sphericity index = (geometric mean of dimensions) / (max dimension).
    Returns a value in (0,1].
    """
    dims = np.array([morph.length, morph.width, morph.height])
    gm = np.prod(dims) ** (1 / 3)                     # geometric mean
    max_dim = np.max(dims) + 1e-12
    return gm / max_dim

def hybrid_weight(propensity: float, health: float) -> float:
    """Combined scalar weight w = propensity * health."""
    return propensity * health

def adaptive_learning_rate(base_eta: float, fisher: float, sphericity: float) -> float:
    """
    η̂ = η * Fisher_information * sphericity_index
    Guarantees a positive step size.
    """
    return base_eta * fisher * sphericity

def hybrid_loss(
    pred: np.ndarray,
    target: np.ndarray,
    weight: float,
    entropy: float,
    alpha: float = 0.1,
) -> float:
    """
    L = w * ||target - pred||₂² + α * entropy
    """
    l2 = np.linalg.norm(target - pred) ** 2
    return weight * l2 + alpha * entropy

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_training_step(
    theta: float,
    mu: float,
    sigma: float,
    v: float,
    morph: Morphology,
    cb: EndpointCircuitBreaker,
    action: BanditAction,
    text: str,
    Z: np.ndarray,
    base_eta: float = 0.01,
) -> Tuple[float, float]:
    """
    Executes one hybrid update:
      1. Compute Fisher intensity & information.
      2. Compute sphericity from morphology.
      3. Derive combined weight w and adaptive η̂.
      4. Predict Ŷ = θ * Z (linear proxy for the latent graph model).
      5. Compute loss mixing L2 error and Shannon entropy of the provided text.
      6. Perform a simple gradient step on θ using η̂.
    Returns updated θ and the scalar loss.
    """
    # 1. Fisher core
    intensity, fisher = compute_fisher_information(theta, mu, sigma, v)

    # 2. Morphology‑driven geometry
    sphericity = compute_sphericity(morph)

    # 3. Confidence/health blending
    health = cb.health_score()
    w = hybrid_weight(action.propensity, health)

    # 4. Latent prediction (linear proxy)
    pred = theta * Z

    # 5. Entropy component from text
    tokens = tokenize(text)
    ent = shannon_entropy(tokens)

    # 6. Loss
    loss = hybrid_loss(pred, Z, w, ent)

    # 7. Adaptive step size
    eta_hat = adaptive_learning_rate(base_eta, fisher, sphericity)

    # 8. Gradient (∂L/∂θ ≈ 2 w (θZ - Z)·Z )
    grad = 2 * w * (pred - Z) * Z
    theta_new = theta - eta_hat * grad.mean()  # mean over vector components

    return theta_new, loss

# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Initialise components
    theta0 = 0.5
    mu = 0.0
    sigma = 1.0
    v = 1.0

    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)
    cb = EndpointCircuitBreaker(failure_threshold=3)
    # Simulate a couple of successes and one failure
    cb.record_success()
    cb.record_failure()
    cb.record_success()

    action = BanditAction(
        action_id="A1",
        propensity=0.8,
        expected_reward=1.2,
        confidence_bound=0.1,
        algorithm="demo_bandit"
    )

    text = "The quick brown fox jumps over the lazy dog. The quick brown fox is swift."
    Z = np.random.randn(10)  # latent representation vector

    theta_updated, loss_val = hybrid_training_step(
        theta=theta0,
        mu=mu,
        sigma=sigma,
        v=v,
        morph=morph,
        cb=cb,
        action=action,
        text=text,
        Z=Z,
        base_eta=0.01,
    )

    print(f"Initial θ = {theta0:.4f}")
    print(f"Updated θ = {theta_updated:.4f}")
    print(f"Loss = {loss_val:.6f}")
    print(f"Health score = {cb.health_score():.2f}, Propensity = {action.propensity:.2f}")
    print(f"Sphericity = {compute_sphericity(morph):.4f}")
    intensity, fisher = compute_fisher_information(theta0, mu, sigma, v)
    print(f"Fisher intensity = {intensity:.4f}, Fisher information = {fisher:.4f}")