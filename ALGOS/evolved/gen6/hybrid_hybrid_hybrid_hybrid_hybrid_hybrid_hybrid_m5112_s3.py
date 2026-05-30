# DARWIN HAMMER — match 5112, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m838_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1482_s1.py (gen5)
# born: 2026-05-29T23:59:50Z

"""
Hybrid Adaptive Filter with Entropic Pheromone Modulation

This module fuses the core of **Parent Algorithm A** (NLMS adaptive filtering with a
Clifford geometric product on multivectors) and **Parent Algorithm B**
(pheromone‑based entropy calculations).  

Mathematical bridge
------------------
* The NLMS update modifies a weight vector **w** using a learning rate **μ**.
* In the pheromone subsystem, a set of signal values **s_i** defines a
  probability distribution whose Shannon entropy **H** quantifies uncertainty.
* We let the entropy **H** modulate the effective learning rate:
  \[
      \mu_{\text{eff}} = \mu \, (1 + H)
  \]
  so that higher uncertainty in the pheromone field yields a larger adaptive
  step.
* The weight vector **w** and the input vector **x** are lifted to multivectors
  **W**, **X**. Their Clifford geometric product **W ⊙ X** (implemented here as
  component‑wise multiplication, a valid geometric product for a diagonal
  metric) is returned alongside the NLMS update, providing a richer
  representation of the interaction between adaptive parameters and the data
  stream.

The resulting hybrid algorithm therefore adapts its coefficients with an
entropy‑aware learning rate while simultaneously exposing the multivector
product that can be used for downstream geometric reasoning.

Author: computational physicist & AI architect
"""

import math
import random
import sys
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Multivector utilities (from Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple multivector where the geometric product is implemented as
    component‑wise multiplication (valid for a diagonal metric).
    """
    def __init__(self, components: np.ndarray):
        self.components = np.asarray(components, dtype=float)

    def __mul__(self, other: "Multivector") -> "Multivector":
        return Multivector(self.components * other.components)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Component‑wise geometric product."""
    return a * b


# ----------------------------------------------------------------------
# NLMS core (from Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = wᵀx."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Standard Normalized LMS update.
    Returns (new_weights, error).
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


# ----------------------------------------------------------------------
# Pheromone & entropy utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
    ):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        return math.exp(-self.age_seconds() / self.half_life_seconds)

    def decayed_value(self) -> float:
        """Current value after applying decay."""
        return self.signal_value * self.decay_factor()


def pheromone_distribution(pheromones: List[PheromoneEntry]) -> np.ndarray:
    """
    Returns a normalized probability vector derived from the decayed signal values.
    If the total mass is zero, returns a uniform distribution.
    """
    if not pheromones:
        return np.array([], dtype=float)

    values = np.array([p.decayed_value() for p in pheromones], dtype=float)
    total = values.sum()
    if total <= 0.0:
        return np.full_like(values, 1.0 / len(values))
    return values / total


def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy, handling zero probabilities safely."""
    if probs.size == 0:
        return 0.0
    mask = probs > 0
    return -float(np.sum(probs[mask] * np.log2(probs[mask])))


def normalized_entropy(probs: np.ndarray) -> float:
    """
    Normalized entropy in [0, 1] by dividing by log2(N) where N is the number of states.
    """
    if probs.size <= 1:
        return 0.0
    max_entropy = math.log2(probs.size)
    return shannon_entropy(probs) / max_entropy


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_learning_rate(base_mu: float, pheromones: List[PheromoneEntry]) -> float:
    """
    Adjust the base learning rate using the normalized entropy of the pheromone
    distribution.
    """
    probs = pheromone_distribution(pheromones)
    H = normalized_entropy(probs)
    # Scale mu: higher entropy -> larger step, bounded to [base_mu, 2*base_mu]
    return base_mu * (1.0 + H)


def hybrid_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    pheromones: List[PheromoneEntry],
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Multivector, float]:
    """
    Perform an NLMS update where the learning rate is modulated by pheromone entropy.
    Additionally, return the geometric product of the weight and input multivectors.
    Returns (new_weights, multivector_product, error).
    """
    mu_eff = hybrid_learning_rate(base_mu, pheromones)
    new_weights, error = nlms_update(weights, x, target, mu=mu_eff, eps=eps)

    # Lift to multivectors (pad to same length if necessary)
    max_len = max(new_weights.size, x.size)
    w_pad = np.pad(new_weights, (0, max_len - new_weights.size), constant_values=0.0)
    x_pad = np.pad(x, (0, max_len - x.size), constant_values=0.0)

    W = Multivector(w_pad)
    X = Multivector(x_pad)
    prod = geometric_product(W, X)

    return new_weights, prod, error


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Transform a (T, d) path into a lead‑lag representation.
    This utility is retained from Parent A for potential downstream use.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Random NLMS scenario
    dim = 5
    w = np.random.randn(dim)
    x = np.random.randn(dim)
    target = float(np.dot(w, x) + np.random.randn() * 0.1)  # noisy observation

    # Create a small pheromone field
    pheromones = [
        PheromoneEntry(
            surface_key="node_A",
            signal_kind="type_1",
            signal_value=random.uniform(0.5, 1.5),
            half_life_seconds=30,
        )
        for _ in range(4)
    ]

    # Perform hybrid update
    new_w, mv_product, err = hybrid_update(w, x, target, pheromones)

    print("Original weights :", w)
    print("Input vector     :", x)
    print("Target           :", target)
    print("Error            :", err)
    print("Updated weights  :", new_w)
    print("Multivector product (W ⊙ X):", mv_product)
    print("Effective μ used :", hybrid_learning_rate(0.5, pheromones))
    # Verify that the code runs without exception
    sys.exit(0)