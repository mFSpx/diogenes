# DARWIN HAMMER — match 5707, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_privac_m1943_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m881_s2.py (gen5)
# born: 2026-05-30T00:04:20Z

"""Hybrid Path‑Fisher‑Hoeffding Algorithm
=====================================

Parents
-------
* **Algorithm A** – *hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s0.py*  
  Provides a lead‑lag transformation for time‑series paths and a model‑pool
  infrastructure. Its core mathematical object is the *path signature* which
  can be analysed with information‑theoretic measures (entropy).

* **Algorithm B** – *hybrid_hybrid_hoeffd_m881_s2.py*  
  Supplies Fisher information for a Gaussian‑like signal and a Hoeffding‑bound
  formulation that quantifies uncertainty given a known range.

Mathematical Bridge
-------------------
The bridge is built on **information theory**:

* The **entropy** of a path (A) yields a scalar *range* \\(R\\) that quantifies the
  variability of the data.
* This range is inserted into the **Hoeffding bound** (B) to obtain a
  confidence‑interval width \\(\\varepsilon\\).
* Simultaneously, each lead‑lag transformed point is treated as a Gaussian
  observation; its **Fisher information** measures the certainty of that
  observation.

The hybrid algorithm therefore:
1. Transforms the raw path with the lead‑lag map.
2. Computes Fisher information for each transformed point.
3. Uses the path entropy as the range in a Hoeffding bound.
4. Compares the average Fisher information to the Hoeffding bound to obtain
   a principled *split decision*.

The code below implements this fusion and provides three demonstration
functions."""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Data structures from the parents (light‑weight re‑implementation)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class ModelPool:
    """A minimal replica of the ModelPool from Algorithm A."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

# ----------------------------------------------------------------------
# Core of Algorithm A – lead‑lag transform
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a 2‑D path ``(T, d)`` into its lead‑lag representation
    ``(2·T‑1, 2·d)``.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time, dimension)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t], path[t]])          # lag
        out[2 * t + 1] = np.concatenate([path[t], path[t + 1]])      # lead
    out[-1] = np.concatenate([path[-1], path[-1]])                  # final lag
    return out

# ----------------------------------------------------------------------
# Core of Algorithm B – Gaussian beam, Fisher information, Hoeffding bound
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) for a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single scalar observation ``theta`` assuming a
    Gaussian beam model with parameters ``center`` and ``width``.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """
    Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) ).

    Parameters
    ----------
    range_ : float
        Known range R of the bounded random variable (max - min).
    delta : float
        Desired error probability (0 < δ < 1).
    n : int
        Number of independent observations.

    Returns
    -------
    float
        The bound ε.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Information‑theoretic bridge utilities
# ----------------------------------------------------------------------
def path_entropy(path: np.ndarray, bins: int = 20) -> float:
    """
    Estimate the Shannon entropy of a path by histogramming the Euclidean
    distances between consecutive points.
    """
    path = np.asarray(path, dtype=float)
    if path.shape[0] < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    hist, _ = np.histogram(diffs, bins=bins, density=True)
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log(probs))

def average_fisher_on_path(
    path: np.ndarray,
    center: float = 0.0,
    width: float = 1.0
) -> float:
    """
    Apply the lead‑lag transform to *path* and compute the mean Fisher
    information over all transformed dimensions.
    """
    ll = lead_lag_transform(path)                     # shape (2T‑1, 2d)
    # Treat each coordinate of each transformed point as a separate theta
    thetas = ll.ravel()
    fisher_vals = np.vectorize(lambda th: fisher_score(th, center, width))(thetas)
    return float(fisher_vals.mean())

# ----------------------------------------------------------------------
# Hybrid decision logic
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Fisher split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def hybrid_split_decision(
    path: np.ndarray,
    center: float = 0.0,
    width: float = 1.0,
    delta: float = 0.05,
    bins: int = 20
) -> SplitDecision:
    """
    Combine entropy‑derived Hoeffding bound with average Fisher information
    to decide whether a path segment is *informative enough* to be split.

    Returns
    -------
    SplitDecision
        ``should_split`` is True when the mean Fisher information exceeds the
        Hoeffding bound ε.
    """
    # 1. Entropy provides a data‑driven range R
    R = path_entropy(path, bins=bins)
    # 2. Hoeffding bound ε based on R
    n = lead_lag_transform(path).shape[0]   # number of transformed points
    epsilon = hoeffding_bound(R, delta, n)

    # 3. Average Fisher information on the transformed path
    avg_fisher = average_fisher_on_path(path, center=center, width=width)

    # 4. Decision
    should = avg_fisher > epsilon
    gain_gap = avg_fisher - epsilon
    reason = (
        "Fisher > ε" if should else
        f"Fisher ({avg_fisher:.4f}) ≤ ε ({epsilon:.4f})"
    )
    return SplitDecision(should, epsilon, gain_gap, reason)

# ----------------------------------------------------------------------
# Demonstration functions (three required)
# ----------------------------------------------------------------------
def demo_lead_lag():
    """Show the lead‑lag transformation on a simple 2‑D path."""
    path = np.array([[0, 0], [1, 2], [3, 1], [4, 4]], dtype=float)
    ll = lead_lag_transform(path)
    print("Original path:\n", path)
    print("Lead‑lag representation:\n", ll)

def demo_fisher_entropy():
    """Compute entropy, average Fisher, and the Hoeffding bound for a random path."""
    rng = np.random.default_rng(42)
    path = rng.normal(loc=0.0, scale=1.0, size=(50, 3))
    ent = path_entropy(path)
    avg_f = average_fisher_on_path(path, center=0.0, width=1.0)
    eps = hoeffding_bound(ent, delta=0.05, n=lead_lag_transform(path).shape[0])
    print(f"Path entropy (R): {ent:.4f}")
    print(f"Average Fisher information: {avg_f:.4f}")
    print(f"Hoeffding bound ε: {eps:.4f}")

def demo_hybrid_decision():
    """Run the full hybrid split decision on synthetic data."""
    rng = np.random.default_rng(123)
    path = rng.uniform(-2, 2, size=(30, 2))
    decision = hybrid_split_decision(path, center=0.0, width=1.0, delta=0.01)
    print("Hybrid split decision:")
    print(f"  should_split: {decision.should_split}")
    print(f"  ε (Hoeffding bound): {decision.epsilon:.6f}")
    print(f"  gain_gap (Fisher - ε): {decision.gain_gap:.6f}")
    print(f"  reason: {decision.reason}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Lead‑Lag Transform ===")
    demo_lead_lag()
    print("\n=== Demo: Fisher & Entropy ===")
    demo_fisher_entropy()
    print("\n=== Demo: Hybrid Split Decision ===")
    demo_hybrid_decision()