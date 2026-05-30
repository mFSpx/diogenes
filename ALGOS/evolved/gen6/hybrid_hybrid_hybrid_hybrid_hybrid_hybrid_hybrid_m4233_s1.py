# DARWIN HAMMER — match 4233, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1296_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s4.py (gen5)
# born: 2026-05-29T23:54:22Z

"""Hybrid algorithm merging:
- Parent A: Fisher information scoring and NLMS adaptive weight update.
- Parent B: Energy‑based ModelPool with risk‑adjusted energy, differential‑privacy aggregation,
  Bayesian posterior computation and entropy‑driven guidance.

Mathematical bridge:
Each model receives a Fisher score 𝑓_i(θ) (Parent A).  A reconstruction‑risk probability
r_i (Parent B) weights the model’s energy cost e_i, producing a risk‑adjusted energy
E_i = e_i·r_i·f_i.  The vector E is privately aggregated with Laplace noise
(ε‑DP) to obtain Ẽ.  A Bayesian update with a uniform prior yields a posterior
p_i ∝ Ẽ_i.  The entropy H(p) guides a global allocation factor, while an
NLMS filter adapts a per‑model allocation weight w_i based on the instantaneous
error between desired allocation (p_i) and current weight.  The resulting
weights drive the ModelPool loading decisions, thus fusing the core topologies
of both parents into a single unified system."""
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    mu: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares weight adaptation.
    w   – current weight vector
    x   – input (feature) vector
    d   – desired scalar output
    mu  – step size (0<mu<2)
    Returns updated weight vector.
    """
    y = np.dot(w, x)
    e = d - y
    norm = np.dot(x, x) + eps
    w_new = w + (mu / norm) * e * x
    return w_new


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """Manages a pool of models with an energy ledger."""

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier, energy_cost: float) -> None:
        """Add a model if RAM permits; apply tier‑conflict penalty."""
        if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        # Tier conflict penalty (Parent B rule)
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        self.loaded[model.name] = model
        self._energy += energy_cost

    @property
    def total_energy(self) -> float:
        return self._energy

    def reset(self) -> None:
        self.loaded.clear()
        self._energy = 0.0


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------


def differential_privacy_aggregate(values: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Laplace mechanism: add i.i.d. Laplace(0, 1/ε) noise to each entry.
    """
    scale = 1.0 / max(epsilon, 1e-12)
    noise = np.random.laplace(loc=0.0, scale=scale, size=values.shape)
    return values + noise


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Posterior ∝ prior * likelihood, normalized."""
    unnorm = prior * likelihood
    total = np.sum(unnorm)
    if total == 0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p_safe * np.log(p_safe))


# ----------------------------------------------------------------------
# Core hybrid operation (three public functions)
# ----------------------------------------------------------------------


def compute_fisher_scores(models: List[ModelTier], theta: float) -> np.ndarray:
    """
    Produce a Fisher score for each model using a deterministic mapping
    from model name to Gaussian parameters.
    """
    scores = []
    for m in models:
        # Map name hash to centre and width in a reproducible way
        h = abs(hash(m.name))
        center = (h % 360) * math.pi / 180.0  # centre in radians
        width = 0.1 + (h % 100) / 500.0       # width ∈ [0.1,0.3]
        scores.append(fisher_score(theta, center, width))
    return np.array(scores)


def risk_adjusted_energy(
    models: List[ModelTier],
    base_energies: np.ndarray,
    risk_probs: np.ndarray,
    fisher_scores: np.ndarray,
) -> np.ndarray:
    """
    Combine energy, risk probability and Fisher information:
        E_i = e_i * r_i * f_i
    """
    if not (len(models) == base_energies.size == risk_probs.size == fisher_scores.size):
        raise ValueError("All input arrays must have the same length")
    return base_energies * risk_probs * fisher_scores


def hybrid_step(
    pool: ModelPool,
    models: List[ModelTier],
    input_vec: np.ndarray,
    theta: float,
    risk_probs: np.ndarray,
    base_energies: np.ndarray,
    nlms_mu: float = 0.5,
    dp_epsilon: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid iteration:
    1. Compute Fisher scores.
    2. Form risk‑adjusted energy vector.
    3. Apply differential‑privacy aggregation.
    4. Bayesian posterior → allocation probabilities.
    5. Entropy → global scaling factor.
    6. NLMS weight update using the posterior as desired output.
    7. Load models whose updated weight exceeds a threshold.
    Returns the final posterior and the entropy metric.
    """
    # 1. Fisher scores
    f_scores = compute_fisher_scores(models, theta)

    # 2. Risk‑adjusted energy
    adj_energy = risk_adjusted_energy(models, base_energies, risk_probs, f_scores)

    # 3. Private aggregation
    noisy_energy = differential_privacy_aggregate(adj_energy, dp_epsilon)

    # 4. Bayesian posterior (uniform prior)
    prior = np.full_like(noisy_energy, 1.0 / noisy_energy.size)
    posterior = bayesian_update(prior, noisy_energy)

    # 5. Entropy as a guidance metric
    H = entropy(posterior)

    # 6. NLMS adaptation: treat posterior as desired scalar output for each model
    #    We maintain a weight vector w of same length as models.
    if not hasattr(pool, "_nlms_weights"):
        pool._nlms_weights = np.random.rand(len(models))
    w = pool._nlms_weights
    # Desired output is the posterior probability for the corresponding model.
    # NLMS is applied element‑wise using the same input vector for all models.
    for i in range(len(models)):
        w = nlms_update(w, input_vec, posterior[i], mu=nlms_mu)
    pool._nlms_weights = w

    # 7. Loading decision: weight > 0.5 triggers loading (arbitrary threshold)
    threshold = 0.5
    for i, model in enumerate(models):
        if w[i] > threshold and not pool.is_loaded(model.name):
            try:
                pool.add_model(model, base_energies[i])
            except RuntimeError:
                # RAM ceiling reached – stop loading further models
                break

    return posterior, H


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a small set of dummy models
    model_list = [
        ModelTier(name="alpha", ram_mb=500, tier="T1"),
        ModelTier(name="beta", ram_mb=800, tier="T2"),
        ModelTier(name="gamma", ram_mb=1200, tier="T3"),
        ModelTier(name="delta", ram_mb=400, tier="T1"),
    ]

    # Base energy costs (arbitrary)
    base_energy = np.array([2.0, 3.5, 5.0, 1.5])

    # Simulated reconstruction‑risk probabilities (0‑1)
    risk = np.array([0.2, 0.6, 0.4, 0.1])

    # Input feature vector for NLMS (e.g., system load indicators)
    x = np.array([0.7, 0.3, 0.5])

    # Initialise pool
    pool = ModelPool(ram_ceiling_mb=3000)

    # Run a few hybrid iterations
    for step in range(5):
        theta_val = random.uniform(0, 2 * math.pi)
        posterior, entropy_metric = hybrid_step(
            pool,
            model_list,
            input_vec=x,
            theta=theta_val,
            risk_probs=risk,
            base_energies=base_energy,
            nlms_mu=0.4,
            dp_epsilon=0.8,
        )
        print(f"Step {step+1}: posterior={posterior.round(3)} entropy={entropy_metric:.3f}")
        print(f"  Loaded models: {list(pool.loaded.keys())}")
        print(f"  Total energy: {pool.total_energy:.2f}\n")