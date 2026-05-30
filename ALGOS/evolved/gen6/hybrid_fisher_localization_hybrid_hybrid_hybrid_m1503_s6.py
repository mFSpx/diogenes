# DARWIN HAMMER — match 1503, survivor 6
# gen: 6
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m902_s0.py (gen5)
# born: 2026-05-29T23:36:49Z

"""Hybrid Fisher-VRAM Allocation Module

This module fuses the core mathematics of two parent algorithms:

* **Parent A (fisher_localization.py)** – provides a Gaussian beam model and the
  Fisher‑information score `fisher_score(theta, center, width)` used to rank
  candidate angles.

* **Parent B (hybrid_hybrid_hybrid_fisher_hybrid_hybrid_m902_s0.py)** – extends the
  Gaussian model to a Bayesian update of the probability of successful VRAM
  allocation, introducing `Entity` objects that carry spatial and privacy loads.

**Mathematical bridge**

The Gaussian beam defines an intensity `I(θ)`.  
The Fisher information `F(θ) = (∂I/∂θ)² / I` quantifies how sensitively the intensity
changes with the angle. In the hybrid system we treat this Fisher information as
a *likelihood* for a Bayesian update of the VRAM‑allocation probability.  The
posterior for a candidate angle `θ` is therefore proportional to

    posterior(θ) ∝ prior(θ) × F(θ) × feasibility(entity, model_complexity)

where `feasibility` encodes the VRAM planning part of Parent B (spatial + privacy
load).  The resulting hybrid score unifies the localization scoring of Parent A
with the resource‑allocation reasoning of Parent B.

The functions below implement this unified model.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np


@dataclass
class Entity:
    """Represents a workload element influencing VRAM allocation."""
    timestamp: float
    spatial_load: float   # normalized [0, 1]
    privacy_load: float   # normalized [0, 1]


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) with centre `center` and standard deviation `width`."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for the Gaussian beam.
    F(θ) = (∂I/∂θ)² / I  (with a small epsilon to avoid division by zero).
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def feasibility_score(entity: Entity, model_complexity: float) -> float:
    """
    VRAM‑allocation feasibility derived from Parent B.
    Higher combined load reduces feasibility exponentially.
    """
    if model_complexity <= 0:
        raise ValueError('model_complexity must be positive')
    load = entity.spatial_load + entity.privacy_load
    return math.exp(-load / model_complexity)


def bayesian_fisher_update(prior: float, theta: float, center: float, width: float) -> float:
    """
    Bayesian update where the Fisher score acts as the likelihood.
    Returns an unnormalized posterior.
    """
    likelihood = fisher_score(theta, center, width)
    return prior * likelihood


def hybrid_score(theta: float,
                 center: float,
                 width: float,
                 entity: Entity,
                 model_complexity: float,
                 prior: float) -> float:
    """
    Unified hybrid score = prior × Fisher‑likelihood × feasibility.
    This fuses the localization metric (Parent A) with VRAM feasibility (Parent B).
    """
    posterior = bayesian_fisher_update(prior, theta, center, width)
    feas = feasibility_score(entity, model_complexity)
    return posterior * feas


def compute_posterior_distribution(thetas: List[float],
                                   priors: Dict[float, float],
                                   center: float,
                                   width: float,
                                   entity: Entity,
                                   model_complexity: float) -> Dict[float, float]:
    """
    Returns a normalized posterior distribution over `thetas` using the hybrid score.
    """
    unnorm = {
        t: hybrid_score(t, center, width, entity, model_complexity, priors.get(t, 1.0))
        for t in thetas
    }
    total = sum(unnorm.values())
    if total == 0:
        # avoid division by zero – fall back to uniform distribution
        return {t: 1.0 / len(thetas) for t in thetas}
    return {t: v / total for t, v in unnorm.items()}


def best_hybrid_angle(candidates: List[float],
                      center: float,
                      width: float,
                      entity: Entity,
                      model_complexity: float,
                      priors: Dict[float, float]) -> Tuple[float, float]:
    """
    Select the angle with the highest hybrid posterior score.
    Ties are broken by choosing the angle closest to the centre.
    Returns (best_angle, its_normalized_posterior).
    """
    posterior = compute_posterior_distribution(
        candidates, priors, center, width, entity, model_complexity
    )
    best = max(posterior.keys(),
               key=lambda t: (posterior[t], -abs(t - center)))
    return best, posterior[best]


if __name__ == "__main__":
    # Smoke test --------------------------------------------------------------
    random.seed(0)
    # Define a synthetic set of candidate angles (degrees)
    candidates = [i * 5.0 for i in range(-6, 7)]  # -30, -25, ..., 30
    centre = 0.0
    width = 10.0

    # Uniform prior over candidates
    uniform_prior = {t: 1.0 for t in candidates}

    # Create a dummy Entity representing current workload
    entity = Entity(
        timestamp=1234567890.0,
        spatial_load=0.3,   # 30% spatial pressure
        privacy_load=0.2    # 20% privacy pressure
    )
    model_complexity = 1.5  # arbitrary scaling factor

    best_angle, prob = best_hybrid_angle(
        candidates,
        centre,
        width,
        entity,
        model_complexity,
        uniform_prior
    )
    print(f"Best hybrid angle: {best_angle:.2f}° with posterior probability {prob:.4f}")
    # Verify that the posterior distribution sums to ~1
    posterior = compute_posterior_distribution(
        candidates, uniform_prior, centre, width, entity, model_complexity
    )
    print(f"Sum of posterior probabilities: {sum(posterior.values()):.6f}")