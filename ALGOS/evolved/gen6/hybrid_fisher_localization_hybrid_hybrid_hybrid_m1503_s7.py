# DARWIN HAMMER — match 1503, survivor 7
# gen: 6
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m902_s0.py (gen5)
# born: 2026-05-29T23:36:49Z

"""Hybrid Fisher–Bayesian VRAM Scheduler
=====================================

This module fuses **Parent Algorithm A** (`fisher_localization.py`) and **Parent
Algorithm B** (`hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m902_s0.py`).

Both parents share the *Gaussian beam* (`gaussian_beam`) and the *Fisher‑information
score* (`fisher_score`).  The second parent extends this core with a Bayesian
update that evaluates the probability of a successful VRAM allocation for a
given *Entity* (timestamp, spatial load, privacy load).

The mathematical bridge is therefore:

* Use the Fisher information of a scalar “angle” (here derived from the
  entity timestamp) as a **weight** for a Bayesian update of the allocation
  likelihood.  High Fisher information → the likelihood is trusted more,
  low Fisher information → the prior dominates.

The resulting hybrid system provides:
1. `bayesian_update` – Bayesian posterior with a Fisher‑weighted likelihood.
2. `vram_success_probability` – per‑entity probability of successful VRAM
   allocation.
3. `best_entity` – selection of the entity (or combination) that maximises the
   hybrid metric.

All functions are pure Python and depend only on the standard library and
NumPy.  The module can be executed directly; a smoke‑test runs at the bottom."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Any, Dict

import numpy as np


# ----------------------------------------------------------------------
# Core shared primitives (from Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information score for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Data model (from Parent B)
# ----------------------------------------------------------------------
@dataclass
class Entity:
    """A lightweight record describing a request for VRAM."""
    timestamp: float          # seconds since epoch (or any monotonic scalar)
    spatial_load: float       # normalized [0, 1] load on spatial resources
    privacy_load: float       # normalized [0, 1] load on privacy‑budget


# ----------------------------------------------------------------------
# Hybrid mathematical building blocks
# ----------------------------------------------------------------------
def bayesian_update(prior: float, likelihood: float, fisher_weight: float, eps: float = 1e-12) -> float:
    """
    Return the Bayesian posterior where the likelihood is raised to a power
    given by the Fisher information.  This implements the bridge:

        posterior ∝ prior × likelihood^{fisher_weight}

    The exponent modulates how strongly the data (likelihood) influences the
    posterior; a high Fisher weight forces the posterior toward the likelihood,
    while a low weight keeps it close to the prior.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be in [0, 1]")
    if not 0.0 <= likelihood <= 1.0:
        raise ValueError("likelihood must be in [0, 1]")
    # Clamp fisher_weight to a reasonable range to avoid overflow/underflow
    fisher_weight = max(eps, fisher_weight)

    # Compute unnormalised posterior
    # Use log‑space for numerical stability when fisher_weight is large
    log_num = math.log(prior + eps) + fisher_weight * math.log(likelihood + eps)
    log_den = math.log(prior + eps) + fisher_weight * math.log(likelihood + eps)
    log_den += math.log(1 - prior + eps) + fisher_weight * math.log(1 - likelihood + eps)

    # Convert back from log‑space
    num = math.exp(log_num)
    den = math.exp(log_den)
    posterior = num / (den + eps)
    # Clamp to [0,1] to guard against rounding errors
    return max(0.0, min(1.0, posterior))


def vram_success_probability(entity: Entity,
                             center: float,
                             width: float,
                             base_prior: float = 0.5) -> float:
    """
    Compute the probability that a VRAM allocation request represented by
    *entity* will succeed.

    Steps
    -----
    1. Derive an “angle” from the timestamp (modulo 2π) – this supplies the
       scalar argument for the Gaussian beam.
    2. Compute the Fisher information of that angle.
    3. Build a raw likelihood from the entity’s loads: higher loads → lower
       likelihood, modelled with an exponential decay.
    4. Apply the Fisher‑weighted Bayesian update.
    """
    # 1. Angle from timestamp
    theta = (entity.timestamp % (2 * math.pi))

    # 2. Fisher information (acts as weight)
    fisher = fisher_score(theta, center, width)

    # 3. Load‑based likelihood (bounded in (0,1])
    load_sum = entity.spatial_load + entity.privacy_load
    likelihood = math.exp(-load_sum)  # 0 < likelihood ≤ 1

    # 4. Posterior probability
    return bayesian_update(base_prior, likelihood, fisher)


def best_entity(entities: List[Entity],
                center: float,
                width: float) -> Tuple[Entity, float]:
    """
    Return the entity with the highest hybrid VRAM‑success probability and the
    associated probability value.
    """
    if not entities:
        raise ValueError("entities list must not be empty")
    best = max(entities, key=lambda e: vram_success_probability(e, center, width))
    prob = vram_success_probability(best, center, width)
    return best, prob


def aggregate_hybrid_score(entities: List[Entity],
                           center: float,
                           width: float) -> float:
    """
    Compute a global hybrid score for a collection of entities.
    The score is the sum of Fisher‑weighted probabilities, providing a scalar
    measure of overall feasibility for the batch.
    """
    scores = [
        fisher_score((e.timestamp % (2 * math.pi)), center, width) *
        vram_success_probability(e, center, width)
        for e in entities
    ]
    return float(np.sum(scores))


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic workload of 10 entities with random timestamps and loads
    random.seed(42)
    entities = [
        Entity(
            timestamp=random.uniform(0, 1000),
            spatial_load=random.random(),
            privacy_load=random.random()
        )
        for _ in range(10)
    ]

    # Parameters for the Gaussian beam (tuned arbitrarily for the demo)
    CENTER = math.pi          # centre of the beam
    WIDTH = 0.8               # spread

    # Compute and display the best entity
    best, prob = best_entity(entities, CENTER, WIDTH)
    print("Best entity:")
    print(f"  timestamp    = {best.timestamp:.3f}")
    print(f"  spatial_load = {best.spatial_load:.3f}")
    print(f"  privacy_load = {best.privacy_load:.3f}")
    print(f"  hybrid success probability = {prob:.6f}")

    # Global score for the batch
    total_score = aggregate_hybrid_score(entities, CENTER, WIDTH)
    print(f"\nAggregate hybrid score for {len(entities)} entities: {total_score:.6f}")

    # Ensure the module runs without raising exceptions
    sys.exit(0)