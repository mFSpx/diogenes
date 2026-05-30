# DARWIN HAMMER — match 3124, survivor 0
# gen: 7
# parent_a: hybrid_fisher_localization_hybrid_hybrid_hybrid_m1503_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2694_s0.py (gen4)
# born: 2026-05-29T23:47:57Z

"""Hybrid Fisher–State Certainty Fusion (Parents: hybrid_fisher_localization_hybrid_hybrid_hybrid_m1503_s3.py,
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2694_s0.py)

Mathematical bridge
-------------------
* **Parent A** supplies a *Fisher information* score derived from a Gaussian beam
  (timestamp ≈ θ) and a Bayesian update that mixes the score with the probability
  of successful VRAM allocation.
* **Parent B** supplies a *state‑space* style certainty propagation based on
  morphological indices (righting‑time, recovery‑priority) that act as a
  process‑noise scale and a certainty flag.

The fusion treats the Fisher information as the *precision* (inverse variance) of a
measurement of the entity’s timestamp.  The morphological recovery‑priority is
interpreted as a *certainty weight* that scales the process variance of a simple
one‑dimensional Kalman‑like filter.  The Bayesian update from Parent A becomes the
measurement update of the filter, while the certainty‑weighted process model
represents the prediction step from Parent B.

The resulting hybrid system yields a posterior estimate of the “effective
timestamp” (or any scalar state) together with a confidence that simultaneously
encodes VRAM‑allocation likelihood, Fisher information, and morphological
certainty.  This posterior can be used to rank candidates, drive NLMS‑style
predictors, or guide allocation decisions.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures (shared from both parents)
# ----------------------------------------------------------------------
@dataclass
class Entity:
    """Temporal‑spatial observation used by the Fisher side."""
    timestamp: float          # θ
    spatial_load: float       # load on VRAM
    privacy_load: float       # not used directly in the hybrid core


@dataclass(frozen=True)
class Morphology:
    """Geometric descriptor used by the certainty side."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent‑A utilities (Fisher + Bayesian)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel value."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def vram_allocation_probability(entity: Entity, vram_capacity: float, eps: float = 1e-12) -> float:
    """Probability that the entity fits into the available VRAM."""
    return max(math.exp(-entity.spatial_load / vram_capacity), eps)


def bayesian_update(prior: float, likelihood: float, eps: float = 1e-12) -> float:
    """Simple multiplicative Bayesian update (unnormalised)."""
    return max(prior * likelihood, eps)


# ----------------------------------------------------------------------
# Parent‑B utilities (Morphology‑based certainty)
# ----------------------------------------------------------------------
def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2·height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical righting‑time proxy."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized certainty weight in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid core (three required functions)
# ----------------------------------------------------------------------
def hybrid_fisher_measure_precision(entity: Entity,
                                    center: float,
                                    width: float,
                                    eps: float = 1e-12) -> float:
    """
    Convert Fisher information into a measurement precision (inverse variance).

    The Fisher score from Parent A is exactly the *information* about the
    underlying θ.  In a Kalman‑like filter the measurement precision is
    `I = Fisher`.  To keep the scale numerically stable we bound it from below.
    """
    fisher = fisher_score(entity.timestamp, center, width, eps)
    # Prevent division by zero later; treat very small Fisher as minimal precision.
    return max(fisher, eps)


def hybrid_state_predict(mean_prior: float,
                         var_prior: float,
                         morphology: Morphology,
                         process_scale: float = 1.0) -> Tuple[float, float]:
    """
    Prediction step using morphology‑derived certainty.

    The recovery_priority acts as a certainty weight ω∈[0,1].
    Higher certainty ⇒ smaller process variance (more trust in the prior).
    The process variance is scaled by (1‑ω) and an optional user factor.
    """
    certainty = recovery_priority(morphology)
    var_process = process_scale * (1.0 - certainty)  # larger when certainty low
    mean_pred = mean_prior                     # simple random‑walk model
    var_pred = var_prior + var_process
    return mean_pred, var_pred


def hybrid_state_update(entity: Entity,
                        morphology: Morphology,
                        mean_prior: float,
                        var_prior: float,
                        vram_capacity: float,
                        center: float,
                        width: float) -> Tuple[float, float]:
    """
    Full hybrid Bayesian/Kalman update.

    1. Predict using morphology‑derived process variance.
    2. Compute measurement precision from Fisher information.
    3. Weight the measurement by VRAM‑allocation probability (Parent A).
    4. Perform the standard Kalman update with the resulting precision.
    """
    # ----- Prediction -----
    mean_pred, var_pred = hybrid_state_predict(mean_prior, var_prior, morphology)

    # ----- Measurement preparation -----
    meas_prec = hybrid_fisher_measure_precision(entity, center, width)  # 1/σ²_meas
    vram_prob = vram_allocation_probability(entity, vram_capacity)
    # Incorporate VRAM probability as an additional scaling of precision.
    meas_prec_scaled = meas_prec * vram_prob

    # ----- Kalman update (scalar) -----
    # Posterior variance: 1/σ²_post = 1/σ²_pred + meas_prec_scaled
    var_post = 1.0 / (1.0 / var_pred + meas_prec_scaled)
    # Posterior mean: weighted average
    mean_post = var_post * (mean_pred / var_pred + meas_prec_scaled * entity.timestamp)

    return mean_post, var_post


def rank_entities(entities: List[Entity],
                  morphologies: List[Morphology],
                  vram_capacity: float,
                  center: float,
                  width: float,
                  init_mean: float = 0.0,
                  init_var: float = 1e3) -> List[Tuple[Entity, float]]:
    """
    Rank a list of (entity, morphology) pairs by the posterior mean obtained from
    the hybrid update.  Higher posterior mean is interpreted as a more favorable
    candidate (e.g., higher likelihood of successful allocation under certainty).

    Returns a list of tuples (Entity, posterior_mean) sorted descending.
    """
    if len(entities) != len(morphologies):
        raise ValueError("entities and morphologies must have the same length")

    posterior_means: List[Tuple[Entity, float]] = []
    mean, var = init_mean, init_var
    for ent, morph in zip(entities, morphologies):
        mean, var = hybrid_state_update(ent, morph, mean, var,
                                        vram_capacity, center, width)
        posterior_means.append((ent, mean))
    # Sort descending by posterior mean
    posterior_means.sort(key=lambda pair: pair[1], reverse=True)
    return posterior_means


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic entities
    random.seed(42)
    entities = [
        Entity(timestamp=random.uniform(0, 10),
               spatial_load=random.uniform(0, 8),
               privacy_load=random.uniform(0, 1))
        for _ in range(5)
    ]

    # Corresponding morphologies (random but physically plausible)
    morphologies = [
        Morphology(length=random.uniform(0.5, 2.0),
                   width=random.uniform(0.5, 2.0),
                   height=random.uniform(0.2, 1.0),
                   mass=random.uniform(0.1, 5.0))
        for _ in range(5)
    ]

    # Hyper‑parameters for the hybrid model
    vram_capacity = 4.0          # arbitrary VRAM capacity
    center = 5.0                 # centre of Gaussian beam
    width = 2.0                  # width of Gaussian beam

    # Run ranking
    ranked = rank_entities(entities, morphologies,
                           vram_capacity, center, width)

    print("Ranked entities (posterior mean descending):")
    for ent, post_mean in ranked:
        print(f"Entity(ts={ent.timestamp:.3f}, load={ent.spatial_load:.3f}) → mean={post_mean:.4f}")