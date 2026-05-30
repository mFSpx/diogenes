# DARWIN HAMMER — match 5379, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m2390_s1.py (gen6)
# parent_b: hybrid_temporal_motifs_possum_filter_m87_s1.py (gen1)
# born: 2026-05-30T00:01:36Z

"""
Hybrid module unifying the Hybrid Bandit-RBF-HDC Model with the temporal_spatial_fusion algorithm.

The mathematical bridge between the two parents lies in the combination of 
the bandit's expected reward and the temporal motif's support count. 
The bandit's expected reward is used to modulate the support count of the temporal motif, 
while the temporal motif's support count is used to update the bandit's expected reward.

The dot product of the hyperdimensional similarity and the sparse WTA hypervector 
is used to drive the hybrid recovery priority and decision-making of the ModelPool. 
The trust factor is used to modulate the step size of the Euler integrator toward the target style.

This module provides three hybrid functions:
1. `hybrid_style_target` – compute the trust-weighted style target.
2. `hybrid_bandit_priority` – fuses the bandit's expected reward and the temporal motif's support count into a single priority value.
3. `hybrid_euler_step`   – Euler integration toward the target style, with a step size modulated by the trust factor.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
BipolarVector = List[int]

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    prop: float


# ----------------------------------------------------------------------
# Temporal motif utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int


@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio‑temporal motif."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    score: float  # combined temporal‑spatial score


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_style_target(trust_factor: float, style_vector: Vector) -> Vector:
    """
    Compute the trust-weighted style target.
    
    :param trust_factor: The trust factor to modulate the step size of the Euler integrator.
    :param style_vector: The style vector to compute the target.
    :return: The trust-weighted style target.
    """
    return [x * trust_factor for x in style_vector]


def hybrid_bandit_priority(bandit_action: BanditAction, temporal_motif: TemporalMotif) -> float:
    """
    Fuses the bandit's expected reward and the temporal motif's support count into a single priority value.
    
    :param bandit_action: The bandit action to compute the priority.
    :param temporal_motif: The temporal motif to compute the priority.
    :return: The hybrid bandit priority.
    """
    return bandit_action.expected_reward * temporal_motif.support


def hybrid_euler_step(trust_factor: float, style_vector: Vector, target_vector: Vector) -> Vector:
    """
    Euler integration toward the target style, with a step size modulated by the trust factor.
    
    :param trust_factor: The trust factor to modulate the step size of the Euler integrator.
    :param style_vector: The style vector to integrate.
    :param target_vector: The target style vector.
    :return: The integrated style vector.
    """
    return [x + trust_factor * (y - x) for x, y in zip(style_vector, target_vector)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    trust_factor = 0.5
    style_vector = [1.0, 2.0, 3.0]
    target_vector = [4.0, 5.0, 6.0]
    bandit_action = BanditAction("action1", 0.5, 10.0, 5.0, "algorithm1")
    temporal_motif = TemporalMotif(("pattern1",), 10)
    
    print(hybrid_style_target(trust_factor, style_vector))
    print(hybrid_bandit_priority(bandit_action, temporal_motif))
    print(hybrid_euler_step(trust_factor, style_vector, target_vector))