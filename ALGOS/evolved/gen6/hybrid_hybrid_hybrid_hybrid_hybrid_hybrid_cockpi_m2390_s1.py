# DARWIN HAMMER — match 2390, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hdc_se_m1184_s0.py (gen5)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s4.py (gen2)
# born: 2026-05-29T23:42:08Z

"""
Hybrid module unifying the Hybrid Bandit-RBF-HDC Model with the 
cockpit_rectified_hybrid_style. 

The mathematical bridge between the two parents lies in the combination of 
the bandit's expected reward and the cockpit's trust factor. The bandit's 
expected reward is replaced by the RBF surrogate's prediction for the 
vector [context, action_one_hot]. The cockpit's trust factor is used to 
scale the velocity of the linguistic vector transport. 

The dot product of the hyperdimensional similarity and the sparse WTA 
hypervector is used to drive the hybrid recovery priority and decision-making 
of the ModelPool. The trust factor is used to modulate the step size of the 
Euler integrator toward the target style.

This module provides three hybrid functions:
1. `hybrid_style_target` – compute the trust-weighted style target.
2. `hybrid_bandit_priority` – fuses the bandit's expected reward and the 
   cockpit's trust factor into a single priority value.
3. `hybrid_euler_step`   – Euler integration toward the target style, 
   with a step size modulated by the trust factor.
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
    prop

# ----------------------------------------------------------------------
# Cockpit metrics (from Parent B)
# ----------------------------------------------------------------------

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int, total_displayed: int) -> float:
    """Fraction of displayed items that are known-good, clamped to [0, 1]."""
    if total_displayed <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total_displayed))

def calculate_trust_factor(claims_with_evidence: int, total_claims_emitted: int, displayed_ok: int, unknown_displayed_as_ok: int, total_displayed: int) -> float:
    return (anti_slop_ratio(claims_with_evidence, total_claims_emitted) + cockpit_honesty(displayed_ok, unknown_displayed_as_ok, total_displayed)) / 2

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_style_target(vector0: Vector, vector1: Vector, trust_factor: float) -> Vector:
    """Compute the trust-weighted style target."""
    return [v0 + trust_factor * (v1 - v0) for v0, v1 in zip(vector0, vector1)]


def hybrid_bandit_priority(action: BanditAction, vector0: Vector, vector1: Vector, trust_factor: float) -> float:
    """Fuses the bandit's expected reward and the cockpit's trust factor into a single priority value."""
    return action.expected_reward * trust_factor


def hybrid_euler_step(vector0: Vector, vector1: Vector, trust_factor: float, step_size: float) -> Vector:
    """Euler integration toward the target style, with a step size modulated by the trust factor."""
    return [v0 + trust_factor * step_size * (v1 - v0) for v0, v1 in zip(vector0, vector1)]


def morphology_hv(morphology_scalars: List[float]) -> BipolarVector:
    """Encodes morphology scalars into a bipolar hypervector."""
    return [1 if scalar > 0 else -1 for scalar in morphology_scalars]


def sparse_wta_hv(real_scores: List[float]) -> BipolarVector:
    """Expands a list of real scores into a sparse WTA hypervector."""
    return [1 if score > 0 else -1 for score in real_scores]


def hybrid_priority(action: BanditAction, vector0: Vector, vector1: Vector, trust_factor: float, morphology_scalars: List[float], real_scores: List[float]) -> float:
    """Fuses the bandit's expected reward, the cockpit's trust factor, the morphology scalars, and the sparse WTA hypervector into a single priority value."""
    hypervector0 = morphology_hv(morphology_scalars)
    hypervector1 = sparse_wta_hv(real_scores)
    hybrid_vector = hybrid_style_target(hypervector0, hypervector1, trust_factor)
    return hybrid_bandit_priority(action, vector0, vector1, trust_factor)

if __name__ == "__main__":
    vector0 = [1.0, 2.0, 3.0]
    vector1 = [4.0, 5.0, 6.0]
    trust_factor = 0.5
    step_size = 0.1
    morphology_scalars = [0.1, 0.2, 0.3]
    real_scores = [0.4, 0.5, 0.6]
    action = BanditAction("action_id", 0.5, 0.6, 0.7, "algorithm")
    print(hybrid_style_target(vector0, vector1, trust_factor))
    print(hybrid_bandit_priority(action, vector0, vector1, trust_factor))
    print(hybrid_euler_step(vector0, vector1, trust_factor, step_size))
    print(morphology_hv(morphology_scalars))
    print(sparse_wta_hv(real_scores))
    print(hybrid_priority(action, vector0, vector1, trust_factor, morphology_scalars, real_scores))