# DARWIN HAMMER — match 1606, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_krampus_brain_m1053_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s0.py (gen5)
# born: 2026-05-29T23:37:42Z

"""Hybrid Fusion of Algorithm A and Algorithm B

This module fuses the feature‑extraction / endpoint logic of *hybrid_hybrid_hybrid_hybrid_hybrid_krampus_brain_m1053_s2.py*
(Parent A) with the geometric‑algebra / Shannon‑entropy driven rotor mechanics of
*hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s0.py* (Parent B).

Mathematical bridge:
- Parent A produces a high‑dimensional feature dictionary `f`.
- Parent B computes Shannon entropy `H(f)` of a normalized feature count vector.
- The entropy is used as a rotation angle `θ = π·H` to build a simple rotor (2‑D rotation
  matrix) that transforms the action‑expectation vector derived from Parent A.
- The endpoint failure rate `ρ = failures / (threshold+ε)` modulates the learning rates
  in the GA‑rotor update, providing a feedback loop between reliability (A) and geometric
  transformation (B).

The result is a unified hybrid system where statistical uncertainty directly drives
geometric transformations of decision vectors.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Parent A structures (trimmed / adapted)
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    failures: int
    failure_threshold: int
    righting_time_index: float

    @property
    def failure_rate(self) -> float:
        """Failure rate ρ ∈ [0,1]"""
        return self.failures / (self.failure_threshold + 1e-9)


@dataclass(frozen=True, slots=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True, slots=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


def extract_full_features(text: str) -> Dict[str, float]:
    """Generate a synthetic feature dictionary using Beta‑distributed draws."""
    features: Dict[str, float] = {}
    # operator domain
    features.update({
        "operator_visceral_ratio": np.random.beta(1, 1),
        "operator_tech_ratio": np.random.beta(1, 1),
        "operator_legal_osint_ratio": np.random.beta(1, 1),
    })
    # psyche domain
    features.update({
        "psyche_forensic_shield_ratio": np.random.beta(1, 1),
        "psyche_poetic_entropy": np.random.beta(1, 1),
        "psyche_dissociative_index": np.random.beta(1, 1),
    })
    # resilience domain
    features.update({
        "resilience_bureaucratic_weaponization_index": np.random.beta(1, 1),
        "resilience_resource_exhaustion_metric": np.random.beta(1, 1),
        "resilience_swarm_orchestration_density": np.random.beta(1, 1),
    })
    # rainmaker domain
    features.update({
        "rainmaker_corporate_grit_tension": np.random.beta(1, 1),
        "rainmaker_countdown_density": np.random.beta(1, 1),
        "rainmaker_asset_structuring_weight": np.random.beta(1, 1),
    })
    # telemetry domain
    features.update({
        "telemetry_agent_symmetry_ratio": np.random.beta(1, 1),
        "telemetry_protocol_discipline": np.random.beta(1, 1),
        "telemetry_manic_velocity": np.random.beta(1, 1),
    })
    return features


def extract_master_vector(text: str) -> Dict[str, float]:
    """Compact vector derived from the full feature set."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
    }


# ----------------------------------------------------------------------
# Parent B utilities (entropy + rotor)
# ----------------------------------------------------------------------
def shannon_entropy(prob_dist: np.ndarray) -> float:
    """Calculate Shannon entropy H = -∑ p·log2(p) for a probability distribution."""
    # Guard against zero entries
    prob = prob_dist[prob_dist > 0]
    return -np.sum(prob * np.log2(prob))


def compute_feature_entropy(features: Dict[str, float]) -> float:
    """Normalize feature values to a probability distribution and compute entropy."""
    values = np.array(list(features.values()), dtype=float)
    total = np.sum(values) + 1e-12
    prob = values / total
    return shannon_entropy(prob)


def rotor_matrix(theta: float, dim: int = 2) -> np.ndarray:
    """
    Build a simple planar rotor (rotation matrix) of size `dim`.
    For dim > 2 the rotation is applied to the first two axes; remaining axes are identity.
    """
    R = np.eye(dim)
    c, s = math.cos(theta), math.sin(theta)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def apply_rotor(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply rotor matrix R to vector x."""
    return R @ x


# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate the fused behaviour)
# ----------------------------------------------------------------------
def hybrid_action_vector(actions: List[MathAction]) -> np.ndarray:
    """
    Convert a list of MathAction objects into a numeric vector.
    The vector contains expected values, costs and risks stacked.
    """
    ev = np.array([a.expected_value for a in actions], dtype=float)
    cost = np.array([a.cost for a in actions], dtype=float)
    risk = np.array([a.risk for a in actions], dtype=float)
    # Pad to the same length if needed (here they are equal)
    return np.concatenate([ev, cost, risk])


def hybrid_update(
    actions: List[MathAction],
    endpoint: Endpoint,
    feature_dict: Dict[str, float],
    eta_w: float = 0.01,
    eta_r: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    Perform one hybrid update step:
    1. Build the action vector `a`.
    2. Compute feature entropy `H`.
    3. Build rotor angle θ = π·H and rotor matrix R.
    4. Rotate the action vector → `a_rot`.
    5. Modulate learning rates by endpoint failure_rate ρ.
    6. Return the transformed vector and a scalar “score” that mixes
       the rotated expectation with the failure rate.
    """
    # 1. Action vector
    a = hybrid_action_vector(actions)

    # 2. Entropy from features
    H = compute_feature_entropy(feature_dict)  # H ∈ [0, log2(N)]

    # Normalise H to [0,1] for angle scaling
    max_entropy = math.log2(len(feature_dict)) if feature_dict else 1.0
    H_norm = H / (max_entropy + 1e-12)

    # 3. Rotor
    theta = math.pi * H_norm  # angle in radians
    dim = min(2, a.shape[0])  # ensure rotor fits dimension
    R = rotor_matrix(theta, dim=dim)

    # 4. Rotate (only first two components are rotated)
    a_rot = a.copy()
    a_rot[:dim] = apply_rotor(R, a[:dim])

    # 5. Failure‑rate modulation
    rho = endpoint.failure_rate  # ρ ∈ [0,1]
    eta_w_mod = eta_w * (1.0 - rho)
    eta_r_mod = eta_r * (1.0 - rho)

    # 6. Simple GA‑rotor‑like update (linear blend)
    updated = (1 - eta_w_mod) * a + eta_w_mod * a_rot
    score = np.mean(updated) * (1.0 - rho) + rho * np.std(updated)

    return updated, score


def hybrid_decision_score(
    actions: List[MathAction],
    endpoint: Endpoint,
    text: str,
) -> float:
    """
    End‑to‑end decision score:
    - Extract master feature vector from `text`.
    - Run `hybrid_update` to obtain a transformed action vector.
    - Return a scalar that combines the mean of the transformed vector with the
      endpoint failure rate, mirroring the risk‑adjusted utility used in Parent A.
    """
    features = extract_master_vector(text)
    transformed, _ = hybrid_update(actions, endpoint, features)
    mean_val = np.mean(transformed)
    # Adjust by failure rate (higher failure → lower confidence)
    return mean_val * (1.0 - endpoint.failure_rate)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy text for feature extraction
    dummy_text = "synthetic input for hybrid fusion test"

    # Create a few sample actions
    sample_actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=0.1),
        MathAction(id="A2", expected_value=5.0, cost=1.5, risk=0.2),
        MathAction(id="A3", expected_value=7.5, cost=2.5, risk=0.15),
    ]

    # Endpoint with moderate failures
    ep = Endpoint(failures=3, failure_threshold=10, righting_time_index=0.8)

    # Run the end‑to‑end decision score
    score = hybrid_decision_score(sample_actions, ep, dummy_text)
    print(f"Hybrid decision score: {score:.4f}")

    # Additionally demonstrate the raw update output
    features = extract_master_vector(dummy_text)
    updated_vec, entropy_score = hybrid_update(sample_actions, ep, features)
    print(f"Updated vector (first 5 elements): {updated_vec[:5]}")
    print(f"Entropy‑derived score: {entropy_score:.4f}")