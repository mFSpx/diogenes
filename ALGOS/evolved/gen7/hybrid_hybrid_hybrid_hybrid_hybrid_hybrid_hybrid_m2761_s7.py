# DARWIN HAMMER — match 2761, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_hybrid_gliner_hybrid_hybrid_hybrid_m1138_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1491_s1.py (gen6)
# born: 2026-05-29T23:45:46Z

"""Hybrid Fisher-Entropy Bandit Algorithm
Integrates:
- Parent A: pheromone‑based bandit with Shannon entropy and Schoolfield developmental rate.
- Parent B: Fisher information from Gaussian beams applied to feature scores (e.g., from a count‑min sketch).

Mathematical bridge:
1. Compute Shannon entropy **H** of the normalized pheromone signal distribution.
2. Compute a temperature‑dependent developmental rate **R(T)** using the Schoolfield equation.
3. For each feature extracted (via the count‑min sketch) compute Fisher information **I(θ)** with the Gaussian‑beam formulation.
4. Fuse the three scalars into a unified weight **W = H * R(T) * I(θ)** that rescales the propensity of each BanditAction.

The resulting hybrid update simultaneously respects information‑theoretic signal richness (entropy),
thermodynamic activation (developmental rate), and statistical sensitivity (Fisher score)."""

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridFisherEntropyBandit"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C
    delta_h_activation: float = 12000.0   # J mol⁻¹
    t_low: float = 283.15             # K
    t_high: float = 307.15            # K
    delta_h_low: float = -45000.0     # J mol⁻¹
    delta_h_high: float = 65000.0     # J mol⁻¹
    r_cal: float = 1.987              # cal mol⁻¹ K⁻¹ → convert to J (≈8.314)

@dataclass(frozen=True)
class PheromoneEntry:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float

class PheromoneStore:
    @staticmethod
    def add(pheromone_entry: PheromoneEntry):
        # Placeholder for persistence; no‑op in this hybrid demo.
        pass

# ----------------------------------------------------------------------
# Helper functions (Parent A)
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def shannon_entropy(values: Iterable[float]) -> float:
    """Compute Shannon entropy of a non‑negative value list."""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log(p, 2) for p in probs)

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield temperature‑dependence model.
    R(T) = (ρ25 * exp(-ΔH‡/(R) * (1/T - 1/298.15))) /
           (1 + exp(ΔHL/(R) * (1/TL - 1/T)) + exp(ΔHH/(R) * (1/T - 1/TH)))
    """
    R = params.r_cal * 4.184  # convert cal·mol⁻¹·K⁻¹ to J·mol⁻¹·K⁻¹
    inv_T = 1.0 / temp_k
    inv_298 = 1.0 / 298.15

    numerator = params.rho_25 * math.exp(
        -params.delta_h_activation / R * (inv_T - inv_298)
    )
    denom = (
        1.0
        + math.exp(params.delta_h_low / R * (1.0 / params.t_low - inv_T))
        + math.exp(params.delta_h_high / R * (inv_T - 1.0 / params.t_high))
    )
    return numerator / denom

# ----------------------------------------------------------------------
# Functions from Parent B (Fisher information & sketch)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    I(θ) = (∂/∂θ ln f(θ))² = (derivative²) / intensity
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Simple count‑min sketch."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = hash(item) % width
            table[d][index] += 1
    return table


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Mock feature extractor – in a real system this would parse `text`.
    Returns a dictionary of synthetic feature values.
    """
    rng = random.Random(hash(text))
    features = {
        "operator_visceral_ratio": rng.random(),
        "operator_tech_ratio": rng.random(),
        "operator_legal_osint_ratio": rng.random(),
        "psyche_forensic_shield_ratio": rng.random(),
        "psyche_poetic_entropy": rng.random(),
        "psyche_dissociative_index": rng.random(),
        "resilience_bureaucratic_weaponization_index": rng.random(),
        "resilience_resource_exhaustion_metric": rng.random(),
        "resilience_swarm_orchestration_density": rng.random(),
    }
    return features

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_pheromone_entropy(store: List[PheromoneEntry]) -> float:
    """Aggregate pheromone signal values and return Shannon entropy."""
    values = [entry.signal_value for entry in store if entry.signal_value > 0]
    return shannon_entropy(values)


def compute_feature_fisher_weights(features: Dict[str, float]) -> Dict[str, float]:
    """
    Map each feature to a Fisher information weight.
    The raw feature value is treated as θ for the Gaussian beam.
    """
    weights = {}
    for name, val in features.items():
        # Use a moderate width; centre at 0.5 (mid‑range of random[0,1])
        weights[name] = fisher_score(theta=val, center=0.5, width=0.2)
    return weights


def hybrid_bandit_update(
    actions: List[BanditAction],
    pheromones: List[PheromoneEntry],
    temperature_c: float,
    features: Dict[str, float],
) -> List[BanditAction]:
    """
    Produce a new list of BanditAction objects whose propensities are
    rescaled by the fused weight:
        W = H * R(T) * I_feature
    where H is the pheromone entropy, R(T) the developmental rate,
    and I_feature the Fisher information of the feature linked to the action.
    The linking is done via a simple naming convention: action_id must match
    a feature key; if not found, I_feature defaults to 1.0.
    """
    # 1. Entropy of pheromone distribution
    H = compute_pheromone_entropy(pheromones)

    # 2. Developmental rate at given temperature
    T_k = c_to_k(temperature_c)
    R = developmental_rate(T_k)

    # 3. Fisher information per feature
    fisher_weights = compute_feature_fisher_weights(features)

    # 4. Fuse and produce updated actions
    updated = []
    for act in actions:
        I = fisher_weights.get(act.action_id, 1.0)
        new_propensity = act.propensity * H * R * I
        # Clamp to a reasonable range to avoid overflow
        new_propensity = max(min(new_propensity, 1e6), 0.0)
        updated.append(
            BanditAction(
                action_id=act.action_id,
                propensity=new_propensity,
                expected_reward=act.expected_reward,
                confidence_bound=act.confidence_bound,
                algorithm=act.algorithm,
            )
        )
    return updated


def hybrid_feature_sketch(text_corpus: Iterable[str]) -> Dict[str, float]:
    """
    Build a count‑min sketch from a corpus of strings,
    then derive synthetic features by counting sketch frequencies
    and feeding them through Fisher information.
    """
    # Flatten all words (simple split) into a list
    items = [word for doc in text_corpus for word in doc.split()]
    sketch = count_min_sketch(items, width=128, depth=6)

    # Approximate a feature: total count per depth normalized
    depth_sums = [sum(row) for row in sketch]
    total = sum(depth_sums) or 1.0
    raw_features = {f"depth_{i}_density": s / total for i, s in enumerate(depth_sums)}
    # Convert raw densities to Fisher scores
    fisher_features = {
        name: fisher_score(theta=val, center=0.5, width=0.1) for name, val in raw_features.items()
    }
    return fisher_features

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate synthetic pheromone entries
    pheromones = [
        PheromoneEntry(
            surface_key=f"node_{i}",
            signal_kind="trail",
            signal_value=random.random(),
            half_life_seconds=30.0 + i,
        )
        for i in range(10)
    ]

    # 2. Create dummy bandit actions; use feature keys that will appear later
    actions = [
        BanditAction(
            action_id=f"depth_{i}_density",
            propensity=1.0 + i * 0.1,
            expected_reward=random.random(),
            confidence_bound=0.5,
        )
        for i in range(6)
    ]

    # 3. Build features from a tiny corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "entropy and fisher information guide hybrid decisions",
    ]
    features = hybrid_feature_sketch(corpus)

    # 4. Run hybrid update at 22 °C
    updated_actions = hybrid_bandit_update(
        actions=actions,
        pheromones=pheromones,
        temperature_c=22.0,
        features=features,
    )

    # 5. Display results
    for act in updated_actions:
        print(f"Action {act.action_id}: propensity={act.propensity:.4f}")

    print("\nAll steps executed without error.")