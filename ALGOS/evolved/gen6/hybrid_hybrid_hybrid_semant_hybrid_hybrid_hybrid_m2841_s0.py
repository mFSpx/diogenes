# DARWIN HAMMER — match 2841, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1314_s0.py (gen5)
# born: 2026-05-29T23:46:11Z

"""Hybrid Semantic‑Krampus‑Morphology Algorithm

Parents:
- hybrid_hybrid_semantic_neig_hybrid_hybrid_krampu_m540_s0 (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1314_s0 (Algorithm B)

Mathematical bridge:
Algorithm A provides a semantic feature space, pheromone probabilities and a cosine‑based
approximation of Ollivier‑Ricci curvature. Algorithm B contributes a Fisher‑information based
confidence weight (fisher_score) and morphology‑driven descriptors (sphericity, flatness).
The hybrid algorithm multiplies the pheromone distribution by the Fisher confidence weight,
uses this weighted distribution to scale the curvature‑adjusted similarity between two
semantic‑morphology vectors, and finally performs a JEPA‑style representation update:

    Δrep = w_fisher · p_weighted · (target – rep)

where `p_weighted` are pheromone probabilities normalized and scaled by `w_fisher`. This
unifies the stochastic search of A with the information‑theoretic confidence of B while
embedding geometric morphology into the semantic similarity measure.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Algorithm A
# ----------------------------------------------------------------------
def _cos(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    """Normalize raw pheromone strengths to a probability distribution."""
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone list must contain at least one positive value")
    return [p / total for p in pheromones]


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Expectation of entropy conditioned on a hit probability."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Feature extraction (semantic) – trimmed version of A's extractor
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature map derived from the input string."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countd"
    ]
    # Produce a float in [0,1) for each key
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Core utilities from Algorithm B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity based on the cubic root of volume over longest edge."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio (average of base sides over height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def gaussian_beam(theta: float, center: float = 0.0, width: float = 1.0) -> float:
    """Gaussian intensity profile."""
    return math.exp(-((theta - center) ** 2) / (2 * width * width))


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam – serves as a confidence weight."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def morphology_features(morph: Morphology) -> Dict[str, float]:
    """Derive morphology‑driven scalar descriptors."""
    return {
        "sphericity": sphericity_index(morph.length, morph.width, morph.height),
        "flatness": flatness_index(morph.length, morph.width, morph.height),
        "mass_norm": morph.mass / (morph.length * morph.width * morph.height + 1e-12)
    }


def combine_features(text: str, morph: Morphology) -> np.ndarray:
    """
    Concatenate deterministic semantic features with morphology descriptors
    into a single dense vector.
    """
    sem_feat = extract_full_features(text)
    morph_feat = morphology_features(morph)
    # Order is deterministic: semantic keys sorted, then morphology keys sorted
    combined = [sem_feat[k] for k in sorted(sem_feat.keys())] + \
               [morph_feat[k] for k in sorted(morph_feat.keys())]
    return np.array(combined, dtype=float)


def weighted_curvature(vec_a: np.ndarray, vec_b: np.ndarray, fisher_w: float) -> float:
    """
    Approximate Ollivier‑Ricci curvature between two vectors.
    The curvature is taken as the Fisher‑weighted deviation of cosine similarity
    from its maximal value (1).
    """
    cos_sim = _cos(vec_a.tolist(), vec_b.tolist())
    # Curvature proxy: negative when vectors diverge, zero when identical
    curvature = fisher_w * (cos_sim - 1.0)
    return curvature


def jepa_update(rep: np.ndarray,
                target: np.ndarray,
                pheromones: List[float],
                fisher_w: float,
                learning_rate: float = 0.1) -> np.ndarray:
    """
    JEPA‑style representation update.
    The pheromone distribution is first normalised, then scaled by the Fisher confidence.
    The update moves the representation toward the target proportionally to the weighted
    error term.
    """
    p_norm = np.array(pheromone_probabilities(pheromones), dtype=float)
    weight = fisher_w * p_norm.mean()  # simple scalar confidence
    error = target - rep
    return rep + learning_rate * weight * error


def hybrid_step(text_a: str,
                text_b: str,
                morph_a: Morphology,
                morph_b: Morphology,
                pheromones: List[float],
                theta: float) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid iteration:
      1. Build combined feature vectors for the two agents.
      2. Compute Fisher confidence from `theta`.
      3. Evaluate curvature‑adjusted similarity.
      4. Update the representation of agent A towards agent B using JEPA dynamics.
    Returns the updated representation of A and the curvature value.
    """
    vec_a = combine_features(text_a, morph_a)
    vec_b = combine_features(text_b, morph_b)

    fisher_w = fisher_score(theta)

    curvature = weighted_curvature(vec_a, vec_b, fisher_w)

    updated_a = jepa_update(vec_a, vec_b, pheromones, fisher_w)

    return updated_a, curvature


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample inputs
    txt1 = "Quantum entanglement in cryptographic protocols."
    txt2 = "Neural manifold learning for adaptive control."
    morph1 = Morphology(length=2.3, width=1.5, height=0.9, mass=4.2)
    morph2 = Morphology(length=1.8, width=1.2, height=1.0, mass=3.5)
    pher = [0.2, 0.5, 0.3]          # example pheromone strengths
    theta_val = 0.7                # angle / parameter for Fisher score

    updated_vec, curv = hybrid_step(txt1, txt2, morph1, morph2, pher, theta_val)

    print("Updated representation (first 5 elements):", updated_vec[:5])
    print("Curvature (Fisher‑weighted):", curv)