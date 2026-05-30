# DARWIN HAMMER — match 1439, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_krampus_brain_m945_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_distributed_l_m1145_s0.py (gen3)
# born: 2026-05-29T23:36:19Z

"""Hybrid Algorithm: Fisher-NLMS-Tree Fusion
This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Fisher‑information scoring (gaussian_beam, fisher_score) applied to
  extracted feature densities.
* **Parent B** – Normalised Least‑Mean‑Squares (NLMS) adaptive update and a
  similarity‑based tree construction over Span objects.

**Mathematical Bridge**

The Fisher information scores are used as *information‑density weights* that
modulate the NLMS step‑size (μ) and act as a regulariser for the similarity
measure in the tree construction.  Concretely:

* μₕ = μ₀ · ⟨I⟩ where ⟨I⟩ is the mean Fisher score of the features present in
  the current input vector.
* The similarity between two spans sᵢ and sⱼ becomes  
  Sₕ(sᵢ,sⱼ) = (w ⊙ I)·[scoreᵢ, scoreⱼ] where “⊙” denotes element‑wise multiplication
  of the NLMS weight vector w with the Fisher‑information vector I.

Thus the algorithm jointly adapts its linear predictor (NLMS) while remaining
aware of the informational relevance of each feature (Fisher), and builds a
graph whose edges respect both geometric similarity and information density.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Gaussian‑beam based Fisher information for a single scalar feature."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def extract_full_features(text: str) -> dict[str, float]:
    """
    Deterministically generate a dense feature vector from arbitrary text.
    The hash of the text seeds a local RNG so that the same input always
    yields the same feature values.
    """
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
        "resilience_chaotic_good_tax"
    ]
    return {k: rnd.random() for k in keys}

# ----------------------------------------------------------------------
# Parent B – NLMS and Span utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float

Node = Hashable
Graph = Mapping[Node, set[Node]]

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Normalised LMS weight update:
        w_{t+1} = w_t + μ·e·x / (‖x‖²+ε)
    """
    y = nlms_predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power
    return next_weights, error

def broadcast_probability(phase: int, step: int) -> float:
    """
    Simple decreasing broadcast probability used in the original leader‑election
    parent.  It caps at 1.0 and falls off as 1/(phase·step).
    """
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (phase * step))

# ----------------------------------------------------------------------
# Hybrid core functions (bridge between the two parents)
# ----------------------------------------------------------------------
def compute_fisher_vector(
    feature_names: list[str],
    feature_values: dict[str, float],
    centers: dict[str, float],
    widths: dict[str, float],
) -> np.ndarray:
    """
    Convert raw feature values into a Fisher‑information vector I.
    Each component I_k = fisher_score(value_k, center_k, width_k).
    """
    scores = []
    for name in feature_names:
        val = feature_values.get(name, 0.0)
        cen = centers.get(name, 0.5)   # default centre
        wid = widths.get(name, 0.1)    # default width
        scores.append(fisher_score(val, cen, wid))
    return np.array(scores, dtype=float)

def hybrid_nlms_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    fisher_vec: np.ndarray,
    base_mu: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    NLMS update where the step‑size μ is scaled by the mean Fisher information
    of the active features:
        μₕ = μ₀ · mean(I)
    """
    mean_fisher = float(np.mean(fisher_vec)) if fisher_vec.size else 1.0
    mu_h = base_mu * mean_fisher
    return nlms_update(weights, x, target, mu=mu_h)

def construct_information_tree(
    spans: list[Span],
    weights: np.ndarray,
    fisher_vec: np.ndarray,
) -> dict[Span, list[tuple[Span, float]]]:
    """
    Build a similarity tree where each edge weight incorporates both the NLMS
    weight vector and the Fisher information vector:
        similarity = (w ⊙ I)·[score_i, score_j]
    """
    if weights.shape != fisher_vec.shape:
        raise ValueError("weights and fisher_vec must have the same shape")
    weighted_vec = weights * fisher_vec  # element‑wise product
    tree: dict[Span, list[tuple[Span, float]]] = {}
    for span in spans:
        neighbours: list[tuple[Span, float]] = []
        for other in spans:
            if span is other:
                continue
            # similarity uses the two scalar scores of the spans
            sim = float(np.dot(weighted_vec, np.array([span.score, other.score])))
            neighbours.append((other, sim))
        tree[span] = neighbours
    return tree

def adaptive_step_size(
    phase: int,
    step: int,
    fisher_vec: np.ndarray,
    base_mu: float = 0.5,
) -> float:
    """
    Combine the leader‑election broadcast probability with Fisher information
    to obtain a globally adaptive NLMS step‑size:
        μₐ = μ₀ · broadcast_probability(phase, step) · mean(I)
    """
    prob = broadcast_probability(phase, step)
    mean_fisher = float(np.mean(fisher_vec)) if fisher_vec.size else 1.0
    return base_mu * prob * mean_fisher

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Feature extraction and Fisher vector construction
    sample_text = "The quick brown fox jumps over the lazy dog."
    features = extract_full_features(sample_text)
    feature_names = list(features.keys())

    # Arbitrary centres and widths for demonstration
    centres = {k: 0.5 for k in feature_names}
    widths = {k: 0.2 for k in feature_names}

    fisher_vec = compute_fisher_vector(feature_names, features, centres, widths)

    # 2. Initialise NLMS weights (same dimensionality as feature vector)
    w = np.zeros_like(fisher_vec)

    # Dummy input vector x (use the raw feature values)
    x = np.array([features[k] for k in feature_names], dtype=float)

    # Target value – arbitrary
    target_val = 0.7

    # Perform a hybrid NLMS step
    w_new, err = hybrid_nlms_step(w, x, target_val, fisher_vec, base_mu=0.3)
    print(f"Hybrid NLMS update error: {err:.4f}")

    # 3. Build a small set of Span objects
    spans = [
        Span(0, 5, "alpha", "A", 0.1),
        Span(6, 10, "beta", "B", 0.4),
        Span(11, 15, "gamma", "C", 0.9),
    ]

    # Construct the information‑aware similarity tree
    tree = construct_information_tree(spans, w_new, fisher_vec[:2])  # use first 2 dims for demo
    for span, neighbours in tree.items():
        sims = ", ".join(f"{nbr.label}:{sim:.3f}" for nbr, sim in neighbours)
        print(f"Span {span.label} -> {sims}")

    # 4. Demonstrate adaptive global step size
    mu_adapt = adaptive_step_size(phase=2, step=5, fisher_vec=fisher_vec, base_mu=0.3)
    print(f"Adaptive global step size μ: {mu_adapt:.4f}")