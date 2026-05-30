# DARWIN HAMMER — match 2762, survivor 7
# gen: 5
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s0.py (gen4)
# born: 2026-05-29T23:45:36Z

"""
Hybrid Algorithm: Krampus-Ollivier-Ricci BrainMap + Pheromone Fisher‑Gaussian Fusion

Parent A:   hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py  
Parent B:   hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s0.py  

Mathematical Bridge
-------------------
The bridge is built on the observation that both parents manipulate *probability
distributions*:

* Parent A produces a high‑dimensional feature vector that can be interpreted as a
  probability‑like weighting of operator/psyche/resilience dimensions.
* Parent B normalises raw pheromone signals into a probability distribution,
  computes its entropy, and evaluates a Fisher‑information‑based score using a
  Gaussian beam intensity model.

We therefore treat the normalized feature vector from A as a *prior* weighting,
and the pheromone probability distribution from B as a *likelihood*.  The
hybrid metric multiplies three mathematically compatible quantities:

1. **Feature mass** – the ℓ₁‑norm of the master feature vector (a scalar weight).
2. **Entropy H(p)** – Shannon entropy of the pheromone distribution (information
   content).
3. **Fisher‑Gaussian coupling** – the Fisher score of a Gaussian beam evaluated
   at a decision angle 𝜃, modulated by the Gaussian intensity profile.

The product  

  M = ‖v‖₁ · H(p) · G(θ) · F(θ)

provides a unified scalar that simultaneously reflects brain‑map topology,
information‑theoretic diversity of surface usage, and decision‑hygiene
sensitivity to angular deviation.

The implementation below realises this fusion while respecting the import
constraints.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Parent A – feature extraction (kept verbatim, truncated safely)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Generate a synthetic high‑dimensional feature dictionary."""
    features: dict[str, float] = {}
    # Operator group
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    # Psyche group
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    # Resilience group
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    # Rainmaker group
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    # Telemetry group
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def extract_master_vector(text: str) -> dict[str, float]:
    """Condense the full feature dict to a flat master vector."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "bureaucratic_weaponization_index": f.get("resilience_bureaucratic_weaponization_index", 0.0),
        "resource_exhaustion_metric": f.get("resilience_resource_exhaustion_metric", 0.0),
        "swarm_orchestration_density": f.get("resilience_swarm_orchestration_density", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }

# ----------------------------------------------------------------------
# Parent B – probability handling and Gaussian‑Fisher utilities
# ----------------------------------------------------------------------
def normalize_distribution(values: list[float]) -> list[float]:
    """Convert a raw list of non‑negative numbers into a probability distribution."""
    total = sum(values)
    if total <= 0:
        raise ValueError("Sum of values must be positive for normalization.")
    return [v / total for v in values]


def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    """Shannon entropy H(p) = - Σ p_i log p_i."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Probability mass must be positive.")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile G(θ) = exp( -½ ((θ‑center)/width)² )."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam with respect to the angle θ.
    For a Gaussian G(θ) the derivative dG/dθ = -((θ‑center)/width²)·G(θ).
    The Fisher score I(θ) = (d ln G / dθ)² = ((θ‑center)/width²)².
    """
    if width <= 0:
        raise ValueError("width must be positive")
    diff = theta - center
    return (diff / (width * width)) ** 2 + eps  # eps prevents exact zero


# ----------------------------------------------------------------------
# Hybrid Core – mathematically fusing the two parent topologies
# ----------------------------------------------------------------------
def compute_feature_mass(text: str) -> float:
    """
    L1‑norm of the master feature vector.
    Acts as a scalar weight summarising the brain‑map topology.
    """
    vec = extract_master_vector(text)
    return float(np.linalg.norm(list(vec.values()), ord=1))


def hybrid_metric(
    text: str,
    pheromone_values: list[float],
    theta: float,
    center: float,
    width: float,
) -> float:
    """
    Unified hybrid metric M = ‖v‖₁ · H(p) · G(θ) · I(θ).

    Parameters
    ----------
    text : str
        Arbitrary input used for feature extraction (Parent A).
    pheromone_values : list[float]
        Raw pheromone signal strengths (Parent B). They are normalised internally.
    theta, center, width : float
        Parameters of the Gaussian beam that modulate decision hygiene (Parent B).

    Returns
    -------
    float
        The scalar hybrid score.
    """
    # 1️⃣ Feature mass from Parent A
    feature_mass = compute_feature_mass(text)

    # 2️⃣ Normalised pheromone distribution and its entropy from Parent B
    probs = normalize_distribution(pheromone_values)
    ent = entropy(probs)

    # 3️⃣ Gaussian intensity and Fisher information (Parent B)
    g_intensity = gaussian_beam(theta, center, width)
    fisher = fisher_score(theta, center, width)

    # Combine multiplicatively – the mathematical bridge
    return feature_mass * ent * g_intensity * fisher


def hybrid_decision_surface(
    texts: list[str],
    pheromone_matrix: list[list[float]],
    theta_grid: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Evaluate the hybrid metric over a grid of decision angles for multiple
    inputs.  Returns a 2‑D array where rows correspond to `texts` and columns
    correspond to `theta_grid` values.
    """
    results = np.zeros((len(texts), len(theta_grid)), dtype=float)
    for i, txt in enumerate(texts):
        # Average pheromone distribution across the provided matrix row
        raw_vals = pheromone_matrix[i]
        probs = normalize_distribution(raw_vals)
        ent = entropy(probs)  # entropy is independent of theta, compute once
        feat_mass = compute_feature_mass(txt)

        for j, th in enumerate(theta_grid):
            g = gaussian_beam(th, center, width)
            f = fisher_score(th, center, width)
            results[i, j] = feat_mass * ent * g * f
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Example inputs
    sample_text = "Synthetic operational briefing for hybrid metric."
    pheromone_vals = [random.expovariate(1.0) for _ in range(7)]

    # Decision parameters
    theta_val = 0.3
    center_val = 0.0
    width_val = 0.5

    # Compute single hybrid metric
    score = hybrid_metric(
        text=sample_text,
        pheromone_values=pheromone_vals,
        theta=theta_val,
        center=center_val,
        width=width_val,
    )
    print(f"Hybrid metric score: {score:.6f}")

    # Grid evaluation
    texts = [f"Scenario {i}" for i in range(3)]
    pher_matrix = [
        [random.expovariate(1.0) for _ in range(5)] for _ in range(3)
    ]
    theta_grid = np.linspace(-1.0, 1.0, 9)

    surface = hybrid_decision_surface(
        texts=texts,
        pheromone_matrix=pher_matrix,
        theta_grid=theta_grid,
        center=center_val,
        width=width_val,
    )
    print("Hybrid decision surface (rows: texts, cols: theta):")
    print(surface)