# DARWIN HAMMER — match 2762, survivor 6
# gen: 5
# parent_a: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_fisher_m22_s0.py (gen4)
# born: 2026-05-29T23:45:36Z

"""
Hybrid algorithm combining:
- Parent A (krampus_brainmap / Ollivier‑Ricci curvature feature extraction)
- Parent B (pheromone probability, entropy, Gaussian beam, Fisher information)

Mathematical bridge:
Both parents operate on probability‑like vectors.
Parent A builds a feature vector that can be normalised to a distribution over
semantic dimensions. Parent B produces a pheromone probability distribution.
The hybrid computes an Ollivier‑Ricci curvature approximation between the two
distributions (using total‑variation distance as the Wasserstein‑1 metric),
weights the resulting curvature with a Gaussian beam (θ‑dependent) and
evaluates a Fisher‑information‑style score. The final hybrid metric fuses
entropy, curvature and Fisher information into a single decision value.
"""

import sys
import math
import random
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Parent A – feature extraction (kept unchanged except for typing)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """Generate a synthetic high‑dimensional feature dictionary."""
    features: dict[str, float] = {}
    features.update({
        "operator_visceral_ratio": random.random(),
        "operator_tech_ratio": random.random(),
        "operator_legal_osint_ratio": random.random(),
    })
    features.update({
        "psyche_forensic_shield_ratio": random.random(),
        "psyche_poetic_entropy": random.random(),
        "psyche_dissociative_index": random.random(),
    })
    features.update({
        "resilience_bureaucratic_weaponization_index": random.random(),
        "resilience_resource_exhaustion_metric": random.random(),
        "resilience_swarm_orchestration_density": random.random(),
    })
    features.update({
        "rainmaker_corporate_grit_tension": random.random(),
        "rainmaker_countdown_density": random.random(),
        "rainmaker_asset_structuring_weight": random.random(),
    })
    features.update({
        "telemetry_agent_symmetry_ratio": random.random(),
        "telemetry_protocol_discipline": random.random(),
        "telemetry_manic_velocity": random.random(),
    })
    return features


def extract_master_vector(text: str) -> dict[str, float]:
    """Map the full feature set onto the canonical 12‑dimensional vector."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
    }

# ----------------------------------------------------------------------
# Parent B – pheromone handling, entropy, Gaussian beam, Fisher score
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(
    surface_key: str, limit: int = 10, db_url: str | None = None
) -> list[float]:
    """
    Return a synthetic pheromone probability distribution.
    In the original code this queried a PostgreSQL DB; here we fall back to a
    random positive vector to stay within the allowed import set.
    """
    # Guard against non‑positive limit
    limit = max(1, limit)
    # Generate positive random values and normalise
    raw = [random.random() + 0.01 for _ in range(limit)]
    total = sum(raw)
    return [v / total for v in raw]


def entropy(probabilities: list[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = [(p / total) for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in probs)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile used as a weighting kernel."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian family.
    For a Gaussian N(center, width^2) the Fisher information w.r.t. the mean
    is 1/width^2. We weight it by the Gaussian beam value to couple both parents.
    """
    g = gaussian_beam(theta, center, width)
    if width <= eps:
        raise ValueError("width must be positive")
    return g * (1.0 / (width * width))

# ----------------------------------------------------------------------
# Hybrid core – curvature, entropy, Fisher‑Gaussian fusion
# ----------------------------------------------------------------------
def _normalize_vector(vec: list[float]) -> list[float]:
    """Utility: L1‑normalise a vector to a probability distribution."""
    total = sum(vec)
    if total == 0:
        # fallback to uniform distribution of same length
        n = len(vec)
        return [1.0 / n] * n
    return [v / total for v in vec]


def ollivier_ricci_curvature(p: list[float], q: list[float]) -> float:
    """
    Approximate Ollivier‑Ricci curvature between two discrete measures.
    Using total‑variation distance TV(p,q) = 0.5 * L1(p−q) as the 1‑Wasserstein
    distance on a unit graph, curvature κ = 1 − TV(p,q) / d where d=1.
    """
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))
    return 1.0 - tv  # d = 1


def hybrid_metric(
    text: str,
    surface_key: str,
    limit: int = 10,
    theta: float = 0.0,
    beam_center: float = 0.0,
    beam_width: float = 1.0,
) -> dict[str, float]:
    """
    Compute a fused metric that blends:
    • Feature‑derived distribution (Parent A)
    • Pheromone distribution (Parent B)
    • Ollivier‑Ricci curvature between them
    • Entropy of the pheromone distribution
    • Fisher information weighted by a Gaussian beam

    Returns a dictionary with the intermediate and final values.
    """
    # 1. Feature vector → probability distribution
    feature_dict = extract_master_vector(text)
    feature_vec = list(feature_dict.values())
    p = _normalize_vector(feature_vec)

    # 2. Pheromone distribution
    q = calculate_pheromone_probabilities(surface_key, limit)

    # 3. Curvature between the two distributions
    curvature = ollivier_ricci_curvature(p, q)

    # 4. Entropy of pheromone distribution
    pheromone_entropy = entropy(q)

    # 5. Fisher‑Gaussian weighting
    g_weight = gaussian_beam(theta, beam_center, beam_width)
    fisher = fisher_score(theta, beam_center, beam_width)

    # 6. Fuse everything into a single decision score
    #    The formula is deliberately simple yet respects all components:
    #    score = (curvature * g_weight) - (pheromone_entropy * (1 - g_weight)) + fisher
    decision_score = (curvature * g_weight) - (pheromone_entropy * (1.0 - g_weight)) + fisher

    return {
        "curvature": curvature,
        "pheromone_entropy": pheromone_entropy,
        "gaussian_weight": g_weight,
        "fisher_information": fisher,
        "decision_score": decision_score,
    }


def hybrid_decision(text: str, surface_key: str) -> str:
    """
    High‑level decision routine.
    Returns a categorical recommendation based on the hybrid decision score.
    """
    metrics = hybrid_metric(text, surface_key, limit=12, theta=0.3, beam_center=0.0, beam_width=0.5)
    score = metrics["decision_score"]
    if score > 0.7:
        return "PROCEED"
    if score > 0.3:
        return "REVIEW"
    return "ABORT"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "Synthetic input for hybrid evaluation."
    sample_surface = "surface_XYZ"
    print("=== Hybrid Metric ===")
    result = hybrid_metric(
        sample_text,
        sample_surface,
        limit=8,
        theta=0.2,
        beam_center=0.0,
        beam_width=0.4,
    )
    for k, v in result.items():
        print(f"{k}: {v:.6f}")

    print("\nDecision:", hybrid_decision(sample_text, sample_surface))