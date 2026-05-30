# DARWIN HAMMER — match 945, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py (gen2)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py (gen1)
# born: 2026-05-29T23:31:49Z

"""
Hybrid Fisher‑Ricci Algorithm
============================

Parents:
- **hybrid_hybrid_fisher_locali_jepa_energy_m52_s0.py** – provides a Fisher‑information
  based scoring function for angular (temporal) parameters.
- **hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py** – extracts a high‑dimensional
  feature vector from free‑form text and supplies a toy implementation of Ollivier‑Ricci
  curvature on that representation space.

Mathematical Bridge
-------------------
In information geometry the Fisher information matrix *I(θ)* plays the role of a
Riemannian metric on the statistical manifold of parameters *θ*.  Ollivier‑Ricci
curvature *κ(x, y)* is defined on a metric space *(X, d)* by comparing the
Wasserstein‑1 distance between local probability measures with the base distance
*d(x, y)*.  By treating the scalar Fisher score *F(θ)* (the trace of *I(θ)*) as a
local “metric density”, we can weight the curvature computed on the feature
space.  The hybrid score therefore reads

    S(θ, t) = F(θ) · κ( φ(t) , μ ),

where *θ* is an angular representation of a datetime, *t* is the raw text,
*φ(t)* is the extracted feature vector, and *μ* is a reference vector (e.g. the
mean of a corpus).  This fuses the temporal‑information density of the Fisher
component with the geometric‑information density of the Ricci component.

The implementation below provides:
1. `fisher_score` – unchanged from parent A.
2. `extract_master_vector` – unchanged from parent B.
3. `angular_representation` – converts a `datetime` to an angle *θ*.
4. `ricci_curvature` – a lightweight Ollivier‑Ricci estimator using Euclidean
   distances between noisy neighbourhoods of two vectors.
5. `hybrid_information_curvature` – the combined hybrid score *S(θ, t)*.

All functions are pure NumPy/Python and require only the standard library and
`numpy`.
"""

import math
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Return a normalized Gaussian evaluated at *theta*."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a one‑dimensional Gaussian beam.
    Implements (∂I/∂θ)² / I where I is the beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Parent B – Feature extraction utilities
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> dict[str, float]:
    """
    Deterministic pseudo‑random feature generation based on the hash of *text*.
    Returns a dictionary with 24 scalar descriptors.
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
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}

def extract_master_vector(text: str) -> dict[str, float]:
    """
    Condenses the full feature set to a smaller, semantically grouped vector.
    """
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }

def vector_to_numpy(vec: dict[str, float]) -> np.ndarray:
    """Deterministically order the dict keys and return a NumPy vector."""
    if not vec:
        return np.zeros(0)
    keys = sorted(vec.keys())
    return np.array([vec[k] for k in keys], dtype=float)

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def angular_representation(dt: datetime) -> float:
    """
    Convert a UTC datetime to an angle θ ∈ [0, 2π) representing the time‑of‑day.
    Midnight → 0, Noon → π.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    seconds = (
        dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000
    )
    fraction = seconds / 86_400.0  # seconds per day
    return fraction * 2.0 * math.pi

def ricci_curvature(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    sigma: float = 0.5,
    n_samples: int = 30,
) -> float:
    """
    Toy Ollivier‑Ricci curvature between two points in Euclidean space.

    The local probability measure μ_x is approximated by drawing *n_samples*
    points from a spherical Gaussian N(x, σ² I).  The 1‑Wasserstein distance
    between the two empirical measures is estimated by the average Euclidean
    distance between paired samples (optimal transport for i.i.d. Gaussians
    reduces to this simple pairing).  Curvature is

        κ = 1 - W₁(μ_a, μ_b) / d(a, b)

    where *d* is the Euclidean distance between the centers.
    """
    d_center = np.linalg.norm(vec_a - vec_b)
    if d_center < 1e-12:
        return 0.0  # coincident points ⇒ zero curvature by definition

    # Sample neighbourhoods
    rng = np.random.default_rng(seed=0)  # deterministic for reproducibility
    neigh_a = rng.normal(loc=vec_a, scale=sigma, size=(n_samples, vec_a.size))
    neigh_b = rng.normal(loc=vec_b, scale=sigma, size=(n_samples, vec_b.size))

    # Pairwise distance (optimal coupling for identical i.i.d. samples)
    pairwise = np.linalg.norm(neigh_a - neigh_b, axis=1)
    w1 = pairwise.mean()
    return 1.0 - (w1 / d_center)

def hybrid_information_curvature(
    dt: datetime,
    text: str,
    reference_vec: dict[str, float] | None = None,
    fisher_center: float = math.pi,
    fisher_width: float = math.pi / 4,
) -> float:
    """
    Compute the hybrid score S(θ, t) = F(θ) · κ( φ(t), μ ).

    Parameters
    ----------
    dt : datetime
        Timestamp that is transformed into an angle θ.
    text : str
        Raw textual input from which a feature vector φ(t) is derived.
    reference_vec : dict or None
        Optional reference feature vector μ.  If omitted, the function uses the
        vector extracted from the *text* itself, yielding κ≈0 (self‑curvature).
    fisher_center, fisher_width : float
        Parameters of the Gaussian beam that defines the Fisher metric density.
    """
    # 1. Fisher component
    theta = angular_representation(dt)
    F = fisher_score(theta, fisher_center, fisher_width)

    # 2. Feature vectors
    vec_t_dict = extract_master_vector(text)
    vec_t = vector_to_numpy(vec_t_dict)

    if reference_vec is None:
        # Use the same vector as a degenerate reference – curvature will be zero.
        vec_ref = vec_t.copy()
    else:
        vec_ref = vector_to_numpy(reference_vec)

    # Guard against empty vectors (e.g., empty text)
    if vec_t.size == 0 or vec_ref.size == 0:
        return 0.0

    # 3. Ricci curvature component
    kappa = ricci_curvature(vec_t, vec_ref)

    # 4. Hybrid score
    return F * kappa

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample datetime (UTC) and text payload
    sample_dt = datetime(2023, 3, 14, 15, 9, 26, tzinfo=timezone.utc)  # Pi day
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    In cryptographic lore, entropy is the fox, and the dog is the
    deterministic algorithm that never learns.
    """

    # Compute a reference vector from a different piece of text to obtain
    # a non‑trivial curvature.
    reference_text = "A completely unrelated sentence that serves as a baseline."
    reference_vector = extract_master_vector(reference_text)

    hybrid_score = hybrid_information_curvature(
        dt=sample_dt,
        text=sample_text,
        reference_vec=reference_vector,
    )

    print(f"Hybrid Fisher‑Ricci score: {hybrid_score:.6f}")

    # Additional sanity checks
    assert isinstance(hybrid_score, float)
    # Verify that the score is bounded (Fisher score ≥0, curvature ∈ [-∞, 1])
    assert hybrid_score >= -1e-6  # numerical tolerance

    # Show intermediate components for curiosity
    theta = angular_representation(sample_dt)
    fisher = fisher_score(theta, math.pi, math.pi / 4)
    vec_t = vector_to_numpy(extract_master_vector(sample_text))
    vec_ref = vector_to_numpy(reference_vector)
    curvature = ricci_curvature(vec_t, vec_ref)
    print(f"  θ (rad): {theta:.4f}")
    print(f"  Fisher score F(θ): {fisher:.6f}")
    print(f"  Ricci curvature κ: {curvature:.6f}")
    print(f"  Hybrid score S = F·κ: {fisher * curvature:.6f}")