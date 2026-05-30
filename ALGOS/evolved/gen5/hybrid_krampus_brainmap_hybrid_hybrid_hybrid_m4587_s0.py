# DARWIN HAMMER — match 4587, survivor 0
# gen: 5
# parent_a: krampus_brainmap.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_fractional_hd_m1594_s1.py (gen4)
# born: 2026-05-29T23:56:41Z

"""Hybrid Krampus‑Hyperdimensional Engine

Parents:
* **krampus_brainmap.py** – extracts a 20+‑dimensional scalar feature vector from
  free‑form text via domain‑specific operators.
* **hybrid_hybrid_hybrid_hybrid_hybrid_fractional_hd_m1594_s1.py** – builds a
  weighted hypervector using a sinusoidal weekday weight vector, GPU‑aware
  hypervectors and element‑wise binding (the “hyperdimensional” multiplication).

**Mathematical bridge**

Both parents ultimately produce *vectors* that can be combined by the same
operation: element‑wise multiplication (binding).  The Krampus extractor yields a
real‑valued scalar vector **f** ∈ ℝⁿ (n ≈ 20).  The weekday allocator yields a
scalar weight vector **w** ∈ ℝⁿ.  Their Hadamard product **s = f ⊙ w** is a
scaled feature vector.  Each GPU is encoded as a random hypervector
**hᵢ** ∈ {−1,+1}ⁿ whose amplitude is raised to a fractional power proportional to
its free memory.  Binding **s** with each **hᵢ** (again a Hadamard product)
produces a set of weighted hypervectors **bᵢ = s ⊙ hᵢ**.  Summation (bundling)
of all **bᵢ** yields a single composite hypervector **H** that simultaneously
encodes textual semantics (Krampus) and resource‑allocation context (Hybrid A).

The functions below implement this pipeline and provide a causal‑effect
estimate via cosine similarity against a reference hypervector.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import the Krampus feature extractor; fall back to a stub if missing.
# ---------------------------------------------------------------------------
try:
    from krampus_brainmap import extract_master_vector  # type: ignore
except Exception:  # pragma: no cover
    def extract_master_vector(text: str) -> Dict[str, float]:
        """Fallback stub – returns a deterministic pseudo‑feature vector."""
        random.seed(hash(text) & 0xFFFFFFFF)
        keys = [
            "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
            "recursion_score", "directive_ratio", "target_density",
            "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
            "wrath_velocity", "bureaucratic_weaponization_index",
            "resource_exhaustion_metric", "swarm_orchestration_density",
            "logic_crucifixion_index", "conspiracy_grounding_ratio",
            "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
            "asset_structuring_weight", "pitch_formatting_ratio",
        ]
        return {k: random.random() for k in keys}


# ---------------------------------------------------------------------------
# Parent A constants (weekday allocation)
# ---------------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
N_DIM: int = 22  # dimensionality of the Krampus master vector (fixed above)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _to_numpy_vector(feature_dict: Dict[str, float]) -> np.ndarray:
    """Convert a Krampus feature dict to an ordered numpy array of length N_DIM."""
    order = [
        "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
        "recursion_score", "directive_ratio", "target_density",
        "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
        "wrath_velocity", "bureaucratic_weaponization_index",
        "resource_exhaustion_metric", "swarm_orchestration_density",
        "logic_crucifixion_index", "conspiracy_grounding_ratio",
        "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
        "asset_structuring_weight", "pitch_formatting_ratio",
    ]
    vec = np.zeros(N_DIM, dtype=float)
    for i, key in enumerate(order):
        vec[i] = feature_dict.get(key, 0.0)
    return vec


def _sinusoidal_weekday_weights(date: dt.date) -> np.ndarray:
    """
    Produce a row‑stochastic sinusoidal weight vector of length N_DIM.
    The weekday (0=Mon … 6=Sun) determines the phase shift.
    """
    weekday = date.weekday()
    phases = np.arange(N_DIM) * (2 * math.pi / N_DIM)
    raw = np.sin(phases + weekday * (2 * math.pi / 7))
    # Shift to positive domain and normalize to sum = 1
    raw = raw - raw.min() + 1e-6
    weights = raw / raw.sum()
    return weights


def _gpu_hypervector(gpu_id: int, free_mem_mb: int, total_mem_mb: int) -> np.ndarray:
    """
    Generate a random bipolar hypervector for a GPU and raise its amplitude
    to a fractional power proportional to its free memory.
    """
    rng = np.random.default_rng(seed=gpu_id)
    hv = rng.choice([-1, 1], size=N_DIM).astype(float)
    fraction = free_mem_mb / max(total_mem_mb, 1)
    # Apply fractional power element‑wise while preserving sign
    hv = np.sign(hv) * (np.abs(hv) ** fraction)
    return hv


# ---------------------------------------------------------------------------
# Hybrid core functions (demonstrate the bridge)
# ---------------------------------------------------------------------------
def weekday_weight_vector(date: dt.date) -> np.ndarray:
    """
    Public wrapper for the sinusoidal weekday weight vector.
    Returns a 1‑D numpy array of length N_DIM whose entries sum to 1.
    """
    return _sinusoidal_weekday_weights(date)


def hybrid_allocation_plan(
    text: str,
    date: dt.date,
    gpu_info: Iterable[Tuple[int, int, int]],
) -> np.ndarray:
    """
    Build a composite hypervector that fuses Krampus textual features with
    weekday‑based allocation weights and GPU‑aware hypervectors.

    Parameters
    ----------
    text: str
        Free‑form input for Krampus feature extraction.
    date: datetime.date
        Calendar date that determines the sinusoidal weight vector.
    gpu_info: iterable of (gpu_id, free_mem_mb, total_mem_mb)
        Information for each GPU that will be encoded.

    Returns
    -------
    np.ndarray
        The bundled hypervector H ∈ ℝⁿ (floating‑point bipolar after fractional
        scaling) representing the hybrid plan.
    """
    # 1️⃣ Krampus feature vector f
    raw_features = extract_master_vector(text)
    f_vec = _to_numpy_vector(raw_features)  # shape (N_DIM,)

    # 2️⃣ Weekday weight vector w
    w_vec = weekday_weight_vector(date)    # shape (N_DIM,)

    # 3️⃣ Scaled scalar vector s = f ⊙ w
    s_vec = f_vec * w_vec

    # 4️⃣ Encode each GPU and bind with s
    bound_vectors: List[np.ndarray] = []
    for gpu_id, free_mb, total_mb in gpu_info:
        hv = _gpu_hypervector(gpu_id, free_mb, total_mb)   # shape (N_DIM,)
        bound = s_vec * hv                                 # element‑wise binding
        bound_vectors.append(bound)

    if not bound_vectors:
        # No GPUs – return the scaled scalar vector itself as a degenerate hypervector
        return s_vec.copy()

    # 5️⃣ Bundle (all‑add) the bound hypervectors
    composite = np.sum(bound_vectors, axis=0)
    return composite


def estimate_plan_effect(composite_hv: np.ndarray, reference_hv: np.ndarray) -> float:
    """
    Treat the composite hypervector as a causal representation and return a
    similarity‑based effect estimate using cosine similarity.

    The result lies in [-1, 1]; values close to 1 indicate a strong positive
    alignment between the plan and the reference causal pattern.
    """
    if composite_hv.shape != reference_hv.shape:
        raise ValueError("Hypervectors must have the same dimensionality")
    dot = np.dot(composite_hv, reference_hv)
    norm_prod = np.linalg.norm(composite_hv) * np.linalg.norm(reference_hv)
    if norm_prod == 0:
        return 0.0
    return dot / norm_prod


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Sample input text
    sample_text = (
        "The operator's visceral ratio surged while the rainmaker's corporate grit tension "
        "remained steady. Recursive loops in the ledger density hinted at a looming "
        "resource exhaustion metric."
    )

    # Current date
    today = dt.date.today()

    # Mock GPU information: (gpu_id, free_mem_mb, total_mem_mb)
    mock_gpus = [
        (0, 2048, 8192),
        (1, 3072, 8192),
        (2, 1024, 4096),
    ]

    # Build the hybrid hypervector
    composite = hybrid_allocation_plan(sample_text, today, mock_gpus)

    # Create a random reference hypervector for effect estimation
    rng = np.random.default_rng(seed=42)
    reference = rng.choice([-1, 1], size=N_DIM).astype(float)

    # Estimate effect
    effect = estimate_plan_effect(composite, reference)

    print(f"Composite hypervector (first 5 components): {composite[:5]}")
    print(f"Effect estimate (cosine similarity): {effect:.4f}")