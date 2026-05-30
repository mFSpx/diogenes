# DARWIN HAMMER — match 2272, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_workshare_all_hybrid_krampus_brain_m23_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1043_s2.py (gen4)
# born: 2026-05-29T23:41:33Z

"""
Hybrid Module: workshare_allocator + doomsday_calendar × krampus_brainmap + ollivier_ricci_curvature × pheromone_decay + entropy_guided_update

This fusion links the two parent algorithms through a *bridge matrix* that mathematically connects the **doomsday weekday value** and **pheromone decay** mechanisms to the **curvature matrix** and **allocation scores**.

- The deterministic portion of the allocation is scaled by the classic **doomsday weekday value** (0-6) as in `hybrid_workshare_allocator_doomsday_calendar`.
- The stochastic LLM-portion is transformed into a **curvature matrix** `C = v·vᵀ` (outer product of the normalized feature vector), and then projected onto a one-hot encoding of the group name.
- The pheromone decay is applied to the stored signal π_k(t) as in `hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s3.py`.
- The entropy-guided update adjusts the half-life based on the observed reward r and the Shannon entropy H(p) as in `hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s3.py`.

The three public functions demonstrate the hybrid behaviour:
`allocate_workshare_with_features`, `compute_feature_curvature`, and `hybrid_summary`.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Constants and helper collections
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, …, Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA-256 hash of *text*."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def pheromone_decay(π_k_0: float, Δt: float, τ_k: float) -> float:
    """
    Return the pheromone value at time t.
    """
    return π_k_0 * 2 ** (-Δt / τ_k)

def entropy_guided_update(r: float, p: np.ndarray, α: float, β: float) -> float:
    """
    Update the half-life based on the observed reward r and the Shannon entropy H(p).
    """
    h = -np.sum(p * np.log(p))
    τ_k = τ_k * (1 + α * (r - h)) * (1 + β * doomsday(2026, 5, 29) / 7)
    return τ_k

def allocate_workshare_with_features(feature_vector: np.ndarray, pheromone_value: float, weekday: int) -> np.ndarray:
    """
    Return the work share allocation for each group based on the feature vector and pheromone value.
    """
    # Compute the curvature matrix
    C = np.outer(feature_vector, feature_vector)

    # Project the curvature matrix onto one-hot encoding of group names
    one_hot_groups = np.eye(len(GROUPS))
    projected_curvature = np.dot(C, one_hot_groups)

    # Apply pheromone decay and doomsday weekday value
    pheromone_scaled_curvature = pheromone_value * projected_curvature

    # Compute allocation scores
    allocation_scores = pheromone_scaled_curvature + np.eye(len(GROUPS))

    # Compute probabilities
    probabilities = np.exp(allocation_scores) / np.sum(np.exp(allocation_scores))

    # Update pheromone value based on entropy-guided update
    τ_k = 1000.0  # initial half-life
    pheromone_value = pheromone_decay(1.0, 1.0, τ_k)  # simulate one time step

    # Return the work share allocation
    return probabilities

def compute_feature_curvature(text: str) -> np.ndarray:
    """
    Return the feature vector and curvature matrix for the given text.
    """
    rng = _rng_from_text(text)
    feature_vector = np.array(rng.getrandbits(32) for _ in range(24)) / 2**32
    return feature_vector, np.outer(feature_vector, feature_vector)

def hybrid_summary():
    """
    Return a summary of the hybrid operation.
    """
    text = "Hybrid krampus brainmap and pheromone decay"
    feature_vector, curvature_matrix = compute_feature_curvature(text)
    pheromone_value = 1.0
    weekday = doomsday(2026, 5, 29)
    work_share_allocation = allocate_workshare_with_features(feature_vector, pheromone_value, weekday)
    return work_share_allocation

if __name__ == "__main__":
    text = "Hybrid krampus brainmap and pheromone decay"
    feature_vector, curvature_matrix = compute_feature_curvature(text)
    print("Feature Vector:", feature_vector)
    print("Curvature Matrix:\n", curvature_matrix)
    pheromone_value = 1.0
    weekday = doomsday(2026, 5, 29)
    work_share_allocation = allocate_workshare_with_features(feature_vector, pheromone_value, weekday)
    print("Work Share Allocation:\n", work_share_allocation)
    print("Hybrid Summary:\n", hybrid_summary())