# DARWIN HAMMER — match 3463, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_ternar_m1293_s1.py (gen5)
# born: 2026-05-29T23:50:13Z

"""Hybrid algorithm merging DARWIN HAMMER's hybrid semantic neighbors & recovery priority
(PARENT ALGORITHM A) with stylometry‑based geometric routing & Voronoi partitioning
(PARENT ALGORITHM B).

Mathematical bridge:
- Morphological descriptors (righting‑time, sphericity, flatness) are treated as
  continuous coordinates and concatenated with the discrete stylometry vector
  derived from a text document.  The resulting high‑dimensional point lives in a
  unified feature space.
- Voronoi partitioning is performed on this unified space; each cell’s centroid
  represents a “text‑morphology archetype”.
- A joint score S(p) = s(p)·(1+z_s) from the original temporal‑motif model is
  re‑interpreted: s(p) is the normalized magnitude of the combined feature vector,
  while z_s is the z‑score of the distances of the point to all centroids (a
  measure of support across patterns).
- The final hybrid recovery score multiplies the original recovery_priority
  (based on morphology) with the joint score, yielding a spatio‑temporal
  evaluation that respects both physical shape and linguistic style.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – Stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb": set(
        "very really quite almost barely simply just".split()
    ),
}


def tokenise(text: str) -> List[str]:
    return [t.strip(".,!?;:()[]\"'").lower() for t in text.split() if t]


def stylometry_vector(text: str) -> np.ndarray:
    """Return a normalized count vector over FUNCTION_CATS."""
    tokens = tokenise(text)
    counts = np.zeros(len(FUNCTION_CATS), dtype=float)
    for i, cat in enumerate(FUNCTION_CATS):
        counts[i] = sum(1 for t in tokens if t in FUNCTION_CATS[cat])
    norm = np.linalg.norm(counts)
    return counts / norm if norm > 0 else counts


# ----------------------------------------------------------------------
# Hybrid core – unified feature construction & scoring
# ----------------------------------------------------------------------
def morphology_feature_vector(m: Morphology) -> np.ndarray:
    """Three continuous morphological descriptors normalized to unit variance."""
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    vec = np.array([sph, flat, rt], dtype=float)
    # Simple min‑max scaling to [0,1] (using plausible bounds)
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([1.0, 5.0, 20.0])
    scaled = (vec - mins) / (maxs - mins)
    return np.clip(scaled, 0.0, 1.0)


def combined_feature_vector(m: Morphology, text: str) -> np.ndarray:
    """Concatenate normalized morphology descriptors with stylometry vector."""
    morph_vec = morphology_feature_vector(m)
    style_vec = stylometry_vector(text)
    return np.concatenate([morph_vec, style_vec])


def voronoi_assignments(points: np.ndarray,
                        centroids: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest centroid (Euclidean distance).
    Returns an array of centroid indices with shape (n_points,).
    """
    if points.ndim != 2 or centroids.ndim != 2:
        raise ValueError("points and centroids must be 2‑D arrays")
    # Compute squared Euclidean distances efficiently
    diff = points[:, None, :] - centroids[None, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1)


def joint_score(point: np.ndarray,
                centroids: np.ndarray) -> float:
    """
    Implements S(p) = s(p)·(1+z_s)

    - s(p)  : L2‑norm of the point (already normalized, so between 0‑1)
    - z_s   : z‑score of the distance distribution of the point to all centroids
    """
    s_p = np.linalg.norm(point)
    dists = np.linalg.norm(centroids - point, axis=1)
    mean = dists.mean()
    std = dists.std(ddof=0) if dists.std(ddof=0) > 0 else 1e-9
    z_s = (dists.min() - mean) / std
    return s_p * (1.0 + z_s)


def hybrid_recovery_score(m: Morphology, text: str,
                          centroids: np.ndarray) -> float:
    """
    Unified score = recovery_priority(m) * joint_score(combined_vector, centroids)
    """
    rp = recovery_priority(m)
    point = combined_feature_vector(m, text)
    js = joint_score(point, centroids)
    return rp * js


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def generate_random_centroids(k: int, dim: int) -> np.ndarray:
    """Create k random centroids uniformly in [0,1]^dim."""
    return np.random.rand(k, dim)


def demo_hybrid_process() -> None:
    """Run a small demo showing the hybrid pipeline."""
    # Sample morphology
    morph = Morphology(length=0.45, width=0.30, height=0.20, mass=2.5)

    # Sample text
    sample_text = (
        "The quick brown fox jumps over the lazy dog while I contemplate the "
        "meaning of existence and the role of auxiliary verbs."
    )

    # Build combined feature vector
    point = combined_feature_vector(morph, sample_text)

    # Generate centroids (10 cells, dimension matches combined vector)
    centroids = generate_random_centroids(k=10, dim=point.shape[0])

    # Assign point to a Voronoi cell (not strictly needed for score but illustrative)
    assignment = voronoi_assignments(point.reshape(1, -1), centroids)[0]

    # Compute hybrid score
    score = hybrid_recovery_score(morph, sample_text, centroids)

    print(f"Morphology: {morph}")
    print(f"Assigned Voronoi cell: {assignment}")
    print(f"Hybrid recovery score: {score:.4f}")


if __name__ == "__main__":
    # Simple smoke test – should run without error
    demo_hybrid_process()