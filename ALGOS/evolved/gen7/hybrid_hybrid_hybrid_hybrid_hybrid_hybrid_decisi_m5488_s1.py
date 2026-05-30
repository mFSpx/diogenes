# DARWIN HAMMER — match 5488, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2275_s0.py (gen6)
# parent_b: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s0.py (gen3)
# born: 2026-05-30T00:02:13Z

"""Hybrid Fusion Module
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2275_s0 (Algorithm A)
- hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s0 (Algorithm B)

Mathematical Bridge:
Algorithm A provides the Gini coefficient (inequality measure) and a Voronoi‑style
partitioning routine. Algorithm B supplies a spatial distance metric (haversine)
and regex‑based linguistic feature extraction.  
The fusion builds a *joint resource matrix* R where each element combines the
geodesic distance between two locations with the inequality weight derived from
the Gini coefficient of privacy‑related scores.  This matrix then modulates the
policy update step, and the extracted textual features act as additional scaling
factors.  In this way the core topologies of both parents are mathematically
intertwined rather than merely concatenated.
"""

import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime as dt

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Gini coefficient and Voronoi‑style adjustment
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D array of non‑negative numbers."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


def voronoi_adjusted(points: np.ndarray, gini: float) -> np.ndarray:
    """
    Simple Voronoi‑style adjustment: each point is scaled by (1 + gini)
    and the Euclidean distance to the original set is returned as a proxy
    for region size.
    """
    adjusted = points * (1.0 + gini)
    distances = np.linalg.norm(adjusted[:, None, :] - points[None, :, :], axis=2)
    # Return the minimal distance to any other point (region radius)
    region_radius = np.min(distances + np.eye(len(points)) * 1e9, axis=1)
    return region_radius


# ----------------------------------------------------------------------
# Algorithm B – Haversine distance and regex feature extraction
# ----------------------------------------------------------------------
# Regex patterns (excerpted from the original parent)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)


def haversine_distance(latlon1: np.ndarray, latlon2: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise haversine distance (in kilometres) between two
    collections of latitude/longitude points.
    latlon arrays are shape (N, 2) with degrees.
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(latlon1[:, 0])[:, None]
    lon1 = np.radians(latlon1[:, 1])[:, None]
    lat2 = np.radians(latlon2[:, 0])[None, :]
    lon2 = np.radians(latlon2[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # shape (N, M)


def extract_text_features(text: str) -> dict:
    """Count occurrences of each regex class and return a feature dict."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome": len(OUTCOME_RE.findall(text)),
    }


# ----------------------------------------------------------------------
# Hybrid Functions – integrating both parents
# ----------------------------------------------------------------------
def joint_resource_matrix(latlon_points: np.ndarray, privacy_scores: np.ndarray) -> np.ndarray:
    """
    Build a joint matrix R where:
        R_ij = haversine_distance(i, j) * (1 + G)
    G is the Gini coefficient of the privacy_scores vector.
    The matrix captures spatial cost weighted by inequality of privacy.
    """
    if latlon_points.shape[0] != privacy_scores.shape[0]:
        raise ValueError("points and privacy_scores must have the same length")
    gini = gini_coefficient(privacy_scores)
    dist = haversine_distance(latlon_points, latlon_points)  # (N, N)
    return dist * (1.0 + gini)


def hybrid_policy_update(
    updates: list[dict],
    text: str,
    latlon_points: np.ndarray,
    privacy_scores: np.ndarray,
) -> dict:
    """
    Perform a policy update that blends:
      • Gini‑scaled spatial costs (via joint_resource_matrix)
      • Linguistic feature scaling (via extract_text_features)
      • The original reward accumulation logic from Algorithm A.
    Returns a dict mapping action_id -> [weighted_reward, count].
    """
    # 1. Core inequality and spatial weighting
    R = joint_resource_matrix(latlon_points, privacy_scores)
    # Use the mean row‑wise cost as a global spatial penalty factor
    spatial_penalty = np.mean(R, axis=1)  # shape (N,)

    # 2. Linguistic feature scaling
    feats = extract_text_features(text)
    # Simple linear combination of feature counts to a single scalar
    linguistic_factor = (
        0.5 * feats["evidence"]
        + 0.3 * feats["planning"]
        + 0.2 * feats["support"]
        - 0.4 * feats["delay"]
        - 0.3 * feats["boundary"]
    )
    # Normalize to a modest multiplier around 1.0
    linguistic_factor = 1.0 + 0.05 * math.tanh(linguistic_factor)

    # 3. Policy accumulation
    policy = defaultdict(lambda: [0.0, 0.0])
    for upd in updates:
        aid = upd["action_id"]
        reward = float(upd["reward"])
        # Incorporate spatial penalty based on the index of the update (if provided)
        idx = upd.get("point_index", 0) % len(spatial_penalty)
        penalty = spatial_penalty[idx]
        # Final weighted reward
        weighted = reward * (1.0 + penalty / 1000.0) * linguistic_factor
        policy[aid][0] += weighted
        policy[aid][1] += 1.0
    return dict(policy)


def voronoi_spatial_regions(latlon_points: np.ndarray, privacy_scores: np.ndarray) -> np.ndarray:
    """
    Produce a region‑size vector that merges:
      – Voronoi‑style Euclidean scaling from Algorithm A (applied on a
        projected planar approximation of the lat/lon points),
      – Gini‑adjusted weighting from the privacy scores.
    The result can be interpreted as an effective “influence radius” for each
    point in the hybrid system.
    """
    # Simple equirectangular projection (good enough for small areas)
    proj = np.radians(latlon_points) * np.array([111.0, 111.0 * np.cos(np.radians(latlon_points[:, 0]))])
    gini = gini_coefficient(privacy_scores)
    return voronoi_adjusted(proj, gini)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    N = 5
    random.seed(42)
    np.random.seed(42)

    # Random lat/lon points roughly around a city
    latlon = np.column_stack(
        (
            np.random.uniform(37.0, 37.8, size=N),  # latitude
            np.random.uniform(-122.5, -121.5, size=N),  # longitude
        )
    )
    # Random privacy scores (non‑negative)
    privacy = np.random.rand(N) * 10.0

    # Dummy updates
    updates = [
        {"action_id": "A", "reward": 1.2, "point_index": i}
        for i in range(N)
    ] + [
        {"action_id": "B", "reward": -0.4, "point_index": i}
        for i in range(N)
    ]

    # Sample text containing a mix of patterns
    sample_text = (
        "The plan was verified and the evidence was logged. "
        "We will wait before the next step, but support from the team is strong. "
        "All boundaries were respected and the outcome is successful."
    )

    # Run hybrid functions
    R = joint_resource_matrix(latlon, privacy)
    print("Joint resource matrix (km weighted):\n", np.round(R, 2))

    policy = hybrid_policy_update(updates, sample_text, latlon, privacy)
    print("\nHybrid policy:", policy)

    regions = voronoi_spatial_regions(latlon, privacy)
    print("\nVoronoi‑adjusted region radii:", np.round(regions, 3))

    print("\nSmoke test completed successfully.")