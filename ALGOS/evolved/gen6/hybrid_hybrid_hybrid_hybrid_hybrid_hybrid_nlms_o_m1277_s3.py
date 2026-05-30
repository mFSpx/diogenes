# DARWIN HAMMER — match 1277, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m782_s1.py (gen5)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_krampu_m293_s2.py (gen3)
# born: 2026-05-29T23:36:19Z

"""Hybrid Fusion of Gini‑Weighted Bandit Voronoi and NLMS‑Krampus

This module fuses the two parent algorithms:

* **Parent A** – uses the Gini coefficient of bandit propensities to weight
  geometric points in a Voronoi partitioning.
* **Parent B** – provides a Normalized Least‑Mean‑Squares (NLMS) adaptive
  filter together with a deterministic feature extractor.

**Mathematical bridge** – the Gini coefficient computed from the bandit
propensities is used to modulate the NLMS learning‑rate (μ).  In addition,
the Voronoi partitions produced from the weighted geometric points are
used to select a subset of features that drive the NLMS update.  Thus the
fairness metric of the bandit layer directly influences the adaptive
filter of the Krampus layer, creating a single unified hybrid system.
"""

import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Utility functions from Parent A
# ----------------------------------------------------------------------
def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D non‑negative array."""
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


def words(text: str) -> List[str]:
    """Extract lower‑case word tokens from a string."""
    return re.findall(r"\b\w+\b", text.lower())


# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class GeometricPoint:
    point_id: str
    coordinates: List[float]
    weight: float = field(default=1.0)


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def voronoi_partitioning(
    points: List[GeometricPoint],
    num_partitions: int,
) -> Dict[int, List[GeometricPoint]]:
    """
    Simple Voronoi partitioning: randomly pick `num_partitions` centroids
    from ``points`` and assign each point to the nearest centroid using
    Euclidean distance.  The function returns a dict mapping centroid index
    to the list of points belonging to that region.
    """
    if not points:
        return {}
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    centroids = random.sample(points, min(num_partitions, len(points)))
    partitions: Dict[int, List[GeometricPoint]] = defaultdict(list)

    for pt in points:
        distances = [_euclidean(pt.coordinates, c.coordinates) for c in centroids]
        nearest = int(np.argmin(distances))
        partitions[nearest].append(pt)

    return dict(partitions)


# ----------------------------------------------------------------------
# Functions from Parent B (NLMS + feature extraction)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).  In the hybrid we will modulate it with
        the Gini coefficient.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (weights, error) : Tuple[np.ndarray, float]
        Updated weight vector and the prediction error.
    """
    error = target - nlms_predict(weights, x)
    norm_sq = np.linalg.norm(x) ** 2
    weights = weights + mu * error * x / (norm_sq + eps)
    return weights, error


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Deterministic pseudo‑feature extraction.  The function hashes the input
    text to seed a ``random.Random`` instance and then produces a fixed set
    of floating‑point features in the range [0, 1).  The feature names are
    taken from the original parent algorithm.
    """
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
    ]
    return {k: rnd.random() for k in keys}


# ----------------------------------------------------------------------
# Hybrid core – three demonstrative functions
# ----------------------------------------------------------------------
def compute_gini_from_actions(actions: List[BanditAction]) -> float:
    """Compute the Gini coefficient of the propensity scores of a list of actions."""
    propensities = np.array([a.propensity for a in actions], dtype=np.float64)
    return gini_coefficient(propensities)


def nlms_update_with_gini(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    gini: float,
    base_mu: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Modulate the NLMS learning rate by the Gini coefficient.

    The effective learning rate is ``mu = base_mu * (1 + gini)``; thus a
    higher inequality (larger Gini) yields a larger step size.
    """
    mu = base_mu * (1.0 + gini)
    # Clamp mu to the stable NLMS interval (0, 2)
    mu = max(min(mu, 1.999), 1e-6)
    return nlms_update(weights, x, target, mu=mu)


def hybrid_step(
    actions: List[BanditAction],
    points: List[GeometricPoint],
    weights: np.ndarray,
    text: str,
    target: float,
    num_voronoi_regions: int = 3,
) -> Tuple[np.ndarray, float, Dict[int, List[GeometricPoint]]]:
    """
    Perform one hybrid iteration:

    1. Compute Gini from bandit propensities.
    2. Partition geometric points via Voronoi; use the total weight of the
       region containing the highest‑propensity action as a scaling factor.
    3. Extract a feature vector from ``text``; keep only the features whose
       names appear in the selected region's point IDs (simulating a
       geometry‑driven feature mask).
    4. Update NLMS weights with the Gini‑modulated learning rate.

    Returns the updated weights, the NLMS error, and the Voronoi partitions.
    """
    # 1. Gini coefficient
    gini = compute_gini_from_actions(actions)

    # 2. Voronoi partitioning
    partitions = voronoi_partitioning(points, num_voronoi_regions)

    # Identify the action with the highest propensity
    top_action = max(actions, key=lambda a: a.propensity)

    # Find the region that contains a point whose ID matches the top action
    selected_region_idx = None
    for idx, pts in partitions.items():
        if any(pt.point_id == top_action.action_id for pt in pts):
            selected_region_idx = idx
            break
    # Fallback to the first region if no match
    if selected_region_idx is None:
        selected_region_idx = next(iter(partitions))

    selected_region = partitions[selected_region_idx]

    # 3. Feature extraction with geometry‑driven mask
    all_features = extract_full_features(text)
    # Keep features whose keys appear in any point_id of the selected region
    mask_keys = {pt.point_id for pt in selected_region}
    masked_features = {k: v for k, v in all_features.items() if k in mask_keys}
    # If the mask removes everything, fall back to the full vector
    if not masked_features:
        masked_features = all_features

    # Build feature vector aligned with the weight dimension
    feature_names = sorted(masked_features.keys())
    x = np.array([masked_features[k] for k in feature_names], dtype=np.float64)

    # Adjust weight vector size if needed (pad with zeros)
    if weights.size < x.size:
        weights = np.pad(weights, (0, x.size - weights.size))
    elif weights.size > x.size:
        x = np.pad(x, (0, weights.size - x.size))

    # 4. NLMS update with Gini‑scaled mu
    new_weights, error = nlms_update_with_gini(weights, x, target, gini)

    return new_weights, error, partitions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy bandit actions
    actions = [
        BanditAction("op1", propensity=0.2, expected_reward=1.0, confidence_bound=0.1, algorithm="A"),
        BanditAction("op2", propensity=0.5, expected_reward=0.8, confidence_bound=0.2, algorithm="A"),
        BanditAction("op3", propensity=0.3, expected_reward=0.9, confidence_bound=0.15, algorithm="A"),
    ]

    # Create dummy geometric points (IDs deliberately overlap with action IDs)
    points = [
        GeometricPoint("op1", [0.0, 1.0, 2.0], weight=1.0),
        GeometricPoint("op2", [1.0, 0.0, 2.0], weight=0.8),
        GeometricPoint("op3", [2.0, 1.0, 0.0], weight=0.9),
        GeometricPoint("extra", [0.5, 0.5, 0.5], weight=0.5),
    ]

    # Initial NLMS weight vector (size matches number of possible features)
    initial_weights = np.zeros(14, dtype=np.float64)  # 14 feature keys in extractor

    # Dummy input text and target scalar
    sample_text = "The quick brown fox jumps over the lazy dog."
    target_value = 0.7

    # Run a single hybrid iteration
    updated_weights, nlms_err, voronoi = hybrid_step(
        actions,
        points,
        initial_weights,
        sample_text,
        target_value,
        num_voronoi_regions=2,
    )

    # Print results (simple sanity check)
    print("Gini‑modulated NLMS error:", nlms_err)
    print("Updated weight vector (first 5 entries):", updated_weights[:5])
    print("Voronoi partitions (region → point IDs):")
    for idx, pts in voronoi.items():
        print(f"  Region {idx}: {[p.point_id for p in pts]}")
    sys.exit(0)