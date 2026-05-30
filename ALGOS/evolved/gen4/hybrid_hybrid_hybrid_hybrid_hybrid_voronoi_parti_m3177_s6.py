# DARWIN HAMMER — match 3177, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_fractional_hd_m2459_s0.py (gen3)
# parent_b: hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py (gen1)
# born: 2026-05-29T23:48:20Z

"""Hybrid Voronoi–Krampus Fractional Engine
========================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A – hybrid_hybrid_hybrid_krampu_hybrid_fractional_hd_m2459_s0.py**  
  Provides high‑dimensional weighted feature vectors (via ``extract_full_features``) that
  can be interpreted as points in a vector space.

* **Parent B – hybrid_voronoi_partition_poikilotherm_schoolf_m49_s0.py**  
  Supplies a 2‑D Voronoi partitioning routine and a temperature‑dependent
  Schoolfield developmental‑rate model.

**Mathematical bridge**

The weighted feature vector ``v ∈ ℝⁿ`` from Parent A is linearly projected onto its
first two components (or a random 2‑D projection) to obtain a planar point
``p = (v₀, v₁) ∈ ℝ²``.  The set of points is fed to the Voronoi engine of Parent B,
producing regions associated with a set of seed points.

Each region is summarised by its centroid temperature ``T`` (derived from the
average magnitude of the original high‑dimensional vectors belonging to the
region).  The Schoolfield equation from Parent B converts ``T`` into an activity
scalar ``a ∈ [0,1]``.  Finally, a random hyper‑vector (high‑dimensional complex
vector) is generated for each region and scaled by ``a`` – thus marrying the
feature‑space geometry of Parent A with the thermodynamic gating of Parent B.

The resulting data structure can be used wherever a weighted, region‑aware
hyper‑vector representation is required.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – feature extraction and hyper‑vector generation
# ----------------------------------------------------------------------


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministically generate a dictionary of 24 pseudo‑features from *text*."""
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


def vector_from_features(feat: Dict[str, float]) -> np.ndarray:
    """Convert the feature dict into a deterministic ordered numpy vector."""
    ordered_keys = sorted(feat.keys())
    return np.array([feat[k] for k in ordered_keys], dtype=float)


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """
    Generate a random high‑dimensional vector.

    Parameters
    ----------
    d: int
        Dimensionality of the vector.
    kind: str
        ``"complex"`` → complex entries with unit magnitude,
        ``"real"``    → real entries in [-1, 1].
    seed: int | None
        Optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        angles = rng.random(d) * 2.0 * math.pi
        return np.exp(1j * angles)  # unit‑circle complex numbers
    else:
        return rng.uniform(-1.0, 1.0, size=d)


# ----------------------------------------------------------------------
# Parent B – Voronoi utilities and Schoolfield temperature model
# ----------------------------------------------------------------------


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Return the index of the seed nearest to *point*."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Assign each point to its nearest seed, returning a region dictionary."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


class SchoolfieldParams:
    """Parameter container for the Schoolfield developmental‑rate model."""
    def __init__(
        self,
        rho_25: float = 1.0,
        delta_h_activation: float = 12_000.0,
        t_low: float = 283.15,
        t_high: float = 307.15,
        delta_h_low: float = -45_000.0,
        delta_h_high: float = 65_000.0,
        r_cal: float = 1.987,
    ):
        self.rho_25 = rho_25
        self.delta_h_activation = delta_h_activation
        self.t_low = t_low
        self.t_high = t_high
        self.delta_h_low = delta_h_low
        self.delta_h_high = delta_h_high
        self.r_cal = r_cal


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield equation – returns a rate proportional to temperature."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map a temperature in Celsius to a normalized activity ∈ [0,1] using the Schoolfield model."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    # Compute the maximal possible rate over the sampling interval
    max_rate = max(
        developmental_rate(
            c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)),
            params,
        )
        for i in range(samples)
    )
    if max_rate <= 0:
        return 0.0
    return max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Hybrid operations – three demonstrative functions
# ----------------------------------------------------------------------


def extract_and_project(text: str) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Extract features from *text*, convert to a high‑dimensional vector,
    and project onto a 2‑D point using the first two components.
    Returns (point_2d, full_vector).
    """
    feats = extract_full_features(text)
    vec = vector_from_features(feats)
    # Simple projection: take the first two components; if vector shorter, pad with zeros
    if vec.size >= 2:
        point = (float(vec[0]), float(vec[1]))
    else:
        point = (float(vec[0]), 0.0)
    return point, vec


def assign_regions(
    texts: List[str],
    seeds: List[Tuple[float, float]],
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[Tuple[float, float]]]]:
    """
    Project a list of *texts* into 2‑D, assign each projection to a Voronoi region,
    and return both the high‑dimensional vectors per region and the 2‑D points per region.
    """
    points: List[Tuple[float, float]] = []
    vectors: List[np.ndarray] = []
    for t in texts:
        pt, vec = extract_and_project(t)
        points.append(pt)
        vectors.append(vec)

    region_points = assign(points, seeds)
    region_vectors: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(seeds))}
    for idx, pt in enumerate(points):
        region_idx = nearest(pt, seeds)
        region_vectors[region_idx].append(vectors[idx])

    return region_vectors, region_points


def region_activity(
    region_vectors: Dict[int, List[np.ndarray]],
    region_points: Dict[int, List[Tuple[float, float]]],
    low_c: float = 5.0,
    high_c: float = 40.0,
) -> Dict[int, float]:
    """
    Compute an activity scalar for each Voronoi region.

    The temperature is inferred from the mean L2‑norm of the high‑dimensional
    vectors belonging to the region, linearly mapped into the Celsius interval
    [low_c, high_c].  The Schoolfield model then yields a normalized activity.
    """
    activities: Dict[int, float] = {}
    for rid, vecs in region_vectors.items():
        if not vecs:
            activities[rid] = 0.0
            continue
        # Mean magnitude of vectors in this region
        magnitudes = [np.linalg.norm(v) for v in vecs]
        mean_mag = float(np.mean(magnitudes))
        # Map magnitude (arbitrary scale) into temperature range
        # First compute min/max possible magnitude given the feature generation (0..~24*10)
        max_possible = 24 * 10.0  # each feature ≤10, 24 features
        temp_c = low_c + (high_c - low_c) * (mean_mag / max_possible)
        activities[rid] = normalized_activity(temp_c, low_c, high_c)
    return activities


def generate_region_hypervectors(
    region_vectors: Dict[int, List[np.ndarray]],
    activities: Dict[int, float],
    hv_dim: int = 10000,
    seed: int | None = None,
) -> Dict[int, np.ndarray]:
    """
    For each region, create a random hyper‑vector and scale it by the region's activity.
    The random seed is offset by the region id to ensure reproducibility per region.
    """
    hv_dict: Dict[int, np.ndarray] = {}
    base_rng = np.random.default_rng(seed)
    base_seed = base_rng.integers(0, 2**31 - 1) if seed is not None else None
    for rid, act in activities.items():
        region_seed = (base_seed + rid) if base_seed is not None else None
        hv = random_hv(d=hv_dim, kind="complex", seed=region_seed)
        hv_dict[rid] = hv * act  # scale magnitude by activity (complex scaling preserves phase)
    return hv_dict


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Sample seed points forming a simple square Voronoi diagram
    seed_points = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]

    # Dummy corpus of texts
    sample_texts = [
        "Alpha protocol engaged.",
        "Beta subsystem offline.",
        "Gamma wavefront detected.",
        "Delta resonance achieved.",
        "Epsilon anomaly logged.",
        "Zeta calibration complete.",
        "Eta feedback loop initiated.",
        "Theta matrix inversion succeeded.",
    ]

    # Hybrid pipeline
    region_vecs, region_pts = assign_regions(sample_texts, seed_points)
    acts = region_activity(region_vecs, region_pts)
    hvectors = generate_region_hypervectors(region_vecs, acts, hv_dim=2048, seed=42)

    # Print a concise summary
    for rid in sorted(hvectors.keys()):
        print(
            f"Region {rid}: "
            f"{len(region_vecs[rid])} vectors, "
            f"activity={acts[rid]:.3f}, "
            f"hv_norm={np.linalg.norm(hvectors[rid]):.3f}"
        )