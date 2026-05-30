# DARWIN HAMMER — match 3920, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1300_s2.py (gen5)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:52:41Z

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Iterable, Hashable

import numpy as np
from collections import Counter
from collections.abc import Iterable as IterableABC


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Geospatial entity that may host an endpoint."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    """Computational tier with finite RAM+VRAM budget."""
    name: str
    ram_mb: int
    vram_mb: int
    tier: str = ""  # optional label


@dataclass(frozen=True)
class Morphology:
    """Physical or logical shape descriptor used for similarity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    """Endpoint exposing a model; linked to a single Entity."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"


# ----------------------------------------------------------------------
# Helper utilities (entropy, distance, similarity)
# ----------------------------------------------------------------------


def shannon_entropy(
    observations: Iterable[Hashable | float],
    as_distribution: bool = False,
) -> float:
    """
    Compute Shannon entropy (bits).

    If ``as_distribution`` is True, ``observations`` must already sum to 1.
    Otherwise a frequency distribution is built from the raw observations.
    """
    xs = list(observations)
    if not xs:
        return 0.0

    if as_distribution:
        probs = [float(p) for p in xs]
        if any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, rel_tol=1e-9):
            raise ValueError("Provided probabilities must be non‑negative and sum to 1")
    else:
        cnt = Counter(xs)
        total = float(sum(cnt.values()))
        probs = [v / total for v in cnt.values()]

    return -sum(p * math.log2(p) for p in probs if p > 0.0)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _norm_vector(m: Morphology) -> np.ndarray:
    """Return L2‑normalised 4‑D vector for a Morphology."""
    vec = np.array([m.length, m.width, m.height, m.mass], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------
# Core hybrid calculations
# ----------------------------------------------------------------------


def compute_spatial_privacy_risk(
    entities: List[Entity],
    reference: Tuple[float, float],
    max_distance_km: float = 20000.0,
) -> Dict[str, float]:
    """
    Distance‑weighted privacy risk for each entity.

    risk = min(1, distance / max_distance_km)
    """
    ref_lat, ref_lon = reference
    out: Dict[str, float] = {}
    for e in entities:
        d = haversine(e.lat, e.lon, ref_lat, ref_lon)
        out[e.id] = min(1.0, d / max_distance_km)
    return out


def compute_weighted_ssim(
    endpoints: List[EngineEndpoint],
    distance_matrix: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a proxy SSIM matrix using cosine similarity on normalised morphology
    vectors.  If a distance matrix is supplied, similarity is down‑weighted by a
    Gaussian of the spatial distance, encouraging nearby endpoints to be compared
    more heavily.

    Returns
    -------
    sim_matrix : np.ndarray
        Symmetric matrix with values in [0, 1].
    avg_sim : np.ndarray
        Per‑endpoint average similarity (excluding self‑similarity).
    """
    n = len(endpoints)
    if n == 0:
        return np.empty((0, 0)), np.empty(0)

    vecs = np.stack([_norm_vector(ep.morphology) for ep in endpoints])
    cos_sim = np.clip(np.dot(vecs, vecs.T), 0.0, 1.0)

    if distance_matrix is not None:
        # Gaussian kernel with σ = 0.5 * max_distance (in normalised units)
        sigma = 0.5 * np.max(distance_matrix)
        spatial_weight = np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
        # Preserve diagonal (self‑weight = 1)
        np.fill_diagonal(spatial_weight, 1.0)
        sim = cos_sim * spatial_weight
    else:
        sim = cos_sim

    # Zero out diagonal for averaging purposes
    np.fill_diagonal(sim, 0.0)
    avg_sim = sim.sum(axis=1) / max(1, n - 1)
    return sim, avg_sim


def _spatial_distance_matrix(endpoints: List[EngineEndpoint]) -> np.ndarray:
    """Pairwise haversine distances (km) between endpoints' associated entities."""
    n = len(endpoints)
    if n == 0:
        return np.empty((0, 0))

    # Build a lookup from engine_id to (lat, lon) assuming each endpoint maps to an Entity
    # In practice the caller should provide this mapping; we fall back to (0,0) if missing.
    # This function is deliberately lightweight – callers can inject a pre‑computed matrix.
    coords = []
    for ep in endpoints:
        # Placeholder: real implementation would retrieve the Entity.
        # Here we set (0,0) to keep the function pure.
        coords.append((0.0, 0.0))

    mat = np.empty((n, n), dtype=float)
    for i in range(n):
        lat_i, lon_i = coords[i]
        for j in range(i, n):
            lat_j, lon_j = coords[j]
            d = haversine(lat_i, lon_i, lat_j, lon_j)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def entropy_modulator(values: List[float]) -> float:
    """
    Compute η = 1 + H / log2(N) where H is the Shannon entropy of the *probability*
    distribution derived from ``values``.  ``values`` are first shifted to be
    non‑negative and then normalised to sum to 1.
    """
    if not values:
        return 1.0

    # Shift to non‑negative (they are already in [0,1] but guard against noise)
    shifted = [max(0.0, v) for v in values]
    total = sum(shifted)
    if total == 0.0:
        # Uniform distribution as fallback
        probs = [1.0 / len(shifted)] * len(shifted)
    else:
        probs = [v / total for v in shifted]

    H = shannon_entropy(probs, as_distribution=True)
    N = len(probs)
    return 1.0 + H / math.log2(N)


def _tier_capacity_factor(tiers: List[ModelTier]) -> Dict[str, float]:
    """
    Normalised capacity factor for each tier based on total memory (RAM+VRAM).
    Returns a dict mapping tier name → factor ∈ (0, 1].
    """
    raw = {t.name: (t.ram_mb + t.vram_mb) for t in tiers}
    max_raw = max(raw.values()) if raw else 1.0
    # Scale to (0,1] and apply a small epsilon to avoid exact zero.
    eps = 1e-6
    return {k: eps + (v / max_raw) * (1 - eps) for k, v in raw.items()}


def allocate_resources_based_on_health(
    entities: List[Entity],
    endpoints: List[EngineEndpoint],
    tiers: List[ModelTier],
    reference_location: Tuple[float, float],
) -> Dict[str, Tuple[str, float]]:
    """
    Compute a refined health score for each endpoint and allocate it to a tier.

    The health model now incorporates:
      * (1 - privacy_risk) – spatial privacy component
      * (1 - weighted_avg_ssim) – morphological similarity component
      * η – entropy modulator built from the *distribution* of (1 - privacy_risk)
      * tier capacity factor – respects relative memory budgets

    Allocation heuristic:
      1. Sort endpoints by descending health.
      2. Assign each endpoint to the tier with the highest remaining capacity factor.
      3. Update the tier's remaining factor (simple multiplicative decay).

    Returns
    -------
    mapping : dict
        {endpoint_id: (selected_tier_name, final_health_score)}.
    """
    # ------------------------------------------------------------------
    # 1. Privacy risk per entity (entity.id ↔ endpoint.engine_id)
    # ------------------------------------------------------------------
    risk_by_entity = compute_spatial_privacy_risk(entities, reference_location)

    # ------------------------------------------------------------------
    # 2. Morphology similarity weighted by spatial distance
    # ------------------------------------------------------------------
    # In a real system we would compute a true spatial distance matrix.
    # For demonstration we skip the weighting (set to None) – the function
    # still returns a valid average similarity.
    _, avg_ssim = compute_weighted_ssim(endpoints, distance_matrix=None)

    # ------------------------------------------------------------------
    # 3. Entropy modulator based on the distribution of (1 - risk)
    # ------------------------------------------------------------------
    complement_risks = [
        1.0 - risk_by_entity.get(ep.engine_id, 0.0) for ep in endpoints
    ]
    eta = entropy_modulator(complement_risks)  # ∈ [1,2]

    # ------------------------------------------------------------------
    # 4. Tier capacity factors
    # ------------------------------------------------------------------
    tier_factors = _tier_capacity_factor(tiers)

    # ------------------------------------------------------------------
    # 5. Compute raw health per endpoint
    # ------------------------------------------------------------------
    raw_health: List[Tuple[EngineEndpoint, float]] = []
    for idx, ep in enumerate(endpoints):
        risk = risk_by_entity.get(ep.engine_id, 0.0)
        health = (1.0 - risk) * (1.0 - avg_ssim[idx]) * eta
        raw_health.append((ep, health))

    # ------------------------------------------------------------------
    # 6. Greedy allocation respecting tier factors
    # ------------------------------------------------------------------
    # Sort by descending health to give the strongest candidates first.
    raw_health.sort(key=lambda pair: pair[1], reverse=True)

    allocation: Dict[str, Tuple[str, float]] = {}
    remaining_factors = tier_factors.copy()

    for ep, health in raw_health:
        # Choose tier with maximal remaining factor
        best_tier = max(remaining_factors, key=remaining_factors.get)
        factor = remaining_factors[best_tier]

        # Apply factor to health (deeper integration)
        final_health = health * factor

        allocation[ep.engine_id] = (best_tier, final_health)

        # Decay the tier's remaining factor to model limited capacity.
        # The decay rate is arbitrary; here we use 0.9 per assignment.
        remaining_factors[best_tier] = factor * 0.9

    return allocation


# ----------------------------------------------------------------------
# Simple sanity test (executed when run as script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Synthetic entities spread across three continents
    entities = [
        Entity(id="e1", lat=34.0522, lon=-118.2437, category="A"),
        Entity(id="e2", lat=40.7128, lon=-74.0060, category="B"),
        Entity(id="e3", lat=51.5074, lon=-0.1278, category="C"),
        Entity(id="e4", lat=-33.8688, lon=151.2093, category="D"),
    ]

    # Endpoints aligned with entities (engine_id == entity.id)
    endpoints = [
        EngineEndpoint(
            engine_id="e1",
            channel="chan1",
            residency="us-west",
            runtime="python3.11",
            resource_class="standard",
            always_on=True,
            endpoint="https://api.example.com/e1",
            capabilities=["infer"],
            morphology=Morphology(length=1.2, width=0.8, height=0.5, mass=2.3),
        ),
        EngineEndpoint(
            engine_id="e2",
            channel="chan2",
            residency="us-east",
            runtime="python3.11",
            resource_class="standard",
            always_on=False,
            endpoint="https://api.example.com/e2",
            capabilities=["infer", "train"],
            morphology=Morphology(length=1.0, width=0.9, height=0.6, mass=2.0),
        ),
        EngineEndpoint(
            engine_id="e3",
            channel="chan3",
            residency="eu-central",
            runtime="python3.11",
            resource_class="highmem",
            always_on=True,
            endpoint="https://api.example.com/e3",
            capabilities=["infer"],
            morphology=Morphology(length=1.3, width=0.7, height=0.4, mass=2.5),
        ),
        EngineEndpoint(
            engine_id="e4",
            channel="chan4",
            residency="ap-southeast",
            runtime="python3.11",
            resource_class="gpu",
            always_on=False,
            endpoint="https://api.example.com/e4",
            capabilities=["infer", "embed"],
            morphology=Morphology(length=0.9, width=0.85, height=0.55, mass=1.8),
        ),
    ]

    tiers = [
        ModelTier(name="tiny", ram_mb=2048, vram_mb=0, tier="low"),
        ModelTier(name="standard", ram_mb=8192, vram_mb=2048, tier="mid"),
        ModelTier(name="gpu", ram_mb=16384, vram_mb=16384, tier="high"),
    ]

    ref_loc = (0.0, 0.0)  # arbitrary reference (e.g., Earth centre)

    allocation = allocate_resources_based_on_health(
        entities=entities,
        endpoints=endpoints,
        tiers=tiers,
        reference_location=ref_loc,
    )

    for ep_id, (tier, health) in allocation.items():
        print(f"Endpoint {ep_id} → Tier '{tier}' with health {health:.4f}")