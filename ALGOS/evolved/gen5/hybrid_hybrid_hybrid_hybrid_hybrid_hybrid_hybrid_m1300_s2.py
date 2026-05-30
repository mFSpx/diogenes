# DARWIN HAMMER — match 1300, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s1.py (gen3)
# born: 2026-05-29T23:35:05Z

"""Hybrid Algorithm: Spatial‑Privacy Risk × Morphological Similarity Scheduler

Parents:
- hybrid_hybrid_hybrid_bayes__... (spatial‑aware Bayesian privacy risk, distance weighting, circuit‑breaker health)
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_... (state‑space morphology, SSIM similarity, recovery priority)

Mathematical Bridge:
The privacy risk of an Entity is distance‑weighted (via haversine) and bounded [0,1].
The morphological similarity of two EngineEndpoints is expressed by the Structural
Similarity Index (SSIM) computed on their normalized morphology vectors.
Both quantities live in the unit interval, therefore we can fuse them by
multiplicative coupling:

    health_i = (1 - privacy_risk_i) * (1 - avg_ssim_i) * resource_factor_i

where `avg_ssim_i` is the mean SSIM of endpoint *i* with all other endpoints,
and `resource_factor_i` scales with the RAM/VRAM budget of the ModelTier that
hosts the endpoint.  The resulting `health_i` drives a combined scheduling
score used for work‑share allocation.

The module implements this unified system and provides three core functions:
1. `compute_spatial_privacy_risk`
2. `compute_morphology_ssim_matrix`
3. `allocate_resources_based_on_health`
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (copied/extended from the parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
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

    def as_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Helper functions (parents)
# ----------------------------------------------------------------------


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two lat/lon points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Bayesian‑style reconstruction risk bounded to [0,1]."""
    if total_records <= 0:
        return 0.0
    raw = unique_quasi_identifiers / total_records
    return max(0.0, min(1.0, raw))


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("Signals must have the same length")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x = x_arr.var()
    sigma_y = y_arr.var()
    sigma_xy = ((x_arr - mu_x) * (y_arr - mu_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


def compute_spatial_privacy_risk(entities: List[Entity]) -> Dict[str, float]:
    """
    For each entity compute a distance‑weighted privacy risk:
        risk_i = reconstruction_risk * (1 + avg_normalized_distance_i)

    The distance is normalized by the maximum pairwise distance in the set.
    """
    n = len(entities)
    if n == 0:
        return {}

    # Pre‑compute pairwise distances
    max_dist = 0.0
    dist_matrix = np.zeros((n, n))
    for i, ei in enumerate(entities):
        for j, ej in enumerate(entities):
            if i >= j:
                continue
            d = haversine_m((ei.lat, ei.lon), (ej.lat, ej.lon))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            max_dist = max(max_dist, d)

    # Avoid division by zero when all points coincide
    if max_dist == 0:
        max_dist = 1.0

    risks: Dict[str, float] = {}
    for idx, e in enumerate(entities):
        # Simple proxy for quasi‑identifier count: length of signature string
        uq = len(signature(e))
        base_risk = reconstruction_risk_score(uq, len(entities))
        avg_norm_dist = dist_matrix[idx].mean() / max_dist
        risks[e.id] = max(0.0, min(1.0, base_risk * (1 + avg_norm_dist)))
    return risks


def compute_morphology_ssim_matrix(endpoints: List[EngineEndpoint]) -> np.ndarray:
    """
    Build an NxN SSIM matrix where N = len(endpoints).
    Each endpoint is represented by a 4‑dimensional vector:
        [length, width, height, mass] normalized to unit range.
    """
    n = len(endpoints)
    if n == 0:
        return np.zeros((0, 0))

    # Extract and normalize morphology vectors
    raw = np.array(
        [
            [
                ep.morphology.length,
                ep.morphology.width,
                ep.morphology.height,
                ep.morphology.mass,
            ]
            for ep in endpoints
        ],
        dtype=float,
    )
    # Min‑max normalization per column
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (raw - mins) / ranges

    # Compute SSIM pairwise
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, j] = 1.0
            else:
                sim = ssim(norm[i].tolist(), norm[j].tolist())
                mat[i, j] = sim
                mat[j, i] = sim
    return mat


def allocate_resources_based_on_health(
    entities: List[Entity],
    endpoints: List[EngineEndpoint],
    tiers: List[ModelTier],
) -> List[Tuple[str, str, float]]:
    """
    Produce a list of allocations (entity_id, engine_id, health_score).

    health_score_i = (1 - privacy_risk_i) *
                    (1 - avg_ssim_i) *
                    resource_factor_i

    where `resource_factor_i` = (ram_mb / max_ram) * (vram_mb / max_vram)
    of the tier that matches the endpoint's `resource_class`.
    """
    # 1️⃣ Compute privacy risks per entity
    privacy_risks = compute_spatial_privacy_risk(entities)

    # 2️⃣ Compute SSIM matrix for endpoints and derive avg similarity per endpoint
    ssim_mat = compute_morphology_ssim_matrix(endpoints)
    avg_ssim = (
        ssim_mat.mean(axis=1) if ssim_mat.size else np.zeros(len(endpoints))
    )  # includes self‑similarity (value 1)

    # 3️⃣ Prepare resource factors from tiers
    max_ram = max(t.ram_mb for t in tiers) or 1
    max_vram = max(t.vram_mb for t in tiers) or 1
    tier_lookup = {t.name: t for t in tiers}
    resource_factors = []
    for ep in endpoints:
        tier = tier_lookup.get(ep.resource_class, None)
        if tier is None:
            # fallback to minimal factor
            resource_factors.append(0.1)
        else:
            ram_factor = tier.ram_mb / max_ram
            vram_factor = tier.vram_mb / max_vram
            resource_factors.append(ram_factor * vram_factor)

    # 4️⃣ Map each entity to the “closest” endpoint (by haversine distance)
    allocations = []
    for ent in entities:
        # Find endpoint with minimal geographic distance
        best_ep_idx = None
        best_dist = float("inf")
        for idx, ep in enumerate(endpoints):
            # Assume endpoint location embedded in its `channel` string as "lat,lon"
            try:
                lat_str, lon_str = ep.channel.split(",")
                ep_lat, ep_lon = float(lat_str), float(lon_str)
            except Exception:
                # If not parsable, assign a large distance
                ep_lat, ep_lon = None, None
            if ep_lat is None:
                continue
            d = haversine_m((ent.lat, ent.lon), (ep_lat, ep_lon))
            if d < best_dist:
                best_dist = d
                best_ep_idx = idx

        if best_ep_idx is None:
            continue  # cannot allocate

        # Assemble health score
        priv = privacy_risks.get(ent.id, 0.0)
        sim = avg_ssim[best_ep_idx] if avg_ssim.size else 0.0
        res = resource_factors[best_ep_idx]
        health = (1 - priv) * (1 - sim) * res
        health = max(0.0, min(1.0, health))
        allocations.append((ent.id, endpoints[best_ep_idx].engine_id, health))

    return allocations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic world
    entities = [
        Entity(id="E1", lat=40.0, lon=-74.0, category="A"),
        Entity(id="E2", lat=40.1, lon=-74.1, category="B"),
        Entity(id="E3", lat=39.9, lon=-73.9, category="C"),
    ]

    morphology_samples = [
        Morphology(length=2.0, width=1.0, height=1.5, mass=5.0),
        Morphology(length=1.8, width=1.2, height=1.4, mass=4.5),
        Morphology(length=2.2, width=0.9, height=1.6, mass=5.5),
    ]

    endpoints = [
        EngineEndpoint(
            engine_id="ENG1",
            channel="40.05,-74.05",
            residency="us-east",
            runtime="docker",
            resource_class="TierA",
            always_on=True,
            endpoint="api/v1",
            capabilities=["compute", "store"],
            morphology=morphology_samples[0],
        ),
        EngineEndpoint(
            engine_id="ENG2",
            channel="40.15,-74.15",
            residency="us-west",
            runtime="k8s",
            resource_class="TierB",
            always_on=False,
            endpoint="api/v2",
            capabilities=["compute"],
            morphology=morphology_samples[1],
        ),
        EngineEndpoint(
            engine_id="ENG3",
            channel="39.95,-73.95",
            residency="eu-central",
            runtime="baremetal",
            resource_class="TierA",
            always_on=True,
            endpoint="api/v3",
            capabilities=["store"],
            morphology=morphology_samples[2],
        ),
    ]

    tiers = [
        ModelTier(name="TierA", ram_mb=8192, tier="high", vram_mb=4096),
        ModelTier(name="TierB", ram_mb=4096, tier="mid", vram_mb=2048),
    ]

    allocations = allocate_resources_based_on_health(entities, endpoints, tiers)
    for ent_id, eng_id, health in allocations:
        print(f"Entity {ent_id} -> Engine {eng_id} | Health Score: {health:.4f}")