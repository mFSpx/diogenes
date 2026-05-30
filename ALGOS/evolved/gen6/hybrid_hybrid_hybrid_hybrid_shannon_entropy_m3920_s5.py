# DARWIN HAMMER — match 3920, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1300_s2.py (gen5)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:52:41Z

"""Hybrid Algorithm: Spatial‑Privacy Risk × Morphological Similarity Scheduler × Entropy Modulator
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1300_s2.py (spatial privacy + SSIM health)
- shannon_entropy.py (Shannon entropy utility)

Mathematical Bridge:
Both parent algorithms produce unit‑interval quantities:
* privacy_risk ∈ [0,1] (distance‑weighted)
* avg_ssim ∈ [0,1] (mean structural similarity)
The Shannon entropy of a discrete distribution of these quantities,
    H = -∑ p_i log₂ p_i,
is also bounded (0 ≤ H ≤ log₂ N).  By normalising H we obtain a factor
    η = 1 + H / log₂ N ∈ [1,2],
which can be used as a multiplicative modulator of the health score.
Thus the unified health for entity *i* is

    health_i = (1 - privacy_risk_i) * (1 - avg_ssim_i) * resource_factor_i * η

where `resource_factor_i` scales with the RAM/VRAM budget of the ModelTier that
hosts the endpoint.  The module implements this fused system and provides three
core functions that demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
from collections import Counter
from collections.abc import Hashable, Iterable

# ----------------------------------------------------------------------
# Data structures (merged from parents)
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


# ----------------------------------------------------------------------
# Helper functions (parent B – Shannon entropy)
# ----------------------------------------------------------------------


def shannon_entropy(observations: Iterable[Hashable | float],
                    is_distribution: bool = False) -> float:
    """Calculate Shannon entropy in bits."""
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ----------------------------------------------------------------------
# Core hybrid functions (parent A + entropy)
# ----------------------------------------------------------------------


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great‑circle distance in kilometres."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_spatial_privacy_risk(entities: List[Entity],
                                 reference: Tuple[float, float],
                                 max_distance_km: float = 20000.0) -> Dict[str, float]:
    """
    Distance‑weighted privacy risk.
    risk_i = min(1, distance_i / max_distance_km)
    Returns a dict {entity_id: risk}.
    """
    ref_lat, ref_lon = reference
    risks = {}
    for e in entities:
        d = haversine(e.lat, e.lon, ref_lat, ref_lon)
        risk = min(1.0, d / max_distance_km)
        risks[e.id] = risk
    return risks


def _morphology_vector(m: Morphology) -> np.ndarray:
    """Normalised 4‑D morphology vector."""
    vec = np.array([m.length, m.width, m.height, m.mass], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_morphology_ssim_matrix(endpoints: List[EngineEndpoint]) -> np.ndarray:
    """
    Proxy SSIM using cosine similarity on normalised morphology vectors.
    Returns an NxN matrix where N = len(endpoints).
    """
    n = len(endpoints)
    if n == 0:
        return np.empty((0, 0))
    mats = np.stack([_morphology_vector(ep.morphology) for ep in endpoints])
    # cosine similarity = dot product because vectors are normalised
    sim = np.clip(np.dot(mats, mats.T), 0.0, 1.0)
    return sim


def allocate_resources_based_on_health(entities: List[Entity],
                                       endpoints: List[EngineEndpoint],
                                       tiers: List[ModelTier],
                                       reference_location: Tuple[float, float]) -> Dict[str, Tuple[str, float]]:
    """
    Compute a health score for each endpoint and allocate it to the most
    suitable ModelTier.  The health incorporates:
      - (1 - privacy_risk)
      - (1 - average SSIM with other endpoints)
      - resource_factor derived from tier RAM+VRAM
      - entropy_modulator η based on the distribution of (1 - privacy_risk) values
    Returns a mapping {endpoint_id: (tier_name, health)}.
    """
    # 1. privacy risk per entity (entity id == endpoint.engine_id assumed)
    risk_by_id = compute_spatial_privacy_risk(entities, reference_location)

    # 2. SSIM matrix and per‑endpoint average
    ssim_mat = compute_morphology_ssim_matrix(endpoints)
    avg_ssim = ssim_mat.mean(axis=1) if ssim_mat.size else np.zeros(len(endpoints))

    # 3. Entropy modulator based on (1 - risk) distribution
    complement_risks = [1 - risk_by_id.get(ep.engine_id, 0.0) for ep in endpoints]
    H = shannon_entropy(complement_risks, is_distribution=False)
    N = max(1, len(complement_risks))
    eta = 1.0 + H / math.log2(N)  # ∈ [1,2]

    # 4. Resource factor per tier (simple linear scaling)
    tier_factors = {t.name: (t.ram_mb + t.vram_mb) / 1024.0 for t in tiers}
    # Normalise to [0,1] then shift to avoid zero
    max_factor = max(tier_factors.values()) if tier_factors else 1.0
    tier_factors = {k: 0.5 + 0.5 * (v / max_factor) for k, v in tier_factors.items()}

    # 5. Compute health and allocate
    allocation: Dict[str, Tuple[str, float]] = {}
    for idx, ep in enumerate(endpoints):
        risk = risk_by_id.get(ep.engine_id, 0.0)
        health = (1 - risk) * (1 - avg_ssim[idx]) * eta
        # pick tier with highest factor (simple heuristic)
        best_tier = max(tier_factors, key=tier_factors.get)
        health *= tier_factors[best_tier]
        allocation[ep.engine_id] = (best_tier, health)

    return allocation


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a few synthetic entities
    entities = [
        Entity(id="e1", lat=34.05, lon=-118.25, category="A"),
        Entity(id="e2", lat=40.71, lon=-74.00, category="B"),
        Entity(id="e3", lat=51.51, lon=-0.13, category="C"),
    ]

    # Corresponding endpoints (engine_id matches entity.id for simplicity)
    endpoints = [
        EngineEndpoint(
            engine_id="e1",
            channel="chan1",
            residency="us-west",
            runtime="python3.11",
            resource_class="standard",
            always_on=True,
            endpoint="http://example.com/e1",
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
            endpoint="http://example.com/e2",
            capabilities=["infer"],
            morphology=Morphology(length=1.0, width=0.9, height=0.4, mass=2.0),
        ),
        EngineEndpoint(
            engine_id="e3",
            channel="chan3",
            residency="eu-central",
            runtime="python3.11",
            resource_class="highmem",
            always_on=True,
            endpoint="http://example.com/e3",
            capabilities=["infer"],
            morphology=Morphology(length=1.3, width=0.7, height=0.6, mass=2.5),
        ),
    ]

    # Define some model tiers
    tiers = [
        ModelTier(name="small", ram_mb=2048, tier="S", vram_mb=1024),
        ModelTier(name="medium", ram_mb=4096, tier="M", vram_mb=2048),
        ModelTier(name="large", ram_mb=8192, tier="L", vram_mb=4096),
    ]

    # Reference location (e.g., datacenter centre)
    reference = (37.7749, -122.4194)  # San Francisco

    # Run allocation
    allocation = allocate_resources_based_on_health(
        entities, endpoints, tiers, reference_location=reference
    )

    for eid, (tier, health) in allocation.items():
        print(f"Endpoint {eid} allocated to tier '{tier}' with health {health:.4f}")