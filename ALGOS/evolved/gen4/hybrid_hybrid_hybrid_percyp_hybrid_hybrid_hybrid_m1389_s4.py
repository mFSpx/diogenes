# DARWIN HAMMER — match 1389, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_percyphon_hyb_pheromone_m337_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s2.py (gen3)
# born: 2026-05-29T23:35:56Z

import math
import random
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float   # decimal degrees, positive north, negative west
    lon: float   # decimal degrees, positive east, negative west
    category: str
    score: float = 0.0
    address_signature: str = ""


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return the classical sphericity (dimensionless, 0 < ψ ≤ 1)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be strictly positive.")
    volume = length * width * height
    surface_area = 2 * (length * width + width * height + height * length)
    # ψ = (π^{1/3} (6V)^{2/3}) / A   → equivalent formulation used below
    return (volume * (36 * math.pi) ** (1 / 3)) / surface_area


def haversine_distance(
    a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    """Great‑circle distance in kilometres between two (lat, lon) pairs."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)

    aa = sin_dlat ** 2 + math.cos(lat1) * math.cos(lat2) * sin_dlon ** 2
    c = 2.0 * math.atan2(math.sqrt(aa), math.sqrt(1.0 - aa))
    return 6371.0 * c  # Earth radius in kilometres


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    if not values:
        return 0.0
    # Ensure all values are non‑negative; negative entries would break the interpretation.
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Gini coefficient is defined for non‑negative values.")
    if np.allclose(values, 0):
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


# ----------------------------------------------------------------------
# Hyperdimensional primitives (continuous, not binary)
# ----------------------------------------------------------------------
def _hash_to_seed(symbol: str) -> int:
    """Deterministic 64‑bit seed derived from SHA‑256 of the symbol."""
    digest = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=False)


def symbol_vector(symbol: str, dim: int = 10_000) -> np.ndarray:
    """
    Produce a *continuous* hyperdimensional vector of length `dim`.
    Elements are drawn from a symmetric uniform distribution on [-1, 1].
    The generation is deterministic via a seed derived from the symbol.
    """
    rng = np.random.default_rng(_hash_to_seed(symbol))
    return rng.uniform(-1.0, 1.0, size=dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element‑wise multiplication (binding) preserving continuous values."""
    if a.shape != b.shape:
        raise ValueError("Vectors must share the same dimensionality for binding.")
    return a * b


# ----------------------------------------------------------------------
# Fusion logic – deeper integration of morphology and semantics
# ----------------------------------------------------------------------
def morphology_scaling_vector(
    morphology: Morphology, dim: int = 10_000
) -> np.ndarray:
    """
    Convert a morphology into a *continuous scaling vector*.
    The sphericity index is used as a global amplitude, while the
    relative proportions of length, width, and height shape a low‑frequency
    sinusoidal pattern that modulates the hypervector.
    """
    ψ = sphericity_index(
        morphology.length, morphology.width, morphology.height
    )
    # Normalise geometric ratios to sum to 1
    total = morphology.length + morphology.width + morphology.height
    ratios = np.array(
        [
            morphology.length / total,
            morphology.width / total,
            morphology.height / total,
        ]
    )
    # Build a smooth sinusoidal envelope across the vector space
    positions = np.linspace(0, 2 * math.pi, dim, endpoint=False)
    envelope = (
        ratios[0] * np.sin(positions)
        + ratios[1] * np.sin(2 * positions)
        + ratios[2] * np.sin(3 * positions)
    )
    # Apply the global sphericity amplitude
    return ψ * envelope


def hybrid_operation(morphology: Morphology, entity: Entity, dim: int = 10_000) -> np.ndarray:
    """
    Produce a fused hyperdimensional representation that intertwines:
      * the morphological scaling vector (continuous, morphology‑driven)
      * the entity identifier vector (continuous, unique per ID)
      * the category vector (continuous, shared per category)
    The three components are bound together, yielding a richer, non‑binary
    embedding.
    """
    scale_vec = morphology_scaling_vector(morphology, dim)
    id_vec = symbol_vector(entity.id, dim)
    cat_vec = symbol_vector(entity.category, dim)

    # Bind sequentially; binding is associative, so order does not matter
    return bind(bind(scale_vec, id_vec), cat_vec)


def compute_gini_on_norms(vectors: List[np.ndarray]) -> float:
    """
    Compute the Gini coefficient on the L2‑norms of a collection of hypervectors.
    This captures dispersion of magnitude across entities, a more meaningful
    statistic than raw element‑wise sums for continuous embeddings.
    """
    norms = [float(np.linalg.norm(v)) for v in vectors]
    return gini_coefficient(norms)


def compute_haversine_distance(entity1: Entity, entity2: Entity) -> float:
    """Convenient wrapper that extracts coordinates from Entity objects."""
    return haversine_distance((entity1.lat, entity1.lon), (entity2.lat, entity2.lon))


# ----------------------------------------------------------------------
# Demonstration / simple test harness
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example morphology (arbitrary but physically plausible)
    morphology = Morphology(length=10.0, width=5.0, height=2.0, mass=100.0)

    # Example entities (coordinates use signed longitudes)
    entity1 = Entity(id="entity_001", lat=40.7128, lon=-74.0060, category="A")
    entity2 = Entity(id="entity_002", lat=34.0522, lon=-118.2437, category="B")

    # Generate fused hypervectors
    vec1 = hybrid_operation(morphology, entity1)
    vec2 = hybrid_operation(morphology, entity2)

    # Statistics
    gini = compute_gini_on_norms([vec1, vec2])
    distance_km = compute_haversine_distance(entity1, entity2)

    print(f"Gini coefficient on vector norms: {gini:.6f}")
    print(f"Haversine distance (km): {distance_km:.2f}")