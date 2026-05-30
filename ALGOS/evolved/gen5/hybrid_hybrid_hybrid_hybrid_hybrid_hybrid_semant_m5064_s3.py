# DARWIN HAMMER — match 5064, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m528_s3.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_hybrid_geomet_m116_s0.py (gen3)
# born: 2026-05-29T23:59:36Z

"""
Hybrid algorithm merging:

- Parent A (hybrid_hybrid_hybrid_bayes__...): spatial‑aware privacy risk vector for Entities.
- Parent B (hybrid_hybrid_semantic_neig_...): Multivector algebra, pheromone‑based probabilities and entropy.

Mathematical bridge:
Each Entity is mapped to a simple Multivector consisting of a scalar component equal to its
privacy‑risk score and a vector component (e1, e2) encoding its geographic coordinates
(lat, lon). ModelTier objects are likewise represented as Multivectors whose scalar part
encodes a tier‑specific reliability weight and whose vector part encodes resource
characteristics (ram_mb, vram_mb). The geometric product of an Entity‑Multivector with a
Tier‑Multivector yields a new Multivector whose scalar part naturally combines privacy
risk, spatial proximity, and tier reliability. This scalar is then weighted by a
pheromone probability (derived from a pheromone trail per tier) and finally aggregated
using an entropy measure to obtain a single hybrid score for the system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ---------- Parent A structures ----------
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
    reliability: float = 1.0   # added for weighting in the hybrid

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def _signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """Return a vector of privacy risk scores, one per entity."""
    risks = []
    for i, entity in enumerate(entities):
        similar = [
            e for j, e in enumerate(entities)
            if i != j
            and _signature(entity) == _signature(e)
            and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m
        ]
        unique_quasi = len(set(e.category for e in similar))
        risk = reconstruction_risk_score(unique_quasi, len(entities))
        risks.append(risk)
    return np.array(risks, dtype=float)

# ---------- Parent B structures ----------
def _cos(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

def pheromone_probabilities(pheromones: List[float]) -> List[float]:
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone list must contain at least one positive value")
    return [p / total for p in pheromones]

def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

class Multivector:
    """
    Simple implementation limited to scalar (grade‑0) and vector (grade‑1) parts.
    Blades are represented by frozensets of integers, e.g. frozenset() for scalar,
    frozenset({1}) for e1, frozenset({2}) for e2, etc.
    """
    def __init__(self, components: Dict[frozenset, float]):
        # keep only non‑zero components
        self.components = {blade: float(v) for blade, v in components.items() if abs(v) > 1e-15}
        self._max_grade = max((len(b) for b in self.components), default=0)

    def scalar(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def vector(self) -> Tuple[float, ...]:
        # assumes at most 2‑dimensional vectors (e1, e2)
        v1 = self.components.get(frozenset({1}), 0.0)
        v2 = self.components.get(frozenset({2}), 0.0)
        return (v1, v2)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                label = "1"
            else:
                label = "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Geometric product limited to scalar+vector grades (no bivectors beyond e12)."""
        result: Dict[frozenset, float] = {}

        # scalar‑scalar
        result[frozenset()] = self.scalar() * other.scalar()

        # scalar‑vector and vector‑scalar
        for i, vi in enumerate(self.vector(), start=1):
            if vi != 0.0:
                # self vector * other scalar
                result[frozenset({i})] = result.get(frozenset({i}), 0.0) + vi * other.scalar()
        for i, wi in enumerate(other.vector(), start=1):
            if wi != 0.0:
                # self scalar * other vector
                result[frozenset({i})] = result.get(frozenset({i}), 0.0) + self.scalar() * wi

        # vector‑vector (dot + wedge)
        v_self = self.vector()
        v_other = other.vector()
        # dot product contributes to scalar
        dot = sum(v_self[i] * v_other[i] for i in range(2))
        result[frozenset()] = result.get(frozenset(), 0.0) + dot
        # wedge (e1^e2) contributes to bivector e12
        wedge = v_self[0] * v_other[1] - v_self[1] * v_other[0]
        if wedge != 0.0:
            result[frozenset({1, 2})] = wedge

        return Multivector(result)

# ---------- Hybrid bridge functions ----------
def entity_to_multivector(entity: Entity, risk: float) -> Multivector:
    """
    Map an Entity to a Multivector:
        scalar part   = privacy risk (0‑1)
        vector part   = (lat, lon) projected onto a 2‑D basis (e1, e2)
    """
    comps = {
        frozenset(): risk,
        frozenset({1}): entity.lat,
        frozenset({2}): entity.lon,
    }
    return Multivector(comps)

def tier_to_multivector(tier: ModelTier) -> Multivector:
    """
    Map a ModelTier to a Multivector:
        scalar part   = reliability weight (0‑1)
        vector part   = normalized (ram_mb, vram_mb) on the same basis
    """
    # Normalization to keep values comparable with lat/lon magnitudes
    max_ram = 64_000.0   # assume an upper bound of 64 GB
    max_vram = 32_000.0  # assume an upper bound of 32 GB
    norm_ram = tier.ram_mb / max_ram
    norm_vram = tier.vram_mb / max_vram
    comps = {
        frozenset(): tier.reliability,
        frozenset({1}): norm_ram,
        frozenset({2}): norm_vram,
    }
    return Multivector(comps)

def hybrid_scores(entities: List[Entity],
                  tiers: List[ModelTier],
                  delta_m: float,
                  pheromones: List[float]) -> Tuple[np.ndarray, float]:
    """
    Compute a hybrid score matrix (entities × tiers) and an overall entropy metric.

    Steps:
    1. Compute privacy risk per entity.
    2. Convert entities and tiers to Multivectors.
    3. For each (entity, tier) pair compute the geometric product.
    4. Extract the scalar part of the product as the raw compatibility score.
    5. Weight the raw score by the pheromone probability of the tier.
    6. Return the weighted score matrix and the entropy of the pheromone distribution.
    """
    if len(tiers) != len(pheromones):
        raise ValueError("Number of pheromone values must match number of tiers")

    risks = spatial_aware_privacy_risk_vector(entities, delta_m)  # shape (E,)
    entity_mvs = [entity_to_multivector(e, r) for e, r in zip(entities, risks)]
    tier_mvs = [tier_to_multivector(t) for t in tiers]

    pher_prob = pheromone_probabilities(pheromones)  # list length T
    E = len(entities)
    T = len(tiers)
    score_matrix = np.zeros((E, T), dtype=float)

    for i, ev in enumerate(entity_mvs):
        for j, tv in enumerate(tier_mvs):
            prod = ev.geometric_product(tv)
            raw = prod.scalar()               # combined scalar
            weighted = raw * pher_prob[j]      # pheromone weighting
            score_matrix[i, j] = weighted

    overall_entropy = entropy(pher_prob)
    return score_matrix, overall_entropy

def best_tier_per_entity(score_matrix: np.ndarray, tiers: List[ModelTier]) -> List[Tuple[Entity, ModelTier, float]]:
    """
    Given the hybrid score matrix, return for each entity the tier with the highest score.
    """
    best_indices = np.argmax(score_matrix, axis=1)
    results = []
    for i, tier_idx in enumerate(best_indices):
        results.append((i, tier_idx, score_matrix[i, tier_idx]))
    return results

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    entities = [
        Entity(id="e1", lat=37.7749, lon=-122.4194, category="A"),
        Entity(id="e2", lat=34.0522, lon=-118.2437, category="A"),
        Entity(id="e3", lat=40.7128, lon=-74.0060, category="B"),
    ]

    tiers = [
        ModelTier(name="tiny", ram_mb=2048, tier="edge", vram_mb=512, reliability=0.8),
        ModelTier(name="small", ram_mb=8192, tier="cloud", vram_mb=2048, reliability=0.9),
        ModelTier(name="large", ram_mb=32768, tier="datacenter", vram_mb=8192, reliability=0.95),
    ]

    # Random pheromone levels (simulating trail intensity)
    pheromones = [random.random() for _ in tiers]

    # Compute hybrid scores
    scores, ent = hybrid_scores(entities, tiers, delta_m=500_000, pheromones=pheromones)

    print("Pheromone probabilities:", pheromone_probabilities(pheromones))
    print("Overall pheromone entropy:", ent)
    print("Hybrid score matrix (entity × tier):")
    print(scores)

    # Identify best tier per entity
    best = best_tier_per_entity(scores, tiers)
    for entity_idx, tier_idx, score in best:
        print(f"Entity {entities[entity_idx].id} best matches Tier '{tiers[tier_idx].name}' with score {score:.4f}")

    sys.exit(0)