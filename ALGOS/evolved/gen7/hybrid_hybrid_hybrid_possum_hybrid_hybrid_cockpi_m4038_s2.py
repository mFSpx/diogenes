# DARWIN HAMMER — match 4038, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hybrid_hybrid_nlms_h_m2426_s2.py (gen6)
# born: 2026-05-29T23:53:14Z

"""Hybrid Spatial‑Privacy & Honesty‑Weighted Pheromone Algorithm
================================================================

Parent A: *hybrid_hybrid_possum_filter_hybrid_caputo_fracti_m1220_s2.py* – provides a
spatial‑aware privacy risk vector based on haversine distances, categorical similarity
and a reconstruction‑risk formula.

Parent B: *hybrid_hybrid_cockpit_metri_hybrid_hybrid_nlms_h_m2426_s2.py* – defines an
honesty‑weighted pheromone signal and an RBF (Radial Basis Function) kernel matrix that
captures similarity between feature vectors.

**Mathematical bridge**

The bridge is the *risk vector* produced by Parent A.  We treat each entity’s risk as a
scalar feature and construct a full RBF kernel matrix **K** over the set of entities.
The kernel values quantify how “close” two entities are in the privacy‑risk space.
These kernel similarities are then injected into the pheromone update of Parent B:
the pheromone strength for an entity is multiplied by the average kernel similarity
to its neighbours, thus biasing the search toward clusters of low‑risk (or high‑risk)
entities while still being modulated by the honesty weight.

The resulting hybrid algorithm therefore fuses spatial‑privacy assessment,
fractional‑memory‑style similarity (via the RBF kernel) and honesty‑weighted pheromone
propagation in a single mathematically coherent pipeline.
"""

import math
import random
import sys
from pathlib import Path, PurePath
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Simple record describing a spatial entity."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def signature(e: Entity) -> str:
    """Canonical signature used for categorical similarity."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """
    For each entity compute a privacy risk that depends on:
      * categorical match (signature)
      * spatial proximity (haversine <= delta_m)
      * reconstruction risk based on the number of distinct IDs among the matching set.
    Returns a 1‑D numpy array of risks aligned with ``entities``.
    """
    total = len(entities)
    risks = np.empty(total, dtype=float)

    for i, ent in enumerate(entities):
        # Find peers that share the same signature and are within the spatial window.
        peers = [
            other
            for j, other in enumerate(entities)
            if i != j
            and signature(ent) == signature(other)
            and haversine_m((ent.lat, ent.lon), (other.lat, other.lon)) <= delta_m
        ]
        unique_ids = {p.id for p in peers}
        risk = reconstruction_risk_score(len(unique_ids), total)
        risks[i] = risk

    return risks


# ----------------------------------------------------------------------
# Kernel & pheromone utilities (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel component."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D numpy arrays."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a - b))


def rbf_kernel_matrix(features: Dict[int, np.ndarray], epsilon: float = 1.0) -> np.ndarray:
    """
    Build a symmetric RBF kernel matrix K where
        K_ij = exp(- (epsilon * ||f_i - f_j||)^2 )
    ``features`` maps an integer index to a feature vector.
    """
    n = len(features)
    K = np.empty((n, n), dtype=float)
    indices = list(features.keys())
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            if j < i:
                K[i, j] = K[j, i]  # symmetry
            else:
                dist = euclidean(features[idx_i], features[idx_j])
                K[i, j] = gaussian(dist, epsilon)
    return K


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Honesty factor in [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def calculate_honesty_weighted_pheromone_signal(
    base_signal: float,
    half_life_seconds: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
) -> float:
    """
    Decays ``base_signal`` exponentially with a half‑life and scales it by the honesty weight.
    """
    honesty = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    elapsed = (datetime.now() - datetime.now()).total_seconds()  # placeholder zero‑delay
    decay = math.pow(0.5, elapsed / half_life_seconds) if half_life_seconds > 0 else 1.0
    return base_signal * decay * honesty


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_compute_entity_features(entities: List[Entity], delta_m: float) -> Dict[int, np.ndarray]:
    """
    Produce a feature dictionary for the kernel where each feature vector consists of:
        [privacy_risk, normalized_score]
    The index used in the dict matches the position of the entity in ``entities``.
    """
    risks = spatial_aware_privacy_risk_vector(entities, delta_m)
    scores = np.array([e.score for e in entities], dtype=float)
    # Normalise scores to [0,1] to keep magnitude comparable with risk.
    if scores.max() > 0:
        scores = scores / scores.max()
    features = {
        i: np.array([risks[i], scores[i]], dtype=float) for i in range(len(entities))
    }
    return features


def hybrid_pheromone_update(
    entities: List[Entity],
    delta_m: float,
    half_life_seconds: float,
    claims_with_evidence: int,
    total_claims_emitted: int,
    epsilon: float = 1.0,
) -> List[float]:
    """
    Core hybrid routine:
      1. Build risk‑augmented feature vectors.
      2. Compute the RBF kernel matrix K over those vectors.
      3. For each entity, derive a base pheromone signal proportional to its raw ``score``.
      4. Modulate the base signal by:
            a) the average kernel similarity to all other entities (captures spatial‑privacy clustering)
            b) the honesty‑weighted decay.
    Returns a list of updated pheromone strengths aligned with ``entities``.
    """
    # 1‑3: raw signals
    base_signals = np.array([e.score for e in entities], dtype=float)
    if base_signals.max() > 0:
        base_signals = base_signals / base_signals.max()  # normalise to [0,1]

    # 1‑2: kernel
    feats = hybrid_compute_entity_features(entities, delta_m)
    K = rbf_kernel_matrix(feats, epsilon)

    # average similarity for each entity (exclude self‑similarity)
    avg_sim = (K.sum(axis=1) - np.diag(K)) / (len(entities) - 1) if len(entities) > 1 else np.zeros(len(entities))

    # 4: combine
    updated = []
    for i, raw in enumerate(base_signals):
        weighted = raw * avg_sim[i]
        final = calculate_honesty_weighted_pheromone_signal(
            weighted,
            half_life_seconds,
            claims_with_evidence,
            total_claims_emitted,
        )
        updated.append(final)

    return updated


def hybrid_select_top_entities(entities: List[Entity], pheromones: List[float], k: int) -> List[Entity]:
    """
    Helper that selects the ``k`` entities with the highest hybrid pheromone strength.
    """
    if k <= 0:
        return []
    paired = list(zip(entities, pheromones))
    paired.sort(key=lambda pair: pair[1], reverse=True)
    return [pair[0] for pair in paired[:k]]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    random.seed(0)
    demo_entities = [
        Entity(id=f"id_{i}", lat=40.0 + random.random(), lon=-74.0 + random.random(),
               category="A" if i % 2 == 0 else "B",
               score=random.random(),
               address_signature="sigA" if i % 3 == 0 else "sigB")
        for i in range(7)
    ]

    # Parameters
    DELTA_M = 5000.0                # 5 km spatial window
    HALF_LIFE = 3600.0              # 1 hour
    CLAIMS_EVID = 8
    CLAIMS_TOTAL = 10
    EPS = 0.8
    TOP_K = 3

    # Run hybrid pipeline
    pheromone_vals = hybrid_pheromone_update(
        demo_entities,
        delta_m=DELTA_M,
        half_life_seconds=HALF_LIFE,
        claims_with_evidence=CLAIMS_EVID,
        total_claims_emitted=CLAIMS_TOTAL,
        epsilon=EPS,
    )

    top_entities = hybrid_select_top_entities(demo_entities, pheromone_vals, TOP_K)

    # Simple verification output (no external libraries)
    print("Hybrid pheromone values:")
    for e, p in zip(demo_entities, pheromone_vals):
        print(f"  {e.id}: {p:.4f}")

    print("\nTop entities:")
    for e in top_entities:
        print(f"  {e.id} (score={e.score:.3f})")