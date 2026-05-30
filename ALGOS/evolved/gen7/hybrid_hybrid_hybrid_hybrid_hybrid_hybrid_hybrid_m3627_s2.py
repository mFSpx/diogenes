# DARWIN HAMMER — match 3627, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1931_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sparse_m2367_s0.py (gen5)
# born: 2026-05-29T23:51:02Z

import numpy as np
import math
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_PERMUTATIONS = 12          # Number of MinHash permutations (higher → more stable)
EARTH_RADIUS_KM = 6371.0       # For haversine distance
MINHASH_SEED_BASE = 0xA5A5A5A5  # Base seed for reproducible hashing

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places for stable printing."""
    return round(float(value), 6)


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash using Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], num_perm: int = NUM_PERMUTATIONS) -> np.ndarray:
    """
    Compute a MinHash signature for an iterable of string tokens.
    Returns an array of uint64 of length ``num_perm``.
    """
    max_uint64 = np.uint64(-1)
    sig = np.full(num_perm, max_uint64, dtype=np.uint64)

    for token in tokens:
        for i in range(num_perm):
            h = _hash(MINHASH_SEED_BASE + i, token)
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard‑like similarity based on equal MinHash components."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match.")
    return float(np.mean(sig1 == sig2))


def _float_features_to_tokens(features: List[float]) -> List[str]:
    """
    Convert a list of numeric features to deterministic string tokens.
    Using a fixed precision avoids hash collisions due to floating‑point noise.
    """
    return [f"{_pct(f):.6f}" for f in features]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great‑circle distance (km) between two lat/lon points.
    """
    rad = math.radians
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


# ----------------------------------------------------------------------
# Core Data Structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Descriptor for a candidate action."""
    action_id: int
    features: List[float]  # e.g. embedding or geographic coordinates


@dataclass(frozen=True)
class Entity:
    """Spatial entity with optional metadata."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


# ----------------------------------------------------------------------
# Regret‑Weighted Bandit Scores
# ----------------------------------------------------------------------
def compute_regret_bandit_scores(
    actions: List[MathAction],
    reference_features: List[float],
    scaling_factor: float,
) -> List[float]:
    """
    Produce regret‑weighted bandit scores for a batch of actions.
    The score is ``similarity * scaling_factor`` where similarity is a
    MinHash Jaccard estimate between the action's feature tokens and the
    reference feature tokens.
    """
    ref_tokens = _float_features_to_tokens(reference_features)
    ref_sig = minhash_signature(ref_tokens)

    scores: List[float] = []
    for act in actions:
        act_tokens = _float_features_to_tokens(act.features)
        act_sig = minhash_signature(act_tokens)
        sim = minhash_similarity(act_sig, ref_sig)
        scores.append(sim * scaling_factor)
    return scores


# ----------------------------------------------------------------------
# Spatial Privacy Risk Vector
# ----------------------------------------------------------------------
def compute_spatial_privacy_risk(entities: List[Entity]) -> np.ndarray:
    """
    Compute a privacy‑risk vector where each entry aggregates the inverse‑distance
    influence of all other entities. The influence decays with haversine distance.
    """
    n = len(entities)
    risk = np.zeros(n, dtype=np.float64)

    # Vectorised pairwise distance matrix (upper triangular)
    for i in range(n):
        lat_i, lon_i = entities[i].lat, entities[i].lon
        for j in range(i + 1, n):
            lat_j, lon_j = entities[j].lat, entities[j].lon
            d = haversine_distance(lat_i, lon_i, lat_j, lon_j)
            influence = 1.0 / (1.0 + d)  # smooth decay
            risk[i] += influence
            risk[j] += influence
    # Normalise to keep values in a comparable range
    total = risk.sum()
    if total > 0:
        risk /= total
    return risk


# ----------------------------------------------------------------------
# Edge Prior Computation (Deep Fusion)
# ----------------------------------------------------------------------
def compute_edge_priors(
    entities: List[Entity],
    risk_vector: np.ndarray,
    bandit_scale: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Fuse spatial privacy risk with regret‑weighted bandit scores to obtain
    Bayesian edge priors.

    For each ordered pair (i → j):
        prior_ij ∝ (risk_i * risk_j) * bandit_score(i, j)

    The proportionality constant is chosen so that the priors sum to 1.
    """
    if len(entities) != risk_vector.shape[0]:
        raise ValueError("Risk vector length must match number of entities.")

    # Pre‑compute MinHash signatures for all entities (based on lat/lon)
    entity_sigs = {}
    for ent in entities:
        tokens = _float_features_to_tokens([ent.lat, ent.lon])
        entity_sigs[ent.id] = minhash_signature(tokens)

    # Compute pairwise bandit scores once (i → j)
    bandit_matrix = np.zeros((len(entities), len(entities)), dtype=np.float64)
    for i, src in enumerate(entities):
        src_sig = entity_sigs[src.id]
        for j, dst in enumerate(entities):
            if i == j:
                continue
            dst_sig = entity_sigs[dst.id]
            sim = minhash_similarity(src_sig, dst_sig)
            # Scale by the destination's privacy risk to reflect "regret" of linking to a risky node
            bandit_matrix[i, j] = sim * (risk_vector[j] ** bandit_scale)

    # Fuse risk and bandit components
    fused = np.outer(risk_vector, risk_vector) * bandit_matrix

    # Normalise to a proper probability distribution
    total = fused.sum()
    if total == 0:
        raise RuntimeError("Degenerate prior matrix: all entries zero.")
    fused /= total

    # Populate dictionary
    edge_priors: Dict[Tuple[str, str], float] = {}
    for i, src in enumerate(entities):
        for j, dst in enumerate(entities):
            if i == j:
                continue
            edge_priors[(src.id, dst.id)] = fused[i, j]

    return edge_priors


# ----------------------------------------------------------------------
# Simple Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny synthetic world
    entities = [
        Entity("A", 37.7749, -122.4194, "city"),
        Entity("B", 37.7859, -122.4364, "city"),
        Entity("C", 37.8044, -122.2712, "city"),
    ]

    risk_vec = compute_spatial_privacy_risk(entities)
    priors = compute_edge_priors(entities, risk_vec, bandit_scale=0.8)

    # Human‑readable output (sorted for reproducibility)
    for (src, dst), val in sorted(priors.items()):
        print(f"{src} → {dst}: {_pct(val)}")