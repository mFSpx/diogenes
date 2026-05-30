# DARWIN HAMMER — match 3834, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1933_s1.py (gen5)
# born: 2026-05-29T23:51:51Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s3.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1933_s1.py

Mathematical Bridge
-------------------
Parent A represents a serpentina morphology as a high‑dimensional random vector and
derives a MinHash signature from a tokenised description of the morphology.
Similarity between two morphologies is obtained via Jaccard‑like equality of their
signatures.

Parent B builds a 3‑dimensional *resource vector* **eᵢ = [dᵢ, pᵢ, sᵢ]** where  
* dᵢ – haversine distance from a reference point,  
* pᵢ – β·σᵢ with σᵢ indicating a signature collision, and  
* sᵢ – a score composed of geometric indices (sphericity, flatness) and the
  righting‑time index.

The fusion treats the MinHash similarity from A as a *modulator* for the
resource‑vector components of B:

- The collision flag σᵢ is derived from the MinHash signature of the current
  morphology against a set of peer signatures.
- The decision‑hygiene score sᵢ incorporates the righting‑time index (from A) and
  the geometric indices (from B).
- The final hybrid metric is a weighted dot‑product of the resource vector,
  where the weights themselves are scaled by the MinHash similarity to a goal
  morphology.

Thus the two topologies are mathematically intertwined: a high‑dimensional
signature informs a low‑dimensional resource vector, and the resource vector
feeds back into the similarity‑driven decision metric.
"""

import math
import random
import hashlib
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Iterable

MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Parent A – MinHash utilities
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Parent B – Geometric and resource‑vector utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """A simple proxy for sphericity: (V)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    longest = max(length, width, height)
    return (volume ** (1.0 / 3.0)) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length + width) / (2·height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """
    Approximate righting‑time index.
    Uses a power‑law of mass and volume, scaled by a neck‑lever factor.
    """
    volume = m.length * m.width * m.height
    return (m.mass ** b) * (volume ** k) * neck_lever


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Distance in metres between two lat/lon points (WGS‑84)."""
    R = 6371000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def morphology_vector(m: Morphology, dim: int = 1024, seed: str | int | None = None) -> List[float]:
    """
    High‑dimensional random vector seeded deterministically by the morphology's
    physical parameters.
    """
    # Derive a 64‑bit integer seed from the morphology attributes
    hash_input = f"{m.length}:{m.width}:{m.height}:{m.mass}"
    seed_int = int.from_bytes(hashlib.sha256(hash_input.encode()).digest()[:8], "big")
    rng = random.Random(seed_int if seed is None else seed)
    return [rng.random() for _ in range(dim)]


def tokenise_morphology(m: Morphology) -> List[str]:
    """Create a simple token set from morphology attributes."""
    return [
        f"len:{m.length:.6f}",
        f"wid:{m.width:.6f}",
        f"hei:{m.height:.6f}",
        f"mass:{m.mass:.6f}",
    ]


def signature_collision(sig: List[int], peer_sigs: Iterable[List[int]]) -> int:
    """
    Returns 1 if *any* hash in `sig` appears in any peer signature, else 0.
    """
    peer_hashes = set()
    for ps in peer_sigs:
        peer_hashes.update(ps)
    return int(any(h in peer_hashes for h in sig))


def compute_resource_vector(
    m: Morphology,
    location: Tuple[float, float],
    reference: Tuple[float, float],
    my_sig: List[int],
    peer_sigs: Iterable[List[int]],
    beta: float = 1.0,
) -> np.ndarray:
    """
    Build the 3‑dimensional resource vector e = [d, p, s] for a given morphology.
    """
    # d – haversine distance
    d = haversine_distance(location[0], location[1], reference[0], reference[1])

    # p – collision flag scaled by β
    sigma = signature_collision(my_sig, peer_sigs)
    p = beta * sigma

    # s – composite decision‑hygiene score
    sphericity = sphericity_index(m.length, m.width, m.height)
    flatness = flatness_index(m.length, m.width, m.height)
    righting = righting_time_index(m)
    # Weighted combination (weights sum to 1)
    s = 0.4 * sphericity + 0.3 * flatness + 0.3 * (righting / (righting + 1.0))

    return np.array([d, p, s], dtype=float)


def hybrid_metric(
    m: Morphology,
    goal_m: Morphology,
    location: Tuple[float, float],
    reference: Tuple[float, float],
    peer_morphologies: List[Morphology],
    beta: float = 1.0,
    weight_similarity: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """
    Compute a unified hybrid score.

    Steps
    -----
    1. Generate MinHash signatures for `m` and the goal morphology.
    2. Compute similarity S ∈ [0,1].
    3. Build the resource vector e = [d, p, s] where:
       - d = geographic distance,
       - p = β·σ (collision flag),
       - s = composite geometric/righting score.
    4. Scale the resource vector by similarity‑dependent weights and return
       the dot product.
    """
    # 1. Signatures
    my_sig = minhash_signature(tokenise_morphology(m))
    goal_sig = minhash_signature(tokenise_morphology(goal_m))
    peer_sigs = [minhash_signature(tokenise_morphology(pm)) for pm in peer_morphologies]

    # 2. Similarity
    S = similarity(my_sig, goal_sig)

    # 3. Resource vector
    e = compute_resource_vector(m, location, reference, my_sig, peer_sigs, beta=beta)

    # 4. Similarity‑modulated weights
    w_base = np.array(weight_similarity, dtype=float)
    w = w_base * (0.5 + 0.5 * S)  # weights increase with similarity, staying within [0,1]
    w /= w.sum()  # re‑normalize to keep a proper convex combination

    return float(e @ w)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a current morphology and a goal morphology
    current = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)
    goal = Morphology(length=1.0, width=0.9, height=0.6, mass=2.0)

    # Geographic positions (lat, lon) in decimal degrees
    loc_current = (37.7749, -122.4194)   # San Francisco
    loc_reference = (34.0522, -118.2437)  # Los Angeles

    # Peer morphologies (could be from a population)
    peers = [
        Morphology(length=1.1, width=0.85, height=0.55, mass=2.1),
        Morphology(length=0.9, width=0.7, height=0.4, mass=1.8),
    ]

    score = hybrid_metric(
        m=current,
        goal_m=goal,
        location=loc_current,
        reference=loc_reference,
        peer_morphologies=peers,
        beta=0.7,
    )
    print(f"Hybrid metric score: {score:.6f}")