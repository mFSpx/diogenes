# DARWIN HAMMER — match 5102, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percyp_hybrid_hybrid_hybrid_m1389_s1.py (gen4)
# born: 2026-05-29T23:59:54Z

import numpy as np
import math
import hashlib
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Morphology:
    """Physical dimensions used for the sphericity‑based blending factor."""
    length: float
    width: float
    height: float
    mass: float = 0.0  # retained for compatibility; not used in the blend


@dataclass(frozen=True)
class Entity:
    """A lightweight representation of a spatially‑located object."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimension‑less sphericity value in (0, 1]."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    volume = length * width * height
    surface = 2 * (length * width + width * height + height * length)
    # Normalised so that a perfect sphere yields 1
    return (volume * (36 * math.pi) ** (1 / 3)) / surface


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in kilometres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    aa = sin_dlat ** 2 + math.cos(lat1) * math.cos(lat2) * sin_dlon ** 2
    c = 2 * math.atan2(math.sqrt(aa), math.sqrt(1 - aa))
    return 6371.0 * c


def _shingles(text: str, width: int = 5) -> List[str]:
    """Return overlapping substrings of length *width*."""
    return [text[i: i + width] for i in range(len(text) - width + 1)]


def _stable_hash(s: str) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    h = hashlib.sha256(s.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)


def minhash_signature(text: str, k: int = 64, width: int = 5) -> np.ndarray:
    """
    Produce a *k*‑dimensional MinHash signature.
    The signature is a NumPy array of unsigned 64‑bit integers.
    """
    if not text:
        return np.zeros(k, dtype=np.uint64)

    shingles = _shingles(text.lower(), width)
    if not shingles:
        return np.zeros(k, dtype=np.uint64)

    # Compute a hash for each shingle
    hashes = np.fromiter((_stable_hash(s) for s in shingles), dtype=np.uint64)

    # Keep the *k* smallest values (MinHash)
    if len(hashes) > k:
        # ``np.partition`` is O(n) and gives the k‑smallest unordered
        kth = np.partition(hashes, k - 1)[:k]
        signature = np.sort(kth)
    else:
        signature = np.sort(hashes)

    # Pad with zeros if we have fewer than k values
    if signature.size < k:
        signature = np.pad(signature, (0, k - signature.size), constant_values=0)

    return signature


def jaccard_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Approximate Jaccard distance from two MinHash signatures.
    ``1 - similarity`` where similarity is the fraction of equal hash values.
    """
    if sig1.shape != sig2.shape:
        raise ValueError("Signatures must have the same shape.")
    matches = np.count_nonzero(sig1 == sig2)
    return 1.0 - matches / sig1.size


def hybrid_distance(
    e1: Entity,
    e2: Entity,
    morphology: Morphology,
    *,
    sphericity_cache: dict = None,
) -> float:
    """
    A blended distance that respects:
      • textual similarity (Jaccard distance on MinHash)
      • geographic proximity (haversine)
      • morphological context (sphericity‑driven weighting)

    The two components are combined via a convex blend:
        d = w * text_dist + (1‑w) * (haversine / H_MAX)

    where ``w`` is derived from the sphericity index and ``H_MAX`` is a
    normalising constant (the Earth's half‑circumference ≈ 20015 km).
    """
    # ---- textual component -------------------------------------------------
    sig1 = minhash_signature(e1.address_signature)
    sig2 = minhash_signature(e2.address_signature)
    text_dist = jaccard_distance(sig1, sig2)  # ∈ [0, 1]

    # ---- geographic component -----------------------------------------------
    geo_dist = haversine_distance((e1.lat, e1.lon), (e2.lat, e2.lon))

    # ---- morphological weighting ---------------------------------------------
    if sphericity_cache is None:
        sphericity_cache = {}
    key = (morphology.length, morphology.width, morphology.height)
    if key not in sphericity_cache:
        sphericity_cache[key] = sphericity_index(*key)
    sph = sphericity_cache[key]  # ∈ (0, 1]

    # Convert sphericity to a convex weight in (0,1)
    w = sph / (sph + 1.0)  # ensures 0 < w < 0.5 for realistic shapes

    # Normalise geographic distance to a comparable scale
    H_MAX = 20015.0  # half the Earth's meridional circumference in km
    geo_norm = geo_dist / H_MAX  # ∈ [0, 1]

    # ---- final blended distance -----------------------------------------------
    return w * text_dist + (1.0 - w) * geo_norm


def ternary_routing(
    src: Entity,
    dst: Entity,
    candidates: List[Entity],
    morphology: Morphology,
) -> int:
    """
    Choose an intermediate node that minimises the sum of blended distances
    from ``src`` → *candidate* → ``dst``.
    The source and destination themselves are excluded from the candidate set.
    """
    if not candidates:
        raise ValueError("Candidate list cannot be empty.")

    # Pre‑compute the distance from src to every candidate
    src_to_cand = [
        hybrid_distance(src, cand, morphology) for cand in candidates
    ]

    # Pre‑compute the distance from every candidate to dst
    cand_to_dst = [
        hybrid_distance(cand, dst, morphology) for cand in candidates
    ]

    # Sum the two legs and pick the index of the minimal total
    total = np.add(src_to_cand, cand_to_dst)
    min_idx = int(np.argmin(total))
    return min_idx


def hybrid_operation(
    entity1: Entity,
    entity2: Entity,
    entities: List[Entity],
    morphology: Morphology,
) -> Tuple[float, int]:
    """
    Return the blended distance between ``entity1`` and ``entity2`` together
    with the index of the best intermediate node chosen from ``entities``.
    """
    distance = hybrid_distance(entity1, entity2, morphology)
    # Exclude the two endpoints from the routing pool
    routing_pool = [e for e in entities if e.id not in {entity1.id, entity2.id}]
    intermediate = ternary_routing(entity1, entity2, routing_pool, morphology)
    # Translate back to the original index in ``entities`` (or -1 if not found)
    try:
        original_index = entities.index(routing_pool[intermediate])
    except (IndexError, ValueError):
        original_index = -1
    return distance, original_index


if __name__ == "__main__":
    # Example usage ------------------------------------------------------------
    morph = Morphology(1.0, 2.0, 3.0, 4.0)

    e1 = Entity("1", 37.7749, -122.4194, "A", 0.5, "123 Main St, San Francisco, CA")
    e2 = Entity("2", 34.0522, -118.2437, "B", 0.7, "456 Sunset Blvd, Los Angeles, CA")

    pool = [
        Entity("3", 40.7128, -74.0060, "C", 0.3, "789 Broadway, New York, NY"),
        Entity("4", 29.7604, -95.3698, "D", 0.9, "321 Elm St, Houston, TX"),
        Entity("5", 41.8781, -87.6298, "E", 0.4, "654 Lake Shore Dr, Chicago, IL"),
    ]

    dist, inter_idx = hybrid_operation(e1, e2, pool, morph)
    print(f"Blended distance: {dist:.6f}")
    print(f"Chosen intermediate node index in original pool: {inter_idx}")