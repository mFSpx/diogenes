# DARWIN HAMMER — match 5173, survivor 2
# gen: 4
# parent_a: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s3.py (gen2)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_temporal_moti_m47_s3.py (gen3)
# born: 2026-05-30T00:00:22Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List


# ---------- Data structures -------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int


@dataclass(frozen=True)
class HybridMotif:
    """Entity representing a spatio‑temporal motif with morphology."""
    pattern: Tuple[str, ...]
    support: int
    centroid_lat: float
    centroid_lon: float
    morphology: Morphology
    vector: Tuple[float, ...]          # semantic feature vector
    score: float = 0.0                 # unified hybrid score (filled after evaluation)


# ---------- Entropy utilities ------------------------------------------------

def shannon_entropy(sequence: bytes) -> float:
    """Shannon entropy (bits) for a byte sequence."""
    if not sequence:
        return 0.0
    freq = np.bincount(np.frombuffer(sequence, dtype=np.uint8), minlength=256)
    prob = freq[freq > 0] / len(sequence)
    return -np.sum(prob * np.log2(prob))


def byte_entropy_normalized(data: bytes, sample: int = 8192) -> float:
    """Normalized entropy in [0, 1] (max 8 bits for a byte)."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(chunk) / 8.0


# ---------- Parent A – morphology & semantic utilities -----------------------

def flatness_index(length: float, width: float, height: float) -> float:
    """A dimension‑based flatness measure in (0, 1]."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (min(length, width, height) ** 2) / (length * width * height)


def cosine_similarity(v1: Tuple[float, ...], v2: Tuple[float, ...]) -> float:
    """Cosine similarity handling zero‑vectors gracefully."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------- Parent B – temporal motif utilities -----------------------------

def temporal_support(motif: TemporalMotif) -> float:
    """Monotonically increasing support mapping into (0, 1)."""
    return motif.support / (1 + motif.support)


def z_score(value: float, mean: float, std: float) -> float:
    """Standard z‑score; returns 0 when variance is zero."""
    if std == 0:
        return 0.0
    return (value - mean) / std


# ---------- Hybrid utilities --------------------------------------------------

def average_cosine_to_neighbors(
    motif_vec: Tuple[float, ...],
    neighbor_vecs: Iterable[Tuple[float, ...]]
) -> float:
    """Mean cosine similarity between a motif vector and its semantic neighbours."""
    neighbor_vecs = list(neighbor_vecs)
    if not neighbor_vecs:
        # No neighbours – treat as perfect self‑similarity to avoid penalising.
        return 1.0
    sims = [cosine_similarity(motif_vec, n) for n in neighbor_vecs]
    return float(np.mean(sims))


def hybrid_score(
    motif: HybridMotif,
    data: bytes,
    mean_support: float,
    std_support: float,
    neighbor_vectors: Iterable[Tuple[float, ...]] = ()
) -> float:
    """
    Compute the unified hybrid score.

    S = s(p) · (1 + z_s) · R(m) · σ · (1 + E)

    where:
    - s(p)            : temporal support (0,1)
    - z_s             : z‑score of support
    - R(m)            : morphology‑based recovery priority,
                        now also modulated by entropy
    - σ               : average cosine similarity to semantic neighbours
    - E               : normalized Shannon entropy (0,1)
    """
    # 1. Temporal component
    s_p = temporal_support(TemporalMotif(motif.pattern, motif.support))

    # 2. Support z‑score
    z_s = z_score(motif.support, mean_support, std_support)

    # 3. Morphology + entropy coupling (deeper integration)
    flat_idx = flatness_index(
        motif.morphology.length,
        motif.morphology.width,
        motif.morphology.height,
    )
    # Base recovery priority
    base_R = 1.0 / (1.0 + flat_idx)          # in (0.5, 1]
    # Entropy‑aware modulation
    E = byte_entropy_normalized(data)      # in [0,1]
    R_m = base_R * (1.0 + E)                # now ranges roughly (0.5, 2]

    # 4. Semantic similarity to neighbours
    sigma = average_cosine_to_neighbors(motif.vector, neighbor_vectors)

    # 5. Final composition
    return s_p * (1.0 + z_s) * R_m * sigma * (1.0 + E)


# ---------- Demonstration ----------------------------------------------------

def demonstrate_hybrid_operation() -> None:
    # Sample motif
    motif = HybridMotif(
        pattern=("A", "B", "C"),
        support=12,
        centroid_lat=37.7749,
        centroid_lon=-122.4194,
        morphology=Morphology(length=1.2, width=2.5, height=3.0, mass=4.0),
        vector=(0.15, 0.22, 0.33),
    )

    # Example byte payload
    data = b"The quick brown fox jumps over the lazy dog."

    # Synthetic semantic neighbours (could come from a graph or embedding lookup)
    neighbours: List[Tuple[float, ...]] = [
        (0.10, 0.20, 0.30),
        (0.12, 0.18, 0.35),
        (0.14, 0.25, 0.31),
    ]

    # Global support statistics (normally derived from the full dataset)
    mean_support = 8.0
    std_support = 3.0

    score = hybrid_score(
        motif,
        data,
        mean_support,
        std_support,
        neighbor_vectors=neighbours,
    )
    print(f"Hybrid score: {score:.6f}")


if __name__ == "__main__":
    demonstrate_hybrid_operation()