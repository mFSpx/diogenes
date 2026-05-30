# DARWIN HAMMER — match 4209, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2232_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s2.py (gen4)
# born: 2026-05-29T23:54:17Z

"""Hybrid Pheromone‑Text Routing

This module fuses two parent algorithms:

* **Parent A** – a pheromone store with exponential decay (half‑life) for each entry.
* **Parent B** – text fingerprinting (min‑hash + Shannon entropy) producing fixed‑size vectors,
  a Euclidean cost matrix, ternary routing, and Voronoi partitioning.

**Mathematical bridge**

Both parents ultimately operate on Euclidean distances between fixed‑size vectors.
Parent B builds a cost matrix `C_ij = ||v_i – v_j||²`.  
Parent A provides a scalar `signal_value` that decays exponentially:
`signal(t) = signal₀·2^{‑Δt/τ}`.

The fusion maps each `PheromoneEntry` to a vector in the same space as a text
vector:


p_vec = [signal_norm, decay_norm, 0, …, 0]   # length = K+1 (K = 64)


`signal_norm` is the current signal value scaled to [0,1] by the maximal
observed value; `decay_norm` is the decay factor (0‑1).  The remaining
components are zero‑padded so that Euclidean distance can be computed
directly against text vectors.  Consequently a single cost matrix can be
built over the union of pheromone‑vectors and text‑vectors, enabling ternary
routing that may select a pheromone entry as an intermediate “bridge” between
two texts.

The module provides three high‑level hybrid functions:

1. `pheromone_to_vector(entry, dim)` – converts a pheromone entry to a vector.
2. `hybrid_cost_matrix(pheromones, texts, k)` – builds a unified Euclidean cost
   matrix for all objects.
3. `hybrid_ternary_route(source_idx, dest_idx, cost_matrix, pivot_range)` –
   performs ternary routing, optionally restricting the intermediate pivot to
   pheromone indices.

A Voronoi partition over the combined vectors is also exposed for clustering.

The `__main__` block demonstrates a smoke test without external files.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone infrastructure
# ----------------------------------------------------------------------
MAX_COMPONENT_TOKENS = 500


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Fraction remaining after exponential half‑life decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return np.power(0.5, self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    _entries: dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_all(cls) -> List[PheromoneEntry]:
        return list(cls._entries.values())

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]


# ----------------------------------------------------------------------
# Parent B – Text fingerprinting utilities
# ----------------------------------------------------------------------
def _shingles(text: str, width: int = 5) -> List[str]:
    return [text[i: i + width] for i in range(len(text) - width + 1)]


def minhash_signature(text: str, k: int = 64, width: int = 5) -> List[int]:
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    return (hashes[:k] + [0] * k)[:k]


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = text[:10000]
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in freq.values())


def text_to_vector(text: str, k: int = 64) -> np.ndarray:
    """Returns a (k+1)-dimensional vector: normalized min‑hash + normalized entropy."""
    sig = minhash_signature(text, k=k)
    sig_arr = np.array(sig, dtype=np.float64) / float(0xFFFFFFFFFFFFFFFF)
    ent = shannon_entropy(text)
    ent_norm = ent / 8.0  # entropy of 8‑bit alphabet maxes at 8 bits
    return np.concatenate([sig_arr, np.array([ent_norm], dtype=np.float64)])


def build_cost_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    """Euclidean squared distance matrix."""
    if not vectors:
        raise ValueError("vectors list must not be empty")
    stacked = np.stack(vectors)
    sq_norms = np.sum(stacked ** 2, axis=1, keepdims=True)
    prod = stacked @ stacked.T
    C = sq_norms + sq_norms.T - 2 * prod
    np.maximum(C, 0.0, out=C)  # numerical safety
    return C


def ternary_route(cost_matrix: np.ndarray, source: int, destination: int,
                  pivot_candidates: List[int] | None = None) -> Tuple[int, float]:
    """
    Finds an intermediate pivot k that minimises
        C[source, k] + C[k, destination]

    If `pivot_candidates` is given, only those indices are considered.
    Returns (pivot_index, total_cost).  When source == destination,
    the pivot is the source itself with zero cost.
    """
    if source == destination:
        return source, 0.0
    if pivot_candidates is None:
        combined = cost_matrix[source, :] + cost_matrix[:, destination]
        k = int(np.argmin(combined))
        total = float(combined[k])
        return k, total
    # limited search
    best_k = source
    best_val = float('inf')
    for cand in pivot_candidates:
        val = float(cost_matrix[source, cand] + cost_matrix[cand, destination])
        if val < best_val:
            best_val = val
            best_k = cand
    return best_k, best_val


def voronoi_partition(points: List[np.ndarray], seed_indices: List[int]) -> Dict[int, List[int]]:
    """Assign each point to the nearest seed (by Euclidean distance)."""
    if not seed_indices:
        raise ValueError("seed_indices must contain at least one index")
    seeds = [points[i] for i in seed_indices]
    assignments: Dict[int, List[int]] = {s_idx: [] for s_idx in seed_indices}
    for idx, pt in enumerate(points):
        dists = [np.linalg.norm(pt - seed) for seed in seeds]
        nearest = int(np.argmin(dists))
        nearest_idx = seed_indices[nearest]
        assignments[nearest_idx].append(idx)
    return assignments


# ----------------------------------------------------------------------
# Hybrid layer – bridging pheromones and text vectors
# ----------------------------------------------------------------------
def pheromone_to_vector(entry: PheromoneEntry, dim: int = 65) -> np.ndarray:
    """
    Map a pheromone entry to a vector compatible with the text vectors.

    The first component encodes the *current* signal value (scaled to [0,1]
    relative to the maximum signal among all entries).  The second component
    encodes the decay factor (0‑1).  Remaining components are zero‑padded to
    reach `dim` (default 65 = 64 min‑hash slots + 1 entropy slot).
    """
    # Normalisation will be performed later when the full list is known.
    # Here we return raw values; the caller will scale.
    return np.array([entry.signal_value, entry.decay_factor()], dtype=np.float64)


def hybrid_cost_matrix(pheromones: List[PheromoneEntry],
                       texts: List[str],
                       k: int = 64) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build a unified cost matrix over pheromone vectors and text vectors.

    Returns:
        cost_matrix – (N+M)×(N+M) Euclidean squared distance matrix
        pher_idx_range – list of indices that correspond to pheromones
        text_idx_range – list of indices that correspond to texts
    """
    # 1. Convert texts to vectors (dim = k+1)
    text_vecs = [text_to_vector(t, k=k) for t in texts]

    # 2. Convert pheromones to raw vectors (2‑dim) then pad/scale
    raw_ph_vecs = [pheromone_to_vector(p, dim=k + 1) for p in pheromones]

    # Normalise pheromone signal component across the set
    if pheromones:
        max_signal = max(p.signal_value for p in pheromones)
        max_signal = max_signal if max_signal > 0 else 1.0
    else:
        max_signal = 1.0

    padded_ph_vecs = []
    for raw in raw_ph_vecs:
        # Scale signal to [0,1] using max_signal, decay already 0‑1
        scaled = np.zeros(k + 1, dtype=np.float64)
        scaled[0] = raw[0] / max_signal
        scaled[1] = raw[1]  # decay factor
        # rest remain zero
        padded_ph_vecs.append(scaled)

    # 3. Stack all vectors
    all_vectors = padded_ph_vecs + text_vecs

    # 4. Build Euclidean cost matrix
    cost = build_cost_matrix(all_vectors)

    pher_idx_range = list(range(len(padded_ph_vecs)))
    text_idx_range = list(range(len(padded_ph_vecs), len(all_vectors)))
    return cost, pher_idx_range, text_idx_range


def hybrid_ternary_route(source_text_idx: int,
                         dest_text_idx: int,
                         cost_matrix: np.ndarray,
                         pher_idx_range: List[int],
                         text_idx_range: List[int]) -> Tuple[int, float]:
    """
    Perform ternary routing between two *text* nodes, forcing the intermediate
    pivot to be a pheromone entry (if any exist).  Returns the chosen pheromone
    index (global index) and the total cost.

    If no pheromones are present, falls back to unrestricted ternary routing.
    """
    # Translate local text indices to global indices inside the combined matrix
    src_global = text_idx_range[source_text_idx]
    dst_global = text_idx_range[dest_text_idx]

    if not pher_idx_range:
        # No pheromones → unrestricted
        pivot, total = ternary_route(cost_matrix, src_global, dst_global)
        return pivot, total

    pivot, total = ternary_route(cost_matrix, src_global, dst_global, pivot_candidates=pher_idx_range)
    return pivot, total


def hybrid_voronoi_clusters(pheromones: List[PheromoneEntry],
                            texts: List[str],
                            k: int = 64,
                            seed_pheromone_count: int = 2) -> Dict[int, List[int]]:
    """
    Run a Voronoi partition on the unified vector set using the first
    `seed_pheromone_count` pheromone entries as seeds.  Returns a mapping from
    seed global index to the list of assigned point indices.
    """
    cost, pher_idx_range, text_idx_range = hybrid_cost_matrix(pheromones, texts, k=k)
    # Voronoi works on raw points, not on the cost matrix.
    # Re‑create the point list from the previous function.
    # (We repeat the vector creation to keep the function self‑contained.)
    text_vecs = [text_to_vector(t, k=k) for t in texts]

    raw_ph_vecs = [pheromone_to_vector(p, dim=k + 1) for p in pheromones]
    max_signal = max((p.signal_value for p in pheromones), default=1.0)
    max_signal = max_signal if max_signal > 0 else 1.0
    padded_ph_vecs = []
    for raw in raw_ph_vecs:
        scaled = np.zeros(k + 1, dtype=np.float64)
        scaled[0] = raw[0] / max_signal
        scaled[1] = raw[1]
        padded_ph_vecs.append(scaled)

    all_points = padded_ph_vecs + text_vecs
    seed_indices = pher_idx_range[:seed_pheromone_count] or [0]  # fallback to first point
    return voronoi_partition(all_points, seed_indices)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few pheromone entries with varying half‑life
    p1 = PheromoneEntry("surfaceA", "typeX", signal_value=10.0, half_life_seconds=30)
    p2 = PheromoneEntry("surfaceB", "typeY", signal_value=5.0, half_life_seconds=60)
    # Simulate some decay
    p1.apply_decay()
    p2.apply_decay()
    PheromoneStore.add(p1)
    PheromoneStore.add(p2)

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs."
    ]

    # Build unified cost matrix
    cost_mat, pher_idxs, text_idxs = hybrid_cost_matrix(PheromoneStore.get_all(), texts)

    # Route between first and third text using pheromone bridge
    pivot, total_cost = hybrid_ternary_route(
        source_text_idx=0,
        dest_text_idx=2,
        cost_matrix=cost_mat,
        pher_idx_range=pher_idxs,
        text_idx_range=text_idxs
    )
    print(f"Chosen pivot index: {pivot} (global), total cost: {total_cost:.4f}")
    if pivot in pher_idxs:
        print("Pivot is a pheromone entry.")
    else:
        print("Pivot is a direct text-to-text shortcut.")

    # Perform a Voronoi clustering using pheromones as seeds
    clusters = hybrid_voronoi_clusters(PheromoneStore.get_all(), texts, seed_pheromone_count=2)
    for seed, members in clusters.items():
        print(f"Seed {seed} owns points {members}")
    sys.exit(0)