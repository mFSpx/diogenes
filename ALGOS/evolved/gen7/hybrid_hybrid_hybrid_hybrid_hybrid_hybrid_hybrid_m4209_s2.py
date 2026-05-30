# DARWIN HAMMER — match 4209, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2232_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_ternar_korpus_text_m1017_s2.py (gen4)
# born: 2026-05-29T23:54:17Z

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import List, Tuple, Dict

import numpy as np

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
    sig = minhash_signature(text, k=k)
    sig_arr = np.array(sig, dtype=np.float64) / float(0xFFFFFFFFFFFFFFFF)
    ent = shannon_entropy(text)
    ent_norm = ent / 8.0  
    return np.concatenate([sig_arr, np.array([ent_norm], dtype=np.float64)])


def build_cost_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("vectors list must not be empty")
    stacked = np.stack(vectors)
    sq_norms = np.sum(stacked ** 2, axis=1, keepdims=True)
    prod = stacked @ stacked.T
    C = sq_norms + sq_norms.T - 2 * prod
    np.maximum(C, 0.0, out=C)  
    return C


def ternary_route(cost_matrix: np.ndarray, source: int, destination: int,
                  pivot_candidates: List[int] | None = None) -> Tuple[int, float]:
    if source == destination:
        return source, 0.0
    if pivot_candidates is None:
        combined = cost_matrix[source, :] + cost_matrix[:, destination]
        k = int(np.argmin(combined))
        total = float(combined[k])
        return k, total
    best_k = source
    best_val = float('inf')
    for cand in pivot_candidates:
        val = float(cost_matrix[source, cand] + cost_matrix[cand, destination])
        if val < best_val:
            best_val = val
            best_k = cand
    return best_k, best_val


def voronoi_partition(points: List[np.ndarray], seed_indices: List[int]) -> Dict[int, List[int]]:
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


def pheromone_to_vector(entry: PheromoneEntry, dim: int = 65, max_signal: float = None) -> np.ndarray:
    signal_norm = entry.signal_value / max_signal if max_signal else entry.signal_value
    decay_norm = entry.decay_factor()
    return np.array([signal_norm, decay_norm] + [0.0] * (dim - 2), dtype=np.float64)


def hybrid_cost_matrix(pheromones: List[PheromoneEntry], texts: List[str], k: int = 64) -> np.ndarray:
    text_vectors = [text_to_vector(text, k=k) for text in texts]
    pheromone_vectors = []
    max_signal = max(entry.signal_value for entry in pheromones)
    for entry in pheromones:
        pheromone_vectors.append(pheromone_to_vector(entry, dim=k + 1, max_signal=max_signal))
    all_vectors = text_vectors + pheromone_vectors
    return build_cost_matrix(all_vectors)


def hybrid_ternary_route(source_idx: int, dest_idx: int, pheromones: List[PheromoneEntry], texts: List[str], k: int = 64) -> Tuple[int, float]:
    cost_matrix = hybrid_cost_matrix(pheromones, texts, k=k)
    return ternary_route(cost_matrix, source_idx, dest_idx)


def main():
    pheromones = [PheromoneEntry("surface1", "kind1", 1.0, 3600), PheromoneEntry("surface2", "kind2", 2.0, 7200)]
    texts = ["text1", "text2"]
    route = hybrid_ternary_route(0, 1, pheromones, texts)
    print(route)


if __name__ == "__main__":
    main()