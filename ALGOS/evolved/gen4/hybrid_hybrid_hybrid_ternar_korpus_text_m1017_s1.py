# DARWIN HAMMER — match 1017, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s1.py (gen3)
# parent_b: korpus_text.py (gen0)
# born: 2026-05-29T23:32:27Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

def _shingles(text: str, width: int = 5) -> List[str]:
    return [text[i : i + width] for i in range(len(text) - width + 1)]

def minhash_signature(text: str, k: int = 64, width: int = 5, seed: int = 42) -> List[int]:
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    hashes = [(hash(s) + seed) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    return (hashes[:k] + [0] * k)[:k]

def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = text[:10000]
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return entropy

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

def ternary_route(cost_matrix: np.ndarray, source: int, destination: int) -> Tuple[int, float]:
    if source == destination:
        return source, 0.0
    combined = cost_matrix[source, :] + cost_matrix[:, destination]
    k = int(np.argmin(combined))
    if k == destination:
        k = np.argmin(combined[combined != combined[k]])
    total = float(combined[k])
    return k, total

def voronoi_partition(points: List[np.ndarray], seed_indices: List[int]) -> Dict[int, List[int]]:
    if not seed_indices:
        raise ValueError("seed_indices must contain at least one index")
    seeds = [points[i] for i in seed_indices]
    assignments: Dict[int, List[int]] = {s_idx: [] for s_idx in seed_indices}
    for idx, pt in enumerate(points):
        dists = [np.linalg.norm(pt - seed) for seed in seeds]
        nearest_seed_pos = int(np.argmin(dists))
        nearest_seed_idx = seed_indices[nearest_seed_pos]
        assignments[nearest_seed_idx].append(idx)
    return assignments

def hybrid_process(
    texts: List[str],
    source_idx: int,
    dest_idx: int,
    seed_indices: List[int],
    k_minhash: int = 64,
) -> Dict[str, any]:
    if not (0 <= source_idx < len(texts)) or not (0 <= dest_idx < len(texts)):
        raise IndexError("source_idx and dest_idx must be valid indices in texts")
    vectors = [text_to_vector(t, k=k_minhash) for t in texts]
    cost_mat = build_cost_matrix(vectors)
    interm, route_cost = ternary_route(cost_mat, source_idx, dest_idx)
    voronoi = voronoi_partition(vectors, seed_indices)

    return {
        "vectors": vectors,
        "cost_matrix": cost_mat,
        "route": {
            "source": source_idx,
            "destination": dest_idx,
            "intermediate": interm,
            "total_cost": route_cost,
        },
        "voronoi": voronoi,
    }

if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
    ]

    src = 0
    dst = 3
    seeds = [1, 2, 4]

    result = hybrid_process(sample_texts, src, dst, seeds)

    print("=== Hybrid Process Result ===")
    print(f"Route: source={src}, destination={dst}, intermediate={result['route']['intermediate']}, cost={result['route']['total_cost']:.4f}")
    print("Voronoi assignment:")
    for seed, members in result["voronoi"].items():
        print(f"  Seed {seed} -> points {members}")
    assert result["cost_matrix"].shape == (len(sample_texts), len(sample_texts))
    print("Smoke test completed successfully.")