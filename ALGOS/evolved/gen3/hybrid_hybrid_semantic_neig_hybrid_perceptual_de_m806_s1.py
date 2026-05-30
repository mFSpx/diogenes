# DARWIN HAMMER — match 806, survivor 1
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# parent_b: hybrid_perceptual_dedupe_hybrid_privacy_sketc_m292_s0.py (gen2)
# born: 2026-05-29T23:31:13Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import hashlib

class Morphology:
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("Both inputs must be 1‑D arrays")
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_ph_dhash(values: List[float]) -> int:
    dhash = 0
    for i in range(len(values) - 1):
        dhash = (dhash << 1) | int(values[i] > values[i + 1])
    return dhash


def compute_ph_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    phash = 0
    for v in values[:64]:
        phash = (phash << 1) | int(v >= avg)
    return phash


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def estimate_unique_quasi_identifiers(sketch: List[List[int]], width: int, depth: int) -> int:
    estimates = [sum(1 for cnt in row if cnt > 0) for row in sketch[:depth]]
    return int(np.mean(estimates))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def hybrid_neighbor_score(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    morph_j: Morphology,
    sketch: List[List[int]],
    total_records: int,
    alpha: float = 0.5,
    beta: float = 0.2,
) -> float:
    c = cosine_similarity(vec_i, vec_j)
    p = recovery_priority(morph_j)
    uniq = estimate_unique_quasi_identifiers(sketch, width=len(sketch[0]), depth=len(sketch))
    r = reconstruction_risk_score(uniq, total_records)
    return max(0.0, min(1.0, alpha * c + (1.0 - alpha) * (beta * p + (1.0 - beta) * (1.0 - r))))


def dedupe_similar_records(
    records: List[List[float]],
    threshold: int = 5,
    alpha: float = 0.6,
    beta: float = 0.2,
) -> List[Tuple[int, int, float]]:
    n = len(records)
    phashes = [compute_ph_phash(rec) for rec in records]
    identifiers = [",".join(f"{v:.2f}" for v in rec) for rec in records]
    sketch = count_min_sketch(identifiers)
    results: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(phashes[i], phashes[j]) > threshold:
                continue
            morph_j = Morphology(
                length=random.uniform(0.5, 2.0),
                width=random.uniform(0.5, 2.0),
                height=random.uniform(0.5, 2.0),
                mass=random.uniform(0.1, 5.0),
            )
            vec_i = np.array(records[i])
            vec_j = np.array(records[j])
            score = hybrid_neighbor_score(
                vec_i, vec_j, morph_j, sketch, total_records=n, alpha=alpha, beta=beta
            )
            if score > 0.5:
                results.append((i, j, score))
    return results


def privacy_aware_morphology_analysis(
    morphologies: List[Morphology],
    quasi_identifiers: List[str],
    alpha: float = 0.4,
    beta: float = 0.2,
) -> List[float]:
    sketch = count_min_sketch(quasi_identifiers)
    total_records = len(quasi_identifiers)
    uniq = estimate_unique_quasi_identifiers(sketch, width=len(sketch[0]), depth=len(sketch))
    r = reconstruction_risk_score(uniq, total_records)
    return [
        max(0.0, min(1.0, alpha * recovery_priority(m) + (1.0 - alpha) * (beta + (1.0 - beta) * (1.0 - r))))
        for m in morphologies
    ]