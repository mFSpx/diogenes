# DARWIN HAMMER — match 1327, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s0.py (gen4)
# parent_b: hybrid_voronoi_partition_percyphon_m779_s1.py (gen1)
# born: 2026-05-29T23:35:24Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence, Tuple, List, Dict
import numpy as np

def shannon_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for cnt in counts:
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log2(p)
    return entropy


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def centroid(points: List[Point]) -> Point:
    if not points:
        raise ValueError("cannot compute centroid of empty set")
    xs, ys = zip(*points)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def compute_region_features(
    regions: Dict[int, List[Point]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    counts_map: Dict[int, List[int]] = {}
    signature_map: Dict[int, List[int]] = {}

    for idx, pts in regions.items():
        if not pts:
            counts_map[idx] = [0, 0, 0, 0]
            signature_map[idx] = signature([], k=64)
            continue

        cen = centroid(pts)
        dists = [distance(p, cen) for p in pts]
        max_dist = max(dists) if dists else 1.0
        bin_edges = np.linspace(0, max_dist, 5)  
        hist, _ = np.histogram(dists, bins=bin_edges)
        counts_map[idx] = hist.tolist()

        tokens = [f"{round(p[0],3)}:{round(p[1],3)}" for p in pts]
        signature_map[idx] = signature(tokens, k=64)

    return counts_map, signature_map


def hybrid_region_score(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
    epsilon: float = 1.0,
) -> float:
    global_cen = centroid(seeds)
    counts_map, signature_map = compute_region_features(regions)
    region_cents: Dict[int, Point] = {}
    for idx, pts in regions.items():
        region_cents[idx] = centroid(pts) if pts else seeds[idx]

    intra = 0.0
    for idx, counts in counts_map.items():
        ent = shannon_entropy(counts)
        dist_to_global = distance(region_cents[idx], global_cen)
        intra += ent * gaussian(dist_to_global, epsilon)

    inter = 0.0
    indices = list(regions.keys())
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            sim = similarity(signature_map[idx_i], signature_map[idx_j])
            d = distance(region_cents[idx_i], region_cents[idx_j])
            inter += sim * gaussian(d, epsilon)

    score = intra + inter
    return score


def hybrid_region_score_improved(
    seeds: List[Point],
    regions: Dict[int, List[Point]],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> float:
    global_cen = centroid(seeds)
    counts_map, signature_map = compute_region_features(regions)
    region_cents: Dict[int, Point] = {}
    for idx, pts in regions.items():
        region_cents[idx] = centroid(pts) if pts else seeds[idx]

    intra = 0.0
    for idx, counts in counts_map.items():
        ent = shannon_entropy(counts)
        dist_to_global = distance(region_cents[idx], global_cen)
        intra += ent * gaussian(dist_to_global, epsilon)

    inter = 0.0
    indices = list(regions.keys())
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            sim = similarity(signature_map[idx_i], signature_map[idx_j])
            d = distance(region_cents[idx_i], region_cents[idx_j])
            inter += sim * gaussian(d, epsilon)

    score = alpha * intra + (1 - alpha) * inter
    return score


def main():
    seeds = [(0, 0), (1, 1), (2, 2)]
    points = [(0.1, 0.1), (0.2, 0.2), (1.1, 1.1), (1.2, 1.2), (2.1, 2.1), (2.2, 2.2)]
    regions = assign(points, seeds)
    print(hybrid_region_score(seeds, regions))
    print(hybrid_region_score_improved(seeds, regions))

if __name__ == "__main__":
    main()