# DARWIN HAMMER — match 5187, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_doomsd_m2387_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s1.py (gen5)
# born: 2026-05-30T00:00:35Z

import math
import random
import hashlib
import sys
import pathlib
from typing import List, Tuple, Dict, Iterable
import numpy as np
from collections import Counter
from datetime import datetime, timezone

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def voronoi_partition(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def region_sizes(regions: Dict[int, List[Point]]) -> List[int]:
    return [len(v) for v in regions.values()]

def gini_coefficient(sizes: List[int]) -> float:
    if not sizes:
        return 0.0
    arr = np.array(sizes, dtype=float)
    if np.all(arr == 0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def ternary_lens(region_points: List[Point], k: int = 8) -> np.ndarray:
    if not region_points:
        return np.zeros(k, dtype=int)
    raw = ''.join(f'{x:.6f}{y:.6f}' for x, y in region_points).encode()
    h = hashlib.sha256(raw).digest()
    bits = int.from_bytes(h, 'big')
    vals = []
    for i in range(k):
        trit = (bits >> (2 * i)) & 0b11
        if trit == 0:
            vals.append(-1)
        elif trit == 1:
            vals.append(0)
        else:
            vals.append(1)
    return np.array(vals, dtype=int)

def text_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return -np.sum(probs * np.log2(probs))

def master_vector(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    for i, ch in enumerate(text):
        h = hashlib.sha256(f'{ch}{i}'.encode()).digest()
        bits = int.from_bytes(h, 'big')
        for d in range(dim):
            if (bits >> d) & 1:
                vec[d] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class PheromoneEntry:
    def __init__(self, index: int, value: float, half_life: float):
        self.index = index
        self.initial = value
        self.half_life = max(half_life, 1e-6)
        self.age = 0.0

    def decay(self, dt: float = 1.0) -> float:
        self.age += dt
        factor = 0.5 ** (self.age / self.half_life)
        return self.initial * factor

def create_pheromones(text: str) -> List[PheromoneEntry]:
    entropy = text_entropy(text)
    half_life = 1.0 + 9.0 * (entropy / 8.0)
    vec = master_vector(text, dim=64)
    entries = [PheromoneEntry(i, float(abs(v)), half_life) for i, v in enumerate(vec) if v != 0]
    return entries

class HybridSheaf:
    def __init__(self, node_ids: Iterable[int], width: int = 64):
        self.nodes = list(node_ids)
        self.width = width
        self._sections: Dict[int, np.ndarray] = {n: np.zeros(width, dtype=float) for n in self.nodes}
        self._restrictions: Dict[Tuple[int, int], np.ndarray] = {}
        self._pheromones: Dict[int, List[PheromoneEntry]] = {}

    def set_section(self, node: int, vec: np.ndarray):
        if vec.shape != (self.width,):
            raise ValueError("section vector has wrong shape")
        self._sections[node] = vec.astype(float)

    def set_restriction(self, src: int, dst: int, weight: float):
        self._restrictions[(src, dst)] = weight * np.identity(self.width, dtype=float)

    def set_pheromones(self, node: int, pheromones: List[PheromoneEntry]):
        self._pheromones[node] = pheromones

    def propagate(self, steps: int = 3, dt: float = 1.0) -> Dict[int, np.ndarray]:
        for _ in range(steps):
            new_sections = {n: np.copy(v) for n, v in self._sections.items()}
            for (src, dst), mat in self._restrictions.items():
                msg = mat @ self._sections[src]
                new_sections[dst] += msg
            for n, v in new_sections.items():
                new_sections[n] = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
            for n, pheromones in self._pheromones.items():
                decayed_pheromones = [p.decay(dt) for p in pheromones]
                new_sections[n] = np.array(decayed_pheromones)
            self._sections = new_sections
        return self._sections

def main():
    points = [(random.random(), random.random()) for _ in range(100)]
    seeds = [(random.random(), random.random()) for _ in range(5)]
    regions = voronoi_partition(points, seeds)
    sizes = region_sizes(regions)
    gini_weight = gini_coefficient(sizes)
    sheaf = HybridSheaf(range(len(seeds)))
    for i, region in regions.items():
        ternary_signature = ternary_lens(region)
        sheaf.set_section(i, ternary_signature)
        sheaf.set_restriction(i, i, gini_weight)
    text = "example text"
    pheromones = create_pheromones(text)
    sheaf.set_pheromones(0, pheromones)
    propagated = sheaf.propagate()
    print(propagated)

if __name__ == "__main__":
    main()