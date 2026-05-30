# DARWIN HAMMER — match 3266, survivor 3
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_rbf_su_m1815_s1.py (gen5)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s1.py (gen2)
# born: 2026-05-29T23:48:59Z

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = Sequence[float]

def distance(a: Point, b: Point) -> float:
    """Euclidean distance in 2‑D."""
    return np.linalg.norm(np.array(a) - np.array(b))

def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed nearest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return np.argmin([distance(point, seed) for seed in seeds])

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Return a mapping seed_index → list of points belonging to that cell."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean norm between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return np.linalg.norm(np.array(a) - np.array(b))

# ----------------------------------------------------------------------
# RBF surrogate
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis."""
    return math.exp(-((epsilon * r) ** 2))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gauss‑Jordan elimination for a dense square system."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    x = np.linalg.solve(a, b)
    return x.tolist()

@dataclass(frozen=True)
class RBFSurrogate:
    """Thin RBF surrogate predicting a scalar (e.g. memory usage)."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def _phi(self, x: Vector) -> List[float]:
        return [gaussian(euclidean(x, c), self.epsilon) for c in self.centers]

    def predict(self, x: Vector) -> float:
        phi = self._phi(x)
        return sum(w * p for w, p in zip(self.weights, phi))

# ----------------------------------------------------------------------
# Simple VRAM planner
# ----------------------------------------------------------------------
@dataclass
class VRAMArtifact:
    name: str
    size_mb: float
    priority: float  # higher = more important

@dataclass
class VRAMPlanner:
    """Keeps registered artifacts inside a memory budget."""
    budget_mb: float
    reserve_mb: float = 0.0
    _registry: List[VRAMArtifact] = field(default_factory=list)

    @property
    def used_mb(self) -> float:
        return sum(a.size_mb for a in self._registry)

    @property
    def free_mb(self) -> float:
        return max(0.0, self.budget_mb - self.used_mb - self.reserve_mb)

    def register(self, artifact: VRAMArtifact) -> None:
        """Attempt to register *artifact*; evict low‑priority items if needed."""
        self._registry.append(artifact)
        self._registry.sort(key=lambda a: (-a.priority, a.size_mb))
        while self.free_mb < 0:
            self._registry.pop(0)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [{'name': a.name, 'size_mb': a.size_mb, 'priority': a.priority} for a in self._registry]

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature
# ----------------------------------------------------------------------
def _neighbor_distribution(G: Dict[int, List[int]], node: int) -> Dict[int, float]:
    """Uniform probability over node itself + its neighbors."""
    neigh = [node] + G.get(node, [])
    prob = 1.0 / len(neigh)
    return {n: prob for n in neigh}

def _wasserstein_distance(p: Dict[int, float], q: Dict[int, float], G: Dict[int, List[int]]) -> float:
    """
    Approximate 1‑Wasserstein distance for two discrete distributions on the
    same node set using shortest‑path distances in *G*.
    """
    nodes = set(p) | set(q)
    dist = {}
    for src in nodes:
        dist[src] = {}
        queue = [(src, 0)]
        visited = set()
        while queue:
            cur, d = queue.pop(0)
            if cur not in visited:
                visited.add(cur)
                dist[src][cur] = d
                for nb in G.get(cur, []):
                    if nb not in visited:
                        queue.append((nb, d + 1))
    total = 0.0
    for i, pi in p.items():
        for j, qj in q.items():
            total += pi * qj * dist[i][j]
    return total

def ricci_curvature_edge(G: Dict[int, List[int]], u: int, v: int) -> float:
    """
    Ollivier‑Ricci curvature κ(u,v) = 1 - W1(m_u, m_v) / d(u,v)
    Here d(u,v)=1 for adjacent nodes.
    """
    pu = _neighbor_distribution(G, u)
    pv = _neighbor_distribution(G, v)
    w1 = _wasserstein_distance(pu, pv, G)
    return 1.0 - w1  # d(u,v)=1

def compute_region_curvature_stats(G: Dict[int, List[int]]) -> Tuple[float, float]:
    """
    Return (mean_curvature, variance) over all edges of *G*.
    Empty graph yields (0.0, 0.0).
    """
    curvs = []
    for u, nbrs in G.items():
        for v in nbrs:
            if u < v:
                curvs.append(ricci_curvature_edge(G, u, v))
    if curvs:
        mean = sum(curvs) / len(curvs)
        variance = sum((c - mean) ** 2 for c in curvs) / len(curvs)
    else:
        mean, variance = 0.0, 0.0
    return mean, variance

def main():
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    seeds = [(0, 0), (4, 4)]
    regions = assign(points, seeds)

    G = {}
    for i, region in regions.items():
        G[i] = [j for j, _ in regions.items() if i != j]

    mean_curvature, variance = compute_region_curvature_stats(G)

    centers = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    rbf = RBFSurrogate(centers, weights)

    vram_planner = VRAMPlanner(budget_mb=100)
    for i, region in regions.items():
        artifact = VRAMArtifact(f'Artifact {i}', size_mb=10, priority=i)
        vram_planner.register(artifact)

if __name__ == '__main__':
    main()