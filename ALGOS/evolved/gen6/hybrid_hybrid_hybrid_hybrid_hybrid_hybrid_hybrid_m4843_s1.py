# DARWIN HAMMER — match 4843, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m614_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_caputo_hybrid_hybrid_sketch_m1405_s0.py (gen5)
# born: 2026-05-29T23:58:27Z

import math
import random
import hashlib
from collections import defaultdict
from typing import List, Tuple, Dict, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometry utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Return the index of the closest seed (break ties by smallest index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(
        range(len(seeds)),
        key=lambda i: (euclidean_distance(point, seeds[i]), i),
    )


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi‑like assignment of points to the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


# ----------------------------------------------------------------------
# Exterior algebra utilities (multivector wedge product)
# ----------------------------------------------------------------------
def _canonical_order(indices: List[int]) -> Tuple[FrozenSet[int], int]:
    """
    Return a sorted tuple of distinct indices together with the sign
    obtained by sorting via swaps (the usual rule for the wedge product).
    If an index appears twice the wedge product is zero.
    """
    sign = 1
    # bubble‑sort while counting swaps
    n = len(indices)
    for i in range(n):
        for j in range(n - 1 - i):
            if indices[j] > indices[j + 1]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
                sign *= -1
    # check for repeated indices (zero blade)
    if len(set(indices)) != len(indices):
        return frozenset(), 0
    return frozenset(indices), sign


def wedge(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Wedge product of two blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _canonical_order(combined)
    return result, sign


class Multivector:
    """Simple multivector supporting addition and wedge product."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # discard zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k}, self.n
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = defaultdict(float, self.components)
        for b, c in other.components.items():
            result[b] += c
        return Multivector(dict(result), self.n)

    def __xor__(self, other: "Multivector") -> "Multivector":
        """Wedge (exterior) product."""
        result: Dict[FrozenSet[int], float] = defaultdict(float)
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                blade, sign = wedge(b1, b2)
                if sign != 0:
                    result[blade] += sign * c1 * c2
        return Multivector(dict(result), self.n)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


# ----------------------------------------------------------------------
# Special functions
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
)


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation for the Gamma function (valid for Re(z) > 0)."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


# ----------------------------------------------------------------------
# Fractional calculus
# ----------------------------------------------------------------------
def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Discrete Caputo derivative of order ``alpha`` (0 < alpha < 1)
    using the Grünwald‑Letnikov approximation with the trapezoidal rule.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1) for the implemented scheme")
    if len(t) != len(f):
        raise ValueError("t and f must have the same length")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("uniform time step required for the simple implementation")
    h = dt[0]

    # coefficients for the Grünwald‑Letnikov series
    coeffs = np.empty(len(f))
    coeffs[0] = 0.0
    for k in range(1, len(f)):
        coeffs[k] = ((k - 1 - alpha) / k) * coeffs[k - 1] + (1 / k) * ((-1) ** k) * math.comb(k, int(alpha))
    # Convolution with the discrete derivative of f
    df = np.diff(f, prepend=f[0])
    derivative = (1 / gamma_lanczos(1 - alpha)) * np.convolve(df, coeffs, mode="full")[: len(f)] / (h ** alpha)
    return derivative


# ----------------------------------------------------------------------
# Probabilistic sketching
# ----------------------------------------------------------------------
def _stable_hash(item: str, width: int) -> int:
    """Deterministic hash based on SHA‑256, mapped to [0,width)."""
    h = hashlib.sha256(item.encode("utf-8")).digest()
    # use first 8 bytes as unsigned integer
    val = int.from_bytes(h[:8], "big")
    return val % width


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """
    Classic Count‑Min sketch with deterministic hashing.
    Returns a ``depth`` × ``width`` matrix of counters.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _stable_hash(f"{d}:{item}", width)
            table[d][idx] += 1
    return table


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (simple discrete approximation)
# ----------------------------------------------------------------------
def _probability_distribution(neighbors: List[int], n: int) -> np.ndarray:
    """Uniform probability on the neighbor set (including self‑loop)."""
    prob = np.zeros(n)
    if not neighbors:
        return prob
    mass = 1.0 / len(neighbors)
    for v in neighbors:
        prob[v] = mass
    return prob


def _wasserstein_distance(p: np.ndarray, q: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute the 1‑Wasserstein distance between two discrete distributions
    using the linear‑programming formulation solved by the
    simple transport plan: mass moves along the shortest edge.
    For the small graphs typical in this demo, a greedy algorithm suffices.
    """
    # cumulative distribution along a shortest‑path ordering
    # Here we use the earth mover's distance on a line induced by the graph
    # (i.e., we treat the distance matrix as a metric on the node set).
    # This is not exact for arbitrary graphs but provides a reasonable
    # approximation while keeping the implementation lightweight.
    diff = p - q
    cumulative = 0.0
    cost = 0.0
    for i in range(len(diff)):
        cumulative += diff[i]
        cost += abs(cumulative) * np.min(distance_matrix[i])
    return cost


def ollivier_ricci_curvature(graph: Dict[int, List[int]]) -> float:
    """
    Approximate Ollivier‑Ricci curvature for an undirected unweighted graph.
    The curvature κ(x,y) for an edge (x,y) is defined as
        κ = 1 - W(m_x, m_y) / d(x,y)
    where m_x,m_y are uniform neighbor measures and d is the graph distance.
    The function returns the average curvature over all edges.
    """
    nodes = list(graph.keys())
    n = len(nodes)
    # build shortest‑path distance matrix via Floyd‑Warshall (small graphs)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        dist[i, i] = 0
    for u, neigh in graph.items():
        for v in neigh:
            dist[u, v] = 1
            dist[v, u] = 1  # ensure undirected

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    total_curvature = 0.0
    edge_count = 0
    for u, neigh in graph.items():
        for v in neigh:
            if u >= v:  # count each undirected edge once
                continue
            mu_u = _probability_distribution([u] + neigh, n)
            mu_v = _probability_distribution([v] + graph[v], n)
            w_dist = _wasserstein_distance(mu_u, mu_v, dist)
            if dist[u, v] == 0:
                continue
            curvature = 1.0 - w_dist / dist[u, v]
            total_curvature += curvature
            edge_count += 1

    return total_curvature / edge_count if edge_count else 0.0


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    points: List[Point], seeds: List[Point], items: List[str]
) -> Tuple[float, np.ndarray]:
    """
    Combine spatial clustering (via Voronoi assignment) with a
    Count‑Min sketch and a fractional‑order Caputo derivative.
    Returns:
        - average Ollivier‑Ricci curvature of the induced region graph,
        - Caputo derivative of the sketch’s total count time‑series.
    """
    # 1. Spatial partitioning
    regions = assign(points, seeds)

    # 2. Build region adjacency graph (undirected)
    graph: Dict[int, List[int]] = defaultdict(list)
    for region, pts in regions.items():
        for p in pts:
            neighbor_region = nearest(p, seeds)
            if neighbor_region != region and neighbor_region not in graph[region]:
                graph[region].append(neighbor_region)
                graph[neighbor_region].append(region)

    # 3. Curvature
    curvature = ollivier_ricci_curvature(graph)

    # 4. Sketch and its dynamics
    sketch = count_min_sketch(items, width=128, depth=6)
    # total count per time step (here each row is a “time slice”)
    f = np.array([sum(row) for row in sketch], dtype=float)
    t = np.arange(len(f), dtype=float)

    # 5. Fractional derivative
    derivative = caputo_derivative(f, t, alpha=0.5)

    return curvature, derivative


# ----------------------------------------------------------------------
# Demo entry point
# ----------------------------------------------------------------------
def main() -> None:
    random.seed(42)
    points = [(random.random(), random.random()) for _ in range(200)]
    seeds = [(random.random(), random.random()) for _ in range(15)]
    items = [f"item_{i}" for i in range(500)]

    curvature, derivative = hybrid_operation(points, seeds, items)

    print("Average Ollivier‑Ricci curvature:", curvature)
    print("Caputo derivative (order 0.5):", derivative)


if __name__ == "__main__":
    main()