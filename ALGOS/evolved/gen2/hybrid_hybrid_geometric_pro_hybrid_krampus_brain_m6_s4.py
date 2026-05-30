# DARWIN HAMMER — match 6, survivor 4
# gen: 2
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s0.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s2.py (gen1)
# born: 2026-05-29T23:22:39Z

import math
import itertools
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, FrozenSet, Iterable

Point = Tuple[float, ...]  # n‑dimensional point


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


def voronoi_partition(seeds: List[Point], points: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the Voronoi cell of the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def _canonical_blade(indices: Iterable[int]) -> Tuple[FrozenSet[int], int]:
    """
    Return a canonical (sorted, duplicate‑free) blade together with the sign
    incurred by re‑ordering the basis vectors.

    In Euclidean Clifford algebra e_i^2 = 1, therefore duplicate indices cancel
    pairwise without affecting the sign.
    """
    # Count occurrences of each index
    cnt = Counter(indices)
    # Remove pairs (e_i e_i = 1)
    for i in list(cnt):
        if cnt[i] % 2 == 0:
            del cnt[i]
        else:
            cnt[i] = 1  # keep a single occurrence

    # The remaining indices are all distinct
    distinct = sorted(cnt.elements())
    # Compute sign of the permutation that brings the original order to sorted order
    # Using a simple bubble‑sort count of swaps
    sign = 1
    original = list(indices)
    # Remove cancelled pairs from original as well
    original = [i for i in original if cnt.get(i, 0) == 1]
    # Bubble sort to sorted list
    for i in range(len(original)):
        for j in range(len(original) - 1 - i):
            if original[j] > original[j + 1]:
                original[j], original[j + 1] = original[j + 1], original[j]
                sign = -sign
    return frozenset(distinct), sign


def _multiply_blades(a: FrozenSet[int], b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades."""
    combined = list(a) + list(b)
    return _canonical_blade(combined)


class Multivector:
    """
    Simple multivector implementation for an n‑dimensional Euclidean
    Clifford algebra (metric +1 on every basis vector).
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # store only non‑zero components
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # --------------------------------------------------------------------- #
    # Basic algebraic operations
    # --------------------------------------------------------------------- #
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result, max(self.n, other.n))

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = {blade: -coeff for blade, coeff in other.components.items()}
        return self + Multivector(neg, other.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        return geometric_product(self, other)

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {blade: coeff for blade, coeff in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar(self) -> float:
        """Return the scalar (grade‑0) part."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coeff:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product of two multivectors."""
    result: Dict[FrozenSet[int], float] = defaultdict(float)
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            blade_out, sign = _multiply_blades(blade_a, blade_b)
            result[blade_out] += sign * coeff_a * coeff_b
    return Multivector(dict(result), max(a.n, b.n))


# -------------------------------------------------------------------------
# Region → multivector conversion
# -------------------------------------------------------------------------
def point_to_blade(point: Point) -> FrozenSet[int]:
    """
    Encode a point as a blade: each non‑zero coordinate contributes its axis index.
    Zero coordinates are ignored, which yields a sparse representation suitable for
    high‑dimensional data.
    """
    return frozenset(i for i, x in enumerate(point) if abs(x) > 1e-12)


def region_multivector(region: List[Point], n: int) -> Multivector:
    """
    Build a multivector for a Voronoi region by summing unit blades
    corresponding to its points.
    """
    components: Dict[FrozenSet[int], float] = {}
    for p in region:
        blade = point_to_blade(p)
        components[blade] = components.get(blade, 0.0) + 1.0
    return Multivector(components, n)


# -------------------------------------------------------------------------
# Graph construction for curvature analysis
# -------------------------------------------------------------------------
def region_centroid(region: List[Point]) -> Point:
    """Arithmetic mean of the points in *region*."""
    if not region:
        raise ValueError("cannot compute centroid of empty region")
    dim = len(region[0])
    sums = [0.0] * dim
    for p in region:
        for i, x in enumerate(p):
            sums[i] += x
    return tuple(s / len(region) for s in sums)


def neighbor_graph(
    seeds: List[Point], points: List[Point], radius_factor: float = 1.5
) -> Dict[int, List[int]]:
    """
    Build an undirected adjacency list of Voronoi cells.
    Two cells are adjacent if the distance between their centroids is
    less than *radius_factor* times the average nearest‑seed distance.
    """
    regions = voronoi_partition(seeds, points)
    centroids = {i: region_centroid(reg) for i, reg in regions.items()}

    # average nearest‑seed distance (used as a scale)
    dists = [
        euclidean_distance(p, seeds[nearest_point(p, seeds)]) for p in points
    ]
    scale = (sum(dists) / len(dists)) * radius_factor

    adjacency: Dict[int, List[int]] = {i: [] for i in regions}
    for i, j in itertools.combinations(regions.keys(), 2):
        if euclidean_distance(centroids[i], centroids[j]) <= scale:
            adjacency[i].append(j)
            adjacency[j].append(i)
    return adjacency


# -------------------------------------------------------------------------
# Ollivier‑Ricci curvature on the region graph
# -------------------------------------------------------------------------
def probability_measure(region: List[Point]) -> Dict[int, float]:
    """
    Uniform probability over the indices of points in *region*.
    The measure is returned as a dict mapping point index (relative to the whole
    dataset) to probability mass.
    """
    if not region:
        return {}
    mass = 1.0 / len(region)
    return {i: mass for i in range(len(region))}


def earth_movers_distance(
    mu: Dict[int, float], nu: Dict[int, float], cost: Dict[Tuple[int, int], float]
) -> float:
    """
    Simple linear‑program‑free approximation of the 1‑Wasserstein distance
    for discrete, equally‑sized supports. It solves the optimal transport
    problem by greedy matching because the cost matrix is metric.
    """
    # Convert to sorted lists for deterministic behaviour
    mu_items = sorted(mu.items())
    nu_items = sorted(nu.items())
    i = j = 0
    dist = 0.0
    while i < len(mu_items) and j < len(nu_items):
        idx_mu, w_mu = mu_items[i]
        idx_nu, w_nu = nu_items[j]
        delta = min(w_mu, w_nu)
        dist += delta * cost[(idx_mu, idx_nu)]
        w_mu -= delta
        w_nu -= delta
        if abs(w_mu) < 1e-12:
            i += 1
        else:
            mu_items[i] = (idx_mu, w_mu)
        if abs(w_nu) < 1e-12:
            j += 1
        else:
            nu_items[j] = (idx_nu, w_nu)
    return dist


def ollivier_ricci_curvature(
    seeds: List[Point],
    points: List[Point],
    radius_factor: float = 1.5,
) -> Dict[Tuple[int, int], float]:
    """
    Compute Ollivier‑Ricci curvature κ(x,y) for each edge (x,y) of the Voronoi
    region graph. The curvature is defined as

        κ(x,y) = 1 - W1(μ_x, μ_y) / d(x,y)

    where μ_x, μ_y are uniform measures on the points of the two regions,
    W1 is the 1‑Wasserstein distance, and d(x,y) is the Euclidean distance
    between region centroids.
    """
    regions = voronoi_partition(seeds, points)
    adjacency = neighbor_graph(seeds, points, radius_factor)

    # Pre‑compute centroids and measures
    centroids = {i: region_centroid(reg) for i, reg in regions.items()}
    measures = {i: probability_measure(reg) for i, reg in regions.items()}

    # Build a global index map from (region, local_index) → global point index
    global_index: Dict[Tuple[int, int], int] = {}
    counter = 0
    for rid, reg in regions.items():
        for local_i in range(len(reg)):
            global_index[(rid, local_i)] = counter
            counter += 1

    # Pre‑compute pairwise point‑wise distances (cost matrix) only when needed
    curvature: Dict[Tuple[int, int], float] = {}
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if (j, i) in curvature:  # avoid double computation
                continue
            d_ij = euclidean_distance(centroids[i], centroids[j])
            if d_ij == 0:
                curvature[(i, j)] = 0.0
                continue

            # Build cost matrix between points of region i and region j
            cost: Dict[Tuple[int, int], float] = {}
            for li, pi in enumerate(regions[i]):
                gi = global_index[(i, li)]
                for lj, pj in enumerate(regions[j]):
                    gj = global_index[(j, lj)]
                    cost[(gi, gj)] = euclidean_distance(pi, pj)

            mu_i = measures[i]
            mu_j = measures[j]

            # Translate local indices to global for the transport routine
            mu_i_global = {global_index[(i, li)]: w for li, w in mu_i.items()}
            mu_j_global = {global_index[(j, lj)]: w for lj, w in mu_j.items()}

            w_distance = earth_movers_distance(mu_i_global, mu_j_global, cost)
            curvature[(i, j)] = 1.0 - w_distance / d_ij
    return curvature


# -------------------------------------------------------------------------
# Integrated analysis pipeline
# -------------------------------------------------------------------------
def integrated_analysis(
    seeds: List[Point],
    points: List[Point],
    n: int,
    radius_factor: float = 1.5,
) -> Tuple[List[Multivector], List[Multivector], Dict[Tuple[int, int], float]]:
    """
    1. Partition the point cloud into Voronoi cells.
    2. Build a multivector for each cell.
    3. Compute the geometric product of each cell multivector with the
       multivector of each of its neighbours (producing a richer set of
       algebraic features).
    4. Evaluate Ollivier‑Ricci curvature on the underlying region graph.

    Returns:
        - list of cell multivectors,
        - list of neighbour‑product multivectors (flattened),
        - dictionary of edge curvatures.
    """
    regions = voronoi_partition(seeds, points)
    adjacency = neighbor_graph(seeds, points, radius_factor)

    # 1. cell multivectors
    cell_mv = [region_multivector(reg, n) for reg in regions.values()]

    # 2. neighbour products (order‑independent, each edge once)
    neighbour_products: List[Multivector] = []
    for i, neigh in adjacency.items():
        for j in neigh:
            if i < j:  # ensure each unordered pair is processed once
                prod = cell_mv[i] * cell_mv[j]
                neighbour_products.append(prod)

    # 3. curvature
    curvature = ollivier_ricci_curvature(seeds, points, radius_factor)

    return cell_mv, neighbour_products, curvature


# -------------------------------------------------------------------------
# Example usage (can be removed or guarded by __name__ check)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple 3‑D example
    seeds = [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (10.0, 0.0, 0.0)]
    points = [
        (1.0, 0.5, 0.2),
        (0.2, 1.5, 0.3),
        (5.1, 5.2, 5.0),
        (9.5, 0.1, 0.2),
        (9.8, 0.3, 0.1),
    ]
    n_dim = 3

    cells, neigh_prods, curv = integrated_analysis(seeds, points, n_dim)

    print("Cell multivectors:")
    for i, mv in enumerate(cells):
        print(f"  Cell {i}: {mv}")

    print("\nNeighbour products:")
    for i, mv in enumerate(neigh_prods):
        print(f"  Edge {i}: {mv}")

    print("\nOllivier‑Ricci curvature on edges:")
    for edge, k in curv.items():
        print(f"  Edge {edge}: κ = {k:.4f}")