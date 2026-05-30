# DARWIN HAMMER — match 1547, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_hybrid_hybrid_fisher_m165_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s3.py (gen3)
# born: 2026-05-29T23:37:26Z

import numpy as np
import math
from typing import Mapping, Hashable, Sequence, List, Dict, Set, Tuple, Iterable

Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Deterministic perceptual hashing based on feature vectors
# ----------------------------------------------------------------------
def compute_phash(values: Sequence[float]) -> int:
    """Return a 64‑bit perceptual hash of a sequence of floats.

    The hash is deterministic: each bit encodes whether the corresponding
    value is above the median of the whole sequence (or the global median
    if the sequence is longer than 64 elements).  The first 64 values are
    used for the hash; excess values affect the median but not the bits.
    """
    if not values:
        return 0
    arr = np.asarray(values, dtype=float)
    median = np.median(arr)
    bits = 0
    for v in arr[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Clifford algebra utilities (Euclidean metric)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[Tuple[int, ...], int]:
    """Sort a list of basis indices and compute the sign produced by swaps.

    Duplicate indices cancel because e_i * e_i = 1.
    Returns a tuple (sorted_unique_indices, sign).
    """
    sign = 1
    # bubble‑sort while tracking swaps
    n = len(indices)
    i = 0
    while i < n - 1:
        if indices[i] > indices[i + 1]:
            indices[i], indices[i + 1] = indices[i + 1], indices[i]
            sign = -sign
            i = max(i - 1, 0)  # step back to re‑check ordering
        elif indices[i] == indices[i + 1]:
            # cancel a pair e_i * e_i = 1
            del indices[i : i + 2]
            n -= 2
            i = max(i - 1, 0)
        else:
            i += 1
    return tuple(indices), sign


class Multivector:
    """Euclidean Clifford algebra element Cℓ(n,0).

    Internally stored as a mapping ``blade -> coefficient`` where a blade is
    represented by a sorted tuple of basis indices (the empty tuple denotes the
    scalar part).  Coefficients are real numbers.
    """

    def __init__(self, blades: Dict[Tuple[int, ...], float] | None = None):
        self._blades: Dict[Tuple[int, ...], float] = {}
        if blades:
            for b, c in blades.items():
                if abs(c) > 1e-15:
                    self._blades[tuple(sorted(b))] = float(c)

    @property
    def blades(self) -> Dict[Tuple[int, ...], float]:
        return self._blades

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self._blades)
        for b, c in other._blades.items():
            result[b] = result.get(b, 0.0) + c
            if abs(result[b]) < 1e-15:
                del result[b]
        return Multivector(result)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self._blades.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[Tuple[int, ...], float] = {}
        for blade_a, coeff_a in self._blades.items():
            for blade_b, coeff_b in other._blades.items():
                combined = list(blade_a) + list(blade_b)
                sorted_blade, sign = _blade_sign(combined)
                coeff = coeff_a * coeff_b * sign
                if abs(coeff) < 1e-15:
                    continue
                result[sorted_blade] = result.get(sorted_blade, 0.0) + coeff
                if abs(result[sorted_blade]) < 1e-15:
                    del result[sorted_blade]
        return Multivector(result)

    def scalar(self) -> float:
        """Return the scalar (grade‑0) part."""
        return self._blades.get((), 0.0)

    def grade(self, k: int) -> "Multivector":
        """Extract the grade‑k part."""
        return Multivector({b: c for b, c in self._blades.items() if len(b) == k})

    def norm(self) -> float:
        """Euclidean norm of the multivector (sqrt of sum of squares)."""
        return math.sqrt(sum(c * c for c in self._blades.values()))

    def __repr__(self) -> str:
        terms = []
        for blade, coeff in sorted(self._blades.items(), key=lambda x: (len(x[0]), x[0])):
            if blade == ():
                term = f"{coeff:.3g}"
            else:
                basis = "∧".join(f"e{i}" for i in blade)
                term = f"{coeff:.3g}{basis}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Convenient wrapper."""
    return a * b


# ----------------------------------------------------------------------
# Similarity and Fisher score utilities
# ----------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity in [‑1, 1] (returns 0 for zero vectors)."""
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_perceptual_similarity(node_a: Node, node_b: Node,
                                 features: Mapping[Node, FeatureVec]) -> float:
    """Similarity based on deterministic perceptual hash of feature vectors."""
    vec_a = np.asarray(features[node_a], dtype=float)
    vec_b = np.asarray(features[node_b], dtype=float)
    hash_a = compute_phash(vec_a)
    hash_b = compute_phash(vec_b)
    # Normalised Hamming similarity (1 = identical)
    return 1.0 - hamming_distance(hash_a, hash_b) / 64.0


def compute_fisher_score(node: Node, nodes: Iterable[Node],
                         features: Mapping[Node, FeatureVec]) -> float:
    """Mean perceptual similarity of *node* to all other nodes."""
    sims = [
        compute_perceptual_similarity(node, other, features)
        for other in nodes
        if other != node
    ]
    return float(np.mean(sims)) if sims else 0.0


# ----------------------------------------------------------------------
# Voronoi‑style clustering in feature space
# ----------------------------------------------------------------------
def voronoi_partition(nodes: List[Node],
                     features: Mapping[Node, FeatureVec],
                     seeds: List[Node]) -> Dict[Node, List[Node]]:
    """Assign each node to the nearest seed (Euclidean distance)."""
    seed_vecs = {s: np.asarray(features[s], dtype=float) for s in seeds}
    clusters: Dict[Node, List[Node]] = {s: [] for s in seeds}
    for n in nodes:
        v = np.asarray(features[n], dtype=float)
        nearest = min(seeds,
                      key=lambda s: np.linalg.norm(v - seed_vecs[s]))
        clusters[nearest].append(n)
    return clusters


# ----------------------------------------------------------------------
# Core hybrid operation
# ----------------------------------------------------------------------
def build_node_multivector(nodes: List[Node],
                           features: Mapping[Node, FeatureVec]) -> Multivector:
    """Encode each node as a basis vector e_i weighted by ‖feature‖."""
    blades: Dict[Tuple[int, ...], float] = {}
    for i, n in enumerate(nodes):
        weight = np.linalg.norm(features[n])
        if weight != 0.0:
            blades[(i,)] = weight
    return Multivector(blades)


def hybrid_operation(graph: Graph,
                     features: Mapping[Node, FeatureVec],
                     seed_fraction: float = 0.3) -> Dict[Node, float]:
    """
    1. Compute Fisher scores for every node using perceptual similarity.
    2. Encode the node set as a multivector (weights = feature norms).
    3. Form the geometric product of the multivector with itself.
       * The scalar part (‖v‖²) acts as a global confidence weight.
       * The bivector norm captures pairwise geometric entanglement.
    4. Choose high‑scoring nodes as Voronoi seeds and obtain clusters.
    5. Return a weighted Fisher score that blends the global and bivector
       contributions, optionally normalised per cluster.
    """
    nodes = list(graph.keys())

    # ---- 1. Fisher scores -------------------------------------------------
    fisher_scores = {
        n: compute_fisher_score(n, nodes, features) for n in nodes
    }

    # ---- 2. Multivector encoding -----------------------------------------
    mv = build_node_multivector(nodes, features)

    # ---- 3. Geometric self‑product ----------------------------------------
    product = geometric_product(mv, mv)
    scalar_weight = product.scalar()                     # ≈ Σ‖f‖²
    bivector_norm = product.grade(2).norm()              # captures pairwise geometry

    # Normalise the two geometric contributions
    if scalar_weight == 0.0:
        scalar_weight = 1.0
    if bivector_norm == 0.0:
        bivector_norm = 1.0
    geom_factor = (scalar_weight + bivector_norm) / 2.0

    # ---- 4. Voronoi seeds -------------------------------------------------
    # Pick top‑k nodes by Fisher score as seeds
    k = max(1, int(len(nodes) * seed_fraction))
    seed_nodes = sorted(nodes,
                        key=lambda n: fisher_scores[n],
                        reverse=True)[:k]
    clusters = voronoi_partition(nodes, features, seed_nodes)

    # ---- 5. Combine --------------------------------------------------------
    weighted_scores: Dict[Node, float] = {}
    for seed, members in clusters.items():
        # cluster‑level normalisation
        cluster_factor = 1.0 + len(members) / len(nodes)
        for n in members:
            base = fisher_scores[n]
            weighted = base * geom_factor * cluster_factor
            weighted_scores[n] = weighted

    return weighted_scores


# ----------------------------------------------------------------------
# Demonstration (self‑contained)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic graph: 7 nodes, each connected to the next two (undirected)
    graph: Graph = {
        i: {(i - 1) % 7, (i + 1) % 7, (i + 2) % 7}
        for i in range(7)
    }

    # Random but reproducible feature vectors (seeded)
    rng = np.random.default_rng(42)
    features: Dict[Node, FeatureVec] = {
        i: rng.normal(loc=0.0, scale=1.0, size=10).tolist()
        for i in range(7)
    }

    result = hybrid_operation(graph, features, seed_fraction=0.3)
    for node, score in sorted(result.items()):
        print(f"Node {node}: weighted Fisher score = {score:.4f}")