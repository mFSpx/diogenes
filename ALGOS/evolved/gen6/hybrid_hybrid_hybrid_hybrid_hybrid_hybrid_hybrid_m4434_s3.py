# DARWIN HAMMER — match 4434, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s0.py (gen4)
# born: 2026-05-29T23:55:47Z

"""Hybrid Algorithm combining Hoeffding‑Gini decision trees with Morphology‑Driven
Distributed Leader Election.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_gini_coeffici_m1406_s0.py (Hoeffding tree,
  Gini coefficient, Euclidean distance)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s0.py (Morphology,
  sphericity index, graph‑based clustering and leader election)

Mathematical Bridge:
Both parents operate on collections of numeric descriptors.  The Gini coefficient
measures inequality of a scalar feature set, while the sphericity index provides
a shape‑based normalisation of 3‑D morphology.  By normalising the Gini value with
the sphericity index we obtain a *shape‑aware impurity* that can be used inside
the Hoeffding bound to decide whether a node in a streaming decision tree
should split.  The same impurity score is also employed as a utility function
for distributed leader election on a similarity graph: nodes with lower
shape‑aware impurity are preferred as cluster representatives.  Thus the core
topologies of the two parents are fused through a single scalar that blends
statistical inequality (Gini) with geometric regularity (sphericity) and is
plugged into both the Hoeffding bound and the leader‑election criterion."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric and statistical primitives (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient as a measure of inequality."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def hoeffding_bound(sample_mean: float, delta: float, sqrt_n: float) -> float:
    """Hoeffding bound for a confidence interval."""
    return math.sqrt(math.log(1.0 / delta) / (2.0 * sqrt_n))


# ----------------------------------------------------------------------
# Morphology utilities (from Parent B)
# ----------------------------------------------------------------------
class Morphology:
    """Stores the morphology of a physical object."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity index = (volume)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0.0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


# ----------------------------------------------------------------------
# Hybrid Core: shape‑aware impurity
# ----------------------------------------------------------------------
def shape_aware_impurity(values: Iterable[float],
                         morph: Morphology) -> float:
    """
    Combine Gini inequality with morphological sphericity.

    The impurity is defined as:
        I = Gini(values) / (1 + sphericity)

    Adding 1 to the denominator prevents division by zero and keeps the
    impurity bounded when the object is highly spherical (sphericity → 1).
    """
    gini = gini_coefficient(values)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    return gini / (1.0 + sph)


# ----------------------------------------------------------------------
# Hybrid Hoeffding split decision (uses shape‑aware impurity)
# ----------------------------------------------------------------------
def should_split(node_feature_samples: List[float],
                 morph: Morphology,
                 delta: float = 0.05,
                 min_samples: int = 30) -> bool:
    """
    Decide whether a streaming decision‑tree node should split.

    1. Compute the shape‑aware impurity of the current node.
    2. Treat the impurity as the sample mean for Hoeffding bound.
    3. If the bound is smaller than a fraction of the impurity, the split is
       statistically safe.

    Returns True if a split is recommended.
    """
    n = len(node_feature_samples)
    if n < min_samples:
        return False

    impurity = shape_aware_impurity(node_feature_samples, morph)
    bound = hoeffding_bound(sample_mean=impurity,
                            delta=delta,
                            sqrt_n=math.sqrt(n))

    # Heuristic: split when bound is less than 20 % of impurity
    return bound < 0.2 * impurity


# ----------------------------------------------------------------------
# Graph utilities for distributed leader election (from Parent B)
# ----------------------------------------------------------------------
def perceptual_hash(vector: Sequence[float]) -> int:
    """
    Very lightweight deterministic hash for a feature vector.
    It mimics a perceptual hash by quantising the vector and packing bits.
    """
    quantised = [int(round(v * 10)) & 0xFF for v in vector]
    h = 0
    for q in quantised:
        h = ((h << 8) | q) & 0xFFFFFFFF
    return h


def similarity(a: Sequence[float], b: Sequence[float],
               threshold: float = 0.1) -> bool:
    """
    Two vectors are considered similar if their Euclidean distance is below
    `threshold`.  The threshold is deliberately small to emulate perceptual‑hash
    clustering.
    """
    return euclidean(a, b) < threshold


def build_similarity_graph(features: List[Sequence[float]]) -> Dict[int, Set[int]]:
    """
    Build an undirected graph where each node is the index of a feature vector.
    An edge exists when the vectors are similar.
    """
    graph: Dict[int, Set[int]] = {i: set() for i in range(len(features))}
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if similarity(features[i], features[j]):
                graph[i].add(j)
                graph[j].add(i)
    return graph


def find_clusters(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """Return connected components of the similarity graph."""
    visited: Set[int] = set()
    clusters: List[Set[int]] = []

    def dfs(v: int, comp: Set[int]) -> None:
        visited.add(v)
        comp.add(v)
        for nb in graph[v]:
            if nb not in visited:
                dfs(nb, comp)

    for node in graph:
        if node not in visited:
            component: Set[int] = set()
            dfs(node, component)
            clusters.append(component)
    return clusters


# ----------------------------------------------------------------------
# Hybrid leader election using shape‑aware impurity as utility
# ----------------------------------------------------------------------
def elect_leaders(features: List[Sequence[float]],
                  morphologies: List[Morphology],
                  delta: float = 0.05) -> Dict[int, int]:
    """
    Perform distributed leader election on clusters of similar feature vectors.

    For each cluster:
        * Compute the shape‑aware impurity of every member (using its own
          morphology and the cluster's feature values).
        * The member with the smallest impurity is elected as the leader.
        * The Hoeffding bound is used as a sanity check: if the bound for the
          chosen leader exceeds 30 % of its impurity, the election is considered
          unstable and the leader is set to -1 (no confident leader).

    Returns a mapping ``node_index -> leader_index`` (leader_index == node_index
    for the elected node, -1 otherwise).
    """
    graph = build_similarity_graph(features)
    clusters = find_clusters(graph)

    leaders: Dict[int, int] = {}

    for cluster in clusters:
        # Evaluate impurity for each member
        impurities: List[Tuple[int, float]] = []
        for idx in cluster:
            imp = shape_aware_impurity(features[idx], morphologies[idx])
            impurities.append((idx, imp))

        # Pick the member with minimal impurity
        leader_idx, leader_imp = min(impurities, key=lambda t: t[1])

        # Hoeffding sanity check (using cluster size as sample count)
        bound = hoeffding_bound(sample_mean=leader_imp,
                                delta=delta,
                                sqrt_n=math.sqrt(len(cluster)))
        if bound > 0.3 * leader_imp:
            # Unstable election
            for idx in cluster:
                leaders[idx] = -1
        else:
            for idx in cluster:
                leaders[idx] = leader_idx

    return leaders


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic streaming data for a node
    node_samples = [random.random() for _ in range(50)]
    node_morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    split = should_split(node_samples, node_morph)
    print(f"Should split node? {split}")

    # Build a small feature set and corresponding morphologies
    feature_vectors = [
        np.random.rand(3).tolist() for _ in range(10)
    ]
    morphs = [
        Morphology(length=random.uniform(0.5, 3.0),
                   width=random.uniform(0.5, 3.0),
                   height=random.uniform(0.5, 3.0),
                   mass=random.uniform(1.0, 10.0))
        for _ in range(10)
    ]

    leaders = elect_leaders(feature_vectors, morphs)
    print("Leader election result (node -> leader):")
    for node, leader in leaders.items():
        print(f"  {node} -> {leader}")

    # Verify that the graph building does not raise
    graph = build_similarity_graph(feature_vectors)
    print(f"Similarity graph has {len(graph)} nodes and "
          f"{sum(len(v) for v in graph.values()) // 2} edges.")