# DARWIN HAMMER — match 145, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (gen3)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s0.py (gen2)
# born: 2026-05-29T23:27:06Z

"""
Hybrid Algorithm: hybrid_endpoint_ssim_distributed_leader_ambush

Parents:
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (Morphology, sphericity, flatness, righting time)
- hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s0.py (Perceptual‑hash graph, leader election, broadcast probability)

Mathematical Bridge:
The bridge is the *similarity* between objects.  The first parent supplies continuous
shape descriptors (sphericity, flatness, righting‑time) that can be assembled into a
feature vector.  The second parent builds a graph from perceptual hashes and uses
graph connectivity for leader election.  Here we hash the *feature vector* (via a
simple perceptual hash) and use the Euclidean distance of the vectors as an
additional edge weight.  The resulting graph therefore fuses the SSIM‑like
continuous similarity with the discrete hash‑based similarity.  Leaders elected
from each connected component are then fed to the ambush‑strike primitive, whose
decision probability is computed with the broadcast‑probability formula that
depends on the leaders’ righting‑time (phase) and flatness (step).

The module provides three core functions that demonstrate this hybrid operation:
1. `compute_morphology_features` – builds the feature vector.
2. `build_morphology_graph` – constructs the hybrid similarity graph.
3. `elect_and_ambush` – performs leader election and evaluates ambush decisions.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections.abc import Mapping, Hashable
from typing import List, Tuple, Dict, Set

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
class Morphology:
    """Stores the morphology of a physical object."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity = (volume)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length + width) / (2 * height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """
    Simplified righting‑time model derived from a torsional pendulum:
        τ = b * k * θ   (restoring torque)
        I = m.mass * neck_lever**2   (moment of inertia)
        ω = sqrt(τ / I)   (natural angular frequency)
        T = 2π / ω       (period ≈ righting time)
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and lever length must be positive")
    I = m.mass * neck_lever ** 2
    # Use a characteristic angle θ = 1 rad for scaling
    tau = b * k * 1.0
    omega = math.sqrt(abs(tau) / I)
    if omega == 0:
        return float('inf')
    return 2.0 * math.pi / omega


# ----------------------------------------------------------------------
# Functions from Parent B (hashing, graph, broadcast probability)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]


def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: compare each value to the mean of the first 64 entries."""
    if not values:
        return 0
    avg = sum(values[:64]) / min(len(values), 64)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Return p = 1 / 2^(phase‑step), clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def compute_morphology_features(m: Morphology) -> np.ndarray:
    """
    Assemble the continuous descriptors of a Morphology into a feature vector.
    Returns a 3‑element NumPy array: [sphericity, flatness, righting_time].
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    return np.array([sph, flat, rt], dtype=float)


def build_morphology_graph(morphologies: List[Morphology],
                           distance_thresh: float = 0.15,
                           hamming_thresh: int = 4) -> Dict[Node, Set[Node]]:
    """
    Construct a hybrid similarity graph.
    - Nodes are string indices of the input list.
    - An undirected edge (i, j) is added iff:
        1) Euclidean distance between feature vectors < distance_thresh, AND
        2) Hamming distance between perceptual hashes of the vectors <= hamming_thresh.
    This fuses the continuous SSIM‑like metric (Euclidean distance) with the
    discrete hash‑based metric from Parent B.
    """
    n = len(morphologies)
    features = [compute_morphology_features(m) for m in morphologies]
    hashes = [compute_phash(f.tolist()) for f in features]

    graph: Dict[Node, Set[Node]] = {str(i): set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            euclid = np.linalg.norm(features[i] - features[j])
            ham = hamming_distance(hashes[i], hashes[j])
            if euclid <= distance_thresh and ham <= hamming_thresh:
                graph[str(i)].add(str(j))
                graph[str(j)].add(str(i))
    return graph


def elect_and_ambush(graph: Graph,
                    morphologies: List[Morphology]) -> List[Tuple[int, bool]]:
    """
    Perform leader election on each connected component of the hybrid graph.
    The leader is the node with the highest degree (most similar neighbours).
    For each leader, compute an ambush decision using broadcast_probability,
    where:
        phase  = ceil(righting_time)   (converted to an integer)
        step   = ceil(flatness * 10)   (scaled to keep it >=1)
    Returns a list of tuples (leader_index, ambush_flag).
    """
    # Helper: depth‑first search to collect components
    visited: Set[Node] = set()
    components: List[Set[Node]] = []

    def dfs(start: Node, comp: Set[Node]) -> None:
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(graph[node] - visited)

    for node in graph:
        if node not in visited:
            comp: Set[Node] = set()
            dfs(node, comp)
            components.append(comp)

    results: List[Tuple[int, bool]] = []

    for comp in components:
        # Leader = node with maximal degree within the component
        leader = max(comp, key=lambda n: len(graph[n]))
        idx = int(leader)  # original integer index
        m = morphologies[idx]

        # Parameters for broadcast probability
        phase = max(1, int(math.ceil(righting_time_index(m))))
        step = max(1, int(math.ceil(flatness_index(m.length, m.width, m.height) * 10)))

        prob = broadcast_probability(phase, step)
        ambush = random.random() < prob
        results.append((idx, ambush))

    return results


# ----------------------------------------------------------------------
# Optional: semiseparable matrix utility (illustrates matrix operation fusion)
# ----------------------------------------------------------------------
def semiseparable_matrix(features: List[np.ndarray]) -> np.ndarray:
    """
    Build a simple semiseparable matrix A where:
        A_ij = exp(-||f_i - f_j||)   for i <= j
        A_ij = A_ji                  for i > j
    This matrix is symmetric and low‑rank‑like, mirroring the semiseparable
    representation used in Parent A's state‑space models.
    """
    n = len(features)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = math.exp(-np.linalg.norm(features[i] - features[j]))
            A[i, j] = val
            A[j, i] = val
    return A


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)

    # Generate a small population of random Morphology objects
    population = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.2, 1.0),
            mass=random.uniform(0.1, 5.0),
        )
        for _ in range(12)
    ]

    # Build the hybrid graph
    g = build_morphology_graph(population)

    # Run leader election + ambush decision
    decisions = elect_and_ambush(g, population)

    # Display results
    print("Hybrid Graph Edges:")
    for node, nbrs in g.items():
        if nbrs:
            print(f"  {node} -> {sorted(nbrs)}")
    print("\nLeaders and Ambush Decisions:")
    for idx, ambush in decisions:
        print(f"  Leader {idx}: ambush = {ambush}")

    # Demonstrate semiseparable matrix creation (optional)
    feats = [compute_morphology_features(m) for m in population]
    S = semiseparable_matrix(feats)
    print("\nSemiseparable matrix (first 4 rows):")
    print(S[:4, :4])