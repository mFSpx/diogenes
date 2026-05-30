# DARWIN HAMMER — match 3917, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s0.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1708_s0.py (gen6)
# born: 2026-05-29T23:52:35Z

"""Hybrid RBF‑Voronoi‑Fisher Algorithm
================================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s0.py``  
  provides a radial‑basis‑function (RBF) surrogate that predicts a scalar
  similarity score for each node feature vector and a similarity matrix based
  on perceptual‑hash (p‑hash) Hamming distances.

* **Parent B** – ``hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m1708_s0.py``  
  offers Voronoi partitioning of points and a Fisher‑score calculation that
  measures the information content of a point with respect to a Gaussian beam.

**Mathematical bridge**

The scalar output of the RBF surrogate (`y = Σ w_i·φ(||x‑c_i||)`) is interpreted
as an angular coordinate ``θ``.  This ``θ`` feeds directly into the Gaussian‑beam
and Fisher‑score formulas of Parent B, allowing a *geometry‑free* Fisher score
to be evaluated on the high‑dimensional feature space.  The resulting Fisher
scores are then used as weights for the Endpoint‑Circuit‑Breaker, whose failure
threshold is modulated by the average pairwise similarity from Parent A.

The three core hybrid functions are:

1. ``predict_node_scores`` – RBF surrogate predictions for all nodes.
2. ``voronoi_partition_by_score`` – 1‑D Voronoi partition of nodes using the
   predicted scores as coordinates.
3. ``fisher_score_for_partition`` – Fisher scores for each node based on its
   distance to the seed (region) centre, together with a circuit‑breaker update
   that reacts to the overall similarity matrix.

All components are pure NumPy / Python and can be executed as a self‑contained
script.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Very small perceptual hash – 64‑bit binary string."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    """Pairwise similarity (1‑Hamming/64) matrix for a set of hashes."""
    nodes = list(hashes.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = hashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = hashes[nj]
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes


@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate model – same interface as Parent A."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Weighted sum of Gaussian RBFs."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
Point = Tuple[float, float]  # kept for API compatibility – not used directly


def distance_1d(a: float, b: float) -> float:
    """Absolute distance on the real line (used for 1‑D Voronoi)."""
    return abs(a - b)


def nearest_1d(value: float, seeds: List[float]) -> int:
    """Index of the nearest seed in 1‑D."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance_1d(value, seeds[i]), i))


def assign_1d(values: List[float], seeds: List[float]) -> Dict[int, List[int]]:
    """Voronoi assignment of indices of ``values`` to the nearest seed."""
    regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for idx, v in enumerate(values):
        region = nearest_1d(v, seeds)
        regions[region].append(idx)
    return regions


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity (Parent B)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information score derived from the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple counter that trips when a dynamic threshold is exceeded."""
    def __init__(self, base_threshold: float = 0.5):
        self.base_threshold = base_threshold
        self.failures = 0
        self.threshold = base_threshold

    def update_threshold(self, similarity: float) -> None:
        """
        Adjust the threshold: higher average similarity ⇒ stricter threshold.
        The update rule is heuristic but respects the mathematical link to
        Parent A's similarity matrix.
        """
        # Clamp similarity to [0,1] and map to a factor in [0.5, 2.0]
        factor = 0.5 + 1.5 * similarity
        self.threshold = self.base_threshold * factor

    def register(self, score: float) -> None:
        """Increment failure counter if the score exceeds the current threshold."""
        if score > self.threshold:
            self.failures += 1

    def is_tripped(self) -> bool:
        """Circuit is open after a single failure (for demonstration)."""
        return self.failures > 0


def predict_node_scores(
    node_features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate,
) -> Dict[Node, float]:
    """
    Apply the RBF surrogate to every node's feature vector.
    The scalar predictions are interpreted as the angular coordinate θ.
    """
    return {node: surrogate.predict(vec) for node, vec in node_features.items()}


def voronoi_partition_by_score(
    predictions: Dict[Node, float],
    seed_nodes: List[Node],
) -> Tuple[Dict[int, List[Node]], List[float]]:
    """
    Perform a 1‑D Voronoi partition of the node predictions.
    ``seed_nodes`` are the representatives of each region; their predictions
    serve as the Voronoi seeds (centres).
    Returns the region mapping and the list of seed values for later use.
    """
    # Extract the scalar values in a deterministic order
    nodes = list(predictions.keys())
    values = [predictions[n] for n in nodes]

    seed_values = [predictions[s] for s in seed_nodes]
    region_indices = assign_1d(values, seed_values)

    # Convert index‑based regions back to node objects
    regions: Dict[int, List[Node]] = {}
    for region_id, idx_list in region_indices.items():
        regions[region_id] = [nodes[i] for i in idx_list]

    return regions, seed_values


def fisher_score_for_partition(
    predictions: Dict[Node, float],
    regions: Dict[int, List[Node]],
    seed_values: List[float],
    width_factor: float = 0.1,
) -> Dict[Node, float]:
    """
    Compute a Fisher score for every node based on its distance to the centre
    of its Voronoi region (the seed value).  The ``width`` of the Gaussian beam
    is chosen as a fraction of the overall spread of predictions.
    """
    all_vals = np.fromiter(predictions.values(), dtype=float)
    overall_spread = max(all_vals) - min(all_vals)
    width = max(overall_spread * width_factor, 1e-6)  # avoid zero

    scores: Dict[Node, float] = {}
    for region_id, nodes in regions.items():
        centre = seed_values[region_id]
        for node in nodes:
            theta = predictions[node]
            scores[node] = fisher_score(theta, centre, width)
    return scores


def run_hybrid(
    node_features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate,
    seed_nodes: List[Node],
    breaker: EndpointCircuitBreaker,
) -> Tuple[Dict[Node, float], Dict[int, List[Node]], Dict[Node, float]]:
    """
    Full hybrid pipeline:
        1. Predict scalar scores with the RBF surrogate.
        2. Partition nodes via 1‑D Voronoi using the predictions of ``seed_nodes``.
        3. Compute Fisher scores per node.
        4. Update the circuit breaker based on the average similarity of the
           perceptual hashes of the predictions.
    Returns (predictions, regions, fisher_scores).
    """
    # 1 – RBF predictions
    predictions = predict_node_scores(node_features, surrogate)

    # 2 – Voronoi partition
    regions, seed_vals = voronoi_partition_by_score(predictions, seed_nodes)

    # 3 – Fisher scores
    fisher_scores = fisher_score_for_partition(predictions, regions, seed_vals)

    # 4 – Similarity matrix & breaker update
    #   Build a tiny perceptual hash from each prediction (treated as a 1‑D vector)
    phashes = {n: compute_phash([predictions[n]]) for n in predictions}
    S, _ = similarity_matrix(phashes)
    avg_similarity = float(np.mean(S))
    breaker.update_threshold(avg_similarity)

    # Register each Fisher score with the breaker
    for sc in fisher_scores.values():
        breaker.register(sc)

    return predictions, regions, fisher_scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- synthetic data ------------------------------------------------
    random.seed(42)
    np.random.seed(42)

    # Create a tiny graph of 10 nodes with 5‑dimensional random features
    nodes = [f"node_{i}" for i in range(10)]
    features: Dict[Node, FeatureVec] = {
        n: np.random.rand(5).tolist() for n in nodes
    }

    # Random RBF surrogate (3 centres, random weights)
    centres = [tuple(np.random.rand(5)) for _ in range(3)]
    weights = list(np.random.rand(3))
    surrogate = RBFSurrogate(centers=centres, weights=weights, epsilon=1.5)

    # Choose 3 seed nodes arbitrarily
    seed_nodes = nodes[:3]

    # Initialise circuit breaker
    breaker = EndpointCircuitBreaker(base_threshold=0.2)

    # Run the hybrid algorithm
    preds, regs, fisher = run_hybrid(features, surrogate, seed_nodes, breaker)

    # Simple sanity prints (no external libraries)
    print("Predictions (first 5):")
    for n in nodes[:5]:
        print(f"  {n}: {preds[n]:.4f}")

    print("\nVoronoi regions:")
    for rid, members in regs.items():
        print(f"  Region {rid}: {members}")

    print("\nFisher scores (first 5):")
    for n in nodes[:5]:
        print(f"  {n}: {fisher[n]:.6f}")

    print("\nCircuit breaker status:")
    print(f"  Threshold after similarity update: {breaker.threshold:.4f}")
    print(f"  Failures counted: {breaker.failures}")
    print(f"  Tripped? {'Yes' if breaker.is_tripped() else 'No'}")