# DARWIN HAMMER — match 2530, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_fracti_m704_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s3.py (gen4)
# born: 2026-05-29T23:42:39Z

"""
Hybrid Algorithm: hybrid_ternary_router_hybrid_minimum_cost_endpoi_ssim_distributed_leader_ambush

Parents:
- hybrid_ternary_router_hybrid_minimum_cost__m36_s3.py (Ternary router, minimum cost path)
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (Morphology, sphericity, flatness, righting time)

Mathematical Bridge:
The bridge is the *similarity* between objects in both ternary routing and morphology. 
The ternary router uses a minimum cost path, while the morphology parent builds a feature vector from continuous shape descriptors.
Here we hash the *feature vector* (via a simple perceptual hash) and use the Euclidean distance of the vectors as an additional edge weight.
The resulting graph therefore fuses the SSIM-like continuous similarity with the discrete ternary routing.
The minimum cost path is computed with the ternary routing formula that depends on the Euclidean distance.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections.abc import Mapping, Hashable

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
    return (length * width * height) ** (1/3) / max(length, width, height)


def ternary_router(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ternary router between two nodes in the graph.

    Parameters
    ----------
    x : np.ndarray
        Feature vector of node x.
    y : np.ndarray
        Feature vector of node y.

    Returns
    -------
    np.ndarray
        Minimum cost path between node x and node y.
    """
    # Hash the feature vectors
    hx = np.hash(x)
    hy = np.hash(y)

    # Compute Euclidean distance
    distance = np.linalg.norm(x - y)

    # Compute minimum cost path
    mc = 1 / (1 + np.exp(-distance))

    # Return the minimum cost path
    return mc * hx + (1 - mc) * hy


def compute_morphology_features(obj: Morphology) -> np.ndarray:
    """
    Computes the feature vector from the morphology of a physical object.

    Parameters
    ----------
    obj : Morphology
        Morphology of the physical object.

    Returns
    -------
    np.ndarray
        Feature vector.
    """
    # Compute sphericity, flatness, and righting time
    sphericity = sphericity_index(obj.length, obj.width, obj.height)
    flatness = obj.width / obj.length
    righting_time = obj.mass / (obj.length ** 2)

    # Return the feature vector
    return np.array([sphericity, flatness, righting_time])


def build_morphology_graph(objects: List[Morphology]) -> np.ndarray:
    """
    Builds the hybrid similarity graph from the morphology of physical objects.

    Parameters
    ----------
    objects : List[Morphology]
        List of physical objects.

    Returns
    -------
    np.ndarray
        Hybrid similarity graph.
    """
    # Initialize the graph
    graph = np.zeros((len(objects), len(objects)))

    # Compute feature vectors
    features = [compute_morphology_features(obj) for obj in objects]

    # Compute Euclidean distances
    for i in range(len(objects)):
        for j in range(len(objects)):
            distance = np.linalg.norm(features[i] - features[j])

            # Compute similarity
            similarity = 1 / (1 + np.exp(distance))

            # Store the similarity in the graph
            graph[i, j] = similarity

    # Compute minimum cost paths using ternary router
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i != j:
                graph[i, j] = ternary_router(features[i], features[j])

    return graph


def elect_and_ambush(graph: np.ndarray, broadcast_prob: float) -> Tuple[List[int], List[float]]:
    """
    Performs leader election and evaluates ambush decisions.

    Parameters
    ----------
    graph : np.ndarray
        Hybrid similarity graph.
    broadcast_prob : float
        Broadcast probability.

    Returns
    -------
    Tuple[List[int], List[float]]
        List of elected leaders and list of ambush decision probabilities.
    """
    # Perform leader election
    leaders = []
    for i in range(len(graph)):
        if np.sum(graph[i]) > 0:
            leaders.append(i)

    # Evaluate ambush decisions
    ambush_decisions = []
    for i in leaders:
        ambush_decisions.append(broadcast_prob * np.mean(graph[i]))

    return leaders, ambush_decisions


# Smoke test
if __name__ == "__main__":
    # Create some objects
    obj1 = Morphology(1.0, 2.0, 3.0, 4.0)
    obj2 = Morphology(5.0, 6.0, 7.0, 8.0)
    obj3 = Morphology(9.0, 10.0, 11.0, 12.0)

    # Build the graph
    graph = build_morphology_graph([obj1, obj2, obj3])

    # Perform leader election and evaluate ambush decisions
    leaders, ambush_decisions = elect_and_ambush(graph, 0.5)

    # Print the results
    print("Leaders:", leaders)
    print("Ambush decisions:", ambush_decisions)