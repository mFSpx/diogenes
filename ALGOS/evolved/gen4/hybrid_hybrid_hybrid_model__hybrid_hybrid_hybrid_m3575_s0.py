# DARWIN HAMMER — match 3575, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s4.py (gen3)
# born: 2026-05-29T23:50:42Z

"""
Hybrid VRAM-SSIM-Curvature Algorithm

This module fuses two parent algorithms:

* **Parent A** – Hybrid VRAM-Curvature Scheduler, which integrates VRAM planning with Ollivier-Ricci curvature.
* **Parent B** – Hybrid Morphology-SSIM-Hygiene Algorithm, which combines morphology-based indices with SSIM similarity and hygiene scores.

The mathematical bridge between the two parents lies in the use of weighted distributions. In Parent A, the VRAM weights are used to compute the Ollivier-Ricci curvature, while in Parent B, the morphology-based indices are used to compute the SSIM similarity. By combining these two approaches, we can create a hybrid algorithm that integrates VRAM planning, Ollivier-Ricci curvature, and SSIM similarity.

The hybrid algorithm works as follows:

1. Each artefact registered in the VRAM planner becomes a node in a graph, with its VRAM weight used to compute the Ollivier-Ricci curvature.
2. The morphology-based indices are used to compute the SSIM similarity between two artefacts.
3. The SSIM similarity is then used to weight the Ollivier-Ricci curvature, creating a hybrid curvature that reflects both the VRAM allocation landscape and the physical similarity between artefacts.
4. The hybrid curvature is then used to make decisions about accepting or rejecting new artefacts, taking into account both the VRAM budget and the physical similarity between artefacts.

This hybrid algorithm provides a powerful tool for optimizing VRAM allocation and artefact placement, while also considering the physical similarity between artefacts.
"""

import json
import os
import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple
import numpy as np

# Define the default budget and reserve values
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

# Define the morphology-based indices
@dataclass
class MorphologyIndices:
    sphericity: float
    flatness: float
    righting_time: float
    recovery_priority: float

# Define the SSIM similarity function
def ssim_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Compute the SSIM similarity between two vectors.

    Args:
    v1: The first vector.
    v2: The second vector.

    Returns:
    The SSIM similarity between the two vectors.
    """
    # Compute the mean and standard deviation of each vector
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    std1 = np.std(v1)
    std2 = np.std(v2)

    # Compute the covariance between the two vectors
    covariance = np.sum((v1 - mean1) * (v2 - mean2)) / len(v1)

    # Compute the SSIM similarity
    ssim = (2 * mean1 * mean2 + 1) * (2 * covariance + 1) / ((mean1 ** 2 + mean2 ** 2 + 1) * (std1 ** 2 + std2 ** 2 + 1))

    return ssim

# Define the Ollivier-Ricci curvature function
def ollivier_ricci_curvature(graph: Dict[int, List[int]], weights: Dict[int, float]) -> Dict[int, float]:
    """
    Compute the Ollivier-Ricci curvature of a graph.

    Args:
    graph: The graph, represented as a dictionary where each key is a node and each value is a list of neighboring nodes.
    weights: The weights of each node, represented as a dictionary where each key is a node and each value is its weight.

    Returns:
    The Ollivier-Ricci curvature of each node, represented as a dictionary where each key is a node and each value is its curvature.
    """
    # Initialize the curvature dictionary
    curvature = {node: 0 for node in graph}

    # Compute the curvature of each node
    for node in graph:
        # Compute the lazy random-walk distribution
        distribution = {neighbor: weights[neighbor] / sum(weights[n] for n in graph[node]) for neighbor in graph[node]}

        # Compute the curvature
        curvature[node] = sum(distribution[neighbor] * (1 - distribution[neighbor]) for neighbor in graph[node])

    return curvature

# Define the hybrid curvature function
def hybrid_curvature(graph: Dict[int, List[int]], weights: Dict[int, float], ssim_similarities: Dict[Tuple[int, int], float]) -> Dict[int, float]:
    """
    Compute the hybrid curvature of a graph.

    Args:
    graph: The graph, represented as a dictionary where each key is a node and each value is a list of neighboring nodes.
    weights: The weights of each node, represented as a dictionary where each key is a node and each value is its weight.
    ssim_similarities: The SSIM similarities between each pair of nodes, represented as a dictionary where each key is a tuple of two nodes and each value is their SSIM similarity.

    Returns:
    The hybrid curvature of each node, represented as a dictionary where each key is a node and each value is its curvature.
    """
    # Initialize the hybrid curvature dictionary
    hybrid_curvature = {node: 0 for node in graph}

    # Compute the hybrid curvature of each node
    for node in graph:
        # Compute the weighted SSIM similarity
        weighted_ssim = sum(ssim_similarities[(node, neighbor)] * weights[neighbor] for neighbor in graph[node])

        # Compute the hybrid curvature
        hybrid_curvature[node] = ollivier_ricci_curvature(graph, weights)[node] * weighted_ssim

    return hybrid_curvature

# Define the artefact registration function
def register_artefact(graph: Dict[int, List[int]], weights: Dict[int, float], ssim_similarities: Dict[Tuple[int, int], float], artefact: int, weight: float) -> None:
    """
    Register a new artefact in the graph.

    Args:
    graph: The graph, represented as a dictionary where each key is a node and each value is a list of neighboring nodes.
    weights: The weights of each node, represented as a dictionary where each key is a node and each value is its weight.
    ssim_similarities: The SSIM similarities between each pair of nodes, represented as a dictionary where each key is a tuple of two nodes and each value is their SSIM similarity.
    artefact: The new artefact to register.
    weight: The weight of the new artefact.
    """
    # Add the new artefact to the graph
    graph[artefact] = []

    # Add the new artefact to the weights dictionary
    weights[artefact] = weight

    # Update the SSIM similarities
    for node in graph:
        ssim_similarities[(node, artefact)] = ssim_similarity([weights[node]], [weights[artefact]])

# Define the artefact rejection function
def reject_artefact(graph: Dict[int, List[int]], weights: Dict[int, float], ssim_similarities: Dict[Tuple[int, int], float], artefact: int) -> None:
    """
    Reject an artefact from the graph.

    Args:
    graph: The graph, represented as a dictionary where each key is a node and each value is a list of neighboring nodes.
    weights: The weights of each node, represented as a dictionary where each key is a node and each value is its weight.
    ssim_similarities: The SSIM similarities between each pair of nodes, represented as a dictionary where each key is a tuple of two nodes and each value is their SSIM similarity.
    artefact: The artefact to reject.
    """
    # Remove the artefact from the graph
    del graph[artefact]

    # Remove the artefact from the weights dictionary
    del weights[artefact]

    # Update the SSIM similarities
    for node in graph:
        del ssim_similarities[(node, artefact)]

if __name__ == "__main__":
    # Create a sample graph
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    # Create a sample weights dictionary
    weights = {0: 1, 1: 2, 2: 3}

    # Create a sample SSIM similarities dictionary
    ssim_similarities = {(0, 1): 0.5, (0, 2): 0.7, (1, 2): 0.3}

    # Register a new artefact
    register_artefact(graph, weights, ssim_similarities, 3, 4)

    # Compute the hybrid curvature
    hybrid_curvature_dict = hybrid_curvature(graph, weights, ssim_similarities)

    # Print the hybrid curvature
    print(hybrid_curvature_dict)

    # Reject an artefact
    reject_artefact(graph, weights, ssim_similarities, 1)

    # Compute the hybrid curvature again
    hybrid_curvature_dict = hybrid_curvature(graph, weights, ssim_similarities)

    # Print the hybrid curvature again
    print(hybrid_curvature_dict)