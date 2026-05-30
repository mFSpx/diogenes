# DARWIN HAMMER — match 835, survivor 2
# gen: 4
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s2.py (gen1)
# parent_b: hybrid_rectified_flow_hybrid_hybrid_hard_t_m184_s1.py (gen3)
# born: 2026-05-29T23:31:18Z

"""
Hybrid Rectified Flow and NLMS-Graph Engine
=============================================

This module fuses the Rectified Flow Matching algorithm (rectified_flow.py) 
with the Normalized Least-Mean-Squares (NLMS) adaptive filter and graph-propagation engine (hybrid_nlms_omni_chaotic_sprint_m59_s2.py). 
The mathematical bridge between the two structures is found by using the 
Rectified Flow's straight-line interpolant to generate input features for 
the NLMS predictor, which attempts to model the wavefront velocity of the 
graph-propagation engine.

The Rectified Flow's straight-line interpolant is used to generate input 
vectors for the NLMS predictor, which are then used to predict the 
wavefront velocity of the graph-propagation engine. The NLMS error is 
then used to adapt the global weight vector.

Imports:
    numpy
    standard library
    math
    random
    sys
    pathlib
"""

import numpy as np
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random
import sys

# ----------------------------------------------------------------------
# Core NLMS utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot-product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.
    """
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e


# ----------------------------------------------------------------------
# Rectified Flow utilities
# ----------------------------------------------------------------------

def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0.

    Broadcasts t over a leading batch dimension.  If x0 has shape (B, d) and t
    has shape (B,), t is reshaped to (B, 1) 
    """
    t = np.reshape(t, (-1, 1))
    return t * x1 + (1 - t) * x0

def flow_target(x0, x1):
    """Target vector field: v_theta(Z_t, t) = (X_1 - X_0)."""
    return x1 - x0

# ----------------------------------------------------------------------
# Graph-propagation engine utilities
# ----------------------------------------------------------------------

def build_adjacency_structure(num_nodes: int, num_edges: int) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build an adjacency structure with impedance-weighted edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of edges in the graph.

    Returns
    -------
    adjacency_structure : Dict[int, List[Tuple[int, float]]]
        Adjacency structure with impedance-weighted edges.
    """
    adjacency_structure = {i: [] for i in range(num_nodes)}
    for _ in range(num_edges):
        node1, node2, impedance = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1), random.uniform(0, 1)
        adjacency_structure[node1].append((node2, impedance))
        adjacency_structure[node2].append((node1, impedance))
    return adjacency_structure

def compute_wavefront_velocity(adjacency_structure: Dict[int, List[Tuple[int, float]]]) -> Dict[int, float]:
    """
    Compute the wavefront velocity for each node in the graph.

    Parameters
    ----------
    adjacency_structure : Dict[int, List[Tuple[int, float]]]
        Adjacency structure with impedance-weighted edges.

    Returns
    -------
    wavefront_velocity : Dict[int, float]
        Wavefront velocity for each node in the graph.
    """
    wavefront_velocity = {}
    for node in adjacency_structure:
        stress = sum(impedance for _, impedance in adjacency_structure[node])
        wavefront_velocity[node] = 1 / max(stress, 1)
    return wavefront_velocity

def generate_input_vector(node: int, adjacency_structure: Dict[int, List[Tuple[int, float]]], feature_vectors: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Generate an input vector for the NLMS predictor.

    Parameters
    ----------
    node : int
        Current node.
    adjacency_structure : Dict[int, List[Tuple[int, float]]]
        Adjacency structure with impedance-weighted edges.
    feature_vectors : Dict[int, np.ndarray]
        Feature vectors for each node.

    Returns
    -------
    input_vector : np.ndarray
        Input vector for the NLMS predictor.
    """
    input_vector = np.zeros(len(feature_vectors[node]))
    for neighbor, impedance in adjacency_structure[node]:
        input_vector += impedance * feature_vectors[neighbor]
    return input_vector

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------

def hybrid_operation(num_nodes: int, num_edges: int, feature_vectors: Dict[int, np.ndarray], weights: np.ndarray) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Perform the hybrid operation.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    num_edges : int
        Number of edges in the graph.
    feature_vectors : Dict[int, np.ndarray]
        Feature vectors for each node.
    weights : np.ndarray
        Initial weight vector.

    Returns
    -------
    weights : np.ndarray
        Updated weight vector.
    wavefront_velocity : Dict[int, float]
        Wavefront velocity for each node in the graph.
    """
    adjacency_structure = build_adjacency_structure(num_nodes, num_edges)
    wavefront_velocity = compute_wavefront_velocity(adjacency_structure)

    for node in range(num_nodes):
        input_vector = generate_input_vector(node, adjacency_structure, feature_vectors)
        prediction = nlms_predict(weights, input_vector)
        error = wavefront_velocity[node] - prediction
        weights, _ = nlms_update(weights, input_vector, wavefront_velocity[node])

    return weights, wavefront_velocity

if __name__ == "__main__":
    num_nodes, num_edges = 10, 20
    feature_vectors = {i: np.random.rand(5) for i in range(num_nodes)}
    weights = np.random.rand(5)
    updated_weights, wavefront_velocity = hybrid_operation(num_nodes, num_edges, feature_vectors, weights)
    print(updated_weights)
    print(wavefront_velocity)