# DARWIN HAMMER — match 2989, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_krampus_brain_m1835_s1.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_krampu_m293_s4.py (gen3)
# born: 2026-05-29T23:47:02Z

"""
This module fuses the two parent algorithms: 
* hybrid_hybrid_hybrid_ternar_hybrid_krampus_brain_m1835_s1.py, 
  a contextual LinUCB router with geometry utilities and 
* hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_krampu_m293_s4.py, 
  a Hybrid NLMS-Ollivier-Ricci Graph Engine that updates a global weight vector 
  with the rule w ← w + μ·e·x / (‖x‖²+ε).

The mathematical bridge between the two algorithms is the *curvature-weighted 
neighbourhood vector* for every node i, constructed as 
x_i = Σ_{j∈N(i)} (imp_ij · κ_ij) · φ_j, where imp_ij is the impedance weight of 
edge (i,j), κ_ij is the Ollivier-Ricci curvature of edge (i,j) and φ_j is the 
master-vector of node j obtained from the text-feature extractor.

This single set of equations unifies both topologies into one adaptive system.
"""

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Basic I/O helpers
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"

def now_z() -> str:
    """Current UTC timestamp in ISO-8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def emit_json(obj: Any) -> None:
    """Print a JSON object with deterministic key order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))

# Types
Point = Tuple[float, float]
Edge = Tuple[str, str]

# Geometry utilities
def euclidean_length(a: Point, b: Point) -> float:
    """Straight-line distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge]
) -> Tuple[np.ndarray, Dict[str, int], List[Edge]]:
    """
    Build a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Returns the matrix, a name-to-index map, and the canonical edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length

    # keep edges in the same order as input for later reference
    return L, idx_map, edges

# Parent B – feature extraction (master vector)
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo-random feature dictionary from a string."""
    features: Dict[str, float] = {}
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_psyche_ratio"
    ]
    for key in keys:
        features[key] = rnd.random()
    return features

# Hybrid operation
def hybrid_update(
    nodes: Dict[str, Point],
    edges: List[Edge],
    impedance_weights: Dict[Edge, float],
    ollivier_ricci_curvatures: Dict[Edge, float],
    text: str,
    mu: float,
    epsilon: float
) -> Dict[str, float]:
    """
    Updates the global weight vector w with the rule 
    w ← w + μ·e·x / (‖x‖²+ε).
    
    :param nodes: dictionary of node names to points
    :param edges: list of edges
    :param impedance_weights: dictionary of edges to impedance weights
    :param ollivier_ricci_curvatures: dictionary of edges to Ollivier-Ricci curvatures
    :param text: input text
    :param mu: learning rate
    :param epsilon: small value to avoid division by zero
    :return: updated global weight vector
    """
    L, idx_map, _ = build_length_matrix(nodes, edges)
    features = extract_full_features(text)
    n = len(nodes)
    w = np.zeros(n)
    x_i = np.zeros(n)

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        imp_ij = impedance_weights[(a, b)]
        kappa_ij = ollivier_ricci_curvatures[(a, b)]
        phi_j = features
        x_i[i] += imp_ij * kappa_ij * phi_j["operator_visceral_ratio"]

    y_i = np.dot(w, x_i)
    e_i = 1 / (np.max(np.abs(L)) + 1) - y_i
    w += mu * e_i * x_i / (np.dot(x_i, x_i) + epsilon)

    return {node: w[i] for i, node in enumerate(nodes)}

def calculate_curvature_weighted_neighbourhood_vector(
    nodes: Dict[str, Point],
    edges: List[Edge],
    impedance_weights: Dict[Edge, float],
    ollivier_ricci_curvatures: Dict[Edge, float],
    text: str
) -> Dict[str, float]:
    """
    Calculates the curvature-weighted neighbourhood vector for every node i.
    
    :param nodes: dictionary of node names to points
    :param edges: list of edges
    :param impedance_weights: dictionary of edges to impedance weights
    :param ollivier_ricci_curvatures: dictionary of edges to Ollivier-Ricci curvatures
    :param text: input text
    :return: curvature-weighted neighbourhood vector for every node
    """
    _, idx_map, _ = build_length_matrix(nodes, edges)
    features = extract_full_features(text)
    n = len(nodes)
    x_i = np.zeros(n)

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        imp_ij = impedance_weights[(a, b)]
        kappa_ij = ollivier_ricci_curvatures[(a, b)]
        phi_j = features
        x_i[i] += imp_ij * kappa_ij * phi_j["operator_visceral_ratio"]

    return {node: x_i[i] for i, node in enumerate(nodes)}

def calculate_wavefront_velocity(
    nodes: Dict[str, Point],
    edges: List[Edge],
    impedance_weights: Dict[Edge, float],
    ollivier_ricci_curvatures: Dict[Edge, float],
    text: str
) -> Dict[str, float]:
    """
    Calculates the wavefront velocity for every node i.
    
    :param nodes: dictionary of node names to points
    :param edges: list of edges
    :param impedance_weights: dictionary of edges to impedance weights
    :param ollivier_ricci_curvatures: dictionary of edges to Ollivier-Ricci curvatures
    :param text: input text
    :return: wavefront velocity for every node
    """
    L, _, _ = build_length_matrix(nodes, edges)
    features = extract_full_features(text)
    n = len(nodes)
    v_i = np.zeros(n)

    for a, b in edges:
        i, j = L.shape[0] - 1, L.shape[1] - 1
        imp_ij = impedance_weights[(a, b)]
        kappa_ij = ollivier_ricci_curvatures[(a, b)]
        phi_j = features
        v_i[i] = 1 / (np.max(np.abs(L)) + 1)

    return {node: v_i[i] for i, node in enumerate(nodes)}

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    impedance_weights = {("A", "B"): 1, ("B", "C"): 2, ("C", "A"): 3}
    ollivier_ricci_curvatures = {("A", "B"): 0.5, ("B", "C"): 0.7, ("C", "A"): 0.9}
    text = "example text"
    mu = 0.1
    epsilon = 0.01

    w = hybrid_update(nodes, edges, impedance_weights, ollivier_ricci_curvatures, text, mu, epsilon)
    x_i = calculate_curvature_weighted_neighbourhood_vector(nodes, edges, impedance_weights, ollivier_ricci_curvatures, text)
    v_i = calculate_wavefront_velocity(nodes, edges, impedance_weights, ollivier_ricci_curvatures, text)

    print("Updated global weight vector:", w)
    print("Curvature-weighted neighbourhood vector:", x_i)
    print("Wavefront velocity:", v_i)