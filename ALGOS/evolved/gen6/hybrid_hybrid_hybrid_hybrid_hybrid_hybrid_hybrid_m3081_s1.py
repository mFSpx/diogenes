# DARWIN HAMMER — match 3081, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_possum_hybrid_hybrid_hybrid_m1533_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s2.py (gen5)
# born: 2026-05-29T23:47:44Z

"""Hybrid NLMS-LSM Minimum-Cost Tree with Morphology and Recovery Priority
=====================================================================

Parents
-------
* **hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1027_s0.py** – Morphology and Recovery Priority.
* **hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s2.py** – Hybrid NLMS-LSM Minimum-Cost Tree.

Mathematical Bridge
-------------------
We define a hybrid edge weight as the weighted sum of the morphology-based term and the NLMS-LSM term.

For an unordered pair of nodes *e = (i, j)* we define a hybrid edge weight

d_ij      = Euclidean distance between the geometric coordinates of i and j
c_ij      = epistemic-certainty factor supplied by the user (0 ≤ c ≤ 1)
p_i, p_j  = NLMS decision scores (scalar) associated with the two nodes
v_i, v_j  = LSM vectors derived from free-form textual descriptors of the nodes
ℓ_ij      = cosine similarity(v_i, v_j) ∈ [-1, 1] (used as a likelihood)
π_ij      = (p_i + p_j) / (p_i + p_j + ε) (Bayesian prior)
ϕ_ij      = c_ij * 0.1 (false-positive term)
m_ij      = bayes_marginal(π_ij, 1-ℓ_ij, ϕ_ij) (marginal probability of error)
w_ij      = d_ij * (1 - m_ij) + ε (final hybrid weight)

The morphology-based term is weighted by the sphericity index of the entity,
while the NLMS-LSM term is weighted by the righting time index of the entity.

We use the following governing equations to compute the hybrid edge weight:

w_ij = (sphericity_index(i) * d_ij) + (righting_time_index(i) * w_ijNLMSLSM(i, j))

where w_ijNLMSLSM(i, j) is the NLMS-LSM term computed using the NLMS and LSM cores.

We then use the Prim-MST algorithm with the hybrid edge weights to construct a minimum-cost spanning tree.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Entity and Morphology Dataclasses
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    mass: float = 0.0

# ----------------------------------------------------------------------
# Morphology and Recovery Priority
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Entity, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (k / (b * fi)) * (m.mass ** (1.0 / 3.0)) * neck_lever

def _pct(value: float) -> float:
    return round(float(value), 6)

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components, n):
        self.components = components

# ----------------------------------------------------------------------
# NLMS core (parent B)
# ----------------------------------------------------------------------
def nlms_predict(p: float, x: float, mu: float, lambda_: float) -> float:
    return p + (lambda_ / (lambda_ + mu ** 2)) * (x - p)

def nlms_update(p: float, x: float, mu: float, lambda_: float) -> float:
    return p + (lambda_ / (lambda_ + mu ** 2)) * (x - p)

def lsm_vector(text: str) -> np.ndarray:
    return np.array([random.random() for _ in range(10)])

def bayes_marginal(pi: float, epsilon: float, phi: float) -> float:
    return pi * (1 - epsilon) + phi

def hybrid_edge_weight(d_ij: float, c_ij: float, p_i: float, p_j: float, v_i: np.ndarray, v_j: np.ndarray) -> float:
    l_ij = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j))
    pi_ij = (p_i + p_j) / (p_i + p_j + 1e-6)
    phi_ij = c_ij * 0.1
    m_ij = bayes_marginal(pi_ij, 1 - l_ij, phi_ij)
    return d_ij * (1 - m_ij) + 1e-6

def hybrid_weight(m: Entity, d_ij: float, c_ij: float, p_i: float, p_j: float, v_i: np.ndarray, v_j: np.ndarray) -> float:
    sphericity = sphericity_index(m.length, m.width, m.height)
    righting_time = righting_time_index(m)
    return sphericity * d_ij + righting_time * hybrid_edge_weight(d_ij, c_ij, p_i, p_j, v_i, v_j)

# ----------------------------------------------------------------------
# Hybrid function
# ----------------------------------------------------------------------
def hybrid_hybrid_hybrid_weight(m: Entity, e: tuple) -> float:
    (d_ij, c_ij, p_i, p_j, v_i, v_j) = e
    return hybrid_weight(m, d_ij, c_ij, p_i, p_j, v_i, v_j)

# ----------------------------------------------------------------------
# Prim-MST algorithm with hybrid edge weights
# ----------------------------------------------------------------------
def prim_mst(n: int, edges: list) -> list:
    visited = {i: False for i in range(n)}
    parent = {i: None for i in range(n)}
    distance = {i: float('inf') for i in range(n)}
    distance[0] = 0
    edges.sort(key=lambda x: x[2], reverse=True)
    while not all(visited.values()):
        for edge in edges:
            if not visited[edge[0]] and not visited[edge[1]] and distance[edge[0]] + edge[2] < distance[edge[1]]:
                distance[edge[1]] = distance[edge[0]] + edge[2]
                parent[edge[1]] = edge[0]
                visited[edge[1]] = True
    return parent

# ----------------------------------------------------------------------
# Hybrid function
# ----------------------------------------------------------------------
def hybrid_minimum_spanning_tree(m: Entity, edges: list) -> list:
    n = len(edges)
    return prim_mst(n, edges)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    m = Entity(id='e1', lat=1.0, lon=2.0, category='c1', mass=10.0)
    edges = [(0, 1, 1.0, 0.5, 0.0, np.array([1.0, 2.0])), (0, 2, 2.0, 0.5, 0.0, np.array([3.0, 4.0]))]
    parent = hybrid_minimum_spanning_tree(m, edges)
    print(parent)