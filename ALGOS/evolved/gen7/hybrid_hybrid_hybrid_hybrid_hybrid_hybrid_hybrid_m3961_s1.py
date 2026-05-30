# DARWIN HAMMER — match 3961, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m2176_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1465_s4.py (gen5)
# born: 2026-05-29T23:52:50Z

"""Hybrid Algorithm: Fusion of DARWIN HAMMER Parent A (graph‑based deduplication & curvature) 
and Parent B (VRAM‑aware regret‑weighted learning with trust‑scaled similarity).

Mathematical Bridge
------------------
Parent A supplies a **graph G = (V,E)** where each vertex represents a data element
and edges connect elements whose perceptual hashes differ by at most 4 bits.
From G we derive a **curvature scalar κ** (average clustering coefficient) that
characterises the geometric density of the similarity manifold.

Parent B supplies a **global step‑size multiplier**
    
    γ = η · h · s
    
with η = LR·w_regret (VRAM‑scaled learning‑rate), h ∈ [0,1] (trust scalar) and
s ∈ [‑1,1] (semantic similarity).  

The hybrid algorithm combines both topologies by using κ to modulate η
(denser graphs receive a smaller effective learning‑rate) and then applying
the scaled update to a vector field v₀ = x₁ – x₀.  The update is first
trust‑weighted, then rotated by a quaternion rotor Q (geometric product from
Parent A) to inject geometric flexibility, and finally multiplied by γ.
The result is an Euler‑style integration step that simultaneously respects
GPU‑memory constraints, graph curvature, cockpit trust and semantic
similarity.

The module provides three core functions that realise this fusion:
* `build_graph` – constructs G from raw float vectors using perceptual hashing.
* `compute_global_multiplier` – builds γ from VRAM budget, regret weight,
  trust, similarity and graph curvature.
* `hybrid_step` – performs the trust‑scaled, quaternion‑rotated, γ‑scaled
  Euler update on a state vector.
"""

import os
import sys
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Mapping, Hashable, Tuple, List, Dict, Set

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Parent A utilities (graph & curvature)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 64‑bit median threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()

def build_graph(elements: List[List[float]]) -> Graph:
    """
    Build an undirected graph where each node is identified by its index
    (as a string) and an edge exists if the perceptual hashes differ by ≤4 bits.
    """
    hashes: Dict[str, int] = {}
    for i, element in enumerate(elements):
        hashes[str(i)] = compute_phash(element)

    graph: Dict[str, Set[str]] = {}
    n = len(elements)
    for i in range(n):
        vi = str(i)
        graph.setdefault(vi, set())
        for j in range(i + 1, n):
            vj = str(j)
            if hamming_distance(hashes[vi], hashes[vj]) <= 4:
                graph[vi].add(vj)
                graph.setdefault(vj, set()).add(vi)
    return graph

def clustering_coefficient(node: str, g: Graph) -> float:
    """Local clustering coefficient of a node."""
    neighbors = g.get(node, set())
    k = len(neighbors)
    if k < 2:
        return 0.0
    links = 0
    for u in neighbors:
        for v in neighbors:
            if u != v and v in g.get(u, set()):
                links += 1
    # each edge counted twice
    return links / (k * (k - 1))

def compute_curvature(g: Graph) -> float:
    """
    Approximate curvature κ as the average clustering coefficient of the graph.
    Values lie in [0,1]; higher κ indicates a tighter similarity manifold.
    """
    if not g:
        return 0.0
    coeffs = [clustering_coefficient(node, g) for node in g]
    return sum(coeffs) / len(coeffs)

# ----------------------------------------------------------------------
# Parent B utilities (VRAM‑aware regret, trust, similarity)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def gpu_memory() -> Tuple[int, int]:
    """
    Mocked GPU memory query.
    Returns (free_mb, total_mb). In a real setting this would query the driver.
    """
    total_mb = 16384  # assume 16 GiB total
    free_mb = random.randint(1024, total_mb - DEFAULT_RESERVE_MB)
    return free_mb, total_mb

def regret_weighted_lr(base_lr: float, regret: float) -> float:
    """
    Scale the base learning rate by a regret factor and by the proportion of
    free VRAM to the allocated budget.
    """
    free_mb, _ = gpu_memory()
    vram_factor = min(1.0, free_mb / (DEFAULT_BUDGET_MB + DEFAULT_RESERVE_MB))
    return base_lr * regret * vram_factor

def trust_scalar(trust_metric: float) -> float:
    """Clamp a raw trust metric into [0,1]."""
    return max(0.0, min(1.0, trust_metric))

def similarity_factor(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity in [-1,1] between two LSM‑style vectors.
    """
    if np.allclose(vec_a, 0) or np.allclose(vec_b, 0):
        return 0.0
    dot = float(np.dot(vec_a, vec_b))
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return max(-1.0, min(1.0, dot / norm))

# ----------------------------------------------------------------------
# Quaternion utilities (geometric product from Parent A)
# ----------------------------------------------------------------------
def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Create a unit quaternion representing rotation around `axis` by `angle_rad`.
    Returns (w, x, y, z).
    """
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    w = math.cos(half)
    xyz = axis * math.sin(half)
    return np.array([w, *xyz], dtype=np.float64)

def quaternion_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vector `v` by unit quaternion `q` using Hamilton product.
    """
    w, x, y, z = q
    # Quaternion representation of vector
    qv = np.array([0.0, *v])
    # q * v * q_conj
    q_conj = np.array([w, -x, -y, -z])
    temp = _quat_mul(q, qv)
    rotated = _quat_mul(temp, q_conj)
    return rotated[1:]  # return vector part

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=np.float64)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_global_multiplier(
    base_lr: float,
    regret: float,
    trust_metric: float,
    lsm_vec_a: np.ndarray,
    lsm_vec_b: np.ndarray,
    graph: Graph,
) -> float:
    """
    Assemble the global step‑size multiplier γ = η·h·s,
    where η incorporates VRAM‑aware regret weighting and graph curvature κ.
    """
    # η from Parent B, attenuated by curvature κ (dense graphs → smaller η)
    η = regret_weighted_lr(base_lr, regret)
    κ = compute_curvature(graph)          # ∈ [0,1]
    η *= (1.0 - κ)                        # reduce learning when curvature high

    h = trust_scalar(trust_metric)        # trust scalar ∈ [0,1]
    s = similarity_factor(lsm_vec_a, lsm_vec_b)  # similarity ∈ [-1,1]

    γ = η * h * s
    return γ

def hybrid_step(
    state: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    base_lr: float,
    regret: float,
    trust_metric: float,
    lsm_vec_a: np.ndarray,
    lsm_vec_b: np.ndarray,
    graph: Graph,
    rotor_axis: np.ndarray,
    rotor_angle: float,
) -> np.ndarray:
    """
    Perform one hybrid Euler integration step:
        v₀ = x₁ – x₀
        v̂ = h·v₀                         (trust weighting)
        v̂ = Q·v̂·Q*                        (quaternion rotation)
        Δ = γ·v̂                           (global multiplier)
        state_{new} = state + Δ
    """
    # Base velocity
    v0 = x1 - x0

    # Trust‑scaled velocity
    h = trust_scalar(trust_metric)
    v_trust = h * v0

    # Quaternion rotation
    q = quaternion_from_axis_angle(rotor_axis, rotor_angle)
    v_rot = quaternion_rotate(v_trust, q)

    # Global multiplier
    γ = compute_global_multiplier(
        base_lr, regret, trust_metric, lsm_vec_a, lsm_vec_b, graph
    )

    # Euler update
    delta = γ * v_rot
    return state + delta

def deduplicate_and_update(
    elements: List[List[float]],
    state: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    base_lr: float = 0.01,
    regret: float = 1.0,
    trust_metric: float = 0.8,
) -> Tuple[Graph, np.ndarray]:
    """
    High‑level helper that builds the similarity graph from `elements`,
    synthesises two random LSM vectors, chooses a random rotor, and returns
    the updated state together with the graph.
    """
    # 1. Build similarity graph (Parent A)
    graph = build_graph(elements)

    # 2. Mock LSM vectors (semantic embeddings)
    dim = 128
    lsm_a = np.random.randn(dim)
    lsm_b = np.random.randn(dim)

    # 3. Random rotor parameters
    axis = np.random.randn(3)
    angle = random.uniform(0, 2 * math.pi)

    # 4. Perform hybrid update
    new_state = hybrid_step(
        state=state,
        x0=x0,
        x1=x1,
        base_lr=base_lr,
        regret=regret,
        trust_metric=trust_metric,
        lsm_vec_a=lsm_a,
        lsm_vec_b=lsm_b,
        graph=graph,
        rotor_axis=axis,
        rotor_angle=angle,
    )
    return graph, new_state

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 20 elements, each 50‑dimensional float vector
    elems = [list(np.random.rand(50) * 255) for _ in range(20)]

    # Initial state vector (e.g., model parameters) – 3‑dimensional for rotation demo
    init_state = np.zeros(3)

    # Positions in a hypothetical space
    pos0 = np.array([0.0, 0.0, 0.0])
    pos1 = np.array([1.0, 0.5, -0.2])

    g, updated = deduplicate_and_update(
        elements=elems,
        state=init_state,
        x0=pos0,
        x1=pos1,
        base_lr=0.02,
        regret=1.5,
        trust_metric=0.9,
    )

    print("Graph nodes:", len(g))
    print("Average degree:", sum(len(v) for v in g.values()) / len(g) if g else 0)
    print("Curvature κ:", compute_curvature(g))
    print("Updated state:", updated)
    print("Timestamp:", now_z())