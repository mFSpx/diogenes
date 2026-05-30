# DARWIN HAMMER — match 812, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py (gen3)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s5.py (gen2)
# born: 2026-05-29T23:31:08Z

"""Hybrid Algorithm: deterministic feature extraction + ternary minimum‑cost routing

Parents:
- **hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py** (Algorithm A)
  Provides a deterministic pseudo‑random feature vector for any text and a reduced
  “master vector”.  Mathematically this is a mapping  f : Σ* → ℝⁿ that can be used
  as a prior distribution in Bayesian updates.

- **hybrid_ternary_router_hybrid_minimum_cost__m36_s5.py** (Algorithm B)
  Supplies utilities for timestamps, JSON emission and geometric helpers for a
  ternary router that builds a minimum‑cost graph (essentially a Minimum Spanning
  Tree, MST) over points in the plane.

**Mathematical Bridge**

Both algorithms operate on *objects* that can be represented as vectors:
- A’s master vector **vᵢ** ∈ ℝⁿ (feature space).
- B’s geometric point **pᵢ** ∈ ℝ² (spatial space).

The hybrid defines a composite state **sᵢ = (pᵢ, vᵢ)** for each text *i*.
Edge cost between nodes *i* and *j* is a weighted sum of Euclidean distance in
ℝ² and Euclidean distance in feature space ℝⁿ:

c_{ij} = α·‖pᵢ−pⱼ‖₂ + β·‖vᵢ−vⱼ‖₂ , α,β ≥ 0.

The MST built on these costs yields a minimal‑cost routing structure.
A Bayesian update then fuses prior node probabilities with likelihoods derived
from the MST (e.g., degree‑based likelihood).  The result is a posterior
distribution over nodes that respects both spatial and semantic similarity.

The code below implements this hybrid pipeline with three core functions:
`extract_master_vector`, `compute_edge_cost`, and `hybrid_route_mst_bayes`.
"""

import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Deterministic feature extraction (Algorithm A)
# ----------------------------------------------------------------------
def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit integer hash for *text* using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a reproducible pseudo‑random feature vector from *text*.
    The same input always yields the same output across Python runs.
    """
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Reduce the full feature set to a compact *master vector*.
    The selection mirrors the original implementation but remains deterministic.
    """
    f = extract_full_features(text)
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "tech_ratio": f["operator_tech_ratio"],
        "legal_osint_ratio": f["operator_legal_osint_ratio"],
        "forensic_shield_ratio": f["psyche_forensic_shield_ratio"],
        "poetic_entropy": f["psyche_poetic_entropy"],
        "dissociative_index": f["psyche_dissociative_index"],
        "bureaucratic_weaponization_index": f[
            "resilience_bureaucratic_weaponization_index"
        ],
        "resource_exhaustion_metric": f["resilience_resource_exhaustion_metric"],
        "swarm_orchestration_density": f["resilience_swarm_orchestration_density"],
        "corporate_grit_tension": f["rainmaker_corporate_grit_tension"],
        "countdown_density": f["rainmaker_countdown_density"],
        "asset_structuring_weight": f["rainmaker_asset_structuring_weight"],
        "agent_symmetry_ratio": f["telemetry_agent_symmetry_ratio"],
        "protocol_discipline": f["telemetry_protocol_discipline"],
        "manic_velocity": f["telemetry_manic_velocity"],
    }

# ----------------------------------------------------------------------
# Geometry & routing utilities (Algorithm B)
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def emit_json(obj: Any) -> None:
    """Print a JSON object in a deterministic order."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


Point = Tuple[float, float]
EdgeKey = Tuple[int, int]  # (node_index_a, node_index_b)


def _hash_to_point(text: str) -> Point:
    """
    Deterministically map *text* to a point in the unit square.
    Uses the first 64 bits of SHA‑256 as two 32‑bit integers.
    """
    h = _deterministic_hash(text)
    x = (h >> 32) & 0xFFFFFFFF
    y = h & 0xFFFFFFFF
    return (x / 0xFFFFFFFF, y / 0xFFFFFFFF)


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard L2 distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def feature_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """L2 distance between two feature vectors."""
    return np.linalg.norm(v1 - v2)


def compute_edge_cost(
    p_a: Point,
    p_b: Point,
    v_a: np.ndarray,
    v_b: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Hybrid edge cost c_{ab} = α·‖p_a−p_b‖₂ + β·‖v_a−v_b‖₂.
    α and β control the relative influence of geometry vs. semantics.
    """
    geom = euclidean_distance(p_a, p_b)
    feat = feature_distance(v_a, v_b)
    return alpha * geom + beta * feat


# ----------------------------------------------------------------------
# Minimum‑Cost (MST) construction using Kruskal's algorithm
# ----------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


def kruskal_mst(
    points: List[Point],
    features: List[np.ndarray],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> List[EdgeKey]:
    """
    Build a Minimum Spanning Tree over nodes defined by (point, feature).
    Returns a list of edges (node indices) that constitute the MST.
    """
    n = len(points)
    if n < 2:
        return []

    # Build all possible edges with their costs
    edges: List[Tuple[float, EdgeKey]] = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = compute_edge_cost(points[i], points[j], features[i], features[j], alpha, beta)
            edges.append((cost, (i, j)))

    # Sort edges by ascending cost
    edges.sort(key=lambda x: x[0])

    uf = _UnionFind(n)
    mst: List[EdgeKey] = []
    for cost, (i, j) in edges:
        if uf.union(i, j):
            mst.append((i, j))
            if len(mst) == n - 1:
                break
    return mst


# ----------------------------------------------------------------------
# Bayesian update utilities
# ----------------------------------------------------------------------
def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform a Bayesian update: posterior ∝ prior * likelihood.
    Both inputs must be 1‑D arrays of the same length.
    Returns a normalized posterior distribution.
    """
    if prior.shape != likelihood.shape:
        raise ValueError("Prior and likelihood must have the same shape")
    unnorm = prior * likelihood
    total = unnorm.sum()
    if total == 0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


def degree_likelihood(mst: List[EdgeKey], n_nodes: int) -> np.ndarray:
    """
    Simple likelihood derived from node degree in the MST.
    Higher degree ⇒ higher likelihood (more central).
    """
    degrees = np.zeros(n_nodes, dtype=float)
    for i, j in mst:
        degrees[i] += 1.0
        degrees[j] += 1.0
    # Add a small epsilon to avoid zero likelihoods
    epsilon = 1e-6
    return degrees + epsilon


# ----------------------------------------------------------------------
# Hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_route_mst_bayes(
    texts: List[str],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[List[EdgeKey], np.ndarray]:
    """
    1. Convert each *text* into a deterministic point and a master feature vector.
    2. Build a minimum‑cost spanning tree using the hybrid edge cost.
    3. Treat node degree in the tree as a likelihood and update a uniform prior.
    Returns the MST edge list and the posterior probability vector over nodes.
    """
    if not texts:
        raise ValueError("Input list 'texts' must contain at least one element")

    # 1️⃣ Deterministic geometry & semantics
    points: List[Point] = [_hash_to_point(t) for t in texts]
    master_dicts: List[Dict[str, float]] = [extract_master_vector(t) for t in texts]
    # Convert dicts to dense numpy arrays (preserve order of keys)
    feature_keys = sorted(master_dicts[0].keys())
    features: List[np.ndarray] = [
        np.array([d[k] for k in feature_keys], dtype=float) for d in master_dicts
    ]

    # 2️⃣ Minimum‑cost spanning tree
    mst = kruskal_mst(points, features, alpha=alpha, beta=beta)

    # 3️⃣ Bayesian posterior
    prior = np.full(len(texts), 1.0 / len(texts), dtype=float)
    likelihood = degree_likelihood(mst, len(texts))
    posterior = bayesian_update(prior, likelihood)

    return mst, posterior


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Quantum entanglement in neural networks",
        "Legal ramifications of AI‑generated art",
        "Swarm robotics for disaster response",
        "Poetic entropy and the human condition",
        "Resource exhaustion in cloud infrastructures",
    ]

    mst_edges, posterior = hybrid_route_mst_bayes(sample_texts, alpha=0.6, beta=0.4)

    emit_json(
        {
            "timestamp": now_z(),
            "mst_edges": mst_edges,
            "posterior": posterior.tolist(),
        }
    )