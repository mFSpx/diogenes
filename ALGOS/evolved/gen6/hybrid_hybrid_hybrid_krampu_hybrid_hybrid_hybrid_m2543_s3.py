# DARWIN HAMMER — match 2543, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_sketches_hybr_m43_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s0.py (gen5)
# born: 2026-05-29T23:42:51Z

"""Hybrid Krampus‑Brain‑Pheromone‑Sheaf Module
Combines:
- Parent A: Krampus master‑vector graph, Ollivier‑Ricci curvature, 3‑D projection,
  count‑min sketch, contextual bandit selector.
- Parent B: Feature‑based pheromone signals with exponential decay, sheaf‑cohomology
  aggregation, and curvature‑aware time‑aware node metrics.

Mathematical bridge:
Both parents operate on the same semantic graph G=(V,E) built from master vectors.
Parent A supplies a scalar curvature κᵢ per node; Parent B supplies a
time‑decaying pheromone value ϕᵢ and a sheaf‑consistency error εᵢ.
We fuse them into an augmented scalar
    ηᵢ = α·κᵢ − β·εᵢ + γ·ϕᵢ,
which is injected as a third coordinate in the linear projection
    pᵢ = L·vᵢ + [0, 0, ηᵢ] .
The set {pᵢ} is hashed into a count‑min sketch, providing a compact
frequency estimate that is later mixed with bandit reward and confidence
terms for action selection.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple

import numpy as np
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Core graph utilities (Parent A)

def build_adjacency(vectors: np.ndarray, eps: float) -> Dict[int, List[int]]:
    """Build an undirected adjacency list where Euclidean distance < eps."""
    n = vectors.shape[0]
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            if np.linalg.norm(vi - vectors[j]) < eps:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def compute_curvature(vectors: np.ndarray, adj: Dict[int, List[int]], eps: float) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature per node as the average
    (1 - d_ij/eps) over incident edges.  This yields κ∈[0,1].
    """
    n = vectors.shape[0]
    curvature = np.zeros(n, dtype=float)
    for i, neighbors in adj.items():
        if not neighbors:
            curvature[i] = 0.0
            continue
        contrib = 0.0
        vi = vectors[i]
        for j in neighbors:
            dij = np.linalg.norm(vi - vectors[j])
            contrib += max(0.0, 1.0 - dij / eps)
        curvature[i] = contrib / len(neighbors)
    return curvature

# ---------------------------------------------------------------------------
# Pheromone & Sheaf machinery (Parent B)

@dataclass
class PheromoneEntry:
    feature: str
    value: float
    half_life: float
    signal: float = 0.0

    def __post_init__(self):
        self.signal = self.value

    def decay(self, dt: float) -> None:
        """Exponential decay with half‑life."""
        decay_factor = 0.5 ** (dt / self.half_life)
        self.signal *= decay_factor


class HybridSheaf:
    """Simple sheaf on the graph where each node holds a scalar section."""

    def __init__(self, node_ids: List[int], edges: List[Tuple[int, int]]):
        self.nodes = set(node_ids)
        self.edges = edges
        self.sections: Dict[int, float] = {i: 0.0 for i in node_ids}
        # Restriction maps are identity for scalar sections
        self.restrictions: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def set_section(self, node: int, value: float) -> None:
        self.sections[node] = float(value)

    def coboundary_error(self) -> Dict[int, float]:
        """
        For each node, sum squared differences with neighbor sections.
        Returns a per‑node error εᵢ.
        """
        error: Dict[int, float] = {i: 0.0 for i in self.nodes}
        for u, v in self.edges:
            diff = self.sections[u] - self.sections[v]
            err = diff * diff
            error[u] += err
            error[v] += err
        # Normalize by degree to keep scale comparable
        for i in self.nodes:
            deg = max(1, len([e for e in self.edges if i in e]))
            error[i] /= deg
        return error

# ---------------------------------------------------------------------------
# Count‑Min Sketch (Parent A)

class CountMinSketch:
    """Deterministic count‑min sketch using hashlib.sha256."""

    def __init__(self, width: int = 256, depth: int = 4):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seeds = [i * 0x9e3779b1 for i in range(depth)]

    def _hash(self, item: str, d: int) -> int:
        h = hashlib.sha256()
        h.update(item.encode('utf-8'))
        h.update(self.seeds[d].to_bytes(4, byteorder='little', signed=False))
        return int.from_bytes(h.digest()[:4], byteorder='little') % self.width

    def add(self, item: str, count: int = 1) -> None:
        for d in range(self.depth):
            idx = self._hash(item, d)
            self.table[d, idx] += count

    def estimate(self, item: str) -> int:
        mins = []
        for d in range(self.depth):
            idx = self._hash(item, d)
            mins.append(self.table[d, idx])
        return min(mins)

# ---------------------------------------------------------------------------
# Hybrid projection & bandit selector (fusion of both parents)

def hybrid_projection(
    vectors: np.ndarray,
    curvature: np.ndarray,
    pheromone_signal: np.ndarray,
    sheaf_error: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Produce 3‑D points pᵢ = L·vᵢ + [0,0, ηᵢ],
    where ηᵢ = α·κᵢ − β·εᵢ + γ·ϕᵢ.
    L is a random (3×d) matrix with fixed seed for reproducibility.
    """
    rng = np.random.default_rng(42)
    d = vectors.shape[1]
    L = rng.standard_normal((3, d))
    base = L @ vectors.T  # shape (3, n)
    eta = alpha * curvature - beta * sheaf_error + gamma * pheromone_signal
    points = base.T.copy()
    points[:, 2] += eta  # inject fused scalar into z‑coordinate
    return points  # shape (n,3)


def build_sketch_from_points(points: np.ndarray, width: int = 256, depth: int = 4) -> CountMinSketch:
    """Hash each 3‑D point (rounded to 4 decimals) into a count‑min sketch."""
    sketch = CountMinSketch(width=width, depth=depth)
    for p in points:
        key = f"{p[0]:.4f}_{p[1]:.4f}_{p[2]:.4f}"
        sketch.add(key, count=1)
    return sketch


def select_action_ucb(
    node_ids: List[int],
    rewards: Dict[int, float],
    pulls: Dict[int, int],
    total_pulls: int,
    sketch: CountMinSketch,
    points: np.ndarray,
    c: float = 2.0,
    sketch_weight: float = 0.5,
) -> int:
    """
    Upper‑confidence‑bound bandit that mixes:
    - empirical reward,
    - confidence term sqrt(c·log(total)/pulls),
    - sketch frequency term proportional to sketch.estimate(key).
    Returns the selected node id.
    """
    best_id = None
    best_score = -math.inf
    for idx, node in enumerate(node_ids):
        r = rewards.get(node, 0.0)
        n = pulls.get(node, 0) + 1e-9  # avoid div‑zero
        ucb = r + math.sqrt(c * math.log(max(1, total_pulls)) / n)
        # sketch term
        p = points[idx]
        key = f"{p[0]:.4f}_{p[1]:.4f}_{p[2]:.4f}"
        freq = sketch.estimate(key)
        score = ucb + sketch_weight * math.log(1 + freq)
        if score > best_score:
            best_score = score
            best_id = node
    return best_id  # type: ignore

# ---------------------------------------------------------------------------
# End‑to‑end hybrid pipeline

def hybrid_pipeline(vectors: np.ndarray, eps: float = 1.0) -> Tuple[int, np.ndarray]:
    """
    Executes the full hybrid workflow:
    1. Build graph & curvature (Parent A).
    2. Initialise pheromones from vector norms (Parent B).
    3. Decay pheromones for a single time step.
    4. Build sheaf sections = decayed pheromone signals.
    5. Compute sheaf inconsistency ε.
    6. Project to 3‑D using fused scalar η.
    7. Insert points into a count‑min sketch.
    8. Run a UCB bandit to pick an action (node id).
    Returns the selected node id and the point matrix.
    """
    n = vectors.shape[0]

    # 1. Graph & curvature
    adj = build_adjacency(vectors, eps)
    curvature = compute_curvature(vectors, adj, eps)

    # 2. Pheromone init (value = norm, half‑life proportional to entropy)
    pheromones: List[PheromoneEntry] = []
    for i, v in enumerate(vectors):
        norm = float(np.linalg.norm(v))
        probs = np.abs(v) / (np.abs(v).sum() + 1e-12)
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        half_life = 1.0 + entropy  # slower decay for high entropy
        pheromones.append(PheromoneEntry(feature=f"node_{i}", value=norm, half_life=half_life))

    # 3. Decay for a fixed Δt = 1.0
    for p in pheromones:
        p.decay(dt=1.0)

    pheromone_signal = np.array([p.signal for p in pheromones])

    # 4. Sheaf sections = pheromone signals (scalar)
    edges = [(u, v) for u, neigh in adj.items() for v in neigh if u < v]
    sheaf = HybridSheaf(node_ids=list(range(n)), edges=edges)
    for i, sig in enumerate(pheromone_signal):
        sheaf.set_section(i, sig)

    # 5. Sheaf inconsistency ε
    sheaf_error_dict = sheaf.coboundary_error()
    sheaf_error = np.array([sheaf_error_dict[i] for i in range(n)])

    # 6. Hybrid projection
    points = hybrid_projection(vectors, curvature, pheromone_signal, sheaf_error)

    # 7. Sketch
    sketch = build_sketch_from_points(points)

    # 8. Bandit selection
    rewards = {i: random.random() for i in range(n)}  # mock reward estimates
    pulls = {i: random.randint(1, 10) for i in range(n)}
    total_pulls = sum(pulls.values())
    selected = select_action_ucb(
        node_ids=list(range(n)),
        rewards=rewards,
        pulls=pulls,
        total_pulls=total_pulls,
        sketch=sketch,
        points=points,
    )
    return selected, points

# ---------------------------------------------------------------------------
# Smoke test

if __name__ == "__main__":
    # Generate a small synthetic corpus of 12 texts represented as 5‑D vectors
    rng = np.random.default_rng(123)
    master_vectors = rng.standard_normal((12, 5))
    chosen_node, pts = hybrid_pipeline(master_vectors, eps=2.5)
    print(f"Selected node id: {chosen_node}")
    print("First three projected points (x, y, z):")
    for i, p in enumerate(pts[:3]):
        print(f"  node {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")