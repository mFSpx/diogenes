# DARWIN HAMMER — match 2949, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_minimum_cost__hybrid_jepa_energy_h_m1737_s1.py (gen4)
# parent_b: hybrid_hybrid_semantic_neig_hybrid_perceptual_de_m806_s0.py (gen3)
# born: 2026-05-29T23:46:58Z

import numpy as np
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """A single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class Point:
    """2‑D coordinate used for both geometric and semantic calculations."""
    x: float
    y: float


class Morphology:
    """Compact container for physical attributes."""
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology dimensions must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


# ----------------------------------------------------------------------
# Morphology‑derived indices
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity – ratio of the geometric mean to the longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness – how spread the object is relative to its thickness."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """
    Physical righting‑time proxy.
    Larger values indicate slower self‑recovery.
    """
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority in [0,1] derived from morphology.
    Higher priority → more urgent recovery.
    """
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Count‑Min sketch (privacy‑aware) utilities
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Simple Count‑Min sketch with Laplace DP noise.
    The sketch is used to approximate the reconstruction‑risk *r*.
    """
    def __init__(self,
                 width: int = 2_048,
                 depth: int = 5,
                 epsilon: float = 0.1):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.width = width
        self.depth = depth
        self.epsilon = epsilon
        self.tables = np.zeros((depth, width), dtype=np.float64)
        # Independent hash seeds for each row
        self._seeds = [random.randint(0, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, key: str, row: int) -> int:
        rng = np.random.RandomState(self._seeds[row])
        # deterministic but simple hash – combine Python's hash with row seed
        return (hash(key) ^ self._seeds[row]) % self.width

    def add(self, key: str, count: float = 1.0) -> None:
        for row in range(self.depth):
            col = self._hash(key, row)
            self.tables[row, col] += count

    def estimate(self, key: str) -> float:
        """Return the DP‑noised minimum count across rows."""
        estimates = []
        for row in range(self.depth):
            col = self._hash(key, row)
            noisy = self.tables[row, col] + np.random.laplace(scale=1.0 / self.epsilon)
            estimates.append(noisy)
        return min(estimates)


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
class HybridBanditTree:
    """
    Integrates a contextual bandit, morphology‑derived recovery priority,
    semantic similarity and a DP‑aware reconstruction‑risk.
    """
    def __init__(self,
                 dp_epsilon: float = 0.1,
                 path_weight: float = 0.2,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 sketch_width: int = 2_048,
                 sketch_depth: int = 5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0,1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must lie in [0,1]")
        self._policy: Dict[str, List[float]] = {}
        self.dp_epsilon = dp_epsilon
        self.path_weight = path_weight
        self.alpha = alpha
        self.beta = beta
        self.sketch = CountMinSketch(width=sketch_width,
                                     depth=sketch_depth,
                                     epsilon=dp_epsilon)

    # ------------------------------------------------------------------
    # Policy utilities
    # ------------------------------------------------------------------
    def reset_policy(self) -> None:
        """Erase all learned statistics."""
        self._policy.clear()

    def _reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n else 0.0

    def _count(self, action: str) -> float:
        return self._policy.get(action, [0.0, 0.0])[1]

    def update_policy(self, updates: Iterable[BanditUpdate]) -> None:
        """Incrementally incorporate new bandit observations."""
        for u in updates:
            stats = self._policy.setdefault(u.action_id, [0.0, 0.0])
            stats[0] += float(u.reward)
            stats[1] += 1.0
            # Also feed the update into the DP sketch – the key can be any
            # deterministic identifier of the (context,action) pair.
            self.sketch.add(f"{u.context_id}:{u.action_id}", count=1.0)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _euclidean(a: Point, b: Point) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    # ------------------------------------------------------------------
    # Semantic similarity – robust to zero vectors
    # ------------------------------------------------------------------
    @staticmethod
    def semantic_similarity(a: Point, b: Point, eps: float = 1e-12) -> float:
        """Cosine similarity between two points treated as vectors."""
        vec_a = np.array([a.x, a.y], dtype=np.float64)
        vec_b = np.array([b.x, b.y], dtype=np.float64)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < eps or norm_b < eps:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Reconstruction risk – DP‑aware estimate from the sketch
    # ------------------------------------------------------------------
    def reconstruction_risk(self, a: Point, b: Point) -> float:
        """
        Returns a risk value in [0,1] based on how often the (a,b) edge
        has been observed. The Count‑Min sketch provides a DP‑noised
        frequency; we map it to a probability via a simple sigmoid.
        """
        key = f"{a.x:.4f},{a.y:.4f}|{b.x:.4f},{b.y:.4f}"
        raw = self.sketch.estimate(key)
        # Clip to a reasonable range before sigmoid to avoid overflow
        raw = max(-30.0, min(30.0, raw))
        return 1.0 / (1.0 + math.exp(-raw))

    # ------------------------------------------------------------------
    # Core cost computation
    # ------------------------------------------------------------------
    def tree_cost(self,
                  nodes: Dict[str, Point],
                  edges: List[Tuple[str, str]],
                  root: str,
                  morphology: Dict[str, Morphology],
                  updates: Optional[Iterable[BanditUpdate]] = None) -> float:
        """
        Compute the hybrid cost of a spanning tree.
        The cost consists of three components:
          1. Material length (pure Euclidean sum)
          2. Path‑weight penalty proportional to node depths
          3. Semantic‑morphology‑privacy hybrid term.
        """
        if updates:
            self.update_policy(updates)

        # ------------------------------------------------------------------
        # Build adjacency list and compute material length
        # ------------------------------------------------------------------
        adj: Dict[str, List[str]] = {node: [] for node in nodes}
        material = 0.0
        for u, v in edges:
            if u not in nodes or v not in nodes:
                raise KeyError(f"Edge ({u},{v}) references undefined node")
            adj[u].append(v)
            adj[v].append(u)
            material += self._euclidean(nodes[u], nodes[v])

        # ------------------------------------------------------------------
        # BFS to obtain depth (distance from root) for path‑weight term
        # ------------------------------------------------------------------
        depth: Dict[str, float] = {root: 0.0}
        stack = [root]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in depth:
                    depth[nxt] = depth[cur] + self._euclidean(nodes[cur], nodes[nxt])
                    stack.append(nxt)

        # ------------------------------------------------------------------
        # Hybrid semantic‑morphology‑privacy term
        # ------------------------------------------------------------------
        hybrid_term = 0.0
        for u, v in edges:
            # Semantic similarity (c)
            c = self.semantic_similarity(nodes[u], nodes[v])

            # Morphology‑derived priority (p) – we use the *target* node's priority
            p = recovery_priority(morphology[v])

            # Reconstruction risk (r) – DP‑aware
            r = self.reconstruction_risk(nodes[u], nodes[v])

            # Integrated neighbor score h(i,j) as defined in the spec
            h = self.alpha * c + (1.0 - self.alpha) * p * (1.0 - r)

            hybrid_term += self.beta * h

        total_cost = material + self.path_weight * sum(depth.values()) + hybrid_term
        return total_cost


# ----------------------------------------------------------------------
# Demonstration / simple sanity check
# ----------------------------------------------------------------------
def _demo() -> None:
    nodes = {
        "A": Point(0.0, 0.0),
        "B": Point(3.0, 4.0),
        "C": Point(6.0, 8.0)
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    morphology = {
        "A": Morphology(1.0, 2.0, 3.0, 4.0),
        "B": Morphology(5.0, 6.0, 7.0, 8.0),
        "C": Morphology(9.0, 10.0, 11.0, 12.0)
    }
    updates = [
        BanditUpdate("ctx1", "A", 1.0, 0.5),
        BanditUpdate("ctx2", "B", 2.0, 0.6)
    ]

    hybrid = HybridBanditTree(dp_epsilon=0.2, alpha=0.6, beta=0.4)
    cost = hybrid.tree_cost(nodes, edges, root, morphology, updates)
    print("Hybrid tree cost:", cost)


if __name__ == "__main__":
    _demo()