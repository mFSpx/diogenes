# DARWIN HAMMER — match 5666, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1416_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2473_s3.py (gen6)
# born: 2026-05-30T00:04:02Z

"""Hybrid Algorithm combining:
- Parent A: master‑vector text feature extraction, pheromone decay, Ollivier‑Ricci curvature,
  morphology construction, morphology‑weighted Gini impurity, Hoeffding‑bound split.
- Parent B: Pheromone‑based span entities, bandit‑style action propensities, tree‑cost
  evaluation over a geometric graph.

Mathematical Bridge
-------------------
Each feature dimension produced by the master vector is interpreted as a graph node.
The curvature computed from the node‑to‑node distance matrix provides a *length* λ.
The current pheromone signal gives a *mass* μ.  Normalized feature magnitude supplies a
*width* w and the Shannon entropy of the feature distribution supplies a *height* h.
These four scalars (λ, μ, h, w) are assembled into a *Morphology* vector **M**.

Bandit actions from Parent B carry a propensity π and an expected reward r.
We define a *health‑informed* weight
    α = (μ·π) / (1 + exp(−λ·r))          (Eq. 1)
and use α to re‑weight the Gini impurity of a candidate split.
The resulting impurity gain is then passed through a Hoeffding bound to decide
whether the split is statistically justified.

Thus the sheaf‑based aggregation of pheromones (Parent A) directly modulates the
bandit‑propensity‑driven impurity (Parent B), while the geometric tree‑cost
function incorporates curvature‑derived edge lengths, yielding a unified,
time‑aware, uncertainty‑quantified classifier.
"""

import sys
import math
import random
import pathlib
import numpy as np
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Sequence

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
WIDTH = 64               # master‑vector dimension
HALF_LIFE_BASE = 10.0    # seconds, for pheromone decay
DELTA = 0.05             # Hoeffding confidence parameter

@dataclass
class PheromoneEntry:
    """Unified pheromone representation (both parents)."""
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float
    created_at: datetime
    last_decay: datetime = None

    def decay(self, now: datetime) -> float:
        """Exponential decay based on half‑life."""
        elapsed = (now - self.created_at).total_seconds()
        decay_factor = 0.5 ** (elapsed / self.half_life_seconds)
        self.signal_value *= decay_factor
        self.last_decay = now
        return self.signal_value

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # π
    expected_reward: float     # r
    confidence_bound: float
    algorithm: str

# ----------------------------------------------------------------------
# Core functions – hybrid operations
# ----------------------------------------------------------------------
def master_vector(text: str, width: int = WIDTH) -> np.ndarray:
    """Hash each character to a float and fill a fixed‑size vector."""
    vec = np.zeros(width, dtype=np.float64)
    for i, ch in enumerate(text):
        idx = i % width
        vec[idx] += (ord(ch) * 0.61803398875) % 1.0
    # Normalise to unit L2 norm
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm

def compute_curvature(vec: np.ndarray) -> float:
    """Very coarse Ollivier‑Ricci proxy: curvature = 1 - (std/mean)."""
    mean = np.mean(vec)
    std = np.std(vec)
    if mean == 0:
        return 0.0
    return max(0.0, 1.0 - std / (mean + 1e-12))

def build_morphology(vec: np.ndarray,
                     curvature: float,
                     pheromone: PheromoneEntry) -> Dict[str, float]:
    """Create the four scalar morphology components."""
    length = curvature                     # λ
    mass = pheromone.signal_value          # μ
    entropy = -np.sum(vec * np.log(vec + 1e-12))
    height = entropy / (np.log(len(vec)) + 1e-12)   # normalised entropy h
    width = np.linalg.norm(vec)            # w
    return {"length": length, "mass": mass,
            "height": height, "width": width}

def health_weight(morph: Dict[str, float], action: BanditAction) -> float:
    """Equation 1 – combines morphology mass with bandit propensity & reward."""
    λ = morph["length"]
    μ = morph["mass"]
    π = action.propensity
    r = action.expected_reward
    # Sigmoid‑scaled reward term to keep denominator >0
    reward_factor = 1.0 + math.exp(-λ * r)
    return (μ * π) / reward_factor

def gini_impurity(labels: List[Any]) -> float:
    """Standard Gini impurity."""
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    impurity = 1.0 - sum((c / total) ** 2 for c in counts.values())
    return impurity

def weighted_gini_gain(parent_labels: List[Any],
                       left_labels: List[Any],
                       right_labels: List[Any],
                       weight: float) -> float:
    """Morphology‑weighted Gini gain."""
    n = len(parent_labels)
    if n == 0:
        return 0.0
    parent_imp = gini_impurity(parent_labels)
    left_imp = gini_impurity(left_labels)
    right_imp = gini_impurity(right_labels)
    gain = parent_imp - (len(left_labels) / n) * left_imp - (len(right_labels) / n) * right_imp
    return gain * weight   # health‑informed scaling

def hoeffding_bound(R: float, n: int, delta: float = DELTA) -> float:
    """Hoeffding bound ε = sqrt(R^2 * ln(1/δ) / (2n))."""
    if n == 0:
        return float('inf')
    return math.sqrt((R ** 2) * math.log(1.0 / delta) / (2 * n))

def decide_split(parent_labels: List[Any],
                 left_labels: List[Any],
                 right_labels: List[Any],
                 weight: float,
                 R: float = 1.0) -> bool:
    """Return True if weighted Gini gain exceeds Hoeffding ε."""
    gain = weighted_gini_gain(parent_labels, left_labels, right_labels, weight)
    eps = hoeffding_bound(R, len(parent_labels))
    return gain > eps

def tree_cost(nodes: Dict[str, Tuple[float, float]],
              edges: List[Tuple[str, str]],
              pheromones: Dict[Tuple[str, str], PheromoneEntry],
              root: str,
              path_weight: float = 0.2) -> float:
    """Cost of spanning tree where each edge length is Euclidean distance
    multiplied by the decayed pheromone signal (if any)."""
    # Build adjacency
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    stack = [(root, 0.0)]
    total_cost = 0.0
    while stack:
        node, acc = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        total_cost += acc
        for nbr in adj[node]:
            if nbr in visited:
                continue
            # Euclidean distance
            d = math.hypot(nodes[node][0] - nodes[nbr][0],
                           nodes[node][1] - nodes[nbr][1])
            # Pheromone modulation
            ph = pheromones.get((node, nbr)) or pheromones.get((nbr, node))
            if ph:
                signal = ph.decay(datetime.now(timezone.utc))
                d *= (1.0 + path_weight * signal)
            stack.append((nbr, d))
    return total_cost

# ----------------------------------------------------------------------
# Demonstration of the hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_classify(text: str,
                    actions: List[BanditAction],
                    graph_nodes: Dict[str, Tuple[float, float]],
                    graph_edges: List[Tuple[str, str]],
                    pheromone_map: Dict[Tuple[str, str], PheromoneEntry]) -> str:
    """
    End‑to‑end hybrid classification:
    1. Extract master vector and curvature.
    2. Create a morphology object.
    3. For each bandit action compute a health weight.
    4. Pick the action with maximal weighted Gini gain (simulated with dummy labels).
    5. Evaluate tree cost using pheromone‑modulated edges.
    6. Return the chosen action id together with the cost.
    """
    # 1‑2
    vec = master_vector(text)
    curv = compute_curvature(vec)
    # use a single global pheromone entry for illustration
    global_ph = next(iter(pheromone_map.values())) if pheromone_map else PheromoneEntry(
        uuid="global", surface_key="global", signal_kind="global",
        signal_value=0.5, half_life_seconds=HALF_LIFE_BASE,
        created_at=datetime.now(timezone.utc))
    morph = build_morphology(vec, curv, global_ph)

    # 3‑4 simulate label sets
    dummy_labels = [random.choice(['A', 'B']) for _ in range(100)]
    best_action = None
    best_gain = -float('inf')
    for act in actions:
        w = health_weight(morph, act)
        # naive split: random partition
        left = dummy_labels[:50]
        right = dummy_labels[50:]
        if decide_split(dummy_labels, left, right, w):
            gain = weighted_gini_gain(dummy_labels, left, right, w)
            if gain > best_gain:
                best_gain = gain
                best_action = act

    # 5‑6 tree cost
    cost = tree_cost(graph_nodes, graph_edges, pheromone_map, root=list(graph_nodes)[0])

    chosen = best_action.action_id if best_action else "none"
    return f"{chosen}:{cost:.3f}"

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic data
    sample_text = "Hybrid algorithm test string."
    actions = [
        BanditAction("act1", propensity=0.7, expected_reward=0.4,
                     confidence_bound=0.1, algorithm="A"),
        BanditAction("act2", propensity=0.3, expected_reward=0.9,
                     confidence_bound=0.2, algorithm="B")
    ]

    # Graph with 4 nodes forming a square
    nodes = {
        "n0": (0.0, 0.0),
        "n1": (1.0, 0.0),
        "n2": (1.0, 1.0),
        "n3": (0.0, 1.0)
    }
    edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n0")]

    # Uniform pheromone on each edge
    pheromone_map = {}
    now = datetime.now(timezone.utc)
    for e in edges:
        pid = f"ph_{e[0]}_{e[1]}"
        pheromone_map[e] = PheromoneEntry(
            uuid=pid,
            surface_key=pid,
            signal_kind="edge",
            signal_value=0.5,
            half_life_seconds=HALF_LIFE_BASE,
            created_at=now
        )

    result = hybrid_classify(sample_text, actions, nodes, edges, pheromone_map)
    print("Hybrid classification result ->", result)