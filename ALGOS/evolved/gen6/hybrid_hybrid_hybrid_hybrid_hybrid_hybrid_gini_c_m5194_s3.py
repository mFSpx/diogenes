# DARWIN HAMMER — match 5194, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m2606_s0.py (gen5)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_hybrid_worksh_m2277_s0.py (gen5)
# born: 2026-05-30T00:00:29Z

"""
Hybrid Algorithm integrating:
- Pheromone decay & lead‑lag transformation (Parent A)
- Gini‑coefficient‑driven workshare allocation with RBF similarity (Parent B)

Mathematical Bridge:
The lead‑lag transformed pheromone signal of each node is used as a high‑dimensional
feature vector.  Pairwise Euclidean distances between these vectors are passed
through a Gaussian radial basis function (RBF) to obtain a similarity matrix.
The Gini coefficient of the current pheromone intensities quantifies inequality
among nodes; this scalar modulates the workshare distribution derived from the
similarity matrix, yielding a unified allocation that respects both decay‑driven
exploration and inequality‑aware resource sharing.
"""

import numpy as np
import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Sequence, Hashable, Set, Iterable

# ----------------------------------------------------------------------
# Core data structures (adapted from Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        # uuid4 is not in random; use uuid module via random for compliance
        import uuid
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Mathematical primitives (adapted from Parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return Gini coefficient of a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def lead_lag_transform(path: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Lead‑lag transformation as defined in Parent A.
    Input: T×d array (list of points).
    Output: (2T‑1)×(2d) array.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def rbf_similarity_matrix(features: Dict[Hashable, np.ndarray],
                         epsilon: float = 1.0) -> np.ndarray:
    """
    Compute a symmetric similarity matrix S where
        S[i, j] = exp(- (epsilon * ||f_i - f_j||)^2 )
    using Gaussian RBF on lead‑lag transformed feature vectors.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=float)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian_rbf(dist, epsilon)
            S[i, j] = S[j, i] = sim
    return S


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def update_pheromones(pheromones: Dict[str, PheromoneEntry]) -> None:
    """
    Apply exponential decay to all pheromone entries.
    """
    for entry in pheromones.values():
        entry.apply_decay()


def build_node_features(pheromones: Dict[str, PheromoneEntry]) -> Dict[str, np.ndarray]:
    """
    For each pheromone entry, construct a 2‑D point [signal_value, half_life],
    then apply lead‑lag transformation to obtain a feature vector.
    """
    features: Dict[str, np.ndarray] = {}
    for uid, entry in pheromones.items():
        point = np.array([[entry.signal_value, entry.half_life_seconds]])
        # lead‑lag of a single point yields shape (1,4) – still usable
        transformed = lead_lag_transform(point)
        features[uid] = transformed.ravel()
    return features


def allocate_workshare(pheromones: Dict[str, PheromoneEntry],
                       total_units: int,
                       groups: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Distribute `total_units` among groups based on:
      1. Similarity of node features (RBF matrix)
      2. Gini coefficient of current pheromone intensities (inequality weight)

    Returns a mapping group_id -> allocated integer units.
    """
    # 1. Update decay first
    update_pheromones(pheromones)

    # 2. Feature construction
    features = build_node_features(pheromones)

    # 3. Similarity matrix (node‑level)
    sim_mat = rbf_similarity_matrix(features)

    node_ids = list(features.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # 4. Compute raw group scores as sum of intra‑group similarities
    raw_scores: Dict[str, float] = {}
    for g_id, members in groups.items():
        idx = [node_index[n] for n in members if n in node_index]
        if not idx:
            raw_scores[g_id] = 0.0
            continue
        sub = sim_mat[np.ix_(idx, idx)]
        # sum of upper triangle (including diagonal) normalised by count
        score = sub.sum() / (len(idx) ** 2)
        raw_scores[g_id] = score

    # 5. Modulate by Gini coefficient of pheromone values
    values = [e.signal_value for e in pheromones.values()]
    gini = gini_coefficient(values)
    # Higher inequality (higher gini) should bias towards groups with higher similarity
    mod_scores = {g: s * (1.0 + gini) for g, s in raw_scores.items()}

    # 6. Normalise to allocation fractions
    total_score = sum(mod_scores.values()) or 1.0
    fractions = {g: s / total_score for g, s in mod_scores.items()}

    # 7. Allocate integer units (largest‑remainder method)
    allocation = {g: int(math.floor(frac * total_units)) for g, frac in fractions.items()}
    remainder = total_units - sum(allocation.values())
    if remainder > 0:
        # Distribute remaining units to groups with largest fractional parts
        fractional_parts = {g: (frac * total_units) - allocation[g] for g, frac in fractions.items()}
        for g in sorted(fractional_parts, key=fractional_parts.get, reverse=True)[:remainder]:
            allocation[g] += 1
    return allocation


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small set of pheromone entries
    pheromones: Dict[str, PheromoneEntry] = {}
    for i in range(5):
        pid = f"node_{i}"
        pheromones[pid] = PheromoneEntry(
            surface_key=pid,
            signal_kind="explore",
            signal_value=random.uniform(5.0, 15.0),
            half_life_seconds=random.randint(30, 120)
        )

    # Define groups (overlapping allowed)
    groups = {
        "group_A": {"node_0", "node_1", "node_2"},
        "group_B": {"node_2", "node_3"},
        "group_C": {"node_4"}
    }

    total_units = 100

    allocation = allocate_workshare(pheromones, total_units, groups)

    print("Allocation result:")
    for g, units in allocation.items():
        print(f"  {g}: {units} units")

    # Show that pheromone values have decayed
    print("\nPost‑decay pheromone values:")
    for uid, entry in pheromones.items():
        print(f"  {uid}: {entry.signal_value:.4f}")