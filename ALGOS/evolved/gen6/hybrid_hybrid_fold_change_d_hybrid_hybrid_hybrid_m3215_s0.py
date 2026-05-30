# DARWIN HAMMER — match 3215, survivor 0
# gen: 6
# parent_a: hybrid_fold_change_detectio_hybrid_infotaxis_hyb_m2365_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m602_s1.py (gen5)
# born: 2026-05-29T23:48:31Z

"""Hybrid Fold-Change Detection & Distributed Hash Graph

Parents:
- PARENT A: fold_change_detection + hybrid semantic‑morphology neighbor system.
- PARENT B: distributed hash‑based graph construction with temperature‑driven features.

Mathematical bridge:
The morphology‑derived *recovery priority* `p ∈ [0,1]` (Parent A) is used as a
multiplicative scaling factor for every weight that participates in the
graph‑based computations of Parent B.  Concretely:

* Edge weights `vram_weights` are multiplied by `p` before graph construction,
  thus the topology adapts to the organism’s righting‑time characteristics.
* The fold‑change gain used in the dynamical update `step` is also scaled by `p`,
  linking the two dynamical systems.
* The feature matrix produced from the graph includes `p` as an additional
  descriptor, enabling downstream learning or analysis to see both
  morphological and informational influences.

The three core hybrid functions below demonstrate this integration.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Morphology & Fold‑Change Detection (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def step(u: float, x: float, y: float,
         dt: float = 1.0,
         gain: float = 1.0,
         decay_x: float = 1.0,
         decay_y: float = 1.0,
         eps: float = 1e-12) -> Tuple[float, float]:
    """
    Euler integration of the fold‑change detection dynamics:

        dx/dt = u - decay_x * x
        dy/dt = gain * (u / max(|x|, eps)) - decay_y * y
    """
    if dt < 0:
        raise ValueError('dt must be non‑negative')
    dx = u - decay_x * x
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    x_new = x + dt * dx
    y_new = y + dt * dy
    return x_new, y_new


# ----------------------------------------------------------------------
# Hash‑Based Distributed Graph (Parent B)
# ----------------------------------------------------------------------
def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def gini_coefficient(rewards: List[float]) -> float:
    rewards = np.array(rewards, dtype=float)
    mean = np.mean(rewards)
    if mean == 0:
        return 0.0
    n = len(rewards)
    diff_sum = np.abs(rewards[:, None] - rewards[None, :]).sum()
    return diff_sum / (2 * n * n * mean)


def schoolfield_rate(temperature: float) -> float:
    return 1.0 / (1.0 + math.exp(temperature - 20.0))


# ----------------------------------------------------------------------
# Hybrid Functions (integration of A & B)
# ----------------------------------------------------------------------
def build_hybrid_graph(elements: List[List[float]],
                       vram_weights: List[float],
                       morphology: Morphology,
                       hamming_thresh: int = 4) -> Dict[str, Dict[str, float]]:
    """
    Constructs a similarity graph where edge weights are scaled by the
    morphology‑derived recovery priority `p`.
    """
    p = recovery_priority(morphology)
    # Scale weights once; keep original list untouched
    scaled_weights = [w * p for w in vram_weights]

    hashes: Dict[str, int] = {}
    for i, element in enumerate(elements):
        hashes[str(i)] = compute_phash(element)

    graph: Dict[str, Dict[str, float]] = {}
    n = len(elements)
    for i in range(n):
        graph[str(i)] = {}
        for j in range(i + 1, n):
            if hamming_distance(hashes[str(i)], hashes[str(j)]) <= hamming_thresh:
                # Use the scaled weight of the *target* node as edge capacity
                graph[str(i)][str(j)] = scaled_weights[j]
                graph[str(j)][str(i)] = scaled_weights[i]
    return graph


def hybrid_step(morphology: Morphology,
                u: float,
                x: float,
                y: float,
                dt: float = 1.0,
                base_gain: float = 1.0,
                decay_x: float = 1.0,
                decay_y: float = 1.0) -> Tuple[float, float]:
    """
    Performs a single fold‑change detection update where the gain is
    modulated by the recovery priority `p`.
    """
    p = recovery_priority(morphology)
    gain = base_gain * p
    return step(u, x, y, dt=dt, gain=gain,
                decay_x=decay_x, decay_y=decay_y)


def compute_hybrid_feature_matrix(graph: Dict[str, Dict[str, float]],
                                  temperature: float,
                                  morphology: Morphology) -> np.ndarray:
    """
    Returns a feature matrix where each node vector consists of:
    1. Phash of its incident edge weights,
    2. Temperature‑dependent schoolfield rate,
    3. Node degree,
    4. Recovery priority `p` (morphology channel).
    """
    p = recovery_priority(morphology)
    feature_rows = []
    for node, neighbors in graph.items():
        edge_weights = [neighbors[nbr] for nbr in neighbors]
        ph = compute_phash(edge_weights) if edge_weights else 0
        degree = len(neighbors)
        row = [
            ph,
            schoolfield_rate(temperature),
            degree,
            p
        ]
        feature_rows.append(row)
    return np.array(feature_rows, dtype=float)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology instance
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Demonstrate hybrid_step
    x0, y0 = 0.5, 0.2
    u_signal = 1.0
    x1, y1 = hybrid_step(morph, u=u_signal, x=x0, y=y0, dt=0.1)
    print(f"Hybrid step result: x={x1:.4f}, y={y1:.4f}")

    # Dummy data for graph construction
    random.seed(42)
    elements = [[random.random() for _ in range(8)] for _ in range(5)]
    vram_weights = [random.random() for _ in range(5)]

    graph = build_hybrid_graph(elements, vram_weights, morph)
    print(f"Constructed graph with {len(graph)} nodes.")

    # Compute feature matrix
    temp = 22.0
    feat_mat = compute_hybrid_feature_matrix(graph, temperature=temp, morphology=morph)
    print("Hybrid feature matrix:")
    print(feat_mat)

    # Compute auxiliary metrics to ensure imports work
    gini = gini_coefficient([random.random() for _ in range(10)])
    broadcast = broadcast_probability(phase=3, step=2)
    print(f"Gini coefficient: {gini:.4f}, Broadcast probability: {broadcast:.4f}")

    sys.exit(0)