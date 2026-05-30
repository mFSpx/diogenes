# DARWIN HAMMER — match 2100, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s3.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_label_foundry_m191_s0.py (gen3)
# born: 2026-05-29T23:40:53Z

"""
Hybrid Voronoi-Bandit Fusion

Parents:
- Parent A (hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s3.py): provides
  a contextual multi‑armed bandit (LinUCB / Thompson / ε‑greedy) with a
  virtual‑VRAM store for policy statistics and a lead‑lag transform that
  converts a sequence of binary labels into a causal path representation.
- Parent B (hybrid_hybrid_voronoi_parti_hybrid_label_foundry_m191_s0.py): provides
  Voronoi partitioning and hybrid endpoint circuit breakers with serpentina self-righting 
  and weak supervision labeling primitives.

Mathematical Bridge:
The mathematical bridge between these structures is the concept of "contextual recovery priority," 
which is used to determine the likelihood of an endpoint recovering from a failure. 
This recovery priority is calculated based on the morphology of the endpoint and 
the expected reward of the bandit, and this value is then used to adjust the Voronoi 
partitioning's distance calculation to ensure robust labeling and endpoint management.

The contextual recovery priority is calculated as:

    recovery_priority = σ(𝑅̂(a)) × (1 + morphology.recovery_priority())

where σ is the logistic sigmoid that maps expected reward to [0,1].

The distance calculation in Voronoi partitioning is then adjusted as:

    distance(a, b) = math.hypot(a[0] - b[0], a[1] - b[1]) * (1 + contextual_recovery_priority)
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – bandit structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]


def lead_lag_transform(labels: List[int]) -> np.ndarray:
    """Convert a sequence of binary labels into a causal path representation."""
    T = len(labels)
    phi = np.zeros(2 * T)
    for t in range(T):
        phi[2 * t] = labels[t]  # lead
        if t > 0:
            phi[2 * t + 1] = labels[t - 1]  # lag
    return phi


def sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    return 1 / (1 + math.exp(-x))


# ----------------------------------------------------------------------
# Parent B – Voronoi partitioning
# ----------------------------------------------------------------------

Point = tuple[float, float]

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float

def distance(a: Point, b: Point, recovery_priority: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * (1 + recovery_priority)

def nearest(point: Point, seeds: list[Point], recovery_priority: float) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i], recovery_priority), i))

def assign(points: list[Point], seeds: list[Morphology], recovery_priorities: list[float]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, [s for s in seeds], recovery_priorities[0])
        regions[idx].append(p)
    return regions

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------

def contextual_recovery_priority(morphology: Morphology, expected_reward: float) -> float:
    """Calculate contextual recovery priority."""
    return sigmoid(expected_reward) * (1 + morphology.recovery_priority())

def hybrid_distance(a: Point, b: Point, morphology: Morphology, expected_reward: float) -> float:
    recovery_priority = contextual_recovery_priority(morphology, expected_reward)
    return distance(a, b, recovery_priority)

def hybrid_nearest(point: Point, seeds: list[Morphology], expected_rewards: list[float]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (hybrid_distance(point, seeds[i], seeds[i], expected_rewards[i]), i))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2)]
    seeds = [Morphology(1, 1, 1), Morphology(2, 2, 2)]
    expected_rewards = [0.5, 0.7]
    regions = assign(points, seeds, expected_rewards)
    print(regions)