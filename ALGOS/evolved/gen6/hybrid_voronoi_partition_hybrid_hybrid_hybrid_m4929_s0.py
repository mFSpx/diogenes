# DARWIN HAMMER — match 4929, survivor 0
# gen: 6
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py (gen5)
# born: 2026-05-29T23:58:53Z

"""
This module fuses the Voronoi partitioning algorithm from voronoi_partition.py and the Hybrid Leader Election framework from hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py.
The mathematical bridge between the two structures is the incorporation of Voronoi regions into the leader election process, 
using the distance metric from Voronoi partitioning to inform the probabilistic labels and confidence scores in the Hybrid Leader Election framework.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable, Hashable
from collections.abc import Mapping

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    """Find the index of the nearest seed to a given point."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    """Assign points to Voronoi regions based on their nearest seed."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

Node = Hashable
Graph = Mapping[Node, set[Node]]

@dataclass(frozen=True)
class LabelingFunctionResult:
    """Result of a labeling function."""
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    """Probabilistic label with confidence."""
    doc_id: str
    label: int
    confidence: float

def labeling_function(name: str | None = None):
    """Decorator for labeling functions."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    """Aggregate labels from batches."""
    votes = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                if r.doc_id not in votes:
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        label = 1 if vs.count(1) > vs.count(0) else 0
        confidence = vs.count(label) / len(vs)
        out.append(ProbabilisticLabel(d, label, confidence))
    return out

def broadcast_probability(phase: int, step: int) -> float:
    """Return p=1/2^(phase-step), clamped to [0, 1]."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

class HybridLeaderElection:
    def __init__(self):
        self._POLICY = {}

    def reset_policy(self):
        self._POLICY = {}

    def voronoi_leader_election(self, points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
        """Perform leader election using Voronoi regions."""
        regions = assign(points, seeds)
        return regions

    def probabilistic_labeling(self, points: list[Point], seeds: list[Point]) -> list[ProbabilisticLabel]:
        """Generate probabilistic labels for points based on their Voronoi regions."""
        regions = assign(points, seeds)
        labels = []
        for i, region in regions.items():
            for point in region:
                label = 1 if distance(point, seeds[i]) < distance(point, seeds[(i+1) % len(seeds)]) else 0
                confidence = 1 - broadcast_probability(len(seeds), i+1)
                labels.append(ProbabilisticLabel(str(point), label, confidence))
        return labels

def hybrid_operation(points: list[Point], seeds: list[Point]) -> tuple[dict[int, list[Point]], list[ProbabilisticLabel]]:
    """Perform hybrid operation combining Voronoi partitioning and probabilistic labeling."""
    regions = assign(points, seeds)
    labels = []
    for i, region in regions.items():
        for point in region:
            label = 1 if distance(point, seeds[i]) < distance(point, seeds[(i+1) % len(seeds)]) else 0
            confidence = 1 - broadcast_probability(len(seeds), i+1)
            labels.append(ProbabilisticLabel(str(point), label, confidence))
    return regions, labels

if __name__ == "__main__":
    points = [(random.random(), random.random()) for _ in range(10)]
    seeds = [(random.random(), random.random()) for _ in range(3)]
    regions, labels = hybrid_operation(points, seeds)
    print("Voronoi Regions:", regions)
    print("Probabilistic Labels:", labels)