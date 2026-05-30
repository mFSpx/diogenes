# DARWIN HAMMER — match 4929, survivor 1
# gen: 6
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py (gen5)
# born: 2026-05-29T23:58:53Z

#!/usr/bin/env python3
"""
This module combines the Voronoi partitioning algorithm from voronoi_partition.py and the hybrid distributed leader election framework from hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py.
The mathematical bridge between the two structures is the incorporation of Voronoi partitions as a means to inform the leader election process in the distributed leader election framework.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable, Hashable
from collections.abc import Mapping

Point = Tuple[float, float]
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

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def labeling_function(name: str | None = None):
    """Decorator for labeling functions."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
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

def hybrid_voronoi_partition(seeds: List[Point], points: List[Point], phase: int, step: int) -> Dict[int, List[Point]]:
    """
    Hybrid function that combines Voronoi partitioning with the leader election framework.
    The Voronoi partitioning is informed by the broadcast probability.
    """
    probability = broadcast_probability(phase, step)
    regions = assign(points, seeds)
    for i, region in regions.items():
        for point in region:
            if random.random() < probability:
                # Apply leader election framework to this point
                label = 1 if random.random() < 0.5 else 0
                yield LabelingFunctionResult(f"voronoi_{i}", str(point), label)

def hybrid_leader_election(seeds: List[Point], points: List[Point], phase: int, step: int) -> List[ProbabilisticLabel]:
    """
    Hybrid function that combines the leader election framework with Voronoi partitioning.
    The leader election framework is informed by the Voronoi partitioning.
    """
    regions = assign(points, seeds)
    batches = []
    for i, region in regions.items():
        batch = []
        for point in region:
            label = 1 if random.random() < 0.5 else 0
            batch.append(LabelingFunctionResult(f"voronoi_{i}", str(point), label))
        batches.append(batch)
    return aggregate_labels(batches)

def main():
    seeds = [(0, 0), (1, 1), (2, 2)]
    points = [(0.1, 0.1), (1.1, 1.1), (2.1, 2.1), (0.2, 0.2), (1.2, 1.2), (2.2, 2.2)]
    phase = 2
    step = 1
    hybrid_voronoi_partition(seeds, points, phase, step)
    hybrid_leader_election(seeds, points, phase, step)

if __name__ == "__main__":
    main()