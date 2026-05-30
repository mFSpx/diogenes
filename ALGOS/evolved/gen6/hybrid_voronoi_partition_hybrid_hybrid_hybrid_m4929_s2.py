# DARWIN HAMMER — match 4929, survivor 2
# gen: 6
# parent_a: voronoi_partition.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py (gen5)
# born: 2026-05-29T23:58:53Z

"""
Hybrid algorithm fusion of voronoi_partition.py and hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m1667_s0.py.
The mathematical bridge between the two structures is the incorporation of probabilistic labels and confidence scores 
to inform the Voronoi partitioning process, using movement primitives from the hybrid algorithm to optimize 
the partitioning based on the confidence scores.

This hybrid algorithm uses the Voronoi partitioning to divide the space into regions and then applies 
the probabilistic labels and confidence scores to optimize the partitioning.
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
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
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
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def hybrid_voronoi_partition(points: list[Point], seeds: list[Point], 
                             batches: list[list[LabelingFunctionResult]]) -> dict[int, list[Point]]:
    regions = assign(points, seeds)
    probabilistic_labels = aggregate_labels(batches)
    for region_idx, points_in_region in regions.items():
        confidence_sum = 0
        for label in probabilistic_labels:
            if label.doc_id in [str(point) for point in points_in_region]:
                confidence_sum += label.confidence
        if confidence_sum > 0:
            # Apply movement primitives to optimize the partitioning based on confidence scores
            optimized_points_in_region = []
            for point in points_in_region:
                if random.random() < broadcast_probability(1, region_idx):
                    optimized_points_in_region.append((point[0] + random.uniform(-1, 1), point[1] + random.uniform(-1, 1)))
                else:
                    optimized_points_in_region.append(point)
            regions[region_idx] = optimized_points_in_region
    return regions

def demonstrate_hybrid_operation():
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    seeds = [(0, 0), (2, 2), (4, 4)]
    batches = [[LabelingFunctionResult('lf1', '0,0', 1), LabelingFunctionResult('lf2', '2,2', 0)], 
               [LabelingFunctionResult('lf1', '1,1', 1), LabelingFunctionResult('lf2', '3,3', 0)]]
    regions = hybrid_voronoi_partition(points, seeds, batches)
    print(regions)

if __name__ == "__main__":
    demonstrate_hybrid_operation()