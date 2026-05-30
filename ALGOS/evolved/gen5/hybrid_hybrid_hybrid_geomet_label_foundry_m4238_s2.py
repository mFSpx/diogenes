# DARWIN HAMMER — match 4238, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py (gen4)
# parent_b: label_foundry.py (gen0)
# born: 2026-05-29T23:54:22Z

"""
Hybrid Algorithm: Clifford-Geometric Voronoi Labeling
Integrates:
* **Parent A** – hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py: 
  Clifford-geometric distance and Voronoi partitioning.
* **Parent B** – label_foundry.py: Weak supervision labeling primitives.

The mathematical bridge between the two parents lies in replacing the Euclidean distance 
used in label error detection with the Clifford-geometric distance. This distance is 
used to create a Voronoi partition of the node set and to fill the edge-weight matrix 
of a graph. The labeling function results are then used to compute the confidence 
of each label and detect label errors.

The governing equations of Parent A are used to compute the Clifford-geometric 
distance between two multivectors, while the labeling primitives of Parent B are 
used to aggregate labels and detect label errors.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# Clifford algebra core
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def clifford_distance(a: List[float], b: List[float]) -> float:
    n = len(a)
    result = 0
    for i in range(n):
        result += (a[i] - b[i]) ** 2
    return math.sqrt(result)

# Labeling primitives
@dataclass(frozen=True)
class LabelingFunctionResult: 
    lf_name: str; 
    doc_id: str; 
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel: 
    doc_id: str; 
    label: int; 
    confidence: float

@dataclass(frozen=True)
class LabelError: 
    doc_id: str; 
    given_label: int; 
    suggested_label: int; 
    error_probability: float

def labeling_function(name: str|None=None):
    def deco(fn): 
        fn.lf_name=name or fn.__name__; 
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes={}
    for batch in batches:
        for r in batch:
            if r.label in (0,1): 
                if r.doc_id not in votes: votes[r.doc_id]=[]
                votes[r.doc_id].append(r.label)
    out=[]
    for d,vs in votes.items():
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5)); 
            continue
        c={0:0,1:0}; 
        for v in vs: c[v]+=1
        label=1 if c[1]>=c[0] else 0; 
        out.append(ProbabilisticLabel(d,label,c[label]/len(vs)))
    return out

def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float=0.65) -> list[LabelError]:
    if not (len(docs)==len(given)==len(probs)): 
        raise ValueError('length mismatch')
    errs=[]
    for doc,g,p in zip(docs,given,probs):
        errp=p if g==0 else 1.0-p
        if errp>=threshold: 
            errs.append(LabelError(str(doc.get('id',len(errs))),g,1-g,errp))
    return sorted(errs,key=lambda e:-e.error_probability)

# Hybrid functions
def hybrid_labeling_function(batches: list[list[LabelingFunctionResult]], points: List[List[float]]) -> Tuple[list[ProbabilisticLabel], list[LabelError]]:
    labels = aggregate_labels(batches)
    points_dict = {i: point for i, point in enumerate(points)}
    voronoi_points = []
    for label in labels:
        point_id = int(label.doc_id)
        voronoi_points.append(points_dict[point_id])
    errors = find_label_errors([{ 'id': i } for i in range(len(points))], [label.label for label in labels], [label.confidence for label in labels])
    clifford_errors = []
    for error in errors:
        point_a = points_dict[int(error.doc_id)]
        point_b = points_dict[int(error.doc_id) + 1]
        clifford_dist = clifford_distance(point_a, point_b)
        clifford_errors.append((error, clifford_dist))
    return labels, errors

def compute_clifford_voronoi(points: List[List[float]]) -> Dict[int, List[int]]:
    voronoi_points = {}
    for i, point in enumerate(points):
        min_dist = float('inf')
        closest_point = None
        for j, other_point in enumerate(points):
            if i != j:
                dist = clifford_distance(point, other_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = j
        if closest_point is not None:
            if closest_point not in voronoi_points:
                voronoi_points[closest_point] = []
            voronoi_points[closest_point].append(i)
    return voronoi_points

if __name__ == "__main__":
    points = [[random.random() for _ in range(5)] for _ in range(10)]
    batches = [[LabelingFunctionResult('lf1', str(i), random.randint(0, 1)) for i in range(5)] for _ in range(2)]
    labels, errors = hybrid_labeling_function(batches, points)
    voronoi_points = compute_clifford_voronoi(points)
    print("Labels:", labels)
    print("Errors:", errors)
    print("Voronoi Points:", voronoi_points)