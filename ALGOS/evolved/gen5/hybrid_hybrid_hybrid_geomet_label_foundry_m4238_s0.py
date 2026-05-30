# DARWIN HAMMER — match 4238, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s4.py (gen4)
# parent_b: label_foundry.py (gen0)
# born: 2026-05-29T23:54:22Z

"""
hybrid_hybrid_geometric_pro_label_foundry.py
Integrates:

* **Parent A** – geometric product from Clifford algebra Cl(n,0) and Voronoi
  partitioning.
* **Parent B** – Weak supervision labeling primitives.

**Mathematical bridge**
The output of Parent A's Voronoi partitioning is used as a metric to
assign weights to the labeling functions in Parent B. Specifically,
the Voronoi distances between nodes are used to construct a
probabilistic labeling function that assigns higher confidence to
labels with lower Voronoi distances.

This fusion enables the use of geometric and topological information
from Parent A to inform the labeling process in Parent B, allowing
for more accurate and robust weak supervision.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra core (subset of Parent A)
# ----------------------------------------------------------------------


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Duplicate indices cancel because e_i·e_i = 1.
    """
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
                # cancel pair
                del lst[j : j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset, blade_b: frozenset
) -> Tuple[frozenset, int]:
    """Multiply two basis blades and return (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def voronoi_partition(graph: np.ndarray) -> Dict[int, List[int]]:
    """Perform Voronoi partitioning on the graph using Clifford-geometric distances."""
    # Compute Clifford-geometric distances between nodes
    distances = np.zeros((graph.shape[0], graph.shape[0]))
    for i in range(graph.shape[0]):
        for j in range(i + 1, graph.shape[0]):
            a = np.array([0] * graph.shape[1])
            b = np.array([0] * graph.shape[1])
            a[:graph.shape[1] // 2] = graph[i, :graph.shape[1] // 2]
            b[:graph.shape[1] // 2] = graph[j, :graph.shape[1] // 2]
            dist = np.linalg.norm(a - b)
            distances[i, j] = dist
            distances[j, i] = dist

    # Perform Voronoi partitioning using the distances
    partition = {}
    for i in range(graph.shape[0]):
        partition[i] = []
        for j in range(graph.shape[0]):
            if distances[i, j] == np.min(distances[i]):
                partition[i].append(j)

    return partition


# ----------------------------------------------------------------------
# Weak supervision labeling primitives (subset of Parent B)
# ----------------------------------------------------------------------


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


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)

    out = []
    for d, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(d, 0, 0.5))
            continue
        c = Counter(vs)
        label = 1 if c[1] >= c[0] else 0
        out.append(ProbabilisticLabel(d, label, c[label] / len(vs)))

    return out


def label_with_voronoi(graph: np.ndarray, batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Assign weights to labeling functions using Voronoi partitioning."""
    partition = voronoi_partition(graph)
    labels = aggregate_labels(batches)

    weighted_labels = []
    for label in labels:
        doc_id = label.doc_id
        dist = np.inf
        best_label = None
        best_confidence = 0.0
        for i, cluster in partition[doc_id].items():
            cluster_labels = [r.label for r in batches[i]]
            cluster_confidences = [r.confidence for r in batches[i]]
            cluster_label = Counter(cluster_labels).most_common(1)[0][0]
            cluster_confidence = np.mean(cluster_confidences)
            if dist > cluster_confidence:
                dist = cluster_confidence
                best_label = cluster_label
                best_confidence = cluster_confidence

        weighted_labels.append(ProbabilisticLabel(doc_id, best_label, best_confidence))

    return weighted_labels


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    graph = np.random.rand(5, 5)
    batches = [[LabelingFunctionResult("lf1", str(i), 0) for i in range(5)] for _ in range(5)]
    labels = label_with_voronoi(graph, batches)
    print(labels)