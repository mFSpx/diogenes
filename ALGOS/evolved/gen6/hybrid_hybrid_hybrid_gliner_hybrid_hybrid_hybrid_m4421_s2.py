# DARWIN HAMMER — match 4421, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_hybrid_hybrid_m2123_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_temporal_moti_m1392_s5.py (gen5)
# born: 2026-05-29T23:55:36Z

import math
import random
import sys
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
from scipy.spatial import distance

@dataclass
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

@dataclass
class Endpoint:
    health_score: float
    failure_rate: float
    recovery_priority: float

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    return math.sqrt((r**2 * math.log(2/delta))/(2*n))

def similarity(span1: Span, span2: Span) -> float:
    return 1 - abs(span1.score - span2.score)

def construct_graph(spans: List[Span]) -> np.ndarray:
    num_spans = len(spans)
    graph = np.zeros((num_spans, num_spans))
    for i in range(num_spans):
        for j in range(i+1, num_spans):
            graph[i, j] = similarity(spans[i], spans[j])
            graph[j, i] = graph[i, j]
    return graph

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return distance.euclidean(a, b)

def gini_coefficient(values: List[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

def bayesian_update(prior: List[float], evidence: List[float]) -> List[float]:
    posterior = [p * e for p, e in zip(prior, evidence)]
    sum_posterior = sum(posterior)
    return [p / sum_posterior for p in posterior]

def prim_mst(graph: np.ndarray) -> np.ndarray:
    num_nodes = len(graph)
    visited = [False] * num_nodes
    mst = np.zeros((num_nodes, num_nodes))
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            edges.append((i, j, graph[i, j]))
    edges.sort(key=lambda x: x[2])
    visited[0] = True
    for edge in edges:
        i, j, weight = edge
        if not visited[i] or not visited[j]:
            mst[i, j] = weight
            mst[j, i] = weight
            if not visited[i]:
                visited[i] = True
            if not visited[j]:
                visited[j] = True
    return mst

def hybrid_algorithm(spans: List[Span], entities: List[Entity]) -> Tuple[List[Span], List[Entity]]:
    graph = construct_graph(spans)
    entity_positions = [(e.lat, e.lon) for e in entities]
    distances = np.zeros((len(entity_positions), len(entity_positions)))
    for i in range(len(entity_positions)):
        for j in range(i+1, len(entity_positions)):
            distances[i, j] = length(entity_positions[i], entity_positions[j])
            distances[j, i] = distances[i, j]

    # Apply minimum-cost tree algorithm using Prim's algorithm
    mst = prim_mst(distances)

    # Apply Bayesian update
    prior = [1.0 / len(entities) for _ in entities]
    evidence = [e.score for e in entities]
    posterior = bayesian_update(prior, evidence)

    # Compute Gini coefficient
    gini = gini_coefficient(posterior)

    # Select most relevant spans and entities
    selected_spans = []
    selected_entities = []
    for i, span in enumerate(spans):
        if np.argmax(graph[i, :]) == i:
            selected_spans.append(span)
    for i, entity in enumerate(entities):
        if posterior[i] > 0.5 * (1 - gini):
            selected_entities.append(entity)

    return selected_spans, selected_entities

if __name__ == "__main__":
    spans = [Span(0, 10, "Hello world", "greeting", 0.8, "backend1"), 
              Span(10, 20, "This is a test", "sentence", 0.6, "backend2")]
    entities = [Entity("1", 37.7749, -122.4194, "city", 0.9), 
                Entity("2", 34.0522, -118.2437, "city", 0.7)]
    selected_spans, selected_entities = hybrid_algorithm(spans, entities)
    print(selected_spans)
    print(selected_entities)