# DARWIN HAMMER — match 2341, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s0.py (gen4)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s2.py (gen1)
# born: 2026-05-29T23:41:51Z

"""
Module hybrid_hdc_gliner: A fusion of the hyperdimensional computing primitives 
from hybrid_hybrid_perceptual_de_hybrid_hdc_serpentin_m327_s0.py and the 
geometric embedding of GLiNER zero-shot extractor from 
hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s2.py. The mathematical 
bridge between these two structures is the use of Euclidean distance and 
geometric embedding to influence the creation of bipolar vectors in hdc.

This hybrid system integrates the governing equations of both parents by 
using the sphericity index from the hyperdimensional computing primitives 
to modulate the weights of the minimum-cost tree scorer.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

Vector = Sequence[float]

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (math.pi ** (1/3) * (6 * length * width * height) ** (1/3)) / (length + width + height)

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float

def compute_phash(values: list[float]) -> int:
    if not values: 
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]: 
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int: 
    return (a ^ b).bit_count()

def hybrid_score(spans: list[Span], morphology: Morphology) -> float:
    points = [(s.start, s.end - s.start) for s in spans]
    distances = [[euclidean(p1, p2) for p2 in points] for p1 in points]
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    weights = [[sphericity * (1 / (1 + d)) for d in row] for row in distances]
    scores = [s.score for s in spans]
    return sum([s * w for s, w in zip(scores, [sum(row) for row in weights])])

def cluster_by_phash(hashes: dict[str, int], max_distance: int = 4) -> list[list[str]]:
    clusters = []
    for k, h in hashes.items():
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance: 
                c.append(k)
                break
        else: 
            clusters.append([k])
    return clusters

def generate_random_span(morphology: Morphology) -> Span:
    start = random.randint(0, int(morphology.length))
    end = start + random.randint(1, int(morphology.width))
    text = f"random text {start} {end}"
    label = "random label"
    score = random.random()
    return Span(start, end, text, label, score)

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 3.0, 1.0)
    spans = [generate_random_span(morphology) for _ in range(10)]
    score = hybrid_score(spans, morphology)
    print(f"Hybrid score: {score}")
    hashes = {f"span {i}": compute_phash([s.start, s.end, s.score]) for i, s in enumerate(spans)}
    clusters = cluster_by_phash(hashes)
    print(f"Clusters: {clusters}")