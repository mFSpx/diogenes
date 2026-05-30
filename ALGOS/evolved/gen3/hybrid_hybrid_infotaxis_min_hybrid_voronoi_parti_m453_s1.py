# DARWIN HAMMER — match 453, survivor 1
# gen: 3
# parent_a: hybrid_infotaxis_minhash_m63_s3.py (gen1)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s5.py (gen2)
# born: 2026-05-29T23:29:06Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict
import hashlib

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)

def weighted_jaccard_similarity(sig_a: List[int], sig_b: List[int], weights_a: List[float], weights_b: List[float]) -> float:
    """Approximate weighted Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b) or len(weights_a) != len(sig_a) or len(weights_b) != len(sig_b):
        raise ValueError("signatures and weights must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    numerator = sum(min(weights_a[i], weights_b[i]) for i, (a, b) in enumerate(zip(sig_a, sig_b)) if a == b)
    denominator = sum(max(weights_a[i], weights_b[i]) for i in range(len(sig_a)))
    return numerator / denominator

def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def reliability(self) -> float:
        """Return a smooth reliability score in (0, 1]."""
        # Linear decay with a floor to avoid exact zero.
        return max(0.01, 1.0 - self.failures / (self.failure_threshold * 2))

def hybrid_similarity(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], 
                     circuit_breaker: EndpointCircuitBreaker, k: int = 128) -> float:
    """Calculate the similarity between two sets of points in a Voronoi diagram,
    weighted by the reliability score of a circuit-breaker."""
    # Create MinHash signatures for the sets of points
    sig_a = signature([f"{x},{y}" for x, y in points_a], k)
    sig_b = signature([f"{x},{y}" for x, y in points_b], k)

    # Calculate the weighted Jaccard similarity
    weights_a = [1.0 / (1 + euclidean_distance(point, (0, 0))) for point in points_a]
    weights_b = [1.0 / (1 + euclidean_distance(point, (0, 0))) for point in points_b]
    sim = weighted_jaccard_similarity(sig_a, sig_b, weights_a, weights_b)

    # Weight the similarity by the reliability score
    return sim * circuit_breaker.reliability()

def hybrid_distance(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], 
                    circuit_breaker: EndpointCircuitBreaker) -> float:
    """Calculate the weighted average Euclidean distance between two sets of points,
    weighted by the reliability score of a circuit-breaker."""
    # Calculate the Euclidean distance between each pair of points
    distances = [euclidean_distance(a, b) for a in points_a for b in points_b]

    # Weight the distances by the reliability score
    return sum(distances) / (len(distances) * circuit_breaker.reliability())

def hybrid_reliability(points_a: List[Tuple[float, float]], points_b: List[Tuple[float, float]], 
                       circuit_breaker: EndpointCircuitBreaker) -> float:
    """Calculate the reliability score of a circuit-breaker based on the similarity
    and distance between two sets of points."""
    sim = hybrid_similarity(points_a, points_b, circuit_breaker)
    dist = hybrid_distance(points_a, points_b, circuit_breaker)
    return circuit_breaker.reliability() * sim / (1 + dist)

if __name__ == "__main__":
    points_a = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    points_b = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]
    circuit_breaker = EndpointCircuitBreaker()

    sim = hybrid_similarity(points_a, points_b, circuit_breaker)
    dist = hybrid_distance(points_a, points_b, circuit_breaker)
    reliab = hybrid_reliability(points_a, points_b, circuit_breaker)

    print(f"Similarity: {sim}")
    print(f"Distance: {dist}")
    print(f"Reliability: {reliab}")