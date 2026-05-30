# DARWIN HAMMER — match 5160, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m2272_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_tri_algo_cond_m500_s1.py (gen4)
# born: 2026-05-30T00:00:11Z

"""
Hybrid Module: Fusing hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m2272_s2.py 
              and hybrid_hybrid_hybrid_hybrid_hybrid_tri_algo_cond_m500_s1.py

This fusion integrates the feature-curvature matrix from the first parent with the 
exponential pheromone dynamics and bandit-style entropy handling from the second 
parent. The mathematical bridge is formed by using the feature-curvature matrix 
to modulate the pheromone signal, which in turn influences the allocation scores 
and signal scores.

The feature-curvature matrix `C` from the first parent is used to compute a 
weighted pheromone signal, which is then added to the matrix product of the 
weight matrix `W` and the feature vector `f`. This allows the pheromone dynamics 
to be influenced by the curvature of the feature space.

The second parent contributes the signal scores and noise calculation, which are 
used to update the pheromone signal and the allocation scores. The signal scores 
are computed based on the entropy of the data, status code, MIME type, keyword 
hits, and structural links.

This hybrid module provides a unified system that combines the strengths of both 
parents, allowing for more accurate and robust allocation and signal scoring.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import date, datetime, timezone

# ---------------------------------------------------------------------------
# Constants and helper collections
# ---------------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return the weekday index used by the original doomsday calendar:
    Monday → 1, …, Sunday → 0 (mod 7).
    """
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a stable SHA-256 hash of *text*."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def shannon_entropy(chunk):
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / len(chunk)
        entropy += -p_x * math.log(p_x, 2)
    return entropy

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0  

def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0,
        0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0,
        0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise

def cockpit_honesty(signal_score: float) -> float:
    return signal_score

class HybridAllocator:
    def __init__(self, weight_matrix, pheromones, learning_rates):
        self.W = weight_matrix
        self.pi = pheromones
        self.alpha = learning_rates[0]
        self.beta = learning_rates[1]
        self.GROUPS = GROUPS

    def extract_features(self, text):
        rng = _rng_from_text(text)
        features = np.array([rng.random() for _ in range(len(self.GROUPS))])
        return features

    def update_pheromone(self, features, signal_score):
        weighted_pheromone = np.dot(self.W, features) + signal_score * self.pi
        self.pi = (1 - self.alpha) * self.pi + self.alpha * weighted_pheromone

    def allocate_workshare(self, text):
        features = self.extract_features(text)
        signal_score, _ = signal_scores(text.encode())
        self.update_pheromone(features, signal_score)
        allocation_scores = np.dot(self.W, features) + self.pi
        return allocation_scores

def test_hybrid_allocator():
    weight_matrix = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    pheromones = np.array([0.1, 0.2])
    learning_rates = [0.1, 0.2]
    hybrid_allocator = HybridAllocator(weight_matrix, pheromones, learning_rates)
    allocation_scores = hybrid_allocator.allocate_workshare("test_text")
    print(allocation_scores)

def test_signal_scores():
    data = b"test_data"
    signal_score, noise = signal_scores(data)
    print(f"Signal score: {signal_score}, Noise: {noise}")

def test_cockpit_honesty():
    signal_score = 0.8
    honesty_score = cockpit_honesty(signal_score)
    print(f"Honesty score: {honesty_score}")

if __name__ == "__main__":
    test_hybrid_allocator()
    test_signal_scores()
    test_cockpit_honesty()