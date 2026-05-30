# DARWIN HAMMER — match 4500, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1380_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1205_s6.py (gen6)
# born: 2026-05-29T23:57:48Z

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Hashable, Mapping, Set, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
DEFAULT_WIDTH = 128   # wider sketch for lower collision probability
DEFAULT_DEPTH = 5
EPS = 1e-12           # numerical stability

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places for reporting."""
    return round(float(value), 6)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _hash_token(token: str, seed: int, width: int) -> int:
    """
    Deterministic hash that varies with ``seed`` (depth index).
    Uses SHA‑256 and returns an index in ``[0, width)``.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(token.encode("utf-8"))
    h.update(seed.to_bytes(4, byteorder="little"))
    return int.from_bytes(h.digest()[:4], "little") % width


# ----------------------------------------------------------------------
# Core mathematical components (Parent A)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    A simple Count‑Min sketch with deterministic hash functions.
    Provides ``update`` and ``query`` operations.
    """
    def __init__(self, width: int = DEFAULT_WIDTH, depth: int = DEFAULT_DEPTH):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.float64)

    def update(self, token: str, increment: float = 1.0) -> None:
        """Increment the estimated count of *token* by *increment*."""
        for d in range(self.depth):
            col = _hash_token(token, d, self.width)
            self.table[d, col] += increment

    def bulk_update(self, tokens: Iterable[str]) -> None:
        """Update the sketch with a collection of tokens."""
        for token in tokens:
            self.update(token)

    def query(self, token: str) -> float:
        """Return the minimum count estimate for *token*."""
        mins = []
        for d in range(self.depth):
            col = _hash_token(token, d, self.width)
            mins.append(self.table[d, col])
        return float(min(mins))

    def flatten(self) -> np.ndarray:
        """Return a 1‑D view of the sketch for similarity calculations."""
        return self.table.ravel()


# ----------------------------------------------------------------------
# Types (Parent B)
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]


# ----------------------------------------------------------------------
# Functions from Parent B
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """Probability that a pruning broadcast occurs at a given phase/step."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Standard Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


# ----------------------------------------------------------------------
# Hybrid Mathematics
# ----------------------------------------------------------------------
def trust_similarity(sketch_a: CountMinSketch, sketch_b: CountMinSketch) -> float:
    """
    Cosine similarity between two Count‑Min sketches.
    Returns a value in [0, 1]; higher means the sketches are more alike.
    """
    a = sketch_a.flatten()
    b = sketch_b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a) + EPS
    norm_b = np.linalg.norm(b) + EPS
    return float(dot_product / (norm_a * norm_b))


def effective_temperature(base_temp: float, similarity: float) -> float:
    """
    Modulate *base_temp* using similarity.
    High similarity (≈1) shrinks the temperature, low similarity (≈0) leaves it unchanged.
    """
    floor = EPS
    return float(floor + (1.0 - similarity) * (base_temp - floor))


def annealed_update(
    sketch: CountMinSketch,
    token: str,
    delta_e: float,
    step: int,
    reference_sketch: CountMinSketch,
    t0: float = 1.0,
    alpha: float = 0.95,
) -> bool:
    """
    Perform a Metropolis‑style update of *sketch* for *token*.

    *delta_e* is the energy change if the update were applied.
    *step* is the annealing iteration index.
    The temperature is first obtained from the geometric schedule,
    then scaled by the similarity between *sketch* and *reference_sketch*.
    The function returns ``True`` if the update was accepted.
    """
    base_temp = cooling_temperature(step, t0, alpha)
    sim = trust_similarity(sketch, reference_sketch)
    temp = effective_temperature(base_temp, sim)
    prob = acceptance_probability(delta_e, temp)

    if random.random() < prob:
        sketch.update(token)
        return True
    return False


def hybrid_process(
    tokens: List[str],
    reference_tokens: List[str],
    steps: int = 100,
    broadcast_phase: int = 3,
) -> CountMinSketch:
    """
    Demonstrate the hybrid dynamics:

    1. Build a *reference* sketch from ``reference_tokens`` (trusted corpus).
    2. Initialise an empty sketch.
    3. Iterate ``steps`` times:
       * pick a random token from ``tokens``,
       * compute an artificial energy change ``delta_e`` as the negative
         of the current count estimate (so high‑frequency tokens are
         energetically favourable),
       * possibly apply a broadcast prune based on ``broadcast_probability``,
       * attempt an annealed update via ``annealed_update``.
    Returns the final sketch.
    """
    ref_sketch = CountMinSketch()
    ref_sketch.bulk_update(reference_tokens)

    sketch = CountMinSketch()
    for k in range(steps):
        token = random.choice(tokens)

        # Energy is defined as -log(count+1) to keep it positive and
        # ensure that high-frequency tokens are favoured
        count = sketch.query(token)
        delta_e = -math.log(count + 1)

        if random.random() < broadcast_probability(broadcast_phase, k):
            # Apply a pruning broadcast (not implemented)
            pass

        annealed_update(sketch, token, delta_e, k, ref_sketch)

    return sketch


def kl_divergence(sketch_a: CountMinSketch, sketch_b: CountMinSketch) -> float:
    """
    Kullback-Leibler divergence between two Count-Min sketches.
    """
    a = sketch_a.flatten()
    b = sketch_b.flatten()
    eps = 1e-15
    kl_div = 0.0
    for i in range(len(a)):
        p = a[i] / (np.sum(a) + EPS)
        q = b[i] / (np.sum(b) + EPS)
        kl_div += p * math.log(p / q + EPS)
    return kl_div


def improved_hybrid_process(
    tokens: List[str],
    reference_tokens: List[str],
    steps: int = 100,
    broadcast_phase: int = 3,
) -> Tuple[CountMinSketch, float]:
    ref_sketch = CountMinSketch()
    ref_sketch.bulk_update(reference_tokens)

    sketch = CountMinSketch()
    kl_divs = []
    for k in range(steps):
        token = random.choice(tokens)

        count = sketch.query(token)
        delta_e = -math.log(count + 1)

        if random.random() < broadcast_probability(broadcast_phase, k):
            pass

        accepted = annealed_update(sketch, token, delta_e, k, ref_sketch)
        kl_div = kl_divergence(sketch, ref_sketch)
        kl_divs.append(kl_div)

    avg_kl_div = np.mean(kl_divs)
    return sketch, avg_kl_div