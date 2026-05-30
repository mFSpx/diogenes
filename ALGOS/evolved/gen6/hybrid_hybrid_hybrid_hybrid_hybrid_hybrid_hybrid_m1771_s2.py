# DARWIN HAMMER — match 1771, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_infota_hybrid_hybrid_krampu_m808_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s2.py (gen5)
# born: 2026-05-29T23:38:42Z

"""Hybrid algorithm combining MinHash uncertainty, Fisher information, and pheromone dynamics.

Parents:
- hybrid_hybrid_hybrid_infota_hybrid_hybrid_krampu_m808_s0.py (MinHash + pheromone)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m587_s2.py (Fisher information + count‑min sketch + graph precision)

Mathematical bridge:
Both parents employ sketching techniques (MinHash and Count‑Min Sketch) to compress high‑dimensional information.
We fuse them by feeding the MinHash signature (a stochastic sketch of a probability distribution) into a
Count‑Min Sketch, obtaining frequency estimates that are then weighted by Fisher information – a measure of
sensitivity/precision. The resulting weighted sketch modulates pheromone signal decay, providing a unified
uncertainty‑aware decision layer.
"""

import numpy as np
import math
import random
import sys
import pathlib
import time
import hashlib

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
class StrikeState:
    def __init__(self, velocity: float, distance: float, peak_velocity: float):
        self.velocity = velocity
        self.distance = distance
        self.peak_velocity = peak_velocity


class PheromoneEntry:
    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        self.created_at = time.time()
        self.last_decay = self.created_at

    def age_seconds(self) -> float:
        return time.time() - self.last_decay

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self):
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = time.time()


# ----------------------------------------------------------------------
# Sketching primitives (MinHash & Count‑Min Sketch)
# ----------------------------------------------------------------------
def compute_minhash_signature(probabilities: list[float], k: int = 128) -> list[int]:
    """
    Compute a MinHash signature for a discrete probability distribution.
    For each of k random hash functions we take the smallest index i
    such that a uniform random number < cumulative probability up to i.
    """
    if not probabilities:
        raise ValueError("probabilities list cannot be empty")
    probs = np.array(probabilities, dtype=float)
    if np.any(probs < 0):
        raise ValueError("probabilities must be non‑negative")
    total = probs.sum()
    if total == 0:
        raise ValueError("sum of probabilities must be positive")
    probs /= total                     # normalize to a proper distribution
    cdf = np.cumsum(probs)

    signature = []
    rng = random.Random(0xDEADBEEF)   # deterministic seed for reproducibility
    for _ in range(k):
        # simulate a random hash by drawing a uniform number in (0,1)
        r = rng.random()
        idx = int(np.searchsorted(cdf, r, side='right'))
        signature.append(idx)
    return signature


def minhash_uncertainty(signature: list[int]) -> float:
    """
    Quantify uncertainty of a MinHash signature as the normalized variance
    of its integer components.
    """
    arr = np.array(signature, dtype=float)
    if arr.size == 0:
        return 0.0
    variance = arr.var()
    # Normalization by the squared range to keep the metric in [0,1]
    rng = arr.max() - arr.min()
    if rng == 0:
        return 0.0
    return variance / (rng * rng)


def count_min_sketch(items: list[object], width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Classic Count‑Min Sketch. Returns a depth×width integer matrix.
    """
    table = np.zeros((depth, width), dtype=int)
    for item in items:
        item_bytes = str(item).encode('utf-8')
        for d in range(depth):
            # Derive a deterministic hash per depth
            h = hashlib.sha256(item_bytes + d.to_bytes(1, 'little')).hexdigest()
            idx = int(h, 16) % width
            table[d, idx] += 1
    return table


def sketch_estimate(sketch: np.ndarray, query: object) -> int:
    """
    Estimate frequency of `query` from a Count‑Min Sketch.
    """
    qbytes = str(query).encode('utf-8')
    estimates = []
    depth, width = sketch.shape
    for d in range(depth):
        h = hashlib.sha256(qbytes + d.to_bytes(1, 'little')).hexdigest()
        idx = int(h, 16) % width
        estimates.append(sketch[d, idx])
    return min(estimates)


# ----------------------------------------------------------------------
# Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ under a Gaussian beam model."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def fisher_precision(fisher_val: float, scale: float = 1.0) -> float:
    """
    Convert Fisher information to a precision (inverse variance) term.
    The scale factor allows tuning the influence of Fisher data.
    """
    if fisher_val <= 0:
        return 0.0
    return scale / fisher_val


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_sketch(probabilities: list[float], items: list[object]) -> tuple[np.ndarray, float]:
    """
    Produce a Count‑Min Sketch whose items are the MinHash signature indices.
    Returns the sketch and the uncertainty derived from the MinHash signature.
    """
    signature = compute_minhash_signature(probabilities)
    uncertainty = minhash_uncertainty(signature)

    # Use signature indices as additional items to enrich the sketch
    enriched_items = items + signature
    sketch = count_min_sketch(enriched_items)
    return sketch, uncertainty


def edge_precision(timestamp: float, fisher_val: float, uncertainty: float, alpha: float = 0.5) -> float:
    """
    Combine temporal information, Fisher precision, and MinHash uncertainty
    into a single edge precision value.

    precision = (fisher_precision) * exp(-alpha * uncertainty) * log(1 + timestamp)
    """
    base_prec = fisher_precision(fisher_val)
    temporal_factor = math.log1p(timestamp)          # log(1 + t) grows slowly
    uncertainty_factor = math.exp(-alpha * uncertainty)
    return base_prec * temporal_factor * uncertainty_factor


def update_pheromone(entry: PheromoneEntry, sketch: np.ndarray, query: object,
                    precision: float, decay_multiplier: float = 1.0) -> None:
    """
    Update a pheromone entry based on sketch frequency estimate and edge precision.
    The new signal value is:
        signal_value = (old * decay) + precision * estimate * decay_multiplier
    """
    entry.apply_decay()
    estimate = sketch_estimate(sketch, query)
    increment = precision * estimate * decay_multiplier
    entry.signal_value += increment


# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(probabilities: list[float],
                   items: list[object],
                   theta: float,
                   center: float,
                   width: float,
                   timestamp: float,
                   surface_key: str,
                   signal_kind: str) -> PheromoneEntry:
    """
    Execute the full hybrid workflow:
    1. Build a unified sketch from probability distribution and raw items.
    2. Compute uncertainty from the MinHash component.
    3. Evaluate Fisher information for the supplied Gaussian beam parameters.
    4. Derive an edge precision that blends Fisher, uncertainty and temporal factors.
    5. Initialise a pheromone entry and update it using the sketch.
    Returns the updated PheromoneEntry.
    """
    # Step 1 & 2
    sketch, uncertainty = hybrid_sketch(probabilities, items)

    # Step 3
    fisher_val = fisher_information(theta, center, width)

    # Step 4
    precision = edge_precision(timestamp, fisher_val, uncertainty)

    # Step 5
    entry = PheromoneEntry(surface_key=surface_key,
                           signal_kind=signal_kind,
                           signal_value=1.0,               # start with a base signal
                           half_life_seconds=30)
    # Use a representative query (here we pick the first raw item)
    query = items[0] if items else 0
    update_pheromone(entry, sketch, query, precision)

    return entry


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    probs = [0.1, 0.3, 0.2, 0.4]
    raw_items = ['alpha', 'beta', 'gamma', 'alpha', 'delta']
    theta_val = 0.75
    center_val = 0.5
    width_val = 0.2
    ts = time.time()

    pheromone = hybrid_process(
        probabilities=probs,
        items=raw_items,
        theta=theta_val,
        center=center_val,
        width=width_val,
        timestamp=ts,
        surface_key="node_42",
        signal_kind="exploration"
    )

    print(f"Pheromone entry after hybrid update:")
    print(f"  surface_key : {pheromone.surface_key}")
    print(f"  signal_kind : {pheromone.signal_kind}")
    print(f"  signal_value: {pheromone.signal_value:.4f}")
    print(f"  half_life   : {pheromone.half_life_seconds}s")
    print(f"  age (s)     : {pheromone.age_seconds():.2f}")