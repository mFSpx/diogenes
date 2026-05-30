# DARWIN HAMMER — match 4422, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m842_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2541_s5.py (gen6)
# born: 2026-05-29T23:55:37Z

import math
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Utility: deterministic hashing
# ----------------------------------------------------------------------
def deterministic_hash(text: str) -> int:
    """SHA‑256 based deterministic integer hash."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Use the full 256‑bit digest as an integer (Python ints are unbounded)
    return int(h, 16)


# ----------------------------------------------------------------------
# Parent A – Stylometry feature generation (deterministic)
# ----------------------------------------------------------------------
def generate_stylometry_features(text: str, dim: int = 32) -> np.ndarray:
    """
    Produce a reproducible pseudo‑random feature vector.

    Uses NumPy's Generator seeded with a deterministic hash of *text*.
    """
    seed = deterministic_hash(text) % (2**32)  # NumPy seeds are 32‑bit
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float64)


# ----------------------------------------------------------------------
# Parent B – Temperature‑dependent NLMS adaptation
# ----------------------------------------------------------------------
def schoolfield_dev_rate(temp_c: float, K: float = 1e4, Ea: float = 0.09) -> float:
    """
    Developmental rate from the Schoolfield model (simplified).

    temp_c : temperature in Celsius
    K      : pre‑exponential factor
    Ea     : activation energy (scaled)
    """
    kelvin = temp_c + 273.15
    return K * math.exp(-Ea * kelvin)


def nlms_update(
    weights: np.ndarray,
    input_signal: np.ndarray,
    desired: float,
    dev_rate: float,
    mu: float = 0.1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Perform one NLMS weight update with temperature‑scaled step size.

    weights     : current weight vector (shape (d,))
    input_signal: current input vector (shape (d,))
    desired     : scalar desired response (e.g., mean reward estimate)
    dev_rate    : developmental rate from Schoolfield equation
    mu          : base learning rate (0 < mu <= 1)
    eps         : regularisation term to avoid division by zero
    """
    # Temperature‑scaled learning rate
    step = mu * dev_rate

    # NLMS error term
    prediction = np.dot(weights, input_signal)
    error = desired - prediction

    norm_sq = np.dot(input_signal, input_signal) + eps
    adaptation = (step / norm_sq) * error * input_signal
    return weights + adaptation


# ----------------------------------------------------------------------
# Count‑Min Sketch implementation (deep integration)
# ----------------------------------------------------------------------
@dataclass
class CountMinSketch:
    depth: int
    width: int
    seed: int = 0

    def __post_init__(self):
        self.tables = np.zeros((self.depth, self.width), dtype=np.uint32)
        # Derive a distinct seed for each hash function
        base_hash = deterministic_hash(str(self.seed))
        self.hash_seeds = [(base_hash + i) % (2**32) for i in range(self.depth)]

    @staticmethod
    def _hash(value: float, seed: int) -> int:
        """
        Simple 32‑bit mix of a float's binary representation and a seed.
        """
        # Convert float to its IEEE 754 binary representation
        bits = np.frombuffer(np.float64(value).tobytes(), dtype=np.uint64)[0]
        h = seed ^ (bits & 0xFFFFFFFF)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h ^= (h >> 13)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= (h >> 16)
        return int(h)

    def update(self, vector: np.ndarray):
        """Insert each component of *vector* into the sketch."""
        for i, val in enumerate(vector):
            for d, seed in enumerate(self.hash_seeds):
                idx = self._hash(val, seed) % self.width
                self.tables[d, idx] += 1

    def estimate_mean(self, total_updates: int) -> float:
        """
        Estimate the average count per cell, then normalise by the number of
        updates to obtain an approximate mean reward (assuming reward=1 per update).
        """
        if total_updates == 0:
            return 0.0
        avg_count = self.tables.mean()
        # Each update increments exactly `depth` counters, so divide by depth
        return (avg_count / self.depth) / total_updates


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_labeling_function(text: str, temp_c: float) -> float:
    """
    Integrated pipeline:
    1. Generate deterministic stylometric features.
    2. Insert them into a Count‑Min sketch.
    3. Estimate a temperature‑aware mean reward.
    4. Adapt NLMS weights using the Schoolfield developmental rate.
    5. Return the average of the adapted weight vector as the label score.
    """
    # 1. Feature generation (deterministic)
    features = generate_stylometry_features(text, dim=32)

    # 2. Count‑Min sketch update
    sketch = CountMinSketch(depth=8, width=256, seed=deterministic_hash(text))
    sketch.update(features)

    # 3. Mean‑reward estimate (temperature‑aware)
    # Assume we have performed one update per feature component
    total_updates = features.size
    mean_reward = sketch.estimate_mean(total_updates)

    # 4. Temperature‑scaled NLMS adaptation
    dev_rate = schoolfield_dev_rate(temp_c)
    # Deterministic initial weights using the same seed as the feature vector
    rng = np.random.default_rng(deterministic_hash(text) % (2**32))
    init_weights = rng.random(features.shape, dtype=np.float64)

    adapted_weights = nlms_update(
        weights=init_weights,
        input_signal=features,
        desired=mean_reward,
        dev_rate=dev_rate,
        mu=0.05,  # modest base learning rate
    )

    # 5. Produce final label (mean of adapted weights)
    return float(np.mean(adapted_weights))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    temperature = 20.0  # Celsius
    label_score = hybrid_labeling_function(sample_text, temperature)
    print(f"Label score: {label_score}")