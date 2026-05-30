# DARWIN HAMMER — match 1920, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s3.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s4.py (gen3)
# born: 2026-05-29T23:39:46Z

"""Hybrid Hyperdimensional RBF‑Fisher Model
========================================

This module fuses the two parent algorithms:

* **Parent A** – hyperdimensional computing (binding, bundling,
  similarity) together with an RBF surrogate that predicts with a
  Gaussian kernel.
* **Parent B** – a count‑min sketch → RLCT estimator and a Fisher‑information
  calculator based on a Gaussian beam.

The mathematical bridge is the *modulation vector* produced from the
count‑min sketch.  The sketch is turned into a high‑dimensional binary
vector (`modulation_vector`).  This vector is used

1. to **modulate** the RBF surrogate (binding the surrogate’s centre
   coordinates with the modulation vector and scaling the weights by the
   similarity to a reference vector), and
2. as a similarity weight for the RLCT estimate.

The modulated surrogate then yields predictions that are fed to a
Gaussian‑beam model; the derivative of that beam provides the Fisher
information.  The three core functions below demonstrate this hybrid
pipeline.

"""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent A)
# ----------------------------------------------------------------------
Vector = List[int]
FloatVector = Sequence[float]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = np.zeros(dim, dtype=int)
    for v in vectors:
        sums += np.array(v, dtype=int)
    return [1 if x >= 0 else -1 for x in sums]


def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


# ----------------------------------------------------------------------
# RBF surrogate (Parent A)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FloatVector, b: FloatVector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class RBFSurrogate:
    def __init__(
        self,
        centers: List[FloatVector],
        weights: List[float],
        epsilon: float = 1.0,
    ):
        if len(centers) != len(weights):
            raise ValueError("centers and weights must have same length")
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: FloatVector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers)
        )


def modulate_surrogate(surrogate: RBFSurrogate, modulation_vector: Vector) -> RBFSurrogate:
    """Bind each centre (interpreted as int‑valued coordinates) with the
    modulation vector and scale the weights by the similarity to the
    all‑ones reference vector."""
    dim = len(modulation_vector)
    # Convert centre coordinates to int vectors of length *dim* by repetition.
    modulated_centers: List[FloatVector] = []
    for c in surrogate.centers:
        int_coords = [int(round(v)) for v in c]
        # repeat / truncate to match *dim*
        repeated = (int_coords * ((dim // len(int_coords)) + 1))[:dim]
        bound = bind(repeated, modulation_vector)
        # cast back to float for the surrogate
        modulated_centers.append([float(v) for v in bound])

    ref = [1] * dim
    sim = similarity(modulation_vector, ref)
    modulated_weights = [w * sim for w in surrogate.weights]

    return RBFSurrogate(modulated_centers, modulated_weights, surrogate.epsilon)


# ----------------------------------------------------------------------
# Count‑min sketch and RLCT estimator (Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: Sequence, width: int = 64, depth: int = 4) -> List[List[int]]:
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def estimate_rlct_from_losses(train_losses_per_n: Sequence[float], n_values: Sequence[int]) -> float:
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def hybrid_fisher_rlct(data: Sequence) -> Tuple[float, float]:
    """Build a sketch from *data*, turn it into a modulation vector,
    and return (rlct, similarity‑based weight)."""
    sketch = count_min_sketch(data)
    flat = [cnt for row in sketch for cnt in row if cnt > 0]
    n_vals = list(range(1, len(flat) + 1))
    rlct = estimate_rlct_from_losses(flat, n_vals) if len(flat) > 1 else 0.0

    # Convert each count into a binary hypervector and bundle them.
    vectors = [symbol_vector(str(cnt)) for cnt in flat]
    modulation_vector = bundle(vectors) if vectors else random_vector()
    weight = similarity(modulation_vector, [1] * len(modulation_vector))
    return rlct, weight, modulation_vector


# ----------------------------------------------------------------------
# Gaussian‑beam Fisher information (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def compute_fisher_from_surrogate(surrogate: RBFSurrogate, theta_grid: np.ndarray) -> float:
    """Treat the surrogate prediction at each theta as the Gaussian‑beam
    centre and accumulate Fisher information."""
    fisher = 0.0
    for theta in theta_grid:
        center = surrogate.predict([theta] * len(surrogate.centers[0]))
        fisher += fisher_score(theta, center, width=0.1)
    return fisher


# ----------------------------------------------------------------------
# Hybrid pipeline exposing at least three public functions
# ----------------------------------------------------------------------
def create_random_surrogate(num_centers: int = 5, dim: int = 10000, epsilon: float = 1.0) -> RBFSurrogate:
    """Generate a surrogate with random integer centres (converted to float)
    and random weights."""
    centers = [list(map(float, random_vector(dim))) for _ in range(num_centers)]
    weights = [random.uniform(-1.0, 1.0) for _ in range(num_centers)]
    return RBFSurrogate(centers, weights, epsilon)


def hybrid_predict(data: Sequence, query_point: FloatVector) -> Tuple[float, float, float]:
    """
    1. Build a sketch from *data* and obtain a modulation vector.
    2. Modulate a random surrogate with that vector.
    3. Predict at *query_point*.
    4. Compute Fisher information over a fixed theta grid.
    5. Return (prediction, rlct, fisher_info).
    """
    # Step 1 – sketch → rlct, weight, modulation vector
    rlct, _, modulation_vector = hybrid_fisher_rlct(data)

    # Step 2 – surrogate modulation
    surrogate = create_random_surrogate(dim=len(modulation_vector))
    mod_surrogate = modulate_surrogate(surrogate, modulation_vector)

    # Step 3 – prediction
    pred = mod_surrogate.predict(query_point)

    # Step 4 – Fisher information
    theta_grid = np.linspace(-1.0, 1.0, 100)
    fisher = compute_fisher_from_surrogate(mod_surrogate, theta_grid)

    return pred, rlct, fisher


def hybrid_bundle_from_sketch(data: Sequence, dim: int = 10000) -> Vector:
    """
    Directly expose the hyperdimensional bundle derived from a count‑min sketch.
    Useful when the user only needs the modulation vector.
    """
    _, _, modulation_vector = hybrid_fisher_rlct(data)
    if len(modulation_vector) != dim:
        # Adjust dimension by truncation or padding with random bits
        if len(modulation_vector) > dim:
            return modulation_vector[:dim]
        else:
            extra = random_vector(dim - len(modulation_vector))
            return modulation_vector + extra
    return modulation_vector


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for sketching
    sample_data = [f"item_{i%20}" for i in range(200)]

    # Query point – a random high‑dimensional float vector
    dim = 10000
    query = [float(v) for v in random_vector(dim)]

    pred, rlct, fisher = hybrid_predict(sample_data, query)

    print(f"Prediction          : {pred:.6f}")
    print(f"Estimated RLCT      : {rlct:.6f}")
    print(f"Fisher information  : {fisher:.6f}")

    # Demonstrate direct bundle extraction
    mod_vec = hybrid_bundle_from_sketch(sample_data, dim=dim)
    print(f"Modulation vector similarity to all‑ones: {similarity(mod_vec, [1]*dim):.4f}")