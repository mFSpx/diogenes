# DARWIN HAMMER — match 5327, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s1.py (gen6)
# born: 2026-05-30T00:01:23Z

"""
Hybrid Algorithm: fisher_pheromone_prune.py

Parents:
- hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s1.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a Gaussian beam intensity I(θ) and its Fisher information
F(θ) = (∂_θ I)^2 / I, which quantify how sharply a parameter (here an angle) is
resolved. Algorithm B models adaptive decay of pheromone signals in a
Krampus‑style brain‑map where each entry decays exponentially with a half‑life.

The hybrid algorithm treats the Fisher information of a set of angles as a
*precision weight* that modulates pheromone decay and dataset pruning.  High
Fisher values (sharp, informative beams) accelerate decay of low‑importance
pheromones and increase the pruning aggressiveness of samples whose feature
vectors align with the Gaussian beam.  The count‑min sketch from Algorithm A
offers a memory‑efficient way to estimate feature frequencies, which we reuse
to bias the pruning mask.

Thus the core topology fuses:
1. Gaussian beam & Fisher score (A)
2. Pheromone decay with half‑life (B)
3. Count‑min sketch for approximate counting (A)
4. Pruning based on weighted importance (B)

The three main functions below demonstrate this unified system.
"""

import math
import random
import sys
import pathlib
import hashlib
import uuid
from datetime import datetime
import numpy as np

# ----------------------------------------------------------------------
# Utilities from Algorithm A
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items, width: int = 64, depth: int = 4):
    """Simple count‑min sketch returning a 2‑D list of counters."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def estimate_rlct_from_losses(train_losses_per_n, n_values):
    """Estimate the RLCT (real log‑canonical threshold) from a series of losses."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")
    # Linear regression on log‑log scale: loss ≈ a * log(log(n)) + b
    X = np.log(np.log(ns)).reshape(-1, 1)
    X = np.hstack([X, np.ones_like(X)])
    coeffs, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    a, b = coeffs
    return a  # The slope is the RLCT estimate


# ----------------------------------------------------------------------
# Utilities from Algorithm B (adapted)
# ----------------------------------------------------------------------


class PheromoneEntry:
    """A pheromone signal that decays with a configurable half‑life."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply exponential decay to the signal value."""
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now()


# ----------------------------------------------------------------------
# Hybrid Functions (core of the new algorithm)
# ----------------------------------------------------------------------


def compute_fisher_weights(angles: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Vectorised Fisher information for a batch of angles.
    Returns an array of the same shape as *angles*.
    """
    vec_gauss = np.vectorize(gaussian_beam)
    vec_fisher = np.vectorize(fisher_score)

    intensity = vec_gauss(angles, center, width)
    # Avoid division by zero inside fisher_score
    intensity = np.clip(intensity, 1e-12, None)
    derivative = intensity * (-(angles - center) / (width * width))
    fisher = (derivative * derivative) / intensity
    return fisher


def apply_pheromone_decay_with_fisher(entry: PheromoneEntry,
                                      angles: np.ndarray,
                                      center: float,
                                      width: float,
                                      fisher_scale: float = 1.0) -> None:
    """
    Modulate the standard exponential decay of a pheromone entry by the
    average Fisher information of a supplied angle distribution.

    The decay factor becomes:
        factor = base_decay ** (1 + fisher_scale * avg_fisher)

    where *base_decay* is the usual 0.5^(age/half_life).
    """
    entry.apply_decay()  # first apply the ordinary decay
    avg_fisher = float(np.mean(compute_fisher_weights(angles, center, width)))
    # Compute an extra multiplicative scaling
    extra = 0.5 ** (fisher_scale * avg_fisher)
    entry.signal_value *= extra
    # No need to update timestamps again; this is a pure scaling step.


def hybrid_prune_dataset(x: np.ndarray,
                         y: np.ndarray,
                         angles: np.ndarray,
                         center: float,
                         width: float,
                         alpha: float,
                         lambda_: float) -> tuple:
    """
    Prune (x, y) pairs using a hybrid mask that combines:

    * Gaussian beam intensity as a relevance score.
    * Fisher information as a sharpness weight.
    * Count‑min sketch frequencies to down‑weight overly common features.
    * An XGBoost‑style margin (lambda_) and probability schedule (alpha).

    Returns the pruned (x_pruned, y_pruned) arrays.
    """
    if x.shape[0] != y.shape[0] or x.shape[0] != angles.shape[0]:
        raise ValueError("x, y and angles must have the same first dimension")

    # 1. Beam relevance
    beam = np.vectorize(gaussian_beam)(angles, center, width)

    # 2. Fisher sharpness
    fisher = compute_fisher_weights(angles, center, width)

    # 3. Approximate feature frequency via count‑min sketch
    #    We hash each row (treated as a tuple) and query the sketch.
    sketch = count_min_sketch(map(tuple, x))
    # Estimate frequency as the minimum count across hash rows
    def sketch_est(row):
        mins = []
        for d in range(len(sketch)):
            idx = int(hashlib.sha256(f"{d}:{row}".encode()).hexdigest(), 16) % len(sketch[0])
            mins.append(sketch[d][idx])
        return min(mins)

    freq_est = np.array([sketch_est(tuple(row)) for row in x])

    # Normalise frequency estimate to [0,1]
    freq_norm = freq_est.astype(float)
    if freq_norm.max() > 0:
        freq_norm /= freq_norm.max()

    # 4. Combine into a single importance score
    #    Higher beam & fisher => keep, higher frequency => discard.
    importance = beam * (1 + fisher) * (1 - freq_norm)

    # 5. Stochastic pruning based on alpha (probability schedule)
    rng = np.random.default_rng()
    keep_prob = np.clip(alpha * importance - lambda_, 0.0, 1.0)
    keep_mask = rng.random(len(keep_prob)) < keep_prob

    return x[keep_mask], y[keep_mask]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    rng = np.random.default_rng(42)
    n_samples = 500
    x = rng.normal(size=(n_samples, 8))
    y = (np.sum(x, axis=1) > 0).astype(int)
    angles = rng.uniform(-3.0, 3.0, size=n_samples)

    # Parameters for the hybrid system
    center = 0.0
    width = 1.0
    alpha = 0.8
    lambda_ = 0.1

    # 1. Compute Fisher weights (demonstration)
    fisher_vals = compute_fisher_weights(angles, center, width)
    print(f"Mean Fisher information: {fisher_vals.mean():.4f}")

    # 2. Create a pheromone entry and apply Fisher‑modulated decay
    pher = PheromoneEntry(surface_key="demo", signal_kind="test",
                          signal_value=1.0, half_life_seconds=30)
    print(f"Initial signal value: {pher.signal_value:.4f}")
    apply_pheromone_decay_with_fisher(pher, angles, center, width, fisher_scale=0.5)
    print(f"Signal value after Fisher‑modulated decay: {pher.signal_value:.4f}")

    # 3. Prune the dataset using the hybrid criteria
    x_pruned, y_pruned = hybrid_prune_dataset(x, y, angles,
                                              center, width,
                                              alpha, lambda_)
    print(f"Original dataset size: {x.shape[0]}")
    print(f"Pruned dataset size:   {x_pruned.shape[0]}")
    # Ensure that the code runs without raising exceptions
    sys.exit(0)