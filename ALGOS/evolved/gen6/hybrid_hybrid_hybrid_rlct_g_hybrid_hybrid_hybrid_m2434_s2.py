# DARWIN HAMMER — match 2434, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s2.py (gen5)
# born: 2026-05-29T23:42:19Z

"""Hybrid RLCT‑Grokking + Pheromone‑Infotaxis + Fisher‑Shapley Hyperdimensional Router
===================================================================================

Parent A: *hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s2.py* –  
provides Real Log Canonical Threshold (RLCT) estimation from training losses and a
pheromone‑infotaxis system that uses information‑theoretic entropy to guide
signal decay.

Parent B: *hybrid_hybrid_hybrid_hybrid_hybrid_ternary_route_m928_s2.py* –  
offers hyperdimensional computing primitives (random hypervectors,
binding, weighted bundling) together with ternary routing and Shapley‑kernel
weighting of features.

**Mathematical bridge**  
Both sides manipulate scalar weights that modulate high‑dimensional objects:

* In Parent A the pheromone strength `φ` (a scalar) multiplies the free‑energy
  term `F = λ·RLCT` where `λ` is a learning‑rate‑like coefficient.  
* In Parent B the Shapley kernel weight `w_i` multiplies a hypervector
  `h_i` before bundling.

The hybrid algorithm treats the pheromone strength as a *dynamic Shapley‑like*
scalar that rescales each feature hypervector before the weighted bundle.
Thus the pheromone decay (controlled by the RLCT‑derived half‑life) directly
influences the hyperdimensional representation, unifying energy‑based
optimization with combinatorial feature weighting.

The code below implements this fusion with three core functions:
`update_pheromone_via_rlct`, `shapley_weighted_hypervector`, and
`hybrid_predictor`.  A small smoke test runs at the end."""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone
import numpy as np
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Parent A – RLCT estimation and pheromone infotaxis
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Simple pheromone storage with exponential decay."""
    def __init__(self):
        self.pheromone_signals: Dict[str, Dict[str, float]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        elapsed_seconds: float = 0.0,
    ) -> float:
        """Return the decayed pheromone strength."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        if signal_kind not in self.pheromone_signals[surface_key]:
            self.pheromone_signals[surface_key][signal_kind] = signal_value
        base = self.pheromone_signals[surface_key][signal_kind]
        decay = math.pow(0.5, elapsed_seconds / half_life_seconds)
        return base * decay

    def update_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
    ) -> None:
        """Overwrite the stored signal."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = signal_value


def estimate_rlct_from_losses(
    train_losses_per_n: List[float],
    n_values: List[int],
) -> float:
    """
    Estimate the Real Log Canonical Threshold (RLCT) by fitting a linear model
    to log(loss) = a - λ·log(n).  The slope λ approximates the RLCT.
    """
    if len(train_losses_per_n) != len(n_values):
        raise ValueError("Lengths of losses and n_values must match")
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(losses <= 0) or np.any(ns <= 1):
        raise ValueError("Losses must be >0 and n_values >1 for log scaling")

    log_losses = np.log(losses)
    log_ns = np.log(ns)

    # Linear regression: log_losses = a - λ·log_ns
    A = np.vstack([np.ones_like(log_ns), -log_ns]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_losses, rcond=None)
    _, rlct_est = coeffs
    return max(rlct_est, 0.0)  # RLCT is non‑negative


def update_pheromone_via_rlct(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    signal_kind: str,
    base_signal: float,
    train_losses: List[float],
    n_samples: List[int],
) -> float:
    """
    Compute an RLCT estimate from the supplied losses, translate it into a
    half‑life (larger RLCT → slower decay), update the pheromone store and
    return the current decayed signal (elapsed time is mocked as zero).
    """
    rlct = estimate_rlct_from_losses(train_losses, n_samples)
    # Map RLCT ∈ [0, ∞) to half‑life ∈ [10, 1000] seconds (arbitrary scaling)
    half_life = 10.0 + 990.0 * (1.0 - math.exp(-rlct))
    decayed = pheromone_system.calculate_pheromone_signal(
        surface_key,
        signal_kind,
        base_signal,
        half_life_seconds=half_life,
        elapsed_seconds=0.0,
    )
    pheromone_system.update_pheromone_signal(surface_key, signal_kind, decayed)
    return decayed


# ----------------------------------------------------------------------
# Parent B – Hyperdimensional primitives, ternary routing, Shapley weighting
# ----------------------------------------------------------------------
Vector = List[int]  # hypervector of ±1


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def weighted_bundle(vectors: List[Vector], weights: List[float]) -> Vector:
    """Weighted majority vote across hypervectors."""
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must share the same dimension")
    acc = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i, val in enumerate(vec):
            acc[i] += w * val
    return [1 if s >= 0 else -1 for s in acc]


def ternary_route(command_id: int, dim: int = 10000) -> Vector:
    """
    Encode an integer command as a ternary hypervector with values ∈ {−1,0,1}.
    The pattern is deterministic: bits are set to 1 where the binary
    representation has 1, to −1 where the bit is 0, and every third position
    forced to 0 for ternary sparsity.
    """
    rng = random.Random(command_id)  # deterministic per command
    vec = []
    for i in range(dim):
        if i % 3 == 0:
            vec.append(0)
        else:
            vec.append(1 if rng.getrandbits(1) else -1)
    return vec


def shapley_kernel_weight(feature_index: int, total_features: int) -> float:
    """
    Simple Shapley‑like kernel: weight = 1 / (index + 1) normalized.
    """
    raw = 1.0 / (feature_index + 1)
    return raw / sum(1.0 / (j + 1) for j in range(total_features))


def compute_fisher_score(loss_series: List[float]) -> float:
    """
    Approximate a Fisher information score as the inverse variance of a loss
    trajectory (higher variance → lower information).
    """
    if len(loss_series) < 2:
        return 0.0
    var = np.var(loss_series, ddof=1)
    return 1.0 / (var + 1e-12)


def shapley_weighted_hypervector(
    feature_index: int,
    total_features: int,
    loss_series: List[float],
    dim: int = 10000,
    seed: int | None = None,
) -> Vector:
    """
    Generate a hypervector for a single feature:
      1. Compute a Fisher‑information‑derived scalar.
      2. Multiply by the Shapley kernel weight.
      3. Bind the result with a random base hypervector.
    """
    fisher = compute_fisher_score(loss_series)
    shapley = shapley_kernel_weight(feature_index, total_features)
    scalar = fisher * shapley
    base_vec = random_vector(dim, seed=seed)
    # Bind the base vector with a scalar by scaling its components (still ±1)
    bound = [int(scalar) * x if scalar >= 0 else -int(abs(scalar)) * x for x in base_vec]
    # Clamp to ±1 to keep hypervector binary
    return [1 if b >= 0 else -1 for b in bound]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_rlct_pheromone_update(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    base_signal: float,
    train_losses: List[float],
    n_samples: List[int],
) -> float:
    """
    Wrapper that updates pheromone strength using the RLCT estimate.
    Returns the current (non‑decayed) pheromone value.
    """
    return update_pheromone_via_rlct(
        pheromone_system,
        surface_key,
        "entropy",
        base_signal,
        train_losses,
        n_samples,
    )


def hybrid_feature_vector(
    feature_index: int,
    total_features: int,
    loss_series: List[float],
    pheromone_strength: float,
    dim: int = 10000,
) -> Vector:
    """
    Produce a feature hypervector that is first Shapley‑weighted (via Fisher)
    and then modulated by the current pheromone strength (acting as an
    additional scalar multiplier).
    """
    hv = shapley_weighted_hypervector(
        feature_index,
        total_features,
        loss_series,
        dim=dim,
        seed=feature_index,
    )
    # Apply pheromone scaling (again clamp to ±1)
    scaled = [int(pheromone_strength) * x for x in hv]
    return [1 if s >= 0 else -1 for s in scaled]


def hybrid_predictor(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    base_signal: float,
    train_losses: List[float],
    n_samples: List[int],
    feature_losses: List[List[float]],
    command_id: int,
    dim: int = 10000,
) -> Vector:
    """
    End‑to‑end hybrid predictor:
      1. Update pheromone based on RLCT.
      2. Generate a ternary command vector.
      3. For each feature, build a pheromone‑modulated hypervector.
      4. Bundle all feature vectors with weights derived from pheromone strength.
      5. Bind the bundled feature representation with the command vector
         to produce the final prediction hypervector.
    """
    # Step 1 – pheromone update
    pheromone = hybrid_rlct_pheromone_update(
        pheromone_system, surface_key, base_signal, train_losses, n_samples
    )

    total_features = len(feature_losses)
    feature_vectors = []
    weights = []

    for idx, loss_series in enumerate(feature_losses):
        fv = hybrid_feature_vector(
            idx, total_features, loss_series, pheromone_strength=pheromone, dim=dim
        )
        feature_vectors.append(fv)
        # Use the same pheromone value as weight (demonstrates the bridge)
        weights.append(pheromone)

    # Step 4 – weighted bundle
    bundled = weighted_bundle(feature_vectors, weights)

    # Step 2 – ternary command
    command_vec = ternary_route(command_id, dim=dim)

    # Step 5 – final binding
    prediction = bind(bundled, command_vec)
    return prediction


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data for a tiny demonstration (dim reduced for speed)
    DIM = 1024

    # Pheromone system
    pheromone_sys = PheromoneSystem()

    # Simulated training losses for RLCT estimation
    n_vals = [10, 20, 40, 80, 160]
    losses = [0.9, 0.7, 0.55, 0.43, 0.35]

    # Simulated per‑feature loss histories (3 features)
    feature_loss_series = [
        [0.9, 0.85, 0.8, 0.78],
        [0.7, 0.68, 0.66, 0.65],
        [0.5, 0.48, 0.47, 0.46],
    ]

    pred_vec = hybrid_predictor(
        pheromone_system=pheromone_sys,
        surface_key="test_surface",
        base_signal=1.0,
        train_losses=losses,
        n_samples=n_vals,
        feature_losses=feature_loss_series,
        command_id=42,
        dim=DIM,
    )

    # Simple sanity check: vector length and values in {−1,1}
    assert len(pred_vec) == DIM
    assert all(v in (-1, 1) for v in pred_vec)
    print("Hybrid prediction vector generated successfully. Sample bits:", pred_vec[:10])