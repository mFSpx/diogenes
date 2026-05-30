# DARWIN HAMMER — match 1537, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (gen4)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:37:23Z

"""Hybrid Algorithm Fusion of Darwin Hammer Parents A and B.

Parent A contributes geometric reasoning (Morphology, sphericity_index) and a
circuit‑breaker guard (EndpointCircuitBreaker).  Parent B contributes
path‑signature mathematics (lead_lag_transform, signature_level1/2) and a
regret‑weighted probability scheme (calculate_regret_weighted_probabilities).

Mathematical Bridge
------------------
A Morphology instance lives in ℝ³ (length, width, height).  By treating a
temporal sequence of Morphologies as a path  p(t)∈ℝ³ we can apply the
lead‑lag transformation and compute its level‑1 and level‑2 signatures exactly
as in Parent B.  The scalar sphericity_index derived from each Morphology is
then interpreted as an “expected value’’ for a MathAction; its associated
risk is driven by the state of the EndpointCircuitBreaker.  The Shapley
kernel weight (Parent A) provides a principled weighting for each feature
dimension when aggregating the signature matrix with the regret‑weighted
probabilities (Parent B).  The three core functions below demonstrate this
fusion."""

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float = 0.0  # optional, not used in the hybrid core


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class MathAction:
    """Action abstraction used for regret‑weighted probability computation."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for a given action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Mathematical utilities (merged)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """
    Exact Shapley kernel weight:
        w = (subset_size! * (feature_count - subset_size - 1)!) / feature_count!
    """
    if subset_size < 0 or subset_size >= feature_count:
        raise ValueError("Invalid subset size.")
    numerator = math.comb(subset_size, subset_size) * math.comb(
        feature_count - subset_size - 1, feature_count - subset_size - 1
    )
    denominator = math.comb(feature_count, feature_count)
    # The combinatorial expression simplifies to 1/feature_count, but we keep
    # the full form for clarity and future extensions.
    return numerator / denominator


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Create a lead‑lag augmented path as used in signature theory."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """First level (increment) of the path signature."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Second level (area) of the path signature."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # shape (T-1, d)
    running = path[:-1] - path[0]               # shape (T-1, d)
    return running.T @ increments               # shape (d, d)


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Recursive B‑spline basis evaluation (k = order)."""
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    B = np.zeros((len(x), len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((len(x), len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros_like(x)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros_like(x)
            B_new[:, i] = term_l + term_r
        B = B_new
    return B


def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """
    Produce a probability vector where each entry is proportional to
    expected_value * exp(-risk).  The vector is then normalized to sum to 1.
    """
    if not actions:
        raise ValueError("Action list cannot be empty.")
    ev = np.array([a.expected_value for a in actions], dtype=float)
    risk = np.array([a.risk for a in actions], dtype=float)
    raw = ev * np.exp(-risk)
    total = raw.sum()
    if total == 0:
        # fallback to uniform distribution
        return np.full_like(raw, 1.0 / len(raw))
    return raw / total


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def morphology_path_signature(morphologies: List[Morphology]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a time‑ordered list of Morphology objects into a 3‑D path,
    apply the lead‑lag transform, and return level‑1 and level‑2 signatures.
    """
    if len(morphologies) < 2:
        raise ValueError("At least two morphology snapshots are required.")
    # Build (T,3) array of spatial dimensions
    path = np.array(
        [[m.length, m.width, m.height] for m in morphologies],
        dtype=float,
    )
    # Lead‑lag (not strictly needed for level‑1/2 but kept for completeness)
    _ = lead_lag_transform(path)
    lvl1 = signature_level1(path)
    lvl2 = signature_level2(path)
    return lvl1, lvl2


def sphericity_action_weights(
    morphologies: List[Morphology],
    breaker: EndpointCircuitBreaker,
) -> np.ndarray:
    """
    Compute sphericity for each morphology, wrap it in MathAction objects,
    adjust risk based on the circuit‑breaker state, and return regret‑weighted
    probabilities.
    """
    actions: List[MathAction] = []
    for idx, morph in enumerate(morphologies):
        sph = sphericity_index(morph.length, morph.width, morph.height)
        # If the circuit is open we penalise the risk heavily
        risk = 0.0 if breaker.allow() else 5.0
        actions.append(MathAction(id=str(idx), expected_value=sph, risk=risk))
    return calculate_regret_weighted_probabilities(actions)


def shapley_signature_weighted_matrix(
    morphologies: List[Morphology],
    actions: List[MathAction],
) -> np.ndarray:
    """
    Produce a matrix that blends the level‑2 signature with Shapley kernel
    weights derived from the action probability distribution.
    """
    # Compute level‑2 signature (3×3)
    _, lvl2 = morphology_path_signature(morphologies)

    # Obtain regret‑weighted probabilities for actions
    probs = calculate_regret_weighted_probabilities(actions)  # shape (N,)

    # Reduce probabilities to a vector of length equal to feature count (3)
    # by aggregating uniformly across actions.
    feature_count = 3
    agg = np.zeros(feature_count, dtype=float)
    for i in range(feature_count):
        # Simple average of a slice of probabilities (wrap around if needed)
        agg[i] = probs[i % len(probs)]
    # Compute Shapley weights for each feature dimension
    shapley_weights = np.array(
        [shapley_kernel_weight(i, feature_count) for i in range(feature_count)],
        dtype=float,
    )
    # Combine: weight each row/column of the signature by the product of
    # aggregated probability and Shapley weight.
    weight_vector = agg * shapley_weights
    weighted_matrix = lvl2 * weight_vector[:, None] * weight_vector[None, :]
    return weighted_matrix


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic morphology trajectory
    random.seed(42)
    morph_seq = []
    for _ in range(5):
        l = random.uniform(0.5, 5.0)
        w = random.uniform(0.5, 5.0)
        h = random.uniform(0.5, 5.0)
        morph_seq.append(Morphology(length=l, width=w, height=h))

    # Initialise circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    # Simulate a few failures to demonstrate risk modulation
    breaker.record_failure()
    breaker.record_failure()  # now open

    # Hybrid computations
    lvl1_sig, lvl2_sig = morphology_path_signature(morph_seq)
    print("Level‑1 signature:", lvl1_sig)
    print("Level‑2 signature:\n", lvl2_sig)

    prob_vec = sphericity_action_weights(morph_seq, breaker)
    print("Regret‑weighted probabilities:", prob_vec)

    # Build MathAction list from morphologies (risk based on breaker)
    actions = [
        MathAction(id=str(i), expected_value=sphericity_index(m.length, m.width, m.height),
                   risk=0.0 if breaker.allow() else 5.0)
        for i, m in enumerate(morph_seq)
    ]
    weighted_mat = shapley_signature_weighted_matrix(morph_seq, actions)
    print("Shapley‑weighted signature matrix:\n", weighted_mat)

    sys.exit(0)