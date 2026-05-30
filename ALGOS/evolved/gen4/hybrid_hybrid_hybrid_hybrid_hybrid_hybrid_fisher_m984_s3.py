# DARWIN HAMMER — match 984, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s2.py (gen3)
# born: 2026-05-29T23:32:11Z

"""Hybrid algorithm merging:
- `hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s1.py` (geometric sphericity,
  Shapley value, endpoint circuit breaker)
- `hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s2.py` (Gaussian beam,
  Fisher information, Count‑Min sketch, RLCT regression)

Mathematical bridge:
Both families treat *information* about a set of features.
Parent A distributes a scalar value over all coalitions via the exact Shapley
formula; Parent B quantifies how much information is retained when a
dimensionality‑reduction mapping (Gaussian beam) is applied, using Fisher
information as a sensitivity measure.  

The hybrid therefore:
1. Uses the geometric *sphericity* of an object (Parent A) to define a
   characteristic angle `θ`.
2. Evaluates Fisher information of a Gaussian beam at `θ` (Parent B) to obtain
   a *relevance weight* `w_f`.
3. Multiplies each exact Shapley contribution by `w_f`, producing a
   *Fisher‑weighted Shapley* value.
4. For large feature spaces the exact enumeration is replaced by a
   Count‑Min sketch; the sketch size is adapted with the RLCT slope
   (Parent B) to keep information loss bounded.

The resulting module supplies three core hybrid operations:
`weighted_shapley`, `sketch_fisher_shapley`, and `adaptive_sketch_width`. """

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import combinations, chain
from typing import Any, Callable, Dict, List, Set, FrozenSet

import numpy as np


# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

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

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Classic Shapley kernel 1 / (C(F,|S|) * |S| * (F-|S|))."""
    return math.factorial(subset_size) * math.factorial(
        feature_count - subset_size - 1
    ) / math.factorial(feature_count)


def exact_shapley_value(
    value_fn: Callable[[FrozenSet[int]], float],
    feature_index: int,
    feature_count: int,
) -> float:
    """Exact Shapley value for a single feature by enumerating all coalitions."""
    total = 0.0
    all_features = set(range(feature_count))
    for r in range(feature_count):
        for coalition in combinations(all_features - {feature_index}, r):
            coalition_set = frozenset(coalition)
            with_feat = coalition_set | {feature_index}
            marginal = value_fn(with_feat) - value_fn(coalition_set)
            weight = shapley_kernel_weight(r, feature_count)
            total += weight * marginal
    return total


# ----------------------------------------------------------------------
# Parent B building blocks
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


def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Very compact frequency sketch."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(),
                16,
            )
            table[d][h % width] += 1
    return table


def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[int]) -> float:
    """Estimate the RLCT slope from a sequence of (n, loss) pairs."""
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


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def _characteristic_angle(morph: Morphology) -> float:
    """
    Convert a 3‑D morphology into a single angle.
    The angle is the polar coordinate of the (length, width) projection,
    scaled to the interval [-π, π].
    """
    return math.atan2(morph.width, morph.length)


def weighted_shapley(
    morph: Morphology,
    value_fn: Callable[[FrozenSet[int]], float],
    feature_count: int,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
) -> Dict[int, float]:
    """
    Compute exact Shapley values for all features and weight each contribution
    by Fisher information evaluated at the morphology‑derived angle.

    Returns a dictionary ``{feature_index: weighted_value}``.
    """
    theta = _characteristic_angle(morph)
    weight = fisher_score(theta, fisher_center, fisher_width)

    # Scale the weight by the object's sphericity – a second geometric factor.
    sph = sphericity_index(morph.length, morph.width, morph.height)
    total_weight = weight * sph

    result: Dict[int, float] = {}
    for idx in range(feature_count):
        raw = exact_shapley_value(value_fn, idx, feature_count)
        result[idx] = total_weight * raw
    return result


def sketch_fisher_shapley(
    data: List[Any],
    morph: Morphology,
    value_fn: Callable[[FrozenSet[int]], float],
    feature_count: int,
    base_width: int = 64,
    depth: int = 4,
    fisher_center: float = 0.0,
    fisher_width: float = 1.0,
) -> Dict[int, float]:
    """
    Approximate Fisher‑weighted Shapley values using a Count‑Min sketch.
    The sketch size can be tuned; larger sketches reduce the bias introduced by
    the sketch approximation.
    """
    # Build the sketch once – it represents the multiset of feature identifiers.
    sketch = count_min_sketch(data, width=base_width, depth=depth)

    # Helper: approximate coalition value by summing sketch frequencies of its members.
    def sketch_value(coalition: FrozenSet[int]) -> float:
        # For a coalition we take the minimum counter across the hash rows,
        # which is the standard Count‑Min estimate.
        mins = []
        for d in range(depth):
            row = sketch[d]
            # hash each element with the same seed used in `count_min_sketch`.
            hashed = [
                int(hashlib.sha256(f"{d}:{elem}".encode()).hexdigest(), 16)
                % base_width
                for elem in coalition
            ]
            if hashed:
                mins.append(min(row[h] for h in hashed))
            else:
                mins.append(0)
        return float(min(mins))

    theta = _characteristic_angle(morph)
    fisher_w = fisher_score(theta, fisher_center, fisher_width)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    scaling = fisher_w * sph

    result: Dict[int, float] = {}
    for idx in range(feature_count):
        # marginal contribution approximated via sketch values
        total = 0.0
        all_features = set(range(feature_count))
        for r in range(feature_count):
            for coalition in combinations(all_features - {idx}, r):
                coal_set = frozenset(coalition)
                with_feat = coal_set | {idx}
                marginal = sketch_value(with_feat) - sketch_value(coal_set)
                weight = shapley_kernel_weight(r, feature_count)
                total += weight * marginal
        result[idx] = scaling * total
    return result


def adaptive_sketch_width(
    train_losses: List[float],
    n_vals: List[int],
    base_width: int = 64,
    factor: float = 1.5,
) -> int:
    """
    Choose a sketch width that adapts to the estimated RLCT slope.
    A larger slope (more information loss) leads to a wider sketch.
    """
    slope = estimate_rlct_from_losses(train_losses, n_vals)
    # Map slope (which can be negative) to a multiplicative factor > 0.
    multiplier = 1.0 + factor * max(0.0, slope)
    new_width = int(max(8, base_width * multiplier))
    # Ensure width stays a power of two for hashing efficiency (optional).
    # Simple rounding up to the next power of two:
    return 1 << (new_width - 1).bit_length()


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple toy value function: sum of feature indices in the coalition.
    def toy_value_fn(coal: FrozenSet[int]) -> float:
        return float(sum(coal))

    # Morphology of a fictitious object.
    obj = Morphology(length=3.0, width=2.0, height=1.0, mass=5.0)

    # Exact Fisher‑weighted Shapley (tiny feature space, enumeration feasible)
    exact = weighted_shapley(
        morph=obj,
        value_fn=toy_value_fn,
        feature_count=4,
    )
    print("Exact Fisher‑weighted Shapley:", exact)

    # Approximate version using a sketch.
    data_items = list(range(4)) * 10  # duplicate identifiers to create frequencies
    approx = sketch_fisher_shapley(
        data=data_items,
        morph=obj,
        value_fn=toy_value_fn,
        feature_count=4,
        base_width=64,
        depth=4,
    )
    print("Sketch Fisher‑weighted Shapley (approx):", approx)

    # Adaptive width based on synthetic training loss curve.
    losses = [0.9, 0.6, 0.4, 0.3, 0.25]
    ns = [100, 500, 2000, 8000, 32000]
    new_w = adaptive_sketch_width(losses, ns, base_width=64)
    print("Adapted sketch width:", new_w)

    # Demonstrate the circuit breaker.
    cb = EndpointCircuitBreaker(failure_threshold=2)
    cb.record_failure()
    print("Circuit open after 1 failure?", not cb.allow())
    cb.record_failure()
    print("Circuit open after 2 failures?", not cb.allow())
    cb.record_success()
    print("Circuit open after reset?", not cb.allow())