# DARWIN HAMMER — match 2770, survivor 6
# gen: 5
# parent_a: hybrid_ternary_router_ssim_m1_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (gen4)
# born: 2026-05-29T23:45:44Z

"""
Hybrid Algorithm: ternary_router + ssim + Fisher‑Weighted Morphology

Parents:
- hybrid_ternary_router_ssim_m1_s0.py (SSIM‑based similarity routing)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (Morphology, Fisher information, circuit breaker)

Mathematical Bridge:
The SSIM formula uses means, variances and covariance of two scalar sequences.
We treat each morphological dimension (length, width, height) as a *feature* that defines a Gaussian
distribution of the underlying signal.  The Fisher information of a Gaussian with variance σ² is
I = 1/σ².  By mapping each dimension to a variance proxy (dim²) we obtain a weight vector wᵢ = Iᵢ.
These weights are injected into the SSIM computation as per‑dimension scaling of the mean and
variance terms, yielding a **Fisher‑weighted SSIM**.  The resulting similarity score is then fed
to a circuit‑breaker‑guarded router that decides whether a packet should be forwarded.

The module therefore fuses:
1. The classic SSIM similarity measure.
2. Fisher‑information‑derived weighting from Morphology.
3. An EndpointCircuitBreaker that protects the routing decision.
"""

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float = 1.0  # default mass if not supplied


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
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "fails": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if any(v <= 0 for v in (length, width, height)):
        raise ValueError("All dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return gm / longest


# ----------------------------------------------------------------------
# Fisher information utilities (bridge to Parent A)
# ----------------------------------------------------------------------
def fisher_weights_from_morphology(morph: Morphology) -> Tuple[float, float, float]:
    """
    Approximate Fisher information for each spatial dimension assuming a Gaussian
    with variance proportional to the square of the dimension.
    Weight w_i = 1 / (dim_i ** 2) and then normalise to sum to 1.
    Returns (w_len, w_wid, w_hei).
    """
    dims = np.array([morph.length, morph.width, morph.height], dtype=float)
    if np.any(dims <= 0):
        raise ValueError("Morphology dimensions must be positive")
    inv_var = 1.0 / (dims ** 2)  # I_i = 1/σ_i²
    total = inv_var.sum()
    return tuple(inv_var / total)  # normalized weights


# ----------------------------------------------------------------------
# SSIM core (Parent A) with Fisher weighting
# ----------------------------------------------------------------------
def ssim(
    x: Sequence[float],
    y: Sequence[float],
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Standard (unweighted) Structural Similarity Index."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def weighted_ssim(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float],
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Fisher‑weighted SSIM.
    The per‑sample values are first scaled by the provided weights (which must sum to 1).
    The scaling biases the mean and variance terms toward dimensions with higher Fisher information.
    """
    if len(x) != len(y) or len(x) != len(weights):
        raise ValueError("All sequences must have the same length")
    if not x:
        raise ValueError("samples must not be empty")
    # Apply weights
    wx = np.multiply(x, weights)
    wy = np.multiply(y, weights)

    n = len(wx)
    mx = float(np.sum(wx) / n)
    my = float(np.sum(wy) / n)
    vx = float(np.sum((wx - mx) ** 2) / n)
    vy = float(np.sum((wy - my) ** 2) / n)
    cov = float(np.sum((wx - mx) * (wy - my)) / n)

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Text → numeric conversion utilities (Parent A)
# ----------------------------------------------------------------------
def text_to_signal(text: str, length: int = 256) -> List[float]:
    """
    Convert a UTF‑8 string to a fixed‑size numeric signal.
    The conversion pads/truncates the ordinal values to `length` elements
    and normalises them to the range [0, dynamic_range].
    """
    # Convert characters to ordinal values
    ords = [ord(ch) for ch in text]
    # Pad or truncate
    if len(ords) < length:
        ords.extend([0] * (length - len(ords)))
    else:
        ords = ords[:length]
    # Normalise to 0‑255 (dynamic_range)
    max_ord = max(ords) if ords else 1
    scale = 255.0 / max_ord if max_ord else 1.0
    return [o * scale for o in ords]


# ----------------------------------------------------------------------
# Hybrid routing function (demonstrates integration)
# ----------------------------------------------------------------------
def route_packet(
    packet: Dict[str, Any],
    reference_text: str,
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> Tuple[bool, float]:
    """
    Decide whether to forward `packet` based on Fisher‑weighted SSIM between the packet's
    textual surface and `reference_text`.

    Returns (allowed, similarity_score).
    If the circuit breaker is open, the packet is rejected regardless of similarity.
    """
    if not breaker.allow():
        breaker.record_failure()
        return False, 0.0

    # Extract textual representation from the packet (Parent A logic)
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    # Convert both texts to numeric signals
    sig_packet = text_to_signal(text)
    sig_ref = text_to_signal(reference_text)

    # Obtain Fisher weights from morphology
    w_len, w_wid, w_hei = fisher_weights_from_morphology(morph)
    # Use the three weights to repeat across the signal length (simple broadcasting)
    # This creates a weight vector of the same length as the signals.
    weight_vector = np.array(
        [w_len, w_wid, w_hei] * (len(sig_packet) // 3 + 1)
    )[: len(sig_packet)]

    # Compute weighted SSIM
    similarity = weighted_ssim(sig_packet, sig_ref, weight_vector)

    # Threshold decision (tunable)
    threshold = 0.5
    allowed = similarity >= threshold
    if allowed:
        breaker.record_success()
    else:
        breaker.record_failure()
    return allowed, similarity


# ----------------------------------------------------------------------
# Simple Shapley‑like attribution using Fisher weights (illustrative)
# ----------------------------------------------------------------------
def shapley_attribution(
    features: Dict[str, float],
    morph: Morphology,
) -> Dict[str, float]:
    """
    Approximate Shapley values for a set of scalar features.
    The contribution of each feature is scaled by the Fisher weight of the corresponding
    morphological dimension (length→'size', width→'breadth', height→'depth').
    Any missing mapping receives an equal share.
    """
    # Map feature names to dimensions
    dim_map = {
        "size": morph.length,
        "breadth": morph.width,
        "depth": morph.height,
    }
    # Compute Fisher weights for the mapped dimensions
    base_weights = fisher_weights_from_morphology(morph)
    dim_weights = {
        "size": base_weights[0],
        "breadth": base_weights[1],
        "depth": base_weights[2],
    }

    # Initial equal allocation
    n = len(features)
    shapley: Dict[str, float] = {k: 0.0 for k in features}

    # For each feature, compute marginal contribution over all coalitions
    feature_items = list(features.items())
    for i, (fname, fvalue) in enumerate(feature_items):
        marginal = 0.0
        # Iterate over all subsets not containing the current feature
        for r in range(n):
            for coalition in combinations([j for j in range(n) if j != i], r):
                # Value of coalition without feature
                val_without = sum(
                    feature_items[j][1] * dim_weights.get(feature_items[j][0], 1.0 / n)
                    for j in coalition
                )
                # Value with feature added
                val_with = val_without + fvalue * dim_weights.get(fname, 1.0 / n)
                marginal += (val_with - val_without) / math.comb(n - 1, r)
        shapley[fname] = marginal / n
    return shapley


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal packet example
    packet_example = {
        "text_surface": "Hello, world!",
        "raw_command": None,
        "source": "sensor_A",
    }
    reference = "Hello, universe!"

    # Morphology instance
    morph = Morphology(length=2.5, width=1.0, height=0.8)

    # Circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    allowed, score = route_packet(packet_example, reference, morph, breaker)
    print(f"Routing decision: {'ALLOWED' if allowed else 'BLOCKED'} (SSIM={score:.4f})")
    print(f"Circuit breaker state: {breaker.as_dict()}")

    # Shapley demo
    feats = {"size": 10.0, "breadth": 5.0, "depth": 3.0}
    shap = shapley_attribution(feats, morph)
    print("Shapley‑like attribution:", shap)