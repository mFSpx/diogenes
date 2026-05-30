# DARWIN HAMMER — match 2770, survivor 5
# gen: 5
# parent_a: hybrid_ternary_router_ssim_m1_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (gen4)
# born: 2026-05-29T23:45:44Z

"""Hybrid algorithm combining SSIM-based similarity routing with Fisher information
dimensionality reduction and geometric scaling.

Parents:
- hybrid_ternary_router_ssim_m1_s0.py (SSIM similarity for packet text)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (Fisher information,
  sphericity index, circuit breaker)

Mathematical bridge:
The scalar SSIM similarity `S ∈ [0,1]` computed between a packet's textual surface
and a reference text is used as a weighting factor for the Fisher Information
Matrix (FIM) of high‑dimensional feature vectors.  The FIM encodes the amount of
information each feature direction carries about class labels.  By scaling the
FIM with `S` and with a geometric factor derived from the sphericity index of a
`Morphology` object, we obtain a *similarity‑aware* information metric:

    F̂ = S · σ_sph · FIM

where `σ_sph = sphericity_index(length, width, height)`.  Eigen‑decomposition of
`F̂` yields principal directions that respect both perceptual similarity and
geometric context, enabling a unified routing‑and‑dimensionality‑reduction
procedure.

The module provides three core functions demonstrating this hybrid operation:
`ssim`, `fisher_information`, and `similarity_aware_reduction`.  An optional
`EndpointCircuitBreaker` can be used to gate routing decisions based on recent
failures.

"""

import sys
import math
import random
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Sequence, Tuple, List, Dict

import numpy as np


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for one‑dimensional signals.

    Returns a value in [0, 1] where 1 means perfect similarity.
    """
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


def text_to_signal(text: str) -> List[float]:
    """Convert a Unicode string to a numeric signal suitable for SSIM."""
    # Simple mapping: Unicode code point normalized to [0, dynamic_range]
    dr = 255.0
    return [float(ord(ch) % int(dr + 1)) for ch in text]


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric scaling factor: (geometric mean) / (max dimension)."""
    dims = np.array([length, width, height], dtype=float)
    if np.any(dims <= 0):
        raise ValueError("All dimensions must be positive")
    geo_mean = np.prod(dims) ** (1.0 / 3.0)
    return geo_mean / dims.max()


@dataclass(frozen=True)
class Morphology:
    """Physical or logical size descriptor."""
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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

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


def fisher_information(features: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
    """Estimate a Fisher Information Matrix under a Gaussian class‑conditional model.

    Parameters
    ----------
    features : (n_samples, n_features) array
        Input data.
    labels : (n_samples,) array
        Discrete class labels (assumed integer‑encoded).

    Returns
    -------
    fim : (n_features, n_features) array
        Symmetric positive‑semidefinite Fisher Information Matrix.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2‑D array")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1‑D array")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of rows")

    classes = np.unique(labels)
    n_features = features.shape[1]
    overall_mean = features.mean(axis=0)

    fim = np.zeros((n_features, n_features), dtype=float)

    for c in classes:
        idx = labels == c
        pc = idx.mean()  # class prior
        if pc == 0:
            continue
        class_feat = features[idx]
        mu_c = class_feat.mean(axis=0)
        diff = (mu_c - overall_mean).reshape(-1, 1)
        fim += pc * (diff @ diff.T)

    # Add a regularization term to keep the matrix invertible
    eps = 1e-6
    fim += eps * np.eye(n_features)
    return fim


def similarity_aware_reduction(packet: Dict[str, Any],
                               reference_text: str,
                               features: np.ndarray,
                               labels: np.ndarray,
                               morphology: Morphology,
                               target_dim: int = 2) -> Tuple[np.ndarray, bool]:
    """Hybrid operation:
    1. Compute SSIM similarity between packet text and reference.
    2. Scale Fisher Information with similarity and sphericity.
    3. Perform eigen‑decomposition and project features onto the top `target_dim`
       components.
    4. Return the reduced feature matrix and a routing decision (True = forward).

    The routing decision is gated by an `EndpointCircuitBreaker` that opens when
    similarity falls below a dynamic threshold.
    """
    # ---- 1. SSIM similarity ----
    pkt_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    pkt_signal = text_to_signal(pkt_text)
    ref_signal = text_to_signal(reference_text)
    # Pad shorter signal to match length
    max_len = max(len(pkt_signal), len(ref_signal))
    pkt_signal += [0.0] * (max_len - len(pkt_signal))
    ref_signal += [0.0] * (max_len - len(ref_signal))
    similarity = ssim(pkt_signal, ref_signal)

    # ---- 2. Geometric scaling ----
    sph_factor = sphericity_index(morphology.length,
                                  morphology.width,
                                  morphology.height)

    # ---- 3. Fisher Information ----
    fim = fisher_information(features, labels)

    # Weighted information matrix
    weighted_fim = similarity * sph_factor * fim

    # ---- 4. Eigen‑decomposition ----
    eig_vals, eig_vecs = np.linalg.eigh(weighted_fim)
    # Sort eigenvalues descending
    idx = np.argsort(eig_vals)[::-1]
    top_vecs = eig_vecs[:, idx[:target_dim]]
    reduced = features @ top_vecs

    # ---- 5. Routing decision ----
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    similarity_threshold = 0.5  # heuristic
    if similarity < similarity_threshold:
        breaker.record_failure()
    else:
        breaker.record_success()
    route_forward = breaker.allow()

    return reduced, route_forward


def demo():
    """Simple smoke test exercising the hybrid pipeline."""
    # Mock packet
    packet = {
        "text_surface": "Hello world! This is a test packet.",
        "source": "sensor_A"
    }
    reference = "Hello world! Reference message for similarity."

    # Random feature matrix (e.g., sensor readings) and binary labels
    rng = np.random.default_rng(42)
    features = rng.normal(loc=0.0, scale=1.0, size=(100, 8))
    labels = rng.integers(low=0, high=2, size=100)

    # Morphology instance
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)

    reduced, forward = similarity_aware_reduction(
        packet=packet,
        reference_text=reference,
        features=features,
        labels=labels,
        morphology=morph,
        target_dim=3,
    )

    print(f"Reduced shape: {reduced.shape}")
    print(f"Routing decision (forward=True): {forward}")
    print(f"First reduced sample: {reduced[0]}")


if __name__ == "__main__":
    demo()