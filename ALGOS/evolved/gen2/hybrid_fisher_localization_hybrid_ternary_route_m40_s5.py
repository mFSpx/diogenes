# DARWIN HAMMER — match 40, survivor 5
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_ternary_router_ssim_m1_s0.py (gen1)
# born: 2026-05-29T23:23:35Z

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    For a Gaussian beam I(θ) the Fisher information reduces to
        F(θ) = (θ‑center)² / width⁴ .
    The implementation follows the definition
        F = (∂I/∂θ)² / I
    but guards against division by zero.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def text_to_signal(text: str) -> List[float]:
    """Convert a Unicode string to a numeric signal (float list of code points)."""
    return [float(ord(ch)) for ch in text]


def weighted_ssim(
    x: Sequence[float],
    y: Sequence[float],
    theta: float,
    center: float,
    width: float,
    dynamic_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Weighted Structural Similarity Index.

    The weight for each sample is the Gaussian beam intensity at *theta*.
    This couples the Fisher‑information side (through the beam) directly to the
    similarity computation, yielding a deeper mathematical fusion.

    Parameters
    ----------
    x, y : sequences of equal length
        Numeric signals to compare.
    theta, center, width : float
        Parameters defining the Gaussian weighting function.
    dynamic_range : float, optional
        If omitted, the range is taken as ``max(x∪y) - min(x∪y)``.
    k1, k2 : float
        Stability constants as in the classic SSIM definition.

    Returns
    -------
    float
        Weighted SSIM value in ``[0, 1]`` (approximately).
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    # Compute per‑sample weights
    w = np.array([gaussian_beam(theta, center, width) for _ in range(len(x))])
    w_sum = w.sum()
    if w_sum == 0:
        raise ValueError("sum of weights is zero; check width and theta")

    # Convert to numpy for vectorised operations
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Weighted means
    mx = np.dot(w, x_arr) / w_sum
    my = np.dot(w, y_arr) / w_sum

    # Weighted variances
    vx = np.dot(w, (x_arr - mx) ** 2) / w_sum
    vy = np.dot(w, (y_arr - my) ** 2) / w_sum

    # Weighted covariance
    cov = np.dot(w, (x_arr - mx) * (y_arr - my)) / w_sum

    # Dynamic range based on actual data if not supplied
    if dynamic_range is None:
        data_min = min(min(x_arr), min(y_arr))
        data_max = max(max(x_arr), max(y_arr))
        dynamic_range = max(1.0, data_max - data_min)

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


def normalize_scores(scores: Sequence[float]) -> List[float]:
    """Scale a sequence of non‑negative scores to the interval [0, 1]."""
    if not scores:
        raise ValueError("scores must not be empty")
    max_score = max(scores)
    if max_score == 0:
        return [0.0 for _ in scores]
    return [s / max_score for s in scores]


def hybrid_metric(
    theta: float,
    center: float,
    width: float,
    packet_text: str,
    reference_text: str,
    gamma: float = 0.5,
) -> float:
    """Deeply fused quality metric.

    The metric combines a *normalized* Fisher information term with a
    *weight‑aware* SSIM term via a weighted geometric mean:

        H = (F_norm)^{γ} · (S_weighted)^{1‑γ}

    where ``γ ∈ [0, 1]`` controls the relative emphasis.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")

    # Fisher term (will be normalized later by the caller)
    f_raw = fisher_score(theta, center, width)

    # Weighted SSIM directly couples the angle to the similarity
    s = weighted_ssim(
        text_to_signal(packet_text),
        text_to_signal(reference_text),
        theta,
        center,
        width,
    )
    # Guard against negative or out‑of‑range values due to numerical issues
    s = max(0.0, min(1.0, s))

    # Return the raw product; normalization is performed in the selection routine
    return (f_raw ** gamma) * (s ** (1.0 - gamma))


def best_hybrid_angle(
    candidates: List[float],
    center: float,
    width: float,
    packet_text: str,
    reference_text: str,
    gamma: float = 0.5,
) -> float:
    """Select the angle that maximises the *normalized* hybrid metric.

    Normalization of the Fisher component across the candidate set ensures
    that the SSIM contribution is not drowned out by raw Fisher magnitudes.
    """
    if not candidates:
        raise ValueError("candidates required")

    # Raw Fisher scores for all candidates
    fisher_raw = [fisher_score(t, center, width) for t in candidates]
    fisher_norm = normalize_scores(fisher_raw)

    # Compute hybrid metric for each candidate using the pre‑normalised Fisher term
    metrics = []
    for t, f_n in zip(candidates, fisher_norm):
        s = weighted_ssim(
            text_to_signal(packet_text),
            text_to_signal(reference_text),
            t,
            center,
            width,
        )
        s = max(0.0, min(1.0, s))
        h = (f_n ** gamma) * (s ** (1.0 - gamma))
        metrics.append(h)

    # Choose the angle with the highest metric; tie‑breaker prefers proximity to centre
    best_idx = max(
        range(len(candidates)),
        key=lambda i: (metrics[i], -abs(candidates[i] - center)),
    )
    return candidates[best_idx]


def route_packet_hybrid(
    packet: Dict[str, Any],
    reference_text: str,
    center: float,
    width: float,
    angle_candidates: List[float],
    gamma: float = 0.5,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """Determine routing based on the deep hybrid metric.

    The function extracts the textual surface, selects the optimal angle,
    and assigns a priority based on the resulting metric.
    """
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("payload", {}).get("text", "")
        or ""
    )
    best_angle = best_hybrid_angle(
        angle_candidates, center, width, text, reference_text, gamma=gamma
    )
    metric = hybrid_metric(best_angle, center, width, text, reference_text, gamma=gamma)

    priority = "high_priority" if metric > threshold else "low_priority"

    return {
        "route": priority,
        "chosen_angle": best_angle,
        "hybrid_metric": metric,
        "intent": packet.get("normalized_intent")
        or packet.get("intent")
        or "unknown",
        "context": {
            "source": packet.get("source"),
            "payload": packet.get("payload") or {},
        },
    }


if __name__ == "__main__":
    # Smoke test: generate synthetic data and run the improved hybrid routing
    random.seed(0)

    centre_angle = 0.0
    beam_width = 1.0
    angle_pool = [i * 0.2 for i in range(-10, 11)]  # -2.0 … 2.0

    reference = "Reference command for routing"
    packet_example = {
        "text_surface": "Example packet payload",
        "normalized_intent": "demo_intent",
        "source": "unit_test",
        "payload": {"data": 42},
    }

    result = route_packet_hybrid(
        packet_example,
        reference,
        centre_angle,
        beam_width,
        angle_pool,
        gamma=0.6,
        threshold=0.02,
    )

    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"{k}: {v}")