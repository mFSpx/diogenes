# DARWIN HAMMER — match 40, survivor 2
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_ternary_router_ssim_m1_s0.py (gen1)
# born: 2026-05-29T23:23:35Z

"""Hybrid Fisher‑SSIM Algorithm

Parent A: fisher_localization.py – provides a Gaussian beam model and Fisher‑information
score for a continuous parameter (angle θ).

Parent B: hybrid_ternary_router_ssim_m1_s0.py – provides a structural similarity index
(SSIM) between two 1‑D signals (here used on the Unicode code‑point series of text).

Mathematical bridge:
Both algorithms produce a scalar “quality” measure for a candidate.  The Fisher score
measures how sharply the intensity I(θ) changes with θ, while SSIM measures how
similar two discrete signals are.  By interpreting the Fisher score as an
information‑weight and the SSIM as a contextual similarity weight, we can fuse them
into a single hybrid metric

    H(θ, text) = F(θ) · S(text, reference)

where F(θ) is the Fisher information for angle θ and S(·) is the SSIM between the
Unicode‑code‑point representation of a packet’s textual surface and a reference
string.  The product preserves the ordering of each component and yields a unified
criterion for selecting the optimal angle *and* routing decision.

The module implements the fused operations and demonstrates their use."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence, List, Dict

import numpy as np


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
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


def text_to_signal(text: str) -> List[float]:
    """Convert a Unicode string to a numeric signal (code‑point float list)."""
    return [float(ord(ch)) for ch in text]


def hybrid_metric(theta: float, center: float, width: float,
                  packet_text: str, reference_text: str) -> float:
    """Combined quality metric H = Fisher(θ) × SSIM(text, reference)."""
    f = fisher_score(theta, center, width)
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))
    return f * s


def best_hybrid_angle(candidates: List[float], center: float, width: float,
                      packet_text: str, reference_text: str) -> float:
    """Select the angle that maximises the hybrid metric.

    Tie‑breaker: choose the angle closest to the centre when metrics are equal.
    """
    if not candidates:
        raise ValueError("candidates required")
    return max(
        candidates,
        key=lambda t: (hybrid_metric(t, center, width, packet_text, reference_text),
                       -abs(t - center))
    )


def route_packet_hybrid(packet: Dict[str, Any], reference_text: str,
                        center: float, width: float,
                        angle_candidates: List[float]) -> Dict[str, Any]:
    """Determine routing based on the hybrid metric.

    The packet's textual surface is used for the SSIM part, while the optimal
    angle (from the Fisher part) influences the priority.
    """
    text = str(packet.get("text_surface") or packet.get("raw_command") or "")
    best_angle = best_hybrid_angle(angle_candidates, center, width, text, reference_text)
    metric = hybrid_metric(best_angle, center, width, text, reference_text)

    # Simple priority rule: high priority if hybrid metric exceeds a heuristic threshold
    priority = "high_priority" if metric > 0.1 else "low_priority"

    return {
        "route": priority,
        "chosen_angle": best_angle,
        "hybrid_metric": metric,
        "intent": packet.get("normalized_intent") or packet.get("intent") or "unknown",
        "context": {
            "source": packet.get("source"),
            "payload": packet.get("payload") or {}
        }
    }


if __name__ == "__main__":
    # Smoke test: generate synthetic data and run the hybrid routing
    random.seed(0)

    # Example parameters
    centre_angle = 0.0
    beam_width = 1.0
    angle_pool = [i * 0.2 for i in range(-10, 11)]  # -2.0 … 2.0

    reference = "Reference command for routing"
    packet_example = {
        "text_surface": "Example packet payload",
        "normalized_intent": "demo_intent",
        "source": "unit_test",
        "payload": {"data": 42}
    }

    result = route_packet_hybrid(packet_example, reference,
                                 centre_angle, beam_width, angle_pool)

    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"{k}: {v}")