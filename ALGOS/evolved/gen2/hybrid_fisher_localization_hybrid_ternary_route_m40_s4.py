# DARWIN HAMMER — match 40, survivor 4
# gen: 2
# parent_a: fisher_localization.py (gen0)
# parent_b: hybrid_ternary_router_ssim_m1_s0.py (gen1)
# born: 2026-05-29T23:23:35Z

import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence, List, Dict
import numpy as np

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
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
    return [float(ord(ch)) for ch in text]

def hybrid_metric(theta: float, center: float, width: float,
                  packet_text: str, reference_text: str, alpha: float = 0.5) -> float:
    f = fisher_score(theta, center, width)
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))
    return alpha * f + (1 - alpha) * s

def best_hybrid_angle(candidates: List[float], center: float, width: float,
                      packet_text: str, reference_text: str, alpha: float = 0.5) -> float:
    if not candidates:
        raise ValueError("candidates required")
    return max(
        candidates,
        key=lambda t: hybrid_metric(t, center, width, packet_text, reference_text, alpha)
    )

def route_packet_hybrid(packet: Dict[str, Any], reference_text: str,
                        center: float, width: float,
                        angle_candidates: List[float], alpha: float = 0.5) -> Dict[str, Any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or "")
    best_angle = best_hybrid_angle(angle_candidates, center, width, text, reference_text, alpha)
    metric = hybrid_metric(best_angle, center, width, text, reference_text, alpha)

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
    random.seed(0)

    centre_angle = 0.0
    beam_width = 1.0
    angle_pool = [i * 0.2 for i in range(-10, 11)]  

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