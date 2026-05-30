# DARWIN HAMMER — match 2365, survivor 0
# gen: 4
# parent_a: fold_change_detection.py (gen0)
# parent_b: hybrid_infotaxis_hybrid_semantic_neig_m739_s3.py (gen3)
# born: 2026-05-29T23:41:56Z

"""
Hybrid Fold-Change Detection and Entropy-Morphology Search
Parents:
- fold_change_detection.py (fold-change detection update equations)
- hybrid_infotaxis_hybrid_semantic_neig_m739_s3.py (hybrid semantic-morphology neighbor system)

Mathematical Bridge:
The recovery priority `p ∈ [0,1]` derived from a document's morphology in the hybrid semantic-morphology 
neighbor system modulates the gain in the fold-change detection update equations.

The governing equations of both parents are integrated through the expected gain calculation, 
which now depends on both the fold-change detection state and the morphology-driven recovery priority.

    gain = p * gain  

where `p` is the recovery priority and `gain` is the original gain in fold-change detection.
"""

import math
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def step(u: float, x: float, y: float, dt: float = 1.0, gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12) -> tuple[float, float]:
    """Advance the feed-forward state using Euler integration."""
    if dt < 0:
        raise ValueError('dt must be non-negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def hybrid_step(u: float, x: float, y: float, m: Morphology, dt: float = 1.0, original_gain: float = 1.0, decay_x: float = 1.0, decay_y: float = 1.0, eps: float = 1e-12, max_index: float = 10.0) -> tuple[float, float, float]:
    p = recovery_priority(m, max_index)
    gain = p * original_gain
    x, y = step(u, x, y, dt, gain, decay_x, decay_y, eps)
    return x, y, p

def response_series(inputs: list[float], m: Morphology, x0: float = 1.0, y0: float = 0.0, **kw) -> list[tuple[float, float, float]]:
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y, p = hybrid_step(u, x, y, m, **kw)
        out.append((x, y, p))
    return out

if __name__ == "__main__":
    m = Morphology(1.0, 2.0, 3.0, 4.0)
    inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = response_series(inputs, m)
    print(result)