# DARWIN HAMMER — match 2716, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m160_s2.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s3.py (gen3)
# born: 2026-05-29T23:43:43Z

"""
Hybrid Algorithm: Certainty‑Geometric Fisher‑SSIM (CGFF)

Parents:
- hybrid_hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s4.py (Hybrid Sheaf‑Certainty Cohomology, HSCC)
- hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s3.py (Fisher‑SSIM routing with Decision‑Hygiene pruning)

Mathematical Bridge:
The HSCC provides a certainty‑weighted coboundary operator that yields a scalar
certainty weight  w_c ∈ [0,1] for a given data section.  This weight is used as
the rotation angle  φ = π·w_c  of a 2‑D geometric‑algebra rotor R(φ).  The rotor
rotates the input signal vector before it is fed to the Fisher‑SSIM decision
metric of the second parent.  Simultaneously, the same certainty weight modulates
the Fisher information contribution w_f → w_f·w_c, creating a unified metric

    M(t) = p(t) · [ (w_f·w_c)·SSIM(R·x, y) + w_h·H·Σ_i w_i·f_i ]

where p(t) is a decreasing‑pruning probability, w_h is the normalized entropy
weight, H is Shannon entropy of extracted feature flags, and f_i are feature
counts.  This single expression fuses the topologies of both parents into a
coherent hybrid system.
"""

import os
import sys
import math
import random
import pathlib
import datetime
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A (certainty handling)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable representation of an epistemic certainty label."""
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str = "default"
    rationale: str = ""
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.datetime.now(
        datetime.timezone.utc).isoformat().replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")


def certainty_weight(flags: Iterable[CertaintyFlag]) -> float:
    """
    Aggregate a collection of CertaintyFlag objects into a single scalar weight
    in the interval [0,1].  The aggregation is the mean of normalized confidences.
    """
    confidences = [f.confidence_bps / 10000.0 for f in flags]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


# ----------------------------------------------------------------------
# Core utilities from Parent B (Fisher‑SSIM & pruning)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator if denominator != 0 else 0.0


def shannon_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy H = - Σ p_i log2 p_i."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def pruning_probability(t: int, decay: float = 0.99) -> float:
    """Decreasing pruning probability p(t) = decay^t."""
    if t < 0:
        raise ValueError("time step t must be non‑negative")
    return decay ** t


# ----------------------------------------------------------------------
# Geometric‑Algebra rotor utilities (2‑D for simplicity)
# ----------------------------------------------------------------------
def rotor_matrix(angle: float) -> np.ndarray:
    """Return a 2×2 rotation matrix for a given angle (radians)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def apply_rotor(vec: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """Rotate a 2‑D vector (or stack of vectors) with the given rotor."""
    if vec.ndim == 1:
        if vec.shape[0] != 2:
            raise ValueError("1‑D vector must have length 2")
        return rotor @ vec
    elif vec.ndim == 2:
        if vec.shape[1] != 2:
            raise ValueError("2‑D array must have shape (n,2)")
        return vec @ rotor.T
    else:
        raise ValueError("vec must be 1‑D or 2‑D array")


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_rotor_from_certainty(flags: Iterable[CertaintyFlag]) -> np.ndarray:
    """
    Build a GA rotor whose rotation angle φ = π·w_c,
    where w_c is the aggregated certainty weight.
    """
    w_c = certainty_weight(flags)          # ∈ [0,1]
    angle = math.pi * w_c                  # map to [0, π]
    return rotor_matrix(angle)


def hybrid_fisher_weight(theta: float, center: float, width: float,
                         eps: float = 1e-12,
                         certainty: float = 1.0) -> float:
    """
    Compute Fisher weight w_f = I/(I+ε) and modulate it by certainty w_c.
    """
    I = fisher_score(theta, center, width, eps)
    w_f = I / (I + eps)
    return w_f * certainty


def hybrid_decision_metric(x: np.ndarray,
                           y: np.ndarray,
                           theta: float,
                           center: float,
                           width: float,
                           flags: List[CertaintyFlag],
                           feature_counts: Dict[str, int],
                           t: int,
                           eps: float = 1e-12) -> float:
    """
    Unified metric M(t) = p(t)·[ (w_f·w_c)·SSIM(R·x, y) + w_h·H·Σ_i w_i·f_i ].
    """
    # 1️⃣ Certainty and rotor
    w_c = certainty_weight(flags)                     # certainty weight
    R = rotor_matrix(math.pi * w_c)                   # rotor from certainty
    x_rot = apply_rotor(x, R)                         # rotate input

    # 2️⃣ Fisher‑SSIM component
    w_f = hybrid_fisher_weight(theta, center, width, eps, w_c)
    ssim_val = ssim(x_rot, y)

    # 3️⃣ Hygiene (entropy) component
    H = shannon_entropy(feature_counts)
    total_feature_weight = sum(feature_counts.values())  # Σ_i w_i·f_i  (simple sum)
    w_h = H / (H + eps)                                   # normalized entropy weight

    # 4️⃣ Pruning probability
    p = pruning_probability(t)

    # 5️⃣ Assemble metric
    metric = p * (w_f * ssim_val + w_h * H * total_feature_weight)
    return metric


def hybrid_step(x: np.ndarray,
                y: np.ndarray,
                theta: float,
                center: float,
                width: float,
                flags: List[CertaintyFlag],
                raw_features: List[str],
                t: int) -> Tuple[float, np.ndarray]:
    """
    Perform a single hybrid iteration:
      * compute feature counts,
      * evaluate the decision metric,
      * update the rotor angle by a tiny gradient step proportional to the metric.
    Returns the metric and the new rotated vector.
    """
    # Feature counting (regex‑style placeholder)
    counts: Dict[str, int] = {}
    for token in raw_features:
        counts[token] = counts.get(token, 0) + 1

    # Metric evaluation
    metric = hybrid_decision_metric(x, y, theta, center, width,
                                    flags, counts, t)

    # Gradient‑like rotor update: increase angle if metric is high, decrease otherwise
    grad = 0.001 * (metric - 0.5)               # simple heuristic
    current_angle = math.pi * certainty_weight(flags)
    new_angle = (current_angle + grad) % (2 * math.pi)
    new_rotor = rotor_matrix(new_angle)
    x_new = apply_rotor(x, new_rotor)

    return metric, x_new


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic 2‑D signals
    rng = np.random.default_rng(42)
    x_signal = rng.random((100, 2)) * 255.0
    y_signal = rng.random((100, 2)) * 255.0

    # Certainty flags (randomly generated)
    sample_flags = [
        CertaintyFlag(label=random.choice(EPISTEMIC_FLAGS),
                      confidence_bps=random.randint(2000, 10000))
        for _ in range(5)
    ]

    # Dummy feature tokens
    dummy_features = ["alpha", "beta", "alpha", "gamma", "beta", "beta"]

    # Parameters for Fisher Gaussian
    theta_val = 0.3
    center_val = 0.0
    width_val = 1.0

    # Run a few hybrid steps
    for step in range(5):
        metric_val, x_signal = hybrid_step(
            x_signal,
            y_signal,
            theta=theta_val,
            center=center_val,
            width=width_val,
            flags=sample_flags,
            raw_features=dummy_features,
            t=step
        )
        print(f"Step {step}: metric = {metric_val:.6f}")
    print("Hybrid CGFF execution completed without errors.")