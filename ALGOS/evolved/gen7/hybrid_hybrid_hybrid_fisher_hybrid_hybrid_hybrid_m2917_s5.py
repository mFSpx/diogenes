# DARWIN HAMMER — match 2917, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1344_s0.py (gen6)
# born: 2026-05-29T23:46:39Z

"""Hybrid Algorithm integrating Parent A (Gaussian beam, Fisher information, SSIM) 
and Parent B (epistemic certainty flags, confidence weighting, Gini coefficient).

Mathematical bridge:
- Parent A provides a scalar information measure I(θ)=Fisher_score(θ) for each
  Gaussian beam parameterisation.
- Parent B supplies a confidence weight w ∈ [0,1] derived from the epistemic
  flag's confidence_bps and a temporal decay factor τ based on the flag's
  generated_at timestamp.
- The fusion builds a weighted Fisher matrix  F_ij = w_i · I_j, where i indexes
  certainty flags and j indexes beam angles.  This matrix is then analysed
  with a Gini coefficient G(F) to capture inequality of information distribution,
  and combined with a Structural Similarity Index Measure (SSIM) between two
  intensity profiles to produce a final hybrid metric H.

The module implements the above pipeline and provides a smoke‑test.
"""

import sys
import math
import random
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple, Dict, Union, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent B components – epistemic certainty
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for an epistemic certainty label."""
    label: str
    confidence_bps: int               # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    @property
    def confidence(self) -> float:
        """Confidence as a proportion in [0,1]."""
        return self.confidence_bps / 10_000.0

    @property
    def age_seconds(self) -> float:
        """Age of the flag in seconds (now – generated_at)."""
        try:
            gen = datetime.fromisoformat(self.generated_at.rstrip('Z')).replace(tzinfo=timezone.utc)
        except Exception:
            gen = datetime.now(timezone.utc)
        return (datetime.now(timezone.utc) - gen).total_seconds()


def temporal_decay(age_seconds: float, half_life: float = 86400.0) -> float:
    """Exponential decay factor based on age; half_life default = 1 day."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    return 0.5 ** (age_seconds / half_life)


# ----------------------------------------------------------------------
# Fusion functions
# ----------------------------------------------------------------------
def build_weighted_fisher_matrix(
    thetas: np.ndarray,
    flags: List[CertaintyFlag],
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """
    Construct a matrix F where
        F[i, j] = w_i * I(theta_j; center_j, width_j)
    with w_i = confidence_i * temporal_decay(age_i).

    Parameters
    ----------
    thetas : (M,) array of angle samples.
    flags  : list of N CertaintyFlag objects.
    centers: (M,) array of beam centers (same length as thetas).
    widths : (M,) array of beam widths (same length as thetas).

    Returns
    -------
    F : (N, M) ndarray of weighted Fisher information.
    """
    if not (len(thetas) == len(centers) == len(widths)):
        raise ValueError("thetas, centers, and widths must have identical length")

    # Vectorised Fisher scores for each theta
    fisher_vec = np.vectorize(fisher_score)
    I = fisher_vec(thetas, centers, widths)          # shape (M,)

    # Build weight vector from flags
    w = np.array([f.confidence * temporal_decay(f.age_seconds) for f in flags])  # (N,)

    # Outer product yields weighted matrix
    F = np.outer(w, I)                                 # (N, M)
    return F


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array.
    The Gini is 0 for perfect equality and 1 for maximal inequality.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    if np.any(values < 0):
        raise ValueError("values must be non‑negative")
    sorted_vals = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


def hybrid_metric(
    thetas: np.ndarray,
    flags: List[CertaintyFlag],
    centers: np.ndarray,
    widths: np.ndarray,
    reference_intensity: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute a hybrid quality metric consisting of:
        1. Gini coefficient of the weighted Fisher matrix (captures epistemic inequality).
        2. SSIM between the aggregated intensity profile and a reference profile.
        3. Mean weighted Fisher information (overall information content).

    Returns
    -------
    (gini, ssim_score, mean_fisher)
    """
    # 1. Weighted Fisher matrix
    F = build_weighted_fisher_matrix(thetas, flags, centers, widths)

    # Gini on flattened matrix
    gini = gini_coefficient(F.ravel())

    # 2. Aggregate intensity profile (sum over flags, i.e. sum over rows)
    aggregated_intensity = np.sum(F, axis=0)

    # Normalise both profiles to the same dynamic range for SSIM
    max_dyn = max(aggregated_intensity.max(), reference_intensity.max(), 1e-12)
    agg_norm = (aggregated_intensity / max_dyn) * 255.0
    ref_norm = (reference_intensity / max_dyn) * 255.0
    ssim_score = ssim(agg_norm, ref_norm)

    # 3. Mean Fisher (weighted)
    mean_fisher = np.mean(F)

    return gini, ssim_score, mean_fisher


# ----------------------------------------------------------------------
# Example / Smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    # Sample angle domain
    thetas = np.linspace(-5.0, 5.0, 101)

    # Beam parameters (centered at 0, width 1.0)
    centers = np.zeros_like(thetas)
    widths = np.full_like(thetas, 1.0)

    # Reference intensity: pure Gaussian beam (no weighting)
    ref_intensity = np.array([gaussian_beam(t, 0.0, 1.0) for t in thetas])

    # Create a few epistemic flags with varying confidence and ages
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    flags = [
        CertaintyFlag(label="FACT", confidence_bps=9000, authority_class="A", rationale="ground truth", generated_at=now_iso),
        CertaintyFlag(label="PROBABLE", confidence_bps=6000, authority_class="B", rationale="model estimate", generated_at=now_iso),
        CertaintyFlag(label="POSSIBLE", confidence_bps=3000, authority_class="C", rationale="speculation", generated_at=now_iso),
    ]

    gini, ssim_score, mean_fisher = hybrid_metric(thetas, flags, centers, widths, ref_intensity)

    print(f"Gini coefficient   : {gini:.4f}")
    print(f"SSIM (vs reference) : {ssim_score:.4f}")
    print(f"Mean weighted Fisher: {mean_fisher:.6e}")


if __name__ == "__main__":
    _smoke_test()