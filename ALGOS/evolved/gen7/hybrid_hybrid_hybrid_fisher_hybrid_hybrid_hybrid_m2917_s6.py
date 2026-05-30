# DARWIN HAMMER — match 2917, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1344_s0.py (gen6)
# born: 2026-05-29T23:46:39Z

"""Hybrid Fusion Module

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Gaussian beam, Fisher information, and 1‑D SSIM.
* **Parent B** – Epistemic certainty flags (confidence expressed in basis points) and
  statistical aggregation (Gini coefficient).

The mathematical bridge is a **certainty‑weighted Fisher information matrix**.
The Gaussian beam evaluated over a vector of angles produces an intensity vector **I**.
Its Fisher information vector **F** is computed element‑wise.  
Each epistemic flag contributes a confidence weight **w = confidence_bps / 10 000**.
These weights are assembled into a diagonal matrix **W** and applied to **F**
via matrix multiplication:  


F̂ = W @ F


The resulting weighted Fisher vector **F̂** is then used in two downstream
operations that combine the two parents’ ideas:

1. **Hybrid SSIM** – structural similarity between the normalized Gaussian
   intensity vector and the normalized weighted Fisher vector.
2. **Gini coefficient** – a dispersion measure applied to **F̂**, reflecting
   epistemic certainty distribution.

All functions are pure NumPy / standard‑library code and can be used independently
or as part of the full hybrid pipeline.
"""

import math
import random
import sys
from pathlib import Path
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple, Dict, Union, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (Gaussian beam, Fisher information, SSIM)
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity for a single angle."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def gaussian_beam_vector(thetas: np.ndarray, center: float, width: float) -> np.ndarray:
    """Vectorised Gaussian beam over an array of angles."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (thetas - center) / width
    return np.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def fisher_information_vector(thetas: np.ndarray, center: float, width: float, eps: float = 1e-12) -> np.ndarray:
    """Vectorised Fisher information for an array of angles."""
    intensity = np.maximum(gaussian_beam_vector(thetas, center, width), eps)
    derivative = intensity * (-(thetas - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim_1d(x: np.ndarray, y: np.ndarray,
            dynamic_range: float = 255.0,
            k1: float = 0.01,
            k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent B components (Epistemic certainty flags, Gini coefficient)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for an epistemic certainty label."""
    label: str
    confidence_bps: int  # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty_weight_matrix(flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Build a diagonal matrix whose diagonal entries are the normalized confidence
    weights (0‑1) derived from a list of CertaintyFlag objects.
    """
    if not flags:
        raise ValueError("flags list must not be empty")
    weights = np.array([f.confidence_bps / 10_000.0 for f in flags], dtype=float)
    return np.diag(weights)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array.
    The implementation follows the standard definition:
        G = (2 * Σ_i i * x_i) / (n * Σ_i x_i) - (n + 1) / n
    where x_i are sorted in non‑decreasing order.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    if np.any(values < 0):
        raise ValueError("values must be non‑negative")
    n = values.size
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_vals)) / (n * sum_vals) - (n + 1) / n
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def weighted_fisher(thetas: np.ndarray,
                    center: float,
                    width: float,
                    flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Compute the certainty‑weighted Fisher information vector.

    Steps:
    1. Compute raw Fisher information **F** over ``thetas``.
    2. Build diagonal weight matrix **W** from ``flags``.
    3. If the number of flags differs from the length of ``thetas``,
       the weight vector is broadcast (repeated) to match the size.
    4. Return ``F̂ = W @ F`` (matrix‑vector product).
    """
    raw_fisher = fisher_information_vector(thetas, center, width)  # shape (n,)
    W = certainty_weight_matrix(flags)  # shape (m, m)

    if W.shape[0] != raw_fisher.shape[0]:
        # Broadcast by repeating the weight pattern to the required length.
        repeats = int(math.ceil(raw_fisher.shape[0] / W.shape[0]))
        W = np.kron(np.eye(repeats, dtype=float), W)[: raw_fisher.shape[0], : raw_fisher.shape[0]]

    weighted = W @ raw_fisher
    return weighted


def hybrid_ssim(thetas: np.ndarray,
                center: float,
                width: float,
                flags: List[CertaintyFlag]) -> float:
    """
    Compute SSIM between the normalized Gaussian intensity vector and the
    normalized certainty‑weighted Fisher vector.
    """
    intensity = gaussian_beam_vector(thetas, center, width)
    weighted_fisher = weighted_fisher(thetas, center, width, flags)

    # Normalise both signals to the same dynamic range (0‑255) for SSIM.
    def _norm_to_range(v):
        if v.max() == v.min():
            return np.zeros_like(v)
        return 255.0 * (v - v.min()) / (v.max() - v.min())

    x = _norm_to_range(intensity)
    y = _norm_to_range(weighted_fisher)
    return ssim_1d(x, y)


def hybrid_dispersion_metric(thetas: np.ndarray,
                             center: float,
                             width: float,
                             flags: List[CertaintyFlag]) -> Tuple[float, float]:
    """
    Return a tuple ``(gini, weighted_mean)`` where:
    * ``gini`` is the Gini coefficient of the certainty‑weighted Fisher vector.
    * ``weighted_mean`` is the mean of that vector, providing a scale reference.
    """
    weighted = weighted_fisher(thetas, center, width, flags)
    gini = gini_coefficient(weighted)
    mean = float(weighted.mean())
    return gini, mean


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a synthetic angle domain.
    thetas = np.linspace(-5.0, 5.0, 101)

    # Define a small set of epistemic flags.
    flags = [
        CertaintyFlag(label="FACT", confidence_bps=9500, authority_class="A1", rationale="empirical"),
        CertaintyFlag(label="PROBABLE", confidence_bps=7000, authority_class="B2", rationale="model"),
        CertaintyFlag(label="POSSIBLE", confidence_bps=4000, authority_class="C3", rationale="speculation"),
    ]

    # Parameters for the Gaussian beam.
    center = 0.0
    width = 1.2

    # Run hybrid calculations.
    wf = weighted_fisher(thetas, center, width, flags)
    print(f"Weighted Fisher (first 5 values): {wf[:5]}")

    ssim_val = hybrid_ssim(thetas, center, width, flags)
    print(f"Hybrid SSIM: {ssim_val:.6f}")

    gini_val, mean_val = hybrid_dispersion_metric(thetas, center, width, flags)
    print(f"Gini of weighted Fisher: {gini_val:.4f}")
    print(f"Mean of weighted Fisher: {mean_val:.6f}")

    # Simple sanity check: all outputs should be finite numbers.
    assert np.all(np.isfinite(wf)), "Weighted Fisher contains non‑finite values"
    assert math.isfinite(ssim_val), "SSIM is not finite"
    assert math.isfinite(gini_val) and math.isfinite(mean_val), "Dispersion metrics not finite"

    print("Hybrid fusion module executed successfully.")