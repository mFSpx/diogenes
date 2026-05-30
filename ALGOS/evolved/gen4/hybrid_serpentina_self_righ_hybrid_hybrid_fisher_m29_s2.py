# DARWIN HAMMER — match 29, survivor 2
# gen: 4
# parent_a: serpentina_self_righting.py (gen0)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s5.py (gen3)
# born: 2026-05-29T23:26:24Z

"""Hybrid Morphology‑Beam Fusion Module

This module merges the core mathematics of two parent algorithms:

* **Parent A – serpentina_self_righting.py** – provides morphology‑based indices
  (sphericity, flatness) and a right‑ing time model `righting_time_index`.
* **Parent B – hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s5.py** – supplies
  Gaussian‑beam optics (`gaussian_beam`), Fisher information for that beam
  (`fisher_score`) and a structural‑similarity metric (`ssim`).

**Mathematical bridge**

The bridge treats the morphology of an object as a *parameterisation* of a
Gaussian beam:

* The *center* of the beam is set to the **sphericity index** (a dimensionless
  measure of compactness).
* The *width* of the beam is taken from the **flatness index** (how flattened the
  shape is).
* The **righting time index** `R` is interpreted as an energy‑scale factor that
  weights the beam’s intensity and the resulting Fisher information.

Thus every hybrid function first builds a Gaussian beam whose shape is dictated
by morphology, then combines the beam’s statistical descriptors with the
morphology‑derived recovery priority. The result is a unified scalar or a
similarity score that simultaneously reflects geometric and informational
aspects of the system.

The module offers three representative hybrid operations:

1. `hybrid_morph_beam` – intensity of a morphology‑driven beam.
2. `hybrid_fisher_morph` – Fisher information weighted by recovery priority.
3. `hybrid_similarity` – SSIM between two signals, modulated by morphology.

All functions are pure Python, rely only on the standard library and NumPy,
and can be used independently or as part of a larger workflow.
"""

from __future__ import annotations

import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology primitives
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑less compactness: cubic root of volume divided by the longest edge."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Ratio of lateral dimensions to vertical dimension (higher → flatter)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical model of self‑righting time."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority (0..1) for external assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent B – Gaussian beam, Fisher information, SSIM
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity I(θ) = exp[-½ ((θ‑center)/width)²]."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(
    theta: float, center: float, width: float, eps: float = 1e-12
) -> float:
    """Fisher information for a Gaussian beam at angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(
    x: np.ndarray,
    y: np.ndarray,
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """1‑D Structural Similarity Index Measure."""
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
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid Functions (mathematical fusion)
# ----------------------------------------------------------------------


def _morph_to_beam_params(m: Morphology) -> tuple[float, float]:
    """
    Convert morphology to Gaussian‑beam parameters.

    Returns
    -------
    center : float
        sphericity index (dimensionless) – serves as beam centre.
    width : float
        flatness index – serves as beam width; clipped to a minimum of 0.01
        to avoid degenerate beams.
    """
    center = sphericity_index(m.length, m.width, m.height)
    width = max(flatness_index(m.length, m.width, m.height), 0.01)
    return center, width


def hybrid_morph_beam(m: Morphology, theta: float) -> float:
    """
    Beam intensity at angle ``theta`` whose geometry is dictated by ``m``.

    The intensity is further scaled by the right‑ing time index `R`,
    treating `R` as an energy‑scale factor.

    Parameters
    ----------
    m : Morphology
        Physical description of the object.
    theta : float
        Observation angle (radians or degrees – consistent with downstream use).

    Returns
    -------
    float
        Scaled Gaussian‑beam intensity.
    """
    center, width = _morph_to_beam_params(m)
    base_intensity = gaussian_beam(theta, center, width)
    R = righting_time_index(m)
    return base_intensity * R


def hybrid_fisher_morph(m: Morphology, theta: float) -> float:
    """
    Fisher information of the morphology‑driven beam, weighted by recovery priority.

    The priority `P` (0..1) acts as a multiplier, emphasising objects that
    are harder to self‑right.

    Parameters
    ----------
    m : Morphology
    theta : float

    Returns
    -------
    float
        Weighted Fisher information.
    """
    center, width = _morph_to_beam_params(m)
    F = fisher_score(theta, center, width)
    P = recovery_priority(m)
    return F * P


def hybrid_similarity(
    m: Morphology,
    ref_signal: np.ndarray,
    test_signal: np.ndarray,
    weight: float = 0.5,
) -> float:
    """
    Morphology‑modulated SSIM between two 1‑D signals.

    The raw SSIM is blended with the morphology’s recovery priority:
    ``S_hybrid = (1‑weight) * SSIM + weight * priority``.

    Parameters
    ----------
    m : Morphology
        Provides the priority term.
    ref_signal, test_signal : np.ndarray
        Signals to compare; must share shape.
    weight : float, optional
        Blend factor (default 0.5). Must be in [0, 1].

    Returns
    -------
    float
        Hybrid similarity score in the range [0, 1].
    """
    if not (0.0 <= weight <= 1.0):
        raise ValueError("weight must be between 0 and 1")
    raw_ssim = ssim(ref_signal, test_signal)
    priority = recovery_priority(m)
    return (1 - weight) * raw_ssim + weight * priority


# ----------------------------------------------------------------------
# Regex‑based evidence extraction (demonstrates integration of Parent B utilities)
# ----------------------------------------------------------------------


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def extract_tags(text: str) -> dict[str, list[str]]:
    """
    Simple regex extraction returning found evidence and planning keywords.

    Returns
    -------
    dict with keys ``'evidence'`` and ``'planning'`` mapping to lists of matches.
    """
    return {
        "evidence": EVIDENCE_RE.findall(text),
        "planning": PLANNING_RE.findall(text),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a representative morphology
    demo_morph = Morphology(length=0.35, width=0.25, height=0.10, mass=1.2)

    # Angle for beam calculations
    theta_demo = 0.3

    # Hybrid beam intensity
    intensity = hybrid_morph_beam(demo_morph, theta_demo)
    print(f"Hybrid beam intensity: {intensity:.4f}")

    # Hybrid Fisher information
    fisher = hybrid_fisher_morph(demo_morph, theta_demo)
    print(f"Hybrid Fisher information: {fisher:.6f}")

    # Generate two synthetic signals
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.0, scale=1.0, size=256)
    test = ref + rng.normal(loc=0.0, scale=0.2, size=256)

    # Hybrid similarity
    sim = hybrid_similarity(demo_morph, ref, test, weight=0.4)
    print(f"Hybrid SSIM-like similarity: {sim:.4f}")

    # Regex extraction demo
    sample_text = (
        "The evidence was logged in the report. We need to plan the next steps and "
        "schedule a verification checkpoint."
    )
    tags = extract_tags(sample_text)
    print(f"Extracted tags: {tags}")

    # Ensure no unhandled exceptions
    sys.exit(0)