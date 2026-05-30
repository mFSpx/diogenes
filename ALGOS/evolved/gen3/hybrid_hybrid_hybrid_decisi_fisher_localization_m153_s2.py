# DARWIN HAMMER — match 153, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s4.py (gen2)
# parent_b: fisher_localization.py (gen0)
# born: 2026-05-29T23:25:52Z

"""Hybrid algorithm combining regex‑based feature scoring (Parent A) with Fisher‑information angle selection (Parent B).

Mathematical bridge:
- Parent A produces a non‑negative feature vector **v** = (v₁,…,vₙ) where vᵢ is the count (or weighted presence) of the i‑th regex feature.
- Parent B treats a scalar angle θ and defines a Gaussian “beam” intensity I(θ; μ, σ) and its Fisher information F(θ) = (∂I/∂θ)² / I.
- We map each feature i to a Gaussian beam centered at μᵢ (chosen uniformly on the angular domain) with width σᵢ proportional to 1/√wᵢ, where wᵢ is the positive weight from Parent A. The feature count vᵢ scales the amplitude of that beam.

The composite intensity is the weighted sum of the individual beams:
 I_c(θ) = Σᵢ vᵢ·I(θ; μᵢ, σᵢ)

Its derivative is the sum of the individual derivatives, yielding a composite Fisher score:
 F_c(θ) = (∂I_c/∂θ)² / I_c

The hybrid algorithm therefore evaluates F_c(θ) over a candidate set of angles and selects the angle that maximizes Fisher information while optionally penalising distance from the nearest feature centre.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature definitions and positive weights
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
)

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)

_REGEX_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def extract_feature_counts(text: str) -> np.ndarray:
    """
    Count occurrences of each regex feature in *text*.
    Returns a NumPy array aligned with ``_FEATURE_ORDER``.
    """
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.float64)
    for idx, name in enumerate(_FEATURE_ORDER):
        pattern = _REGEX_MAP[name]
        matches = pattern.findall(text)
        counts[idx] = len(matches)
    return counts


# ----------------------------------------------------------------------
# Parent B – Gaussian beam and Fisher information utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def gaussian_derivative(theta: float, center: float, width: float) -> float:
    """Derivative of the Gaussian intensity with respect to theta."""
    intensity = gaussian_beam(theta, center, width)
    return intensity * (-(theta - center) / (width * width))


# ----------------------------------------------------------------------
# Hybrid core – composite intensity and Fisher score
# ----------------------------------------------------------------------
def composite_intensity(theta: float, counts: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> float:
    """
    Weighted sum of Gaussian beams.
    *counts* – feature amplitudes (non‑negative).
    *centers* – μᵢ for each feature.
    *widths* – σᵢ for each feature.
    """
    intensity = 0.0
    for amp, mu, sigma in zip(counts, centers, widths):
        if amp == 0:
            continue
        intensity += amp * gaussian_beam(theta, mu, sigma)
    return intensity


def composite_derivative(theta: float, counts: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> float:
    """
    Derivative of the composite intensity with respect to theta.
    """
    deriv = 0.0
    for amp, mu, sigma in zip(counts, centers, widths):
        if amp == 0:
            continue
        deriv += amp * gaussian_derivative(theta, mu, sigma)
    return deriv


def composite_fisher_score(theta: float, counts: np.ndarray, centers: np.ndarray, widths: np.ndarray, eps: float = 1e-12) -> float:
    """
    Fisher information for the composite beam at angle *theta*.
    """
    I = max(composite_intensity(theta, counts, centers, widths), eps)
    dI = composite_derivative(theta, counts, centers, widths)
    return (dI * dI) / I


def best_angle_for_text(
    text: str,
    candidates: List[float],
    angular_span: Tuple[float, float] = (0.0, 180.0),
) -> float:
    """
    Determine the angle from *candidates* that maximises the composite Fisher score
    derived from the textual feature counts.

    The angular domain is split uniformly among the features to obtain μᵢ.
    Width σᵢ is set to ``span / (2 * sqrt(weight_i + 1))`` – larger weights give narrower beams.
    """
    if not candidates:
        raise ValueError("candidates required")

    counts = extract_feature_counts(text)

    # Map features to equally spaced centers within the angular span
    start, stop = angular_span
    num_feat = len(_FEATURE_ORDER)
    centers = np.linspace(start, stop, num_feat)

    # Width inversely related to positive weight (avoid division by zero)
    # Add 1 to weight to keep σ finite for zero‑weight features.
    widths = (stop - start) / (2.0 * np.sqrt(_POSITIVE_WEIGHTS + 1.0))

    # Evaluate Fisher score for each candidate; tie‑break by proximity to nearest centre
    best = max(
        candidates,
        key=lambda t: (
            composite_fisher_score(t, counts, centers, widths),
            -min(abs(t - mu) for mu in centers),
        ),
    )
    return best


# ----------------------------------------------------------------------
# Additional utility exposing the pure Parent‑B behaviour for comparison
# ----------------------------------------------------------------------
def best_angle_fisher(candidates: List[float], center: float, width: float) -> float:
    """
    Wrapper around Parent‑B's ``best_angle`` that uses the same scoring logic.
    """
    if not candidates:
        raise ValueError("candidates required")
    return max(candidates, key=lambda t: (fisher_score(t, center, width), -abs(t - center)))


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Parent‑B Fisher information for a single Gaussian beam.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The team has a plan and checklist ready. "
        "We have verified the evidence and recorded logs. "
        "However, we are waiting for the final approval, so there will be a delay. "
        "If anything goes wrong, we can call a friend for support."
    )
    angle_candidates = [i * 10.0 for i in range(0, 19)]  # 0,10,…,180

    chosen = best_angle_for_text(sample_text, angle_candidates)
    print(f"Hybrid best angle: {chosen:.2f}°")

    # Comparison with a single Gaussian (center=90°, width=30°)
    single_best = best_angle_fisher(angle_candidates, center=90.0, width=30.0)
    print(f"Pure Fisher best angle: {single_best:.2f}°")