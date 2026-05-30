# DARWIN HAMMER — match 935, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s5.py (gen3)
# born: 2026-05-29T23:31:49Z

"""Hybrid algorithm merging epistemic certainty (Parent A) with morphological similarity metrics (Parent B).

Mathematical bridge:
- Parent A provides epistemic certainty flags each carrying a confidence (basis points).
- Parent B supplies quantitative shape descriptors (sphericity, flatness, righting‑time) that can be arranged into a feature vector **x** ∈ ℝ³.
- The hybrid constructs a diagonal weight matrix **W** from the normalized confidences of a set of CertaintyFlag objects.
- The fused score is obtained by the quadratic form **s = xᵀ W x**, i.e. each descriptor is weighted by its epistemic confidence before aggregation.
This single matrix operation fuses the two topologies into one unified decision metric.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Tuple, Iterable, List, Union

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty helpers
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
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Union[str, int, Tuple[str, ...]]]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """Factory shortcut mirroring Parent A."""
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Parent B – Morphology and similarity metrics
# ----------------------------------------------------------------------
class Morphology:
    """Physical dimensions of an object."""

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphological parameters must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Sphericity = (V)^(1/3) / longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length + width) / (2 * height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Simplified right‑ing time model.
    t = b * (mass * g * height) / (k * neck_lever)
    where g = 9.81 m·s⁻².
    """
    g = 9.81
    return b * (m.mass * g * m.height) / (k * neck_lever)


# ----------------------------------------------------------------------
# Hybrid core – matrix‑based fusion of epistemic and morphological data
# ----------------------------------------------------------------------
def feature_vector(m: Morphology) -> np.ndarray:
    """
    Assemble a 3‑dimensional feature vector from the morphology:
    [sphericity, flatness, righting_time].
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    return np.array([sph, flat, rt], dtype=float)


def weight_matrix(flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Build a diagonal weight matrix **W** from a list of CertaintyFlag objects.
    Each flag contributes a normalized confidence (0‑1). If fewer than three
    flags are supplied, the missing entries are filled with a neutral weight 0.5.
    If more than three are supplied, the first three are used.
    """
    # Normalise to [0,1]
    confidences = [f.confidence_bps / 10_000.0 for f in flags[:3]]
    # Pad / truncate to length 3
    while len(confidences) < 3:
        confidences.append(0.5)  # neutral default
    w = np.diag(confidences)
    return w


def epistemic_weighted_score(m: Morphology, flags: List[CertaintyFlag]) -> float:
    """
    Compute the hybrid score s = xᵀ W x where
    x = feature_vector(m) and W = weight_matrix(flags).
    The result blends shape similarity with epistemic confidence.
    """
    x = feature_vector(m)
    W = weight_matrix(flags)
    return float(x.T @ W @ x)


def hybrid_report(m: Morphology, flags: List[CertaintyFlag]) -> Dict[str, Union[float, List[Dict]]]:
    """
    Produce a human‑readable report containing:
    - raw morphological indices,
    - epistemic flags,
    - the final weighted score.
    """
    raw = {
        "sphericity": sphericity_index(m.length, m.width, m.height),
        "flatness": flatness_index(m.length, m.width, m.height),
        "righting_time": righting_time_index(m),
    }
    flag_dicts = [f.as_dict() for f in flags]
    score = epistemic_weighted_score(m, flags)
    return {
        "morphology": raw,
        "epistemic_flags": flag_dicts,
        "weighted_score": score,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple object
    obj = Morphology(length=2.0, width=1.5, height=0.5, mass=3.0)

    # Define three epistemic flags with varying confidence
    flags = [
        certainty(
            "FACT",
            confidence_bps=9000,
            authority_class="physics",
            rationale="experimental measurement",
            evidence_refs=["exp001"],
        ),
        certainty(
            "PROBABLE",
            confidence_bps=6000,
            authority_class="simulation",
            rationale="finite‑element model",
            evidence_refs=["simA"],
        ),
        certainty(
            "POSSIBLE",
            confidence_bps=3000,
            authority_class="theory",
            rationale="analytical approximation",
            evidence_refs=["theoryX"],
        ),
    ]

    # Run hybrid calculations
    vec = feature_vector(obj)
    W = weight_matrix(flags)
    score = epistemic_weighted_score(obj, flags)
    report = hybrid_report(obj, flags)

    # Simple sanity prints (no external dependencies)
    print("Feature vector:", vec)
    print("Weight matrix:\n", W)
    print("Weighted score:", score)
    print("Report dictionary:", report)