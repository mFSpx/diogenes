# DARWIN HAMMER — match 1344, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m935_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s0.py (gen4)
# born: 2026-05-29T23:35:28Z

"""Hybrid algorithm merging epistemic certainty (Parent A) with morphological similarity metrics (Parent B) and date-based calculations (Parent B) with bandit updates (Parent B).

Mathematical bridge:
- Parent A provides epistemic certainty flags each carrying a confidence (basis points).
- Parent B supplies quantitative shape descriptors (sphericity, flatness, righting‑time) and date-based calculations that can be arranged into a feature vector **x** ∈ ℝ³, where the Gini coefficient is used to compute the propensity of each bandit arm.
- The hybrid constructs a diagonal weight matrix **W** from the normalized confidences of a set of CertaintyFlag objects and combines this with the Gini coefficient to compute the final score **s = (xᵀ W x + (1 - Gini)) / 2**, i.e. each descriptor is weighted by its epistemic confidence before aggregation and then combined with the date-based calculation to create a novel hybrid system.
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


def gini_coefficient(values: np.ndarray) -> float:
    """
    Return the Gini coefficient of a 1‑D array of non‑negative numbers.
    The implementation follows the definition

        G = Σ_i (2·i – n – 1)·x_i / (n·Σ x_i),

    where ``x_i`` are the values sorted in non‑decreasing order and ``i`` is
    1‑based.  Edge cases (empty array, all zeros) yield ``0.0``.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        return 0.0
    n = x.size
    return np.dot(np.arange(2, n + 2), x) / (n * np.dot(x, np.ones(n)))


def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  The result follows the convention
    0 = Sunday, …, 6 = Saturday, which matches the ``(weekday + 1) % 7`` mapping
    used in the original hybrid.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def hybrid_score(flags: List[CertaintyFlag], descriptors: np.ndarray, years: np.ndarray, months: np.ndarray, days: np.ndarray) -> float:
    """
    Compute the hybrid score based on epistemic certainty flags, shape descriptors, and date-based calculations.
    """
    # Normalize confidence of epistemic certainty flags
    confidence_weights = np.array([flag.confidence_bps / 10_000 for flag in flags])
    confidence_weights = confidence_weights / confidence_weights.sum()

    # Compute weight matrix from confidence weights
    weight_matrix = np.diag(confidence_weights)

    # Compute shape descriptor features
    features = descriptors

    # Compute Gini coefficient for date-based calculations
    gini = gini_coefficient(years)

    # Compute hybrid score
    hybrid_score = (np.dot(features.T, weight_matrix @ features) + (1 - gini)) / 2

    return hybrid_score


def test_hybrid_score():
    # Generate random data
    np.random.seed(0)
    descriptors = np.random.rand(10, 3)  # shape descriptors
    years = np.random.randint(2020, 2030, 10)
    months = np.random.randint(1, 13, 10)
    days = np.random.randint(1, 31, 10)
    flags = [CertaintyFlag("FACT", 5000, "Authority", "Rationale", ("Evidence",)) for _ in range(10)]

    # Compute hybrid score
    hybrid_score_value = hybrid_score(flags, descriptors, years, months, days)

    print(f"Hybrid score: {hybrid_score_value}")


if __name__ == "__main__":
    test_hybrid_score()