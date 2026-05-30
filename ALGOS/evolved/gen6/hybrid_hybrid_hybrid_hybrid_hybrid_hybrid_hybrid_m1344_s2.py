# DARWIN HAMMER — match 1344, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m935_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s0.py (gen4)
# born: 2026-05-29T23:35:28Z

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Tuple, Iterable, List, Union

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
            "generated_at": self.generated_at
        }


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
        raise ValueError("values must be non‑negative")

    return np.sum((2 * np.arange(1, x.size + 1) - x.size - 1) * x) / (x.size * np.sum(x))


def compute_fused_score(x: np.ndarray, certainty_flags: List[CertaintyFlag]) -> float:
    """
    Compute the fused score by integrating epistemic certainty metrics with morphological similarity descriptors.
    """
    n = len(certainty_flags)
    W = np.diag([flag.confidence_bps / 10_000 for flag in certainty_flags])
    return np.dot(x.T, np.dot(W, x))


def compute_propensity(x: np.ndarray, certainty_flags: List[CertaintyFlag], years: np.ndarray, months: np.ndarray, days: np.ndarray) -> float:
    """
    Compute the propensity of each morphological descriptor by integrating epistemic certainty metrics with date-based calculations.
    """
    weekday_indices = weekday_sakamoto(years, months, days)
    gini_coeff = gini_coefficient(weekday_indices)
    return compute_fused_score(x, certainty_flags) * gini_coeff


def compute_weighted_gini_coefficient(x: np.ndarray, certainty_flags: List[CertaintyFlag], years: np.ndarray, months: np.ndarray, days: np.ndarray) -> float:
    """
    Compute the weighted Gini coefficient by integrating epistemic certainty metrics with date-based calculations.
    """
    weekday_indices = weekday_sakamoto(years, months, days)
    weights = np.array([flag.confidence_bps / 10_000 for flag in certainty_flags])
    weighted_weekday_indices = weekday_indices * weights[:, None]
    return gini_coefficient(weighted_weekday_indices)


def compute_improved_propensity(x: np.ndarray, certainty_flags: List[CertaintyFlag], years: np.ndarray, months: np.ndarray, days: np.ndarray) -> float:
    """
    Compute the improved propensity of each morphological descriptor by integrating epistemic certainty metrics with date-based calculations.
    """
    weighted_gini_coeff = compute_weighted_gini_coefficient(x, certainty_flags, years, months, days)
    return compute_fused_score(x, certainty_flags) * weighted_gini_coeff


def main():
    # Create a list of CertaintyFlag objects
    certainty_flags = [
        CertaintyFlag("FACT", 8000, "HIGH", "rationale1", ("ref1", "ref2")),
        CertaintyFlag("POSSIBLE", 4000, "MEDIUM", "rationale2", ("ref3", "ref4")),
        CertaintyFlag("BULLSHIT", 2000, "LOW", "rationale3", ("ref5", "ref6")),
    ]

    # Create a numpy array for morphological similarity descriptors
    x = np.array([0.5, 0.3, 0.2])

    # Create numpy arrays for date-based calculations
    years = np.array([2022, 2023, 2024])
    months = np.array([1, 2, 3])
    days = np.array([1, 2, 3])

    # Compute the fused score
    fused_score = compute_fused_score(x, certainty_flags)
    print(f"Fused score: {fused_score}")

    # Compute the propensity
    propensity = compute_propensity(x, certainty_flags, years, months, days)
    print(f"Propensity: {propensity}")

    # Compute the improved propensity
    improved_propensity = compute_improved_propensity(x, certainty_flags, years, months, days)
    print(f"Improved propensity: {improved_propensity}")

if __name__ == "__main__":
    main()