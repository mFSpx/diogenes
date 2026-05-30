# DARWIN HAMMER — match 5220, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1344_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1715_s3.py (gen6)
# born: 2026-05-30T00:00:45Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Tuple, Iterable, List, Union
import numpy as np

# ----------------------------------------------------------------------
# Epistemic certainty helpers
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

# ----------------------------------------------------------------------
# Allocation utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return a 0‑based weekday index."""
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    year -= month < 3
    return (year + int(year/4) - int(year/100) + int(year/400) + t[month-1] + day) % 7

def weekday_weight_vector(groups: Tuple[str, ...]) -> np.ndarray:
    """
    Builds a weekday-weighted vector for the given groups.

    Args:
        groups: A tuple of group names.

    Returns:
        A numpy array containing the weighted allocation for each group.
    """
    weights = np.array([random.random() for _ in range(len(groups))])
    return weights / np.sum(weights)

def allocate_hybrid(groups: Tuple[str, ...], certainty_flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Allocates resources to different groups based on epistemic certainty flags.

    Args:
        groups: A tuple of group names.
        certainty_flags: A list of certainty flags.

    Returns:
        A numpy array containing the weighted allocation for each group.
    """
    weights = np.array([random.random() for _ in range(len(groups))])
    certainty_weights = np.array([flag.confidence_bps / 10000 for flag in certainty_flags])
    certainty_weights = np.pad(certainty_weights, (0, len(groups) - len(certainty_flags)), mode='constant')
    return (weights * certainty_weights) / np.sum(weights * certainty_weights)

def calculate_gini_coefficient(allocations: np.ndarray) -> float:
    """
    Calculates the Gini coefficient for the given allocations.

    Args:
        allocations: A numpy array containing the allocations for each group.

    Returns:
        The Gini coefficient.
    """
    allocations = allocations.flatten()
    allocations = allocations + 0.0000001  # to avoid division by zero
    index = np.arange(1, allocations.shape[0] + 1)
    index = index / allocations.shape[0]
    sorted_allocations, _ = zip(*sorted(zip(allocations, index)))
    n = allocations.shape[0]
    return ((np.sum((2 * np.arange(n) + 1) * np.array(sorted_allocations)) / (n * np.sum(sorted_allocations))) - (n + 1) / n)

def calculate_hybrid_score(groups: Tuple[str, ...], certainty_flags: List[CertaintyFlag], allocations: np.ndarray) -> float:
    """
    Calculates the hybrid score by combining the epistemic certainty flags and the allocations.

    Args:
        groups: A tuple of group names.
        certainty_flags: A list of certainty flags.
        allocations: A numpy array containing the allocations for each group.

    Returns:
        The hybrid score.
    """
    certainty_weights = np.array([flag.confidence_bps / 10000 for flag in certainty_flags])
    certainty_weights = np.pad(certainty_weights, (0, len(groups) - len(certainty_flags)), mode='constant')
    gini_coefficient = calculate_gini_coefficient(allocations)
    return (np.sum(allocations * certainty_weights) + (1 - gini_coefficient)) / 2

def koopman_operator(allocations: np.ndarray, certainty_flags: List[CertaintyFlag]) -> np.ndarray:
    """
    Applies the Koopman operator to the allocations.

    Args:
        allocations: A numpy array containing the allocations for each group.
        certainty_flags: A list of certainty flags.

    Returns:
        The allocations after applying the Koopman operator.
    """
    certainty_weights = np.array([flag.confidence_bps / 10000 for flag in certainty_flags])
    certainty_weights = np.pad(certainty_weights, (0, len(allocations) - len(certainty_flags)), mode='constant')
    return allocations * certainty_weights

if __name__ == "__main__":
    certainty_flags = [CertaintyFlag("FACT", 5000, "high", "rationale", ("ref1", "ref2"))]
    groups = GROUPS
    allocations = allocate_hybrid(groups, certainty_flags)
    allocations = koopman_operator(allocations, certainty_flags)
    print(calculate_hybrid_score(groups, certainty_flags, allocations))