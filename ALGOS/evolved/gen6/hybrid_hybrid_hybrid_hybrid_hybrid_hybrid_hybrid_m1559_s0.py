# DARWIN HAMMER — match 1559, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s1.py (gen5)
# born: 2026-05-29T23:37:24Z

"""
HYBRID ALGORITHM A + B FUSION

Parent Algorithm A: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s3.py (gen: 4)
Parent Algorithm B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s1.py (gen: 5)

This hybrid algorithm mathematically fuses the core topologies of both parents by integrating their governing equations and matrix operations.
The mathematical bridge found between their structures lies in the intersection of epistemic certainty and health computation.

The parent A's CertaintyFlag dataclass is reused to represent epistemic flags in the hybrid algorithm, while the parent B's health computation is integrated into the hybrid's health estimation.

The hybrid algorithm's governing equation is a weighted sum of the epistemic certainty and health estimation:
    hybrid_score = (1 - alpha) * health + alpha * certainty

where alpha is a tunable parameter that controls the trade-off between epistemic certainty and health estimation.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set, Callable, Union, Any
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Epistemic certainty helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
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
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


# ----------------------------------------------------------------------
# Labeling function results
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str


# ----------------------------------------------------------------------
# Health computation helpers
# ----------------------------------------------------------------------
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift: Monday=0 → Sunday=0
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[dt.date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, dt.datetime):
            d = d.date()
        counts[d.weekday() % 7] += 1
    # shift to match doomsday_numpy convention (Sun=0)
    return np.roll(counts, 1)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    health = (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

    where failure_rate = failures / failure_threshold.
    """
    failure_rate = min(1.0, failures / max(1, failure_threshold))
    health = (1.0 - (reconstruction_risk_score * failure_rate)) * (1.0 - recovery_priority)
    return max(0.0, min(1.0, health))


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_score(
    certainty_flag: CertaintyFlag,
    health: float,
    alpha: float = 0.5,
) -> float:
    """Hybrid score computation."""
    return (1 - alpha) * health + alpha * certainty_flag.confidence_bps / 10_000


def hybrid_certainty(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
    certainty_flag: CertaintyFlag,
    alpha: float = 0.5,
) -> CertaintyFlag:
    """Hybrid certainty computation."""
    health = compute_health(
        reconstruction_risk_score,
        failures,
        failure_threshold,
        recovery_priority,
    )
    hybrid_cert = certainty_flag.label
    if certainty_flag.label != "FACT":
        hybrid_cert += f" ({health * 100:.2f}%)"
    return CertaintyFlag(
        label=hybrid_cert,
        confidence_bps=int(hybrid_score(certainty_flag, health, alpha) * 10_000),
        authority_class=certainty_flag.authority_class,
        rationale=certainty_flag.rationale,
        evidence_refs=certainty_flag.evidence_refs,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    certainty_flag = certainty(
        label="FACT",
        confidence_bps=10_000,
        authority_class="AUTHORITY",
        rationale="RATIONALE",
    )
    print(hybrid_score(certainty_flag, compute_health(0.5, 1, 1, 0.5)))
    print(hybrid_certainty(0.5, 1, 1, 0.5, certainty_flag))