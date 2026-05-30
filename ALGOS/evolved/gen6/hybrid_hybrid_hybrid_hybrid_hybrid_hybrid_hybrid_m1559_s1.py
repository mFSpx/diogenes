# DARWIN HAMMER — match 1559, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s1.py (gen5)
# born: 2026-05-29T23:37:24Z

"""Hybrid Algorithm integrating Epistemic Certainty (Parent A) with Temporal Health Metrics (Parent B).

Mathematical Bridge:
Both parents expose scalar quantities that can be interpreted as *weights*.
- Parent A provides `confidence_bps` (0‑10 000) for each `CertaintyFlag`.
- Parent B uses scalar risk scores (`reconstruction_risk_score`, `recovery_priority`) and a Gini coefficient derived from a distribution.

The fusion treats the confidence vector **c** ∈ ℝⁿ as a probability weight (c/10 000) and
1. builds a **weighted weekday distribution** of events,
2. evaluates the **inequality of confidence** via the Gini coefficient,
3. injects the inequality and the epistemic‑label‑derived risk multiplier into the
   original health equation from Parent B.

The resulting hybrid health metric is

    H_hybrid = H_base · (1 ‑ Gini(ĉ)) · (1 ‑ ℓ̂),

where H_base is the original health from Parent B,
ĉ is the normalized confidence vector, and ℓ̂ is the average label‑risk
derived from the epistemic flags.  This mathematically fuses the two topologies
into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone, date
from typing import List, Tuple, Iterable, Dict, Union

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty structures
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


class CertaintyFlag:
    __slots__ = (
        "label",
        "confidence_bps",
        "authority_class",
        "rationale",
        "evidence_refs",
        "generated_at",
    )

    def __init__(
        self,
        label: str,
        confidence_bps: int,
        authority_class: str,
        rationale: str,
        evidence_refs: Tuple[str, ...] = (),
        generated_at: str = "",
    ) -> None:
        if label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {label!r}")
        if not 0 <= int(confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        self.label = label
        self.confidence_bps = int(confidence_bps)
        self.authority_class = authority_class
        self.rationale = rationale
        self.evidence_refs = tuple(str(x) for x in evidence_refs if x is not None)
        self.generated_at = (
            generated_at
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
# Parent B – Temporal utilities and health computation
# ----------------------------------------------------------------------
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Vectorised weekday calculation (Sun=0 … Sat=6)."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # shift: Monday=0 → Sunday=0
    return (py_weekday + 1) % 7


def weekday_counts(dates: Iterable[date]) -> np.ndarray:
    """Return a length‑7 array with counts of each weekday (Sun=0 … Sat=6)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, datetime):
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
    if cumulative[-1] == 0:
        return 0.0
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
# Hybrid layer – mathematical fusion of both parents
# ----------------------------------------------------------------------
_LABEL_RISK_MAP: Dict[str, float] = {
    "FACT": 0.0,
    "PROBABLE": 0.2,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.9,
    "SURE_MAYBE": 0.7,
}


def confidence_weights(flags: List[CertaintyFlag]) -> np.ndarray:
    """Convert a list of CertaintyFlag objects into a normalized weight vector."""
    raw = np.array([f.confidence_bps for f in flags], dtype=float)
    if raw.size == 0:
        return np.array([], dtype=float)
    return raw / 10_000.0  # scale to [0,1]


def weighted_weekday_distribution(
    dates: List[date], flags: List[CertaintyFlag]
) -> np.ndarray:
    """
    Compute a weekday histogram where each event contributes its confidence weight.
    Result follows Parent B's Sun=0 … Sat=6 convention.
    """
    if len(dates) != len(flags):
        raise ValueError("dates and flags must have the same length")
    weights = confidence_weights(flags)
    counts = np.zeros(7, dtype=float)
    for d, w in zip(dates, weights):
        idx = (d.weekday() + 1) % 7  # Sunday=0 convention
        counts[idx] += w
    return counts


def hybrid_health_metric(
    dates: List[date],
    flags: List[CertaintyFlag],
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    Combine Parent B's health computation with epistemic certainty information.

    Steps:
    1. Base health using a risk score derived from average label risk.
    2. Modulate by (1 - Gini(confidence_weights)).
    3. Modulate by (1 - average label risk).
    """
    if len(dates) != len(flags):
        raise ValueError("dates and flags must have the same length")

    # 1️⃣ Derive a reconstruction risk score from epistemic labels
    avg_label_risk = np.mean([_LABEL_RISK_MAP[f.label] for f in flags]) if flags else 0.0
    reconstruction_risk_score = avg_label_risk  # in [0,1]

    # Base health from Parent B
    base_health = compute_health(
        reconstruction_risk_score=reconstruction_risk_score,
        failures=failures,
        failure_threshold=failure_threshold,
        recovery_priority=recovery_priority,
    )

    # 2️⃣ Gini of confidence distribution
    weights = confidence_weights(flags)
    gini = gini_coefficient(weights) if weights.size else 0.0

    # 3️⃣ Final hybrid health
    hybrid = base_health * (1.0 - gini) * (1.0 - avg_label_risk)
    return max(0.0, min(1.0, hybrid))


def simulate_random_dataset(size: int = 100) -> Tuple[List[date], List[CertaintyFlag]]:
    """Create a synthetic dataset of dates and epistemic flags for testing."""
    today = date.today()
    dates = [
        today.replace(
            year=today.year - random.randint(0, 3),
            month=random.randint(1, 12),
            day=random.randint(1, 28),
        )
        for _ in range(size)
    ]
    flags = [
        certainty(
            random.choice(EPISTEMIC_FLAGS),
            confidence_bps=random.randint(0, 10_000),
            authority_class="sim",
            rationale="synthetic test",
        )
        for _ in range(size)
    ]
    return dates, flags


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate a tiny synthetic scenario
    dates, flags = simulate_random_dataset(20)

    # 1. weighted weekday histogram
    wd_hist = weighted_weekday_distribution(dates, flags)
    print("Weighted weekday distribution (Sun=0):", wd_hist)

    # 2. compute Gini of confidences
    gini_val = gini_coefficient(confidence_weights(flags))
    print("Gini coefficient of confidence weights:", gini_val)

    # 3. hybrid health metric
    health = hybrid_health_metric(
        dates,
        flags,
        failures=3,
        failure_threshold=10,
        recovery_priority=0.15,
    )
    print("Hybrid health metric:", health)