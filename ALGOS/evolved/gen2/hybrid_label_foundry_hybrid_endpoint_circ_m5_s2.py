# DARWIN HAMMER — match 5, survivor 2
# gen: 2
# parent_a: label_foundry.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:22:18Z

"""HybridLabelCircuit: Fusion of weak‑supervision labeling (parent A) and
endpoint circuit‑breaker recovery priority (parent B).

Mathematical bridge
-------------------
Parent A produces a confidence *c*∈[0,1] for each document via vote
majority.  
Parent B defines a *recovery priority* ρ∈[0,1] based on morphology:
ρ = clamp( righting_time_index(m) / max_index ).

The hybrid uses ρ as a multiplicative scaling factor for the confidence
produced by the labeling aggregation:

    c_hybrid = c · ρ

Thus a well‑shaped (high ρ) endpoint boosts confidence, while a fragile
one attenuates it.  Conversely, the error‑detection threshold is relaxed
by the same factor:

    τ_hybrid = τ_base / (1 + ρ)

so that high‑priority morphologies tolerate fewer apparent errors.

The module implements three core functions that embody this unified
system and a tiny smoke‑test."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from random import random
from typing import Any, Callable, Dict, List

import numpy as np


# ----------------------------------------------------------------------
# Parent A – labeling primitives
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A‑logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out: List[ProbabilisticLabel] = []
    for doc_id, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(doc_id, 0, 0.5))
            continue
        c = Counter(vs)
        label = 1 if c[1] >= c[0] else 0
        confidence = c[label] / len(vs)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out


def find_label_errors(
    docs: List[dict],
    given: List[int],
    probs: List[float],
    threshold: float = 0.65,
) -> List[LabelError]:
    """Pure A‑logic: flag docs where error probability exceeds threshold."""
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError('length mismatch')
    errs: List[LabelError] = []
    for doc, g, p in zip(docs, given, probs):
        errp = p if g == 0 else 1.0 - p
        if errp >= threshold:
            errs.append(LabelError(str(doc.get('id', len(errs))), g, 1 - g, errp))
    return sorted(errs, key=lambda e: -e.error_probability)


# ----------------------------------------------------------------------
# Parent B – morphology & circuit breaker
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """ρ = clamp( righting_time_index / max_index ) ∈ [0,1]"""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


class EndpointCircuitBreaker:
    """Circuit breaker whose opening threshold adapts to morphology."""
    def __init__(self, failure_threshold: int = 3, morphology: Morphology | None = None):
        self.base_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.morphology = morphology

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = _now_z()

    def record_failure(self) -> None:
        self.failures += 1
        if self.morphology is not None:
            ρ = recovery_priority(self.morphology)
            # effective threshold shrinks when ρ is high
            eff_thr = self.base_threshold * (1 - ρ)
            self.open = self.failures >= max(1, int(eff_thr))
        else:
            self.open = self.failures >= self.base_threshold
        self.last_event_at = _now_z()

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.base_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def _now_z() -> str:
    """ISO‑8601 UTC timestamp without offset."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Hybrid layer – mathematical fusion
# ----------------------------------------------------------------------
def hybrid_aggregate_labels(
    batches: List[List[LabelingFunctionResult]],
    morphology: Morphology,
    max_index: float = 10.0,
) -> List[ProbabilisticLabel]:
    """
    Combine A's majority‑vote confidence with B's recovery priority.

    For each document:
        c_raw  = proportion of majority votes  (A)
        ρ      = recovery_priority(morphology) (B)
        c_hyb  = c_raw * ρ

    The resulting confidence is bounded in [0,1].
    """
    raw = aggregate_labels(batches)
    ρ = recovery_priority(morphology, max_index)
    out: List[ProbabilisticLabel] = []
    for pl in raw:
        hybrid_conf = pl.confidence * ρ
        out.append(ProbabilisticLabel(pl.doc_id, pl.label, hybrid_conf))
    return out


def hybrid_find_label_errors(
    docs: List[dict],
    given: List[int],
    probs: List[float],
    morphology: Morphology,
    base_threshold: float = 0.65,
    max_index: float = 10.0,
) -> List[LabelError]:
    """
    Adapt A's error‑detection threshold by B's recovery priority.

        τ_hybrid = base_threshold / (1 + ρ)

    Higher ρ (more robust morphology) yields a stricter threshold,
    i.e. fewer false positives.
    """
    ρ = recovery_priority(morphology, max_index)
    τ_hybrid = base_threshold / (1.0 + ρ)
    return find_label_errors(docs, given, probs, threshold=τ_hybrid)


def hybrid_select_endpoint(
    endpoints: Dict[str, Morphology],
    failures: Dict[str, int],
    failure_threshold: int = 3,
) -> str:
    """
    Choose the endpoint with the highest *effective health*:

        health = (1 - failure_rate) * ρ

    where failure_rate = failures / failure_threshold (capped at 1) and
    ρ = recovery_priority(morphology).

    Returns the endpoint identifier; raises RuntimeError if none are healthy.
    """
    best_id = None
    best_score = -np.inf
    for eid, morph in endpoints.items():
        fail_cnt = failures.get(eid, 0)
        fail_rate = min(1.0, fail_cnt / failure_threshold)
        ρ = recovery_priority(morph)
        health = (1.0 - fail_rate) * ρ
        if health > best_score:
            best_score = health
            best_id = eid
    if best_id is None or best_score <= 0:
        raise RuntimeError("No viable endpoint found")
    return best_id


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Prepare synthetic labeling function results
    lf1 = LabelingFunctionResult("lf_a", "doc1", 1)
    lf2 = LabelingFunctionResult("lf_b", "doc1", 0)
    lf3 = LabelingFunctionResult("lf_c", "doc2", 1)
    batch = [[lf1, lf2], [lf3]]

    # 2. Define a morphology (moderately robust)
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)

    # 3. Hybrid aggregation
    hybrid_probs = hybrid_aggregate_labels(batch, morph)
    print("Hybrid Probabilistic Labels:")
    for p in hybrid_probs:
        print(p)

    # 4. Simulate docs and raw probabilities for error detection
    docs = [{"id": "doc1", "text": "x"}, {"id": "doc2", "text": "y"}]
    given = [0, 1]
    raw_probs = [pl.confidence for pl in hybrid_probs]

    errors = hybrid_find_label_errors(docs, given, raw_probs, morph)
    print("\nDetected Label Errors (hybrid):")
    for e in errors:
        print(e)

    # 5. Endpoint selection demo
    endpoints = {
        "cpu_fast": Morphology(1.0, 1.0, 1.0, 1.0),
        "gpu_heavy": Morphology(3.0, 2.5, 2.0, 5.0),
    }
    failures = {"cpu_fast": 1, "gpu_heavy": 0}
    chosen = hybrid_select_endpoint(endpoints, failures, failure_threshold=3)
    print(f"\nChosen endpoint: {chosen}")

    sys.exit(0)