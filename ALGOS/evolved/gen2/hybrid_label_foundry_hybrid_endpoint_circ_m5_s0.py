# DARWIN HAMMER — match 5, survivor 0
# gen: 2
# parent_a: label_foundry.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:22:18Z

"""This module fuses the weak supervision labeling primitives from label_foundry.py 
and the hybrid endpoint circuit breaker concept from hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py.
The mathematical bridge between the two structures is the concept of "recovery priority," 
which is used to determine the likelihood of an endpoint recovering from a failure.
The recovery priority is calculated based on the morphology of the endpoint, 
and this value is then used to adjust the circuit breaker's threshold for determining when to open or close the circuit.
This fusion enables the integration of weak supervision labeling with endpoint circuit breakers, 
allowing for more robust labeling and endpoint management."""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from random import random
from sys import exit
from pathlib import Path
from collections import Counter, defaultdict
from typing import Callable, Dict, Any


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


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
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(d, 0, 0.5))
            continue
        c = Counter(vs)
        label = 1 if c[1] >= c[0] else 0
        out.append(ProbabilisticLabel(d, label, c[label] / len(vs)))
    return out


def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float = 0.65) -> list[LabelError]:
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError('length mismatch')
    errs = []
    for doc, g, p in zip(docs, given, probs):
        errp = p if g == 0 else 1.0 - p
        if errp >= threshold:
            errs.append(LabelError(str(doc.get('id', len(errs))), g, 1 - g, errp))
    return sorted(errs, key=lambda e: -e.error_probability)


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.morphology = morphology

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        if self.morphology is not None:
            recovery_p = recovery_priority(self.morphology)
            self.open = self.failures >= self.failure_threshold * (1 - recovery_p)
        else:
            self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {"failure_threshold": self.failure_threshold, "failures": self.failures, "open": self.open, "last_event_at": self.last_event_at}


class HybridLabelingEndpoint:
    def __init__(self, endpoint_circuit_breaker: EndpointCircuitBreaker, labeling_function_results: list[LabelingFunctionResult]):
        self.endpoint_circuit_breaker = endpoint_circuit_breaker
        self.labeling_function_results = labeling_function_results

    def aggregate_labels(self) -> list[ProbabilisticLabel]:
        return aggregate_labels([self.labeling_function_results])

    def find_label_errors(self, docs: list[dict], given: list[int], probs: list[float], threshold: float = 0.65) -> list[LabelError]:
        return find_label_errors(docs, given, probs, threshold)

    def record_success(self) -> None:
        self.endpoint_circuit_breaker.record_success()

    def record_failure(self) -> None:
        self.endpoint_circuit_breaker.record_failure()

    def allow(self) -> bool:
        return self.endpoint_circuit_breaker.allow()


def create_hybrid_labeling_endpoint(morphology: Morphology, labeling_function_results: list[LabelingFunctionResult]) -> HybridLabelingEndpoint:
    endpoint_circuit_breaker = EndpointCircuitBreaker(morphology=morphology)
    return HybridLabelingEndpoint(endpoint_circuit_breaker, labeling_function_results)


if __name__ == "__main__":
    morphology = Morphology(length=1.0, width=1.0, height=1.0, mass=1.0)
    labeling_function_results = [LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc2", 0)]
    hybrid_labeling_endpoint = create_hybrid_labeling_endpoint(morphology, labeling_function_results)
    print(hybrid_labeling_endpoint.aggregate_labels())
    print(hybrid_labeling_endpoint.find_label_errors([{"id": "doc1"}], [1], [0.8], threshold=0.65))
    hybrid_labeling_endpoint.record_success()
    print(hybrid_labeling_endpoint.allow())