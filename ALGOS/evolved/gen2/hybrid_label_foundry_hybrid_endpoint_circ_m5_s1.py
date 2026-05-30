# DARWIN HAMMER — match 5, survivor 1
# gen: 2
# parent_a: label_foundry.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:22:18Z

"""
This module fuses the label_foundry.py and hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py algorithms.
The mathematical bridge between the two structures is the concept of "recovery priority," which is used to determine the likelihood of an endpoint recovering from a failure.
The recovery priority is calculated based on the morphology of the endpoint, and this value is then used to adjust the circuit breaker's threshold for determining when to open or close the circuit.
In this fusion, we use the labeling functions from label_foundry.py to determine the labels of the endpoints, and then use the recovery priority to adjust the circuit breaker's behavior.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from random import random
from sys import exit
from pathlib import Path
from typing import Callable, Dict, List

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

def labeling_function(name: str|None=None):
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__; 
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes = {}
    for batch in batches:
        for r in batch:
            if r.label in (0,1): 
                if r.doc_id not in votes: 
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5)); 
            continue
        c = {0: 0, 1: 0}
        for v in vs:
            c[v] += 1
        label = 1 if c[1]>=c[0] else 0
        out.append(ProbabilisticLabel(d,label,c[label]/len(vs)))
    return out

def find_label_errors(docs: List[dict], given: List[int], probs: List[float], threshold: float=0.65) -> List[LabelError]:
    if not (len(docs)==len(given)==len(probs)): 
        raise ValueError('length mismatch')
    errs = []
    for doc,g,p in zip(docs,given,probs):
        errp = p if g==0 else 1.0-p
        if errp>=threshold: 
            errs.append(LabelError(str(doc.get('id',len(errs))),g,1-g,errp))
    return sorted(errs,key=lambda e:-e.error_probability)

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

def adjust_threshold(recovery_p: float, failure_threshold: int) -> int:
    return int(failure_threshold * (1 - recovery_p))

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
            self.open = self.failures >= adjust_threshold(recovery_p, self.failure_threshold)
        else:
            self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> Dict[str, any]:
        return {"failure_threshold": self.failure_threshold, "failures": self.failures, "open": self.open, "last_event_at": self.last_event_at}

class EndpointPool:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.breakers = {e: EndpointCircuitBreaker() for e in endpoints}

    def available(self) -> List[str]:
        return [e for e in self.endpoints if self.breakers[e].allow()]

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    outbound_state: str = "draft_only"
    morphology: Morphology = None

    def as_dict(self) -> Dict[str, any]:
        return {"engine_id": self.engine_id, "channel": self.channel, "residency": self.residency, "runtime": self.runtime, "resource_class": self.resource_class, "always_on": self.always_on, "endpoint": self.endpoint, "capabilities": self.capabilities, "outbound_state": self.outbound_state, "morphology": self.morphology.__dict__ if self.morphology is not None else None}

def label_endpoint(endpoint: EngineEndpoint) -> LabelingFunctionResult:
    # This is a simple labeling function that returns a LabelingFunctionResult
    # In a real-world scenario, this function would be more complex and would depend on the specific requirements of the endpoint
    return LabelingFunctionResult("default_lf", endpoint.engine_id, 1)

def prioritize_endpoints(endpoints: List[EngineEndpoint]) -> List[EngineEndpoint]:
    # This function prioritizes the endpoints based on their recovery priority
    recovery_priorities = []
    for endpoint in endpoints:
        if endpoint.morphology is not None:
            recovery_p = recovery_priority(endpoint.morphology)
            recovery_priorities.append((endpoint, recovery_p))
        else:
            recovery_priorities.append((endpoint, 0.0))
    return sorted(recovery_priorities, key=lambda x: x[1], reverse=True)

def create_endpoint_pool(endpoints: List[EngineEndpoint]) -> EndpointPool:
    # This function creates an EndpointPool from a list of EngineEndpoints
    endpoint_ids = [endpoint.engine_id for endpoint in endpoints]
    return EndpointPool(endpoint_ids)

if __name__ == "__main__":
    morphology = Morphology(1.0, 1.0, 1.0, 1.0)
    endpoint = EngineEndpoint("test", "test", "test", "test", "test", True, "test", ["test"], morphology=morphology)
    circuit_breaker = EndpointCircuitBreaker(morphology=morphology)
    circuit_breaker.record_failure()
    print(circuit_breaker.allow())
    circuit_breaker.record_success()
    print(circuit_breaker.allow())
    endpoints = [EngineEndpoint("test1", "test1", "test1", "test1", "test1", True, "test1", ["test1"], morphology=morphology),
                 EngineEndpoint("test2", "test2", "test2", "test2", "test2", True, "test2", ["test2"], morphology=morphology)]
    prioritized_endpoints = prioritize_endpoints(endpoints)
    print(prioritized_endpoints)
    endpoint_pool = create_endpoint_pool(endpoints)
    print(endpoint_pool.available())
    labeling_function_result = label_endpoint(endpoint)
    print(labeling_function_result)