# DARWIN HAMMER — match 17, survivor 5
# gen: 2
# parent_a: semantic_neighbors.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:25:05Z

"""Hybrid Semantic-Morphology Neighbor System
Parents:
- semantic_neighbors.py (cosine similarity based neighbor retrieval)
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (morphology‑driven recovery priority & circuit breaker)

Mathematical Bridge:
The recovery priority `p ∈ [0,1]` derived from a document's morphology is used as a
multiplicative scaling factor for the cosine similarity `c ∈ [-1,1]` between vectors.
The hybrid affinity is defined as  

    h = c * p_other  

where `p_other` is the recovery priority of the candidate neighbor.  
Thus the topology of the semantic space is modulated by the physical‑morphology
space, and the circuit‑breaker logic can suppress neighbors whose morphology yields
low priority (i.e., high right‑ing time)."""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


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
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a normalized priority in [0,1]."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Endpoint Circuit Breaker (Parent B)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure counter with morphology‑aware threshold."""
    def __init__(self, failure_threshold: int = 3, morphology: Morphology | None = None):
        self.base_threshold = failure_threshold
        self.morphology = morphology
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    @property
    def adjusted_threshold(self) -> float:
        """Threshold scaled by recovery priority (higher priority → stricter)."""
        if self.morphology is None:
            return self.base_threshold
        priority = recovery_priority(self.morphology)
        # Inverse scaling: high priority means lower tolerance for failures
        return self.base_threshold * (1.0 - 0.5 * priority)

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.adjusted_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Hybrid Registry (combines semantic vectors with morphology & breaker)
# ----------------------------------------------------------------------
_Registry: Dict[str, Dict[str, Any]] = {}


def clear_registry() -> None:
    """Remove all stored documents."""
    _Registry.clear()


def register_document(
    doc_id: str,
    vector: List[float],
    morphology: Morphology,
    failure_threshold: int = 3,
) -> None:
    """Store a document together with its vector, morphology and a circuit breaker."""
    _Registry[doc_id] = {
        "vector": np.asarray(vector, dtype=float),
        "morphology": morphology,
        "breaker": EndpointCircuitBreaker(failure_threshold, morphology),
    }


# ----------------------------------------------------------------------
# Core Hybrid Operations
# ----------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def hybrid_affinity(source_id: str, target_id: str) -> float:
    """
    Compute the hybrid affinity h = cosine(source, target) * recovery_priority(target).

    The source's circuit breaker must be closed; otherwise, the function returns -inf
    to indicate that the source is unavailable.
    """
    src = _Registry[source_id]
    tgt = _Registry[target_id]

    if src["breaker"].is_open():
        # Source endpoint is tripped; no reliable affinity can be produced.
        return float("-inf")

    cos_sim = _cosine(src["vector"], tgt["vector"])
    priority = recovery_priority(tgt["morphology"])
    return cos_sim * priority


def hybrid_neighbors(doc_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Return the top‑k neighbors sorted by hybrid affinity.
    Documents whose circuit breaker is open are excluded from the candidate set.
    """
    if doc_id not in _Registry:
        raise KeyError(f"{doc_id!r} not registered")
    candidates = []
    for other_id in _Registry:
        if other_id == doc_id:
            continue
        if _Registry[other_id]["breaker"].is_open():
            continue
        score = hybrid_affinity(doc_id, other_id)
        if score == float("-inf"):
            continue
        candidates.append((other_id, score))
    # Sort by descending score, then by identifier for stability
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[:k]


def record_interaction(source_id: str, target_id: str, success: bool) -> None:
    """
    Update circuit breakers based on the outcome of an interaction between two documents.
    A failure increments the source breaker; a success resets it.
    """
    if source_id not in _Registry or target_id not in _Registry:
        raise KeyError("source or target not registered")
    breaker = _Registry[source_id]["breaker"]
    if success:
        breaker.record_success()
    else:
        breaker.record_failure()


def morphology_summary(doc_id: str) -> Dict[str, float]:
    """
    Return a dictionary summarising the morphology‑derived metrics for a document.
    """
    if doc_id not in _Registry:
        raise KeyError(f"{doc_id!r} not registered")
    m = _Registry[doc_id]["morphology"]
    return {
        "sphericity": sphericity_index(m.length, m.width, m.height),
        "flatness": flatness_index(m.length, m.width, m.height),
        "righting_time": righting_time_index(m),
        "recovery_priority": recovery_priority(m),
    }


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    clear_registry()
    # Define three simple documents
    docs = {
        "alpha": {
            "vector": [1.0, 0.0, 0.0],
            "morph": Morphology(1.0, 0.8, 0.5, 2.0),
        },
        "beta": {
            "vector": [0.9, 0.1, 0.0],
            "morph": Morphology(0.9, 0.7, 0.6, 1.8),
        },
        "gamma": {
            "vector": [0.0, 1.0, 0.0],
            "morph": Morphology(0.5, 0.5, 0.5, 0.5),
        },
    }

    for doc_id, info in docs.items():
        register_document(doc_id, info["vector"], info["morph"])

    # Simulate some interactions
    record_interaction("alpha", "beta", success=True)
    record_interaction("alpha", "gamma", success=False)  # cause a failure
    record_interaction("alpha", "gamma", success=False)
    record_interaction("alpha", "gamma", success=False)  # may open breaker

    # Display breaker state
    for doc_id in docs:
        breaker = _Registry[doc_id]["breaker"]
        print(f"{doc_id}: breaker open={breaker.is_open()}, failures={breaker.failures}")

    # Retrieve hybrid neighbors for 'alpha'
    neigh = hybrid_neighbors("alpha", k=2)
    print("\nHybrid neighbors for 'alpha':")
    for nid, score in neigh:
        print(f"  {nid}: hybrid_score={score:.4f}")

    # Show morphology summary for each doc
    print("\nMorphology summaries:")
    for doc_id in docs:
        print(f"{doc_id}: {morphology_summary(doc_id)}")