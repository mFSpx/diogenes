# DARWIN HAMMER — match 17, survivor 6
# gen: 2
# parent_a: semantic_neighbors.py (gen0)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:25:05Z

"""Hybrid Semantic-Morphology System
Parents:
- semantic_neighbors.py (cosine similarity based neighbor search)
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (morphology‑driven recovery priority & circuit breaker)

Mathematical Bridge:
The bridge is the *recovery priority* `p(m)` derived from morphology (B) and the *cosine similarity* `c(v_i,v_j)` derived from semantic vectors (A).
A unified hybrid score `h(i,j)` is defined as a convex combination:

    h(i,j) = α * c(v_i, v_j) + (1‑α) * p(m_j)

where `α ∈ [0,1]` balances pure semantic closeness against the physical robustness of the neighbor.
This single scalar drives both neighbor ranking and dynamic adjustment of the circuit‑breaker threshold.
"""

from __future__ import annotations
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


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Semantic Vectors & Cosine Similarity (Parent A)
# ----------------------------------------------------------------------
_ENCLAVE: Dict[str, Dict[str, Any]] = {}


def clear_registry() -> None:
    """Remove all stored documents."""
    _ENCLAVE.clear()


def register_document(doc_id: str,
                     vector: List[float],
                     morphology: Morphology) -> None:
    """Store a document together with its semantic vector and morphology."""
    if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
        raise TypeError("vector must be a list of numbers")
    _ENCLAVE[doc_id] = {"vector": vector, "morphology": morphology}


def _cos(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den


# ----------------------------------------------------------------------
# Hybrid Core (Mathematical Fusion)
# ----------------------------------------------------------------------
def hybrid_score(doc_i: str,
                 doc_j: str,
                 alpha: float = 0.5) -> float:
    """
    Compute the hybrid score h(i,j) = α·cosine + (1‑α)·recovery_priority.
    Alpha must be in [0,1].
    """
    if doc_i not in _ENCLAVE or doc_j not in _ENCLAVE:
        raise KeyError("Both document IDs must be registered")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    vi = _ENCLAVE[doc_i]["vector"]
    vj = _ENCLAVE[doc_j]["vector"]
    mj = _ENCLAVE[doc_j]["morphology"]
    cosine = _cos(vi, vj)
    priority = recovery_priority(mj)
    return alpha * cosine + (1.0 - alpha) * priority


def hybrid_neighbors(doc_id: str,
                     k: int = 5,
                     alpha: float = 0.5) -> List[Tuple[str, float]]:
    """
    Return the top‑k neighbors sorted by descending hybrid score.
    Each entry is (neighbor_id, hybrid_score).
    """
    if doc_id not in _ENCLAVE:
        raise KeyError(f"Document {doc_id!r} not found")
    scores = [
        (other_id, hybrid_score(doc_id, other_id, alpha))
        for other_id in _ENCLAVE
        if other_id != doc_id
    ]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:k]


# ----------------------------------------------------------------------
# Adaptive Circuit Breaker Integrated with Hybrid Scores
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Circuit breaker whose failure threshold adapts to the average hybrid
    robustness of the most similar neighbours.
    """

    def __init__(self,
                 base_failure_threshold: int = 3,
                 morphology: Morphology | None = None):
        self.base_failure_threshold = base_failure_threshold
        self.failure_threshold = base_failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.morphology = morphology

    def _now_z(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self, doc_id: str, alpha: float = 0.5) -> None:
        """
        Record a failure for the endpoint associated with `doc_id`.
        The failure threshold is dynamically lowered according to the
        average hybrid score of the top 3 neighbours, making the breaker
        more sensitive when the surrounding ecosystem is weak.
        """
        self.failures += 1
        self.last_event_at = self._now_z()
        # Adjust threshold based on neighbours
        neighbours = hybrid_neighbors(doc_id, k=3, alpha=alpha)
        if neighbours:
            avg_score = sum(score for _, score in neighbours) / len(neighbours)
            # Map avg_score ∈ [0,1] to a multiplier ∈ [0.5, 1.5]
            multiplier = 1.5 - avg_score  # higher score → lower multiplier
            self.failure_threshold = max(1, int(self.base_failure_threshold * multiplier))
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open


# ----------------------------------------------------------------------
# Demonstration Functions (at least three)
# ----------------------------------------------------------------------
def compute_recovery_priority(doc_id: str) -> float:
    """Convenience wrapper returning the recovery priority of a document."""
    if doc_id not in _ENCLAVE:
        raise KeyError(doc_id)
    return recovery_priority(_ENCLAVE[doc_id]["morphology"])


def compute_semantic_similarity(doc_i: str, doc_j: str) -> float:
    """Convenience wrapper returning cosine similarity between two documents."""
    if doc_i not in _ENCLAVE or doc_j not in _ENCLAVE:
        raise KeyError("Both IDs must exist")
    return _cos(_ENCLAVE[doc_i]["vector"], _ENCLAVE[doc_j]["vector"])


def adaptive_breaker_demo(doc_id: str, failures: int = 5, alpha: float = 0.5) -> None:
    """
    Run a small demo where a circuit breaker records a series of failures
    and prints the evolving threshold.
    """
    cb = EndpointCircuitBreaker(morphology=_ENCLAVE[doc_id]["morphology"])
    print(f"Initial failure threshold: {cb.failure_threshold}")
    for i in range(1, failures + 1):
        cb.record_failure(doc_id, alpha=alpha)
        print(f"After failure {i}: failures={cb.failures}, threshold={cb.failure_threshold}, open={cb.is_open()}")
        if cb.is_open():
            print("Circuit breaker opened – stopping demo.")
            break
    # Reset with a success
    cb.record_success()
    print(f"After success: failures={cb.failures}, open={cb.is_open()}, threshold={cb.failure_threshold}")


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Register a few sample documents
    for idx in range(1, 6):
        vec = np.random.rand(8).tolist()
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 10.0)
        )
        register_document(f"doc{idx}", vec, morph)

    # Show hybrid neighbours for doc1
    print("\nHybrid neighbours for doc1 (α=0.6):")
    for nid, score in hybrid_neighbors("doc1", k=3, alpha=0.6):
        print(f"  {nid}: {score:.4f}")

    # Compute individual metrics
    print("\nRecovery priority of doc3:", compute_recovery_priority("doc3"))
    print("Semantic similarity doc2 ↔ doc4:", compute_semantic_similarity("doc2", "doc4"))

    # Run adaptive breaker demo on doc1
    print("\nAdaptive circuit breaker demo for doc1:")
    adaptive_breaker_demo("doc1", failures=7, alpha=0.5)