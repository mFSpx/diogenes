# DARWIN HAMMER — match 1, survivor 6
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0.py (gen2)
# parent_b: hybrid_ssim_hybrid_decision_hygi_m9_s1.py (gen2)
# born: 2026-05-29T23:25:07Z

"""Hybrid Endpoint Similarity & Decision Hygiene

This module fuses the core topologies of two parent algorithms:

* **Parent A** – `hybrid_hybrid_endpoint_circ_state_space_duality_m1_s0.py`  
  Provides morphological descriptors, recovery priority, and a circuit‑breaker
  state model for engine endpoints.

* **Parent B** – `hybrid_ssim_hybrid_decision_hygi_m9_s1.py`  
  Supplies a Structural Similarity Index (SSIM) for numeric vectors and a
  decision‑hygiene scoring system based on regex‑identified evidence,
  planning, delay and support tokens together with Shannon entropy.

**Mathematical Bridge**  
Both parents operate on *vectors* that describe a system state:

* The morphology of an endpoint is a 4‑dimensional vector  
  `v = [length, width, height, mass]`.
* SSIM measures similarity between two equal‑length vectors.
* The hygiene module builds a probability distribution over token categories
  extracted from free‑form logs and evaluates its Shannon entropy.

The hybrid algorithm therefore:

1. Constructs morphology vectors for two endpoints.
2. Computes an SSIM‑like similarity `S` between the vectors (matrix‑based
   covariance formulation).
3. Extracts categorical token frequencies from log messages, builds a
   probability vector `p`, and evaluates the normalized Shannon entropy `H`.
4. Combines `S` and `H` with the individual recovery priorities `R₁, R₂` to
   obtain a unified **Hybrid Recovery Score** `Ψ`:


Ψ = (α·S + (1‑α)·(R₁+R₂)/2) · (1‑β·H)


where `α,β ∈ [0,1]` are tunable blending factors.

The resulting score drives the endpoint circuit‑breaker, which records
failures when `Ψ` falls below a configurable threshold.

The module exposes three primary functions demonstrating this hybrid
operation and includes a lightweight smoke test.
"""

import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Endpoint definitions
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


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


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
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


class EndpointCircuitBreaker:
    """Simple failure counter that trips after `failure_threshold` low scores."""

    def __init__(self, failure_threshold: int = 3, score_cutoff: float = 0.4):
        self.failure_threshold = failure_threshold
        self.score_cutoff = score_cutoff
        self.failures = 0
        self.tripped = False

    def update(self, score: float) -> None:
        """Record a new hybrid score; trip if too many low scores."""
        if score < self.score_cutoff:
            self.failures += 1
        else:
            # successful operation resets the failure streak
            self.failures = max(0, self.failures - 1)

        if self.failures >= self.failure_threshold:
            self.tripped = True

    def reset(self) -> None:
        self.failures = 0
        self.tripped = False

# ----------------------------------------------------------------------
# Parent B – SSIM & Decision Hygiene utilities
# ----------------------------------------------------------------------


def ssim_vector(x: np.ndarray, y: np.ndarray, dynamic_range: float = 1.0,
                k1: float = 0.01, k2: float = 0.03) -> float:
    """SSIM for two equal‑length 1‑D numpy arrays."""
    if x.shape != y.shape:
        raise ValueError("vectors must have equal shape")
    if x.size == 0:
        raise ValueError("vectors must not be empty")
    mx = x.mean()
    my = y.mean()
    vx = ((x - mx) ** 2).mean()
    vy = ((y - my) ** 2).mean()
    cov = ((x - mx) * (y - my)).mean()
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# Regexes from the decision‑hygiene parent
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)


def _category_counts(texts: List[str]) -> Dict[str, int]:
    """Count occurrences of each token category in a list of strings."""
    counts = {"evidence": 0, "planning": 0, "delay": 0, "support": 0}
    for line in texts:
        counts["evidence"] += len(EVIDENCE_RE.findall(line))
        counts["planning"] += len(PLANNING_RE.findall(line))
        counts["delay"] += len(DELAY_RE.findall(line))
        counts["support"] += len(SUPPORT_RE.findall(line))
    return counts


def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy (base‑e). Zero probability terms are ignored."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def hygiene_score(texts: List[str]) -> float:
    """
    Compute a composite hygiene score:
      * Normalized Shannon entropy of the category distribution (lower entropy → more focused).
      * Weighted evidence count (more evidence → higher score).

    The final score lies in [0,1].
    """
    if not texts:
        return 0.0
    counts = _category_counts(texts)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    probs = np.array(list(counts.values())) / total
    entropy = shannon_entropy(probs) / math.log(len(probs))  # normalize to [0,1]

    # Evidence weight: saturating function to keep within [0,1]
    evidence_weight = 1 - math.exp(-counts["evidence"] / 5.0)

    # Combine: we reward low entropy and high evidence
    score = (1 - entropy) * 0.6 + evidence_weight * 0.4
    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# Hybrid core – bridging the two parents
# ----------------------------------------------------------------------


def morphology_vector(m: Morphology) -> np.ndarray:
    """Convert a Morphology instance to a 4‑D numpy column vector."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)


def hybrid_similarity(endpoint_a: EngineEndpoint, endpoint_b: EngineEndpoint) -> float:
    """
    Compute SSIM‑style similarity between the morphologies of two endpoints.
    The dynamic range is set to the max absolute component to keep the metric
    scale‑invariant.
    """
    v1 = morphology_vector(endpoint_a.morphology)
    v2 = morphology_vector(endpoint_b.morphology)
    dr = max(np.abs(v1).max(), np.abs(v2).max())
    return ssim_vector(v1, v2, dynamic_range=dr)


def hybrid_recovery_score(
    endpoint_a: EngineEndpoint,
    endpoint_b: EngineEndpoint,
    logs: List[str],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Unified hybrid recovery score Ψ.

    Parameters
    ----------
    endpoint_a, endpoint_b : EngineEndpoint
        The two endpoints whose morphological similarity is evaluated.
    logs : list[str]
        Recent log messages associated with the pair (used for hygiene scoring).
    alpha : float ∈ [0,1]
        Blend factor between SSIM similarity and average recovery priority.
    beta : float ∈ [0,1]
        Influence of hygiene entropy on the final score.

    Returns
    -------
    float
        Hybrid score in the range [0,1]; higher is better.
    """
    # 1️⃣ Morphological similarity
    S = hybrid_similarity(endpoint_a, endpoint_b)

    # 2️⃣ Average recovery priority from parent A
    R1 = recovery_priority(endpoint_a.morphology)
    R2 = recovery_priority(endpoint_b.morphology)
    R_avg = (R1 + R2) / 2.0

    # 3️⃣ Hygiene/entropy component from parent B
    H = hygiene_score(logs)  # already in [0,1]; higher = better hygiene

    # Blend similarity and priority, then attenuate by entropy influence
    blended = alpha * S + (1 - alpha) * R_avg
    Ψ = blended * (1 - beta * (1 - H))  # when H is low (high entropy) score drops
    return max(0.0, min(1.0, Ψ))


def evaluate_endpoint_pair(
    endpoint_a: EngineEndpoint,
    endpoint_b: EngineEndpoint,
    logs: List[str],
    breaker: EndpointCircuitBreaker,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the full hybrid evaluation pipeline and update the circuit breaker.

    Returns a dictionary with intermediate values for introspection.
    """
    S = hybrid_similarity(endpoint_a, endpoint_b)
    H = hygiene_score(logs)
    Ψ = hybrid_recovery_score(endpoint_a, endpoint_b, logs, **kwargs)
    breaker.update(Ψ)

    return {
        "similarity": S,
        "hygiene_score": H,
        "hybrid_score": Ψ,
        "circuit_breaker_tripped": breaker.tripped,
        "failures": breaker.failures,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two synthetic endpoints with random morphologies
    def rand_morph() -> Morphology:
        return Morphology(
            length=random.uniform(0.5, 2.5),
            width=random.uniform(0.5, 2.5),
            height=random.uniform(0.5, 2.5),
            mass=random.uniform(1.0, 10.0),
        )

    ep1 = EngineEndpoint(
        engine_id="E1",
        channel="alpha",
        residency="us-east",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="https://api.example.com/ep1",
        capabilities=["compute", "store"],
        morphology=rand_morph(),
    )

    ep2 = EngineEndpoint(
        engine_id="E2",
        channel="beta",
        residency="eu-west",
        runtime="python3.11",
        resource_class="standard",
        always_on=False,
        endpoint="https://api.example.com/ep2",
        capabilities=["compute"],
        morphology=rand_morph(),
    )

    # Sample log messages mimicking operational chatter
    sample_logs = [
        "Verified the source hash and recorded the checksum.",
        "Planning the rollout sequence for the next deployment.",
        "Will pause the pipeline tomorrow for maintenance.",
        "Asked the support team to review the audit logs.",
        "Evidence collected from the previous run looks solid.",
        "No delay expected after the checkpoint.",
    ]

    breaker = EndpointCircuitBreaker(failure_threshold=2, score_cutoff=0.45)

    result = evaluate_endpoint_pair(ep1, ep2, sample_logs, breaker, alpha=0.6, beta=0.3)

    print("Hybrid evaluation results:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Demonstrate circuit‑breaker behaviour by feeding a low‑score scenario
    low_score_logs = ["delay", "later", "wait", "pause"] * 5
    for _ in range(3):
        evaluate_endpoint_pair(ep1, ep2, low_score_logs, breaker, alpha=0.5, beta=0.7)

    print("\nAfter feeding low‑score logs:")
    print(f"  circuit_breaker_tripped: {breaker.tripped}")
    print(f"  failures: {breaker.failures}")