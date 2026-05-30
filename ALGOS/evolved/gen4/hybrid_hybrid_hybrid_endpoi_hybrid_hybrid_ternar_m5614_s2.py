# DARWIN HAMMER — match 5614, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s1.py (gen2)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s0.py (gen3)
# born: 2026-05-30T00:03:34Z

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Classifications used by the pruning schedule. In a real system this would
# probably be loaded from a config file or an enum.
CLASSIFICATIONS = ["usable_now", "research_only", "needs_conversion"]


def now_z() -> str:
    """Return the current UTC timestamp as an ISO‑8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Simple failure counter that opens after a configurable threshold.
    It also provides a *recovery priority* – a continuous value in [0, 1]
    that reflects how eager the system should be to re‑enable the endpoint.
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures: int = 0
        self.open: bool = False
        self.last_event_at: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    # ------------------------------------------------------------------
    # Derived metric
    # ------------------------------------------------------------------
    def recovery_priority(self) -> float:
        """
        Compute a recovery priority in the interval [0, 1].

        The priority grows as the number of consecutive failures drops.
        A simple sigmoid is used to keep the value smooth:

            p = 1 / (1 + exp(k * (failures - t)))

        where *k* controls steepness and *t* is the failure threshold.
        """
        k = 0.8
        t = self.failure_threshold
        return 1.0 / (1.0 + math.exp(k * (self.failures - t)))

    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
            "recovery_priority": self.recovery_priority(),
        }


# ----------------------------------------------------------------------
# Parent B – resource pruning and VRAM scheduler
# ----------------------------------------------------------------------
def compute_prune_probability(time_step: int, alpha: float, lam: float) -> float:
    """
    Decreasing‑rate pruning probability.

    The original implementation used `min(1, lam * exp(-alpha * t))`.
    Here we keep the same functional form but expose the *effective*
    decay constant `beta = alpha / (1 + alpha)` to make the curve
    more tunable without breaking backward compatibility.
    """
    beta = alpha / (1.0 + alpha)
    prob = lam * math.exp(-beta * time_step)
    return min(1.0, prob)


def modulate_preemption_probabilities(
    manifest: List[Dict[str, Any]],
    time_step: int,
    alpha: float,
    lam: float,
) -> Tuple[Dict[str, float], float]:
    """
    Compute per‑candidate pre‑emption probabilities and return the
    *global* prune probability used later for the fusion step.

    The probability for each candidate is proportional to the
    classification weight (empirical frequency) and the global prune
    probability.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Build a histogram of classifications
    # ------------------------------------------------------------------
    audit_summary = {cls: 0 for cls in CLASSIFICATIONS}
    for cand in manifest:
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            # Unknown classes are treated as “research_only” – a safe default.
            cls = "research_only"
        audit_summary[cls] += 1

    total = max(1, len(manifest))  # avoid division by zero
    weight_vector = {cls: cnt / total for cls, cnt in audit_summary.items()}

    # ------------------------------------------------------------------
    # 2️⃣  Global prune probability (same for every candidate)
    # ------------------------------------------------------------------
    prune_prob = compute_prune_probability(time_step, alpha, lam)

    # ------------------------------------------------------------------
    # 3️⃣  Per‑candidate pre‑emption probability
    # ------------------------------------------------------------------
    preempt = {}
    for cand in manifest:
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            cls = "research_only"
        preempt[cand["id"]] = prune_prob * weight_vector[cls]

    return preempt, prune_prob


# ----------------------------------------------------------------------
# Mathematical bridge – a deeper Bayesian‑style fusion
# ----------------------------------------------------------------------
def fuse_recovery_priority_and_pruning_schedule(
    recovery_priority: float,
    prune_probability: float,
    curvature_factor: float = 0.5,
) -> float:
    """
    Fuse the recovery priority (from Parent A) with the pruning schedule
    (from Parent B) using a *log‑odds* combination that respects the
    probabilistic nature of both quantities.

    Let `p_r` be the recovery priority and `p_p` the prune probability.
    Their log‑odds are:

        o_r = log(p_r / (1 - p_r))
        o_p = log(p_p / (1 - p_p))

    The fused log‑odds is a weighted sum:

        o_f = (1 - c) * o_r + c * o_p

    where `c ∈ [0,1]` is the curvature factor controlling how much
    geometric curvature (Parent A) influences the final value.
    The result is mapped back to a probability with the sigmoid.

    This yields a smoother, more expressive integration than a plain
    multiplication while keeping the output in [0,1].
    """
    # Guard against exact 0/1 which would cause log(0)
    eps = 1e-12
    p_r = min(max(recovery_priority, eps), 1 - eps)
    p_p = min(max(prune_probability, eps), 1 - eps)

    log_odds_r = math.log(p_r / (1.0 - p_r))
    log_odds_p = math.log(p_p / (1.0 - p_p))

    fused_log_odds = (1.0 - curvature_factor) * log_odds_r + curvature_factor * log_odds_p
    fused_probability = 1.0 / (1.0 + math.exp(-fused_log_odds))
    return fused_probability


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_endpoint_morphology_and_resource_pruning(
    manifest: Dict[str, List[Dict[str, Any]]],
    time_step: int,
    alpha: float,
    lam: float,
    curvature_factor: float = 0.5,
) -> Dict[str, Any]:
    """
    Combine geometric properties (recovery priority) with resource
    pruning and VRAM scheduling.

    Parameters
    ----------
    manifest : dict
        Expected shape ``{'candidates': [{ 'id': str, 'classification': str }, …]}``.
    time_step : int
        Discrete time step used by the exponential decay.
    alpha, lam : float
        Parameters of the pruning schedule.
    curvature_factor : float, optional
        Weight of the pruning schedule in the Bayesian fusion (default 0.5).

    Returns
    -------
    dict
        Contains the fused recovery priority, per‑candidate pre‑emption
        probabilities, and a synthetic *endpoint morphology* vector that
        encodes geometric curvature derived from classification frequencies.
    """
    # ------------------------------------------------------------------
    # Normalise input manifest
    # ------------------------------------------------------------------
    candidates = manifest.get("candidates", [])
    if not isinstance(candidates, list):
        raise TypeError("manifest['candidates'] must be a list of dicts")

    # Ensure each candidate has a unique identifier
    for idx, cand in enumerate(candidates):
        if "id" not in cand:
            cand["id"] = f"cand_{idx}"
        if "classification" not in cand:
            cand["classification"] = "research_only"

    # ------------------------------------------------------------------
    # 1️⃣  Circuit‑breaker state (Parent A)
    # ------------------------------------------------------------------
    circuit_breaker = EndpointCircuitBreaker()
    # In a realistic scenario we would feed real success/failure events.
    # For the hybrid demo we keep the breaker closed.
    recovery_prio = circuit_breaker.recovery_priority()

    # ------------------------------------------------------------------
    # 2️⃣  Pruning schedule & pre‑emption probabilities (Parent B)
    # ------------------------------------------------------------------
    preemption_probs, prune_prob = modulate_preemption_probabilities(
        candidates, time_step, alpha, lam
    )

    # ------------------------------------------------------------------
    # 3️⃣  Fuse the two probabilistic signals
    # ------------------------------------------------------------------
    fused_priority = fuse_recovery_priority_and_pruning_schedule(
        recovery_prio, prune_prob, curvature_factor
    )

    # ------------------------------------------------------------------
    # 4️⃣  Derive a synthetic geometric “endpoint morphology”
    # ------------------------------------------------------------------
    # We treat the classification histogram as a point in a simplex and
    # compute its curvature via the normalized entropy. Higher entropy
    # (more uniform distribution) yields a “flatter” morphology.
    counts = np.array([sum(1 for c in candidates if c["classification"] == cls) for cls in CLASSIFICATIONS])
    probs = counts / max(1, counts.sum())
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = math.log2(len(CLASSIFICATIONS))
    curvature = 1.0 - (entropy / max_entropy)  # 0 = flat, 1 = peaked

    endpoint_morphology = {
        "classification_distribution": dict(zip(CLASSIFICATIONS, counts.tolist())),
        "entropy": entropy,
        "curvature": curvature,
        "fused_recovery_priority": fused_priority,
    }

    # ------------------------------------------------------------------
    # 5️⃣  Assemble final result
    # ------------------------------------------------------------------
    return {
        "endpoint_morphology": endpoint_morphology,
        "resource_pruning_schedule": {
            "global_prune_probability": prune_prob,
            "curvature_factor": curvature_factor,
        },
        "preemption_probabilities": preemption_probs,
        "modulated_recovery_priority": fused_priority,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example manifest – note the required 'id' field for deterministic output
    manifest_example = {
        "candidates": [
            {"id": "c1", "classification": "usable_now"},
            {"id": "c2", "classification": "research_only"},
            {"id": "c3", "classification": "needs_conversion"},
            {"id": "c4", "classification": "usable_now"},
        ]
    }

    # Hyper‑parameters
    time_step = 10
    alpha = 0.5
    lam = 0.2
    curvature_factor = 0.4

    result = hybrid_endpoint_morphology_and_resource_pruning(
        manifest_example,
        time_step,
        alpha,
        lam,
        curvature_factor,
    )

    # Pretty‑print the result
    import json

    print(json.dumps(result, indent=2, default=str))