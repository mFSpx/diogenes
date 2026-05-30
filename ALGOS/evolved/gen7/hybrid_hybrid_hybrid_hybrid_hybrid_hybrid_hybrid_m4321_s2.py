# DARWIN HAMMER — match 4321, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2138_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_distributed_l_m2622_s0.py (gen5)
# born: 2026-05-29T23:54:52Z

"""Hybrid Fusion of Darwin Hammer (A) and Distributed Sparse (B)

This module combines the core mathematics of two parent algorithms:

* **Parent A – Hybrid Darwin Hammer**  
  - Uses log‑count ratios as a scalar confidence that multiplies counts, pheromone
    levels and entropy calculations.  
  - Introduces a model‑priority field `f(endpoint, model) = p(m)*(1‑r/R_max)`.

* **Parent B – Distributed Sparse Expansion**  
  - Provides a hash‑based sparse expansion `expand(values, m, salt)` and a
    top‑k mask `top_k_mask`.  
  - Treats a *confidence scalar* to rescale the random coefficients of the
    sparse projection.

**Mathematical Bridge**  
Both parents rely on a single scalar that modulates other quantities:


c = log_count_ratio                # Parent A confidence
c = confidence scalar from signal‑to‑noise gap  # Parent B interpretation


We therefore fuse them by letting `c` (the log‑count ratio) rescale the sparse
expansion coefficients produced by Parent B.  The resulting vector is then
filtered by a top‑k mask and finally combined with the model‑priority field
and the pheromone‑based terms from Parent A.

The unified system therefore follows the pipeline:


values ──► sparse_expand(values, m, c) ──► top_k_mask ──► weighted_sum
      │                                            │
      └─► hybrid_store_factor ──► … ──► fused_score


The three public functions below illustrate the hybrid operation.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Any, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
R_MAX = 1024  # Maximum RAM (MB) allowed in the ModelPool – adjustable as needed


def _log_count_ratio(count: int, total: int) -> float:
    """Utility to compute a stable log‑count ratio."""
    if total <= 0:
        raise ValueError("total must be positive")
    # Add 1 to avoid log(0); use natural log for consistency with Parent A
    return math.log((count + 1) / total)


# ----------------------------------------------------------------------
# Parent A – Model tier and confidence‑driven primitives
# ----------------------------------------------------------------------
@dataclass
class ModelTier:
    """Lightweight descriptor of a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. 'bronze', 'silver', 'gold'

    def priority(self, morphology_score: float) -> float:
        """
        Normalised priority derived from morphology (righting‑time index).

        Parameters
        ----------
        morphology_score : float
            Raw morphology metric; higher means more critical.

        Returns
        -------
        float
            Normalised priority in [0, 1].
        """
        # Clamp and normalise assuming the raw score lies in [0, 100]
        return max(0.0, min(1.0, morphology_score / 100.0))


def hybrid_store_factor(action_id: Any, count: int, log_count_ratio: float) -> float:
    """Parent A: store factor = log_count_ratio * count."""
    return log_count_ratio * count


def pheromone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Parent A: pheromone infotaxis = pheromone * log_count_ratio."""
    return pheromone * log_count_ratio


def decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    """Parent A: entropy term = pheromone * log_count_ratio."""
    return pheromone * log_count_ratio


def fold_change_detection(x: float, eps: float = 1e-12) -> float:
    """Parent A: log‑fold‑change detection."""
    return math.log(x / max(abs(x), eps))


# ----------------------------------------------------------------------
# Parent B – Sparse hash‑based projection utilities
# ----------------------------------------------------------------------
def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  # three independent hash probes per value
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if (h[8] & 1) else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with 1 at indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def broadcast_probability(phases: int, phase: int) -> float:
    """
    Simple exponential decay schedule used in Parent B.
    Returns the probability of broadcasting at a given phase.
    """
    if phases <= 0:
        raise ValueError("phases must be positive")
    decay = math.exp(-phase / phases)
    return decay


# ----------------------------------------------------------------------
# Hybrid Functions – mathematical fusion of A and B
# ----------------------------------------------------------------------
def sparse_hybrid_expand(
    values: List[float],
    m: int,
    log_count_ratio: float,
    salt: str = "",
) -> List[float]:
    """
    Expand `values` sparsely and rescale each contribution by the confidence
    scalar `log_count_ratio` (the bridge between the two parents).

    Equivalent to:  expand(values, m, salt) * log_count_ratio
    """
    base = expand(values, m, salt)
    return [v * log_count_ratio for v in base]


def fused_score(
    endpoint_morphology: float,
    model: ModelTier,
    values: List[float],
    count: int,
    total_actions: int,
    pheromone: float,
    m: int = 256,
    top_k: int = 32,
) -> float:
    """
    Compute a unified score that blends model priority, sparse expansion,
    pheromone infotaxis and the hybrid store factor.

    Steps
    -----
    1. Compute confidence `c = log_count_ratio(count, total_actions)`.
    2. Compute model‑priority field `f = p(m) * (1 - r / R_MAX)`.
    3. Produce a confidence‑scaled sparse vector `s = sparse_hybrid_expand`.
    4. Apply a top‑k mask and sum the retained entries.
    5. Combine all ingredients multiplicatively.

    Returns
    -------
    float
        The fused decision score (higher → more favorable).
    """
    # 1. Confidence scalar
    c = _log_count_ratio(count, total_actions)

    # 2. Model‑priority field
    p = model.priority(endpoint_morphology)
    f = p * (1.0 - model.ram_mb / R_MAX)

    # 3. Confidence‑scaled sparse expansion
    sparse_vec = sparse_hybrid_expand(values, m, c, salt=model.name)

    # 4. Top‑k mask and weighted sum
    mask = top_k_mask(sparse_vec, top_k)
    weighted_sum = sum(v * m_i for v, m_i in zip(sparse_vec, mask))

    # 5. Combine with pheromone terms and store factor
    store = hybrid_store_factor(action_id=None, count=count, log_count_ratio=c)
    infotaxis = pheromone_infotaxis(pheromone, c)
    entropy = decision_hygiene_shannon_entropy(pheromone, c)

    # Final fused score (product of all meaningful components)
    score = (
        weighted_sum
        * store
        * infotaxis
        * entropy
        * f
    )
    return score


def hybrid_update_step(
    model: ModelTier,
    endpoint_morphology: float,
    values: List[float],
    count: int,
    total_actions: int,
    pheromone: float,
    broadcast_phases: int,
    current_phase: int,
) -> Dict[str, Any]:
    """
    Perform a single hybrid update step.

    - Computes the fused score.
    - Decides whether to broadcast the update using the exponential decay
      schedule from Parent B.
    - Returns a dictionary with diagnostic information.
    """
    score = fused_score(
        endpoint_morphology,
        model,
        values,
        count,
        total_actions,
        pheromone,
    )

    prob = broadcast_probability(broadcast_phases, current_phase)
    broadcast = random.random() < prob

    # Optional: adjust pheromone level based on the score (simple heuristic)
    new_pheromone = pheromone * (1 + 0.1 * math.tanh(score))

    return {
        "score": score,
        "broadcast": broadcast,
        "broadcast_probability": prob,
        "new_pheromone": new_pheromone,
        "model_priority": model.priority(endpoint_morphology),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy model tier
    dummy_model = ModelTier(name="alpha", ram_mb=200, tier="silver")

    # Synthetic input data
    endpoint_morphology_score = 73.5  # arbitrary morphology metric
    values = [random.uniform(-1, 1) for _ in range(20)]
    count = 42
    total_actions = 1000
    pheromone_level = 0.8
    broadcast_phases = 10
    current_phase = 3

    result = hybrid_update_step(
        model=dummy_model,
        endpoint_morphology=endpoint_morphology_score,
        values=values,
        count=count,
        total_actions=total_actions,
        pheromone=pheromone_level,
        broadcast_phases=broadcast_phases,
        current_phase=current_phase,
    )

    print("Hybrid update result:")
    for k, v in result.items():
        print(f"  {k}: {v}")