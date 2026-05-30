# DARWIN HAMMER — match 4597, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_ternar_m1677_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2111_s0.py (gen6)
# born: 2026-05-29T23:56:56Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_ternary_lens__hybrid_hybrid_ternar_m1677_s0.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2111_s0.py

Mathematical Bridge
-------------------
Parent A produces a *ternary feature vector* 𝑣∈{‑1,0,1}¹² derived from a command envelope and
evaluates it with a Hoeffding bound (probabilistic confidence interval) and a Gini‑coefficient
metric.  
Parent B manages a pool of models whose resource consumption can be expressed as a *resource
matrix* 𝐑∈ℝ^{k×2} (k = number of loaded models, columns = [RAM, tier‑weight]).

The fusion maps the ternary vector 𝑣 to a *selection vector* 𝑠 by projecting 𝑣 onto a random
hyperplane 𝐡∈ℝ¹² (the “random hyperplane vector” from Parent A).  The scalar projection
p = 𝑣·𝐡 is treated as a stochastic reward estimate for loading a candidate model.
Using the Hoeffding bound we obtain a high‑confidence lower‑bound ℓ(p) on the true reward.
If ℓ(p) exceeds a threshold derived from the model’s normalized RAM demand (a row of 𝐑),
the model is admitted to the pool.

Thus the core topology combines:
1. Ternary → hyperplane projection (vector algebra, Parent A).
2. Hoeffding‑bound confidence (Parent A) as a decision criterion.
3. Model‑resource matrix handling (Parent B) to enforce a global RAM ceiling.

The following module implements this hybrid logic with three public functions:
`ternary_vector`, `hoeffding_bound`, `hybrid_load_decision`, plus supporting utilities."""
import json
import hashlib
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – ternary vector and Hoeffding utilities
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 without microseconds."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def payload_hash(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> str:
    """Deterministic SHA‑256 of the command envelope."""
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ternary_vector(raw_command: str, normalized_intent: str, context: Dict[str, Any]) -> np.ndarray:
    """
    Produce a 12‑dimensional ternary vector (values in {‑1,0,1}) from a command envelope.
    The algorithm mirrors the original Parent A implementation.
    """
    payload = {
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "context": context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    hash_int = int(hashlib.sha256(encoded).hexdigest(), 16)

    vec = np.empty(12, dtype=int)
    for i in range(12):
        # Two bits per entry → values 0‑3
        val = (hash_int >> (2 * i)) & 0b11
        # Map 0→‑1, 1→0, 2→1, 3→0 (arbitrary but deterministic)
        if val == 0:
            vec[i] = -1
        elif val == 1:
            vec[i] = 0
        elif val == 2:
            vec[i] = 1
        else:  # val == 3
            vec[i] = 0
    return vec


def random_hyperplane(dim: int = 12) -> np.ndarray:
    """Generate a random unit hyperplane vector of given dimension."""
    rng = np.random.default_rng()
    h = rng.normal(size=dim)
    norm = np.linalg.norm(h)
    return h / norm if norm != 0 else h


def hoeffding_bound(mean: float, n: int, delta: float) -> float:
    """
    Hoeffding bound for bounded random variables in [0,1].
    Returns the lower confidence bound ℓ such that P(true_mean ≥ ℓ) ≥ 1‑δ.
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive")
    radius = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    return max(0.0, mean - radius)


def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient for a list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = 0.0
    for i, val in enumerate(sorted_vals, 1):
        cumulative += i * val
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2.0 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Parent B – ModelPool and resource matrix handling
# ----------------------------------------------------------------------
@dataclass
class ModelTier:
    """Lightweight descriptor of a model."""
    name: str
    ram_mb: int
    tier: str  # e.g. "base", "large", "xl"


@dataclass
class ModelPool:
    """
    Manages loaded models under a RAM ceiling.
    Internally keeps a resource matrix R where each row is [ram_mb, tier_weight].
    """
    ram_ceiling_mb: int = 6000
    tier_weights: Dict[str, float] = field(default_factory=lambda: {"base": 1.0, "large": 1.5, "xl": 2.0})
    loaded: Dict[str, ModelTier] = field(default_factory=dict)

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """True if adding model stays within RAM ceiling."""
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load model if RAM permits; raise otherwise."""
        if self.can_load(model):
            self.loaded[model.name] = model
        else:
            raise MemoryError(f"Insufficient RAM to load model {model.name}")

    def resource_matrix(self) -> np.ndarray:
        """
        Return the current resource matrix R (rows = models, cols = [ram, tier_weight]).
        """
        rows = []
        for m in self.loaded.values():
            weight = self.tier_weights.get(m.tier, 1.0)
            rows.append([float(m.ram_mb), float(weight)])
        return np.array(rows, dtype=float) if rows else np.empty((0, 2))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_projection_score(
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    hyperplane: np.ndarray | None = None,
) -> float:
    """
    Compute the scalar projection of the ternary vector onto a random hyperplane.
    If `hyperplane` is None a fresh unit hyperplane is generated.
    """
    v = ternary_vector(raw_command, normalized_intent, context).astype(float)
    h = hyperplane if hyperplane is not None else random_hyperplane(len(v))
    # Normalise projection to [0,1] for Hoeffding compatibility
    raw = np.dot(v, h)
    # The raw dot product lies in roughly [‑√d, √d]; map linearly to [0,1]
    d = len(v)
    proj = (raw + math.sqrt(d)) / (2 * math.sqrt(d))
    return float(np.clip(proj, 0.0, 1.0))


def hybrid_load_decision(
    pool: ModelPool,
    candidate: ModelTier,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    delta: float = 0.05,
) -> bool:
    """
    Decide whether to load `candidate` into `pool` using the hybrid criterion:
    1. Compute projection score p ∈ [0,1].
    2. Obtain Hoeffding lower bound ℓ(p) with confidence 1‑δ.
    3. Compare ℓ(p) against a threshold derived from the model's normalized RAM demand.
       Threshold τ = (ram_mb / ram_ceiling_mb) * α, where α ∈ (0,1] is a tunable aggressiveness factor.
    If ℓ(p) ≥ τ and RAM permits, the model is loaded.
    """
    # Step 1 – projection
    p = hybrid_projection_score(raw_command, normalized_intent, context)

    # Step 2 – Hoeffding (treating single observation as n=1)
    lower = hoeffding_bound(p, n=1, delta=delta)

    # Step 3 – threshold
    alpha = 0.8  # aggressiveness; can be tuned
    tau = (candidate.ram_mb / pool.ram_ceiling_mb) * alpha

    decision = lower >= tau and pool.can_load(candidate)
    if decision:
        pool.load(candidate)
    return decision


def hybrid_process(
    pool: ModelPool,
    raw_command: str,
    normalized_intent: str,
    context: Dict[str, Any],
    candidates: List[ModelTier],
) -> List[str]:
    """
    Process a list of candidate models, attempting to load each according to the hybrid rule.
    Returns the names of models successfully loaded.
    Also returns the Gini coefficient of the current RAM distribution as a side metric.
    """
    loaded_names = []
    for cand in candidates:
        if hybrid_load_decision(pool, cand, raw_command, normalized_intent, context):
            loaded_names.append(cand.name)

    # Side metric – diversity of RAM usage among loaded models
    ram_usages = [m.ram_mb for m in pool.loaded.values()]
    gini = gini_coefficient(ram_usages)
    print(f"[Hybrid] Gini coefficient of RAM usage after loading: {gini:.4f}")

    return loaded_names


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal deterministic environment
    command = "summarize the quarterly report"
    intent = "summarize"
    ctx = {"user": "alice", "session_id": "xyz123"}

    # Initialise pool with a modest ceiling
    pool = ModelPool(ram_ceiling_mb=2000)

    # Candidate models
    candidates = [
        ModelTier(name="tiny-base", ram_mb=300, tier="base"),
        ModelTier(name="medium-large", ram_mb=800, tier="large"),
        ModelTier(name="heavy-xl", ram_mb=1500, tier="xl"),
    ]

    # Run hybrid processing
    loaded = hybrid_process(pool, command, intent, ctx, candidates)
    print(f"[Hybrid] Models loaded: {loaded}")
    print(f"[Hybrid] Total RAM used: {pool._used()} / {pool.ram_ceiling_mb} MB")