# DARWIN HAMMER — match 1211, survivor 7
# gen: 4
# parent_a: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s1.py (gen3)
# born: 2026-05-29T23:34:24Z

"""Hybrid Sketch-Bandit RLCT Router

This module fuses the sketching/RLCT core of *hybrid_sketches_rlct_grokking_m5_s0.py*
with the bandit‑router dynamics of *hybrid_hybrid_ternar_hybrid_bandit_router_m202_s1.py*.

Mathematical bridge
------------------
* The Count‑Min sketch produces a low‑dimensional frequency vector **s**∈ℝ^{w·d}.
* The Real Log Canonical Threshold (RLCT) is estimated by linear regression of
  log(loss) versus log(log(n)) on the non‑zero sketch entries; the resulting
  scalar 𝜆̂ quantifies the information loss caused by the dimensionality reduction.
* In a contextual bandit, the uncertainty (confidence bound) of an arm is
  traditionally proportional to √(1/propensity). We replace the √ term by
  √𝜆̂, thereby letting the information‑loss estimate directly modulate exploration.
* The StoreState dynamics (α·∑inflow−β·∑outflow) provide a slowly varying
  resource level **ℓ** that is multiplied with the bandit score, completing a
  single unified decision equation:

    score_i = (expected_reward_i + confidence_i·√𝜆̂) · ℓ

The implementation below embeds these equations in three high‑level functions
that together constitute the hybrid algorithm.
"""

import hashlib
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Sketching primitives (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch matrix of shape (depth, width)."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def minhash_lsh_index(docs: Dict[Any, List[str]]) -> Dict[str, List[Any]]:
    """Very light MinHash LSH: bucket by the minimum SHA‑1 hash of shingles."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)


def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[int]) -> float:
    """Linear regression of log(loss) on log(log(n)) → RLCT estimate."""
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


# ----------------------------------------------------------------------
# Bandit & Store primitives (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float  # probability of being chosen a priori
    expected_reward: float
    confidence_bound: float  # base confidence term (e.g., UCB width)
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = field(default=0.0, init=False, repr=False)

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Integrate inflow/outflow → new level and delta."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded, gain‑scaled version of the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def sketch_vector_from_data(data: List[Any], width: int = 64, depth: int = 4) -> np.ndarray:
    """Flatten a Count‑Min sketch into a 1‑D float vector."""
    sketch = count_min_sketch(data, width, depth)
    flat = np.array([float(v) for row in sketch for v in row], dtype=np.float64)
    return flat


def compute_rlct_from_sketch(sketch_vec: np.ndarray) -> float:
    """Treat non‑zero sketch entries as pseudo‑losses and estimate RLCT."""
    losses = sketch_vec[sketch_vec > 0]
    n_vals = np.arange(1, len(losses) + 1)
    if len(losses) < 2:
        # Degenerate case – no variance → default RLCT of 0 (no penalty)
        return 0.0
    return estimate_rlct_from_losses(losses.tolist(), n_vals.tolist())


def hybrid_bandit_score(
    action: BanditAction,
    rlct: float,
    store: StoreState,
    similarity: float,
) -> float:
    """
    Unified decision metric:

        score = (expected_reward + confidence_bound * sqrt(rlct) * similarity) * store.dance

    - ``similarity`` ∈ [0,1] measures how well the sketch matches the action's
      identifier (via a cheap hash‑based cosine proxy).
    - ``store.dance`` injects the slowly varying resource level.
    """
    # Guard against rlct==0 (no info loss) → treat sqrt as 0 to avoid NaN.
    sqrt_rlct = math.sqrt(rlct) if rlct > 0 else 0.0
    exploration = action.confidence_bound * sqrt_rlct * similarity
    raw_score = action.expected_reward + exploration
    return raw_score * store.dance


def hash_based_similarity(vec: np.ndarray, token: str) -> float:
    """
    Produce a deterministic similarity between a sketch vector and a token.
    We hash the token to a binary mask of the same length as ``vec`` and compute
    cosine similarity between the masked vector and the original.
    """
    # Create a pseudo‑random mask from the token's SHA‑256 digest.
    digest = hashlib.sha256(token.encode()).digest()
    # Expand digest to needed length.
    repeats = (len(vec) // len(digest)) + 1
    mask_bytes = (digest * repeats)[: len(vec)]
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).astype(np.float64)
    # Cosine similarity.
    dot = np.dot(vec, mask)
    norm_vec = np.linalg.norm(vec)
    norm_mask = np.linalg.norm(mask)
    if norm_vec == 0 or norm_mask == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_vec * norm_mask)))


def hybrid_process(
    data: List[Any],
    actions: List[BanditAction],
    store: StoreState,
) -> Tuple[BanditAction, float]:
    """
    End‑to‑end hybrid routine:

    1. Build a Count‑Min sketch vector from ``data``.
    2. Estimate RLCT from the sketch.
    3. For each candidate ``BanditAction`` compute a similarity score.
    4. Combine expected reward, RLCT‑scaled exploration, and store state
       into a unified score.
    5. Return the action with the highest score and the associated score.
    """
    sketch_vec = sketch_vector_from_data(data)
    rlct = compute_rlct_from_sketch(sketch_vec)

    # Update store based on sketch statistics (as a proxy for system load)
    inflow = [float(sketch_vec.sum())]
    outflow = [float(len(data))]
    store.update(inflow, outflow)

    best_action = None
    best_score = -math.inf

    for act in actions:
        sim = hash_based_similarity(sketch_vec, act.action_id)
        score = hybrid_bandit_score(act, rlct, store, sim)
        if score > best_score:
            best_score = score
            best_action = act

    return best_action, best_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data stream
    random.seed(42)
    data_stream = [random.randint(0, 1000) for _ in range(500)]

    # Define a few dummy actions
    dummy_actions = [
        BanditAction(
            action_id=f"action_{i}",
            propensity=1.0 / 3,
            expected_reward=random.uniform(0, 10),
            confidence_bound=random.uniform(0.5, 2.0),
        )
        for i in range(3)
    ]

    # Initialise store state
    store = StoreState(level=5.0, alpha=0.01, beta=0.005, dt=1.0, base=1.0, gain=2.0, limit=15.0)

    # Run the hybrid process
    chosen, score = hybrid_process(data_stream, dummy_actions, store)

    print(f"Chosen action: {chosen.action_id}")
    print(f"Score: {score:.4f}")
    print(f"Store level after update: {store.level:.4f}")
    print(f"RLCT estimate (debug): {compute_rlct_from_sketch(sketch_vector_from_data(data_stream)):.6f}")