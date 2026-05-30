# DARWIN HAMMER — match 415, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_path_signature_m231_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# born: 2026-05-29T23:28:51Z

"""
Hybrid Algorithm: Labeling‑Bandit Fusion

Parents
-------
- **Parent A** – labeling pipeline with path‑signature operations
  (functions: `labeling_function`, `aggregate_labels`, `lead_lag_transform`).
- **Parent B** – contextual multi‑armed bandit with a virtual‑VRAM store
  (functions: `select_action`, `_reward`, policy bookkeeping).

Mathematical Bridge
------------------
A *context vector* for the bandit is built from the **path signature**
derived from the data instance that is to be labeled.  The signature
(`lead_lag_transform`) yields a matrix `S ∈ ℝ^{2·T×2}` (interleaved lead‑lag
channels).  We flatten `S` and normalise it to obtain a feature vector
`c ∈ ℝ^{d}`.  This vector is supplied to `select_action` as the *context*.

The bandit returns an action `a` (a labeling function identifier) together
with an **expected reward** `r̂(a)`.  The hybrid aggregator uses `r̂(a)` to
scale the confidence of the votes contributed by that labeling function.
Thus the final probabilistic label is


p(label = ℓ) ∝ Σ_{i∈F} 𝟙[ℓ_i = ℓ] · confidence_i · r̂(i)


where `F` is the set of labeling functions selected by the bandit for the
current instance.  The *recovery priority* is defined as the inverse of the
right‑most non‑zero entry of the flattened signature, providing a
monotonically decreasing priority that modulates an error‑detection
threshold.

The three public functions below demonstrate this fused workflow:
`hybrid_labeling`, `hybrid_recovery_priority`, and `hybrid_error_detection`. 
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – labeling & path‑signature utilities
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


def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(
    batches: List[List[LabelingFunctionResult]],
    bandit_weights: Dict[str, float] | None = None,
) -> List[ProbabilisticLabel]:
    """
    Hybrid aggregation: majority vote where each vote is weighted by a
    bandit‑derived factor (`bandit_weights`).  If `bandit_weights` is None,
    uniform weighting (original A‑logic) is used.
    """
    votes: Dict[str, Dict[int, float]] = {}
    for batch in batches:
        for r in batch:
            if r.label not in (0, 1):
                continue
            w = 1.0
            if bandit_weights:
                w = bandit_weights.get(r.lf_name, 1.0)
            votes.setdefault(r.doc_id, {0: 0.0, 1: 0.0})
            votes[r.doc_id][r.label] += w

    out: List[ProbabilisticLabel] = []
    for doc_id, weight_dict in votes.items():
        label = max(weight_dict, key=weight_dict.get)
        total = weight_dict[0] + weight_dict[1]
        confidence = weight_dict[label] / total if total > 0 else 0.0
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out


def lead_lag_transform(path: List[Tuple[float, float]]) -> np.ndarray:
    """
    Lead‑lag transform for a 2‑D path.
    For each point (x_t, y_t) we produce a pair:
        lead  = (x_t, y_t)
        lag   = (x_{t-1}, y_{t-1})   (with (0,0) for t=0)
    The result is an interleaved matrix of shape (2·T, 2).
    """
    if not path:
        return np.empty((0, 2))
    lead = np.asarray(path, dtype=float)
    lag = np.vstack([np.zeros((1, 2)), lead[:-1]])
    interleaved = np.empty((lead.shape[0] * 2, 2), dtype=float)
    interleaved[0::2] = lead
    interleaved[1::2] = lag
    return interleaved


# ----------------------------------------------------------------------
# Parent B – contextual bandit with virtual VRAM store
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}  # virtual VRAM store per key


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action given a numeric context.
    The context dictionary is transformed into a flat vector whose Euclidean norm
    scales the LinUCB exploration term.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0.0, _reward(a)),
                1 + max(0.0, 1 - _reward(a)),
            ),
        )
    else:  # linucb‑style surrogate
        scale = math.sqrt(sum(v * v for v in context.values())) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=_reward(chosen),
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def update_policy(update: BanditUpdate) -> None:
    """Online update of the policy statistics."""
    stats = _POLICY.setdefault(update.action_id, [0.0, 0.0])
    # Increment total reward (importance‑weighted) and count
    stats[0] += update.reward / max(update.propensity, 1e-8)
    stats[1] += 1.0
    # Update virtual VRAM store (simulated side‑effect)
    _STORE[update.context_id] = _STORE.get(update.context_id, 0.0) + update.reward


# ----------------------------------------------------------------------
# Hybrid Layer – mathematical fusion of A and B
# ----------------------------------------------------------------------


def _signature_to_context(sig: np.ndarray) -> Dict[str, float]:
    """
    Convert a flattened signature into a dictionary suitable for the bandit.
    Each entry is named ``f{i}`` where *i* is the index.
    """
    flat = sig.ravel()
    # Normalise to unit L2 norm to keep scales comparable
    norm = np.linalg.norm(flat) if np.linalg.norm(flat) > 0 else 1.0
    flat_norm = flat / norm
    return {f"f{i}": float(v) for i, v in enumerate(flat_norm)}


def hybrid_recovery_priority(path: List[Tuple[float, float]]) -> float:
    """
    Compute a *recovery priority* from the right‑most non‑zero entry of the
    lead‑lag signature.  Smaller indices (earlier in the path) yield higher
    priority.
    """
    sig = lead_lag_transform(path)
    if sig.size == 0:
        return 0.0
    flat = sig.ravel()
    # Find last index with magnitude > eps
    eps = 1e-12
    idx = np.max(np.where(np.abs(flat) > eps)[0]) if np.any(np.abs(flat) > eps) else 0
    # Priority decays with index; use exponential decay for smoothness
    priority = math.exp(-0.05 * idx)
    return priority


def hybrid_labeling(
    path: List[Tuple[float, float]],
    labeling_fns: List[Callable[[dict], int]],
    doc_id: str,
) -> ProbabilisticLabel:
    """
    End‑to‑end hybrid labeling:

    1. Build a signature from ``path`` and turn it into a bandit context.
    2. Ask the bandit which labeling function to invoke (one per call).
    3. Execute the chosen function, collect its vote, and repeat for a
       fixed number of rounds (here, equal to the number of available fns).
    4. Aggregate votes using bandit‑derived expected rewards as weights.
    """
    # Step 1 – signature → context
    sig = lead_lag_transform(path)
    context = _signature_to_context(sig)

    # Prepare actions list (labeling function names)
    actions = [fn.lf_name for fn in labeling_fns]

    # Step 2‑4 – iterative bandit‑guided voting
    batches: List[List[LabelingFunctionResult]] = []
    bandit_weights: Dict[str, float] = {}

    for _ in range(len(actions)):
        ba = select_action(context, actions, algorithm="linucb")
        # Find the actual callable
        fn = next(fn for fn in labeling_fns if fn.lf_name == ba.action_id)
        # Simulate a data point for the labeling function
        datum = {"path": path, "doc_id": doc_id}
        label = fn(datum)  # type: ignore[arg-type]
        batches.append([LabelingFunctionResult(fn.lf_name, doc_id, label)])
        # Store weight = expected reward (plus small epsilon to avoid zero)
        bandit_weights[fn.lf_name] = ba.expected_reward + 1e-6

        # Mock reward: 1 if label matches a hidden ground truth (simulated)
        # For demonstration we assume ground truth = 1 when the average x > 0.
        ground_truth = int(np.mean([p[0] for p in path]) > 0)
        reward = 1.0 if label == ground_truth else 0.0
        update_policy(
            BanditUpdate(
                context_id=doc_id,
                action_id=ba.action_id,
                reward=reward,
                propensity=ba.propensity,
            )
        )
        # Remove the chosen action to avoid reselection in this simple loop
        actions.remove(ba.action_id)

    # Final aggregation
    aggregated = aggregate_labels(batches, bandit_weights=bandit_weights)
    # There will be exactly one ProbabilisticLabel for the given doc_id
    return aggregated[0]


def hybrid_error_detection(
    prob_label: ProbabilisticLabel,
    recovery_priority: float,
    base_threshold: float = 0.6,
) -> bool:
    """
    Decide whether the probabilistic label should be flagged as an error.
    The threshold is relaxed (lowered) proportionally to the recovery priority.
    Returns ``True`` if the label is considered *reliable*.
    """
    # Higher priority → more tolerant (lower threshold)
    adjusted_threshold = base_threshold * (1.0 - 0.5 * recovery_priority)
    return prob_label.confidence >= adjusted_threshold


# ----------------------------------------------------------------------
# Example labeling functions (Parent A style)
# ----------------------------------------------------------------------


@labeling_function(name="sign_mean_x")
def sign_mean_x(datum: dict) -> int:
    """Label 1 if mean x > 0, else 0."""
    path: List[Tuple[float, float]] = datum["path"]
    return int(np.mean([p[0] for p in path]) > 0)


@labeling_function(name="sign_last_y")
def sign_last_y(datum: dict) -> int:
    """Label 1 if the last y coordinate is positive."""
    path: List[Tuple[float, float]] = datum["path"]
    return int(path[-1][1] > 0) if path else 0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any lingering state
    reset_policy()

    # Synthetic path: a simple random walk
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(loc=0.0, scale=1.0, size=(20, 2))
    path = list(np.cumsum(steps, axis=0))

    # Perform hybrid labeling
    prob_lbl = hybrid_labeling(
        path=path,
        labeling_fns=[sign_mean_x, sign_last_y],
        doc_id="doc_001",
    )
    print("Probabilistic label:", asdict(prob_lbl))

    # Compute recovery priority and decide reliability
    priority = hybrid_recovery_priority(path)
    reliable = hybrid_error_detection(prob_lbl, priority)
    print(f"Recovery priority: {priority:.4f}, reliable: {reliable}")