# DARWIN HAMMER — match 415, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_label_foundry_path_signature_m231_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# born: 2026-05-29T23:28:51Z

"""
Hybrid Label‑Bandit Fusion

Parents:
- Parent A (hybrid_hybrid_label_foundry_path_signature_m231_s1.py): provides
  labeling functions, majority‑vote aggregation and a lead‑lag transform that
  converts a sequence of binary labels into a causal path representation.
- Parent B (hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py): provides
  a contextual multi‑armed bandit (LinUCB / Thompson / ε‑greedy) with a
  virtual‑VRAM store for policy statistics.

Mathematical Bridge:
For each document we treat the ordered list of binary labels produced by the
labeling functions as a discrete path 𝑥(t)∈{0,1}.  Applying the lead‑lag
transform yields an interleaved vector 𝜙(𝑥)∈ℝ^{2T} that encodes both the
current label (lead) and the previous label (lag), i.e. a simple signature of
the path.  This signature is used as the *context* vector for the bandit
selection algorithm.  Conversely, the bandit’s expected reward 𝑅̂(a) is used
to scale the confidence of the aggregated label, giving a hybrid probabilistic
label:

    confidence_hybrid = confidence_A × σ(𝑅̂(a))

where σ is the logistic sigmoid that maps expected reward to [0,1].

The three core functions below demonstrate this fusion:
1. `document_signature` – builds the lead‑lag signature from label votes.
2. `hybrid_aggregate_labels` – aggregates labels and rescales confidence with
   the bandit’s expected reward.
3. `hybrid_select_action` – selects a bandit action using the signature as
   contextual features.

All state (policy statistics and a virtual VRAM store) lives in module‑level
dictionaries, mirroring Parent B.
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
# Parent A – labeling structures
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


def lead_lag_transform(labels: List[int]) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels.
    For a label sequence [l₁, l₂, …, l_T] we produce
        [l₁, 0, l₂, l₁, l₃, l₂, …, l_T, l_{T‑1}]
    The initial lag is defined as 0.
    """
    if not labels:
        return np.array([], dtype=float)

    lead = np.array(labels, dtype=float)
    lag = np.concatenate(([0.0], lead[:-1]))
    interleaved = np.empty(2 * len(labels), dtype=float)
    interleaved[0::2] = lead
    interleaved[1::2] = lag
    return interleaved


# ----------------------------------------------------------------------
# Parent B – bandit structures and utilities
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


_POLICY: Dict[str, List[float]] = {}   # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}          # virtual VRAM store (unused in this demo)


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: np.ndarray,
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using the supplied context vector.
    The context influences the exploration scale for the LinUCB surrogate.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta‑Bernoulli posterior with pseudo‑counts derived from rewards
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0.0, _reward(a)),
                1 + max(0.0, 1 - _reward(a)),
            ),
        )
    else:  # linucb‑style surrogate
        scale = math.sqrt(np.sum(context.astype(float) ** 2)) if context.size else 1.0
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
    """Incorporate a reward observation into the policy statistics."""
    stats = _POLICY.setdefault(update.action_id, [0.0, 0.0])
    stats[0] += update.reward
    stats[1] += 1.0


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def document_signature(batch: List[LabelingFunctionResult]) -> Tuple[str, np.ndarray]:
    """
    Build a lead‑lag signature for a single document from all its labeling
    function votes.  Returns (doc_id, signature_vector).
    """
    if not batch:
        raise ValueError("empty batch")
    # Preserve the order of appearance (simulates temporal ordering of LFs)
    labels = [r.label for r in batch]
    doc_id = batch[0].doc_id
    sig = lead_lag_transform(labels)
    return doc_id, sig


def hybrid_aggregate_labels(
    batches: List[List[LabelingFunctionResult]],
    actions: List[str],
    algorithm: str = "linucb",
) -> List[ProbabilisticLabel]:
    """
    For each document:
      1. Compute its lead‑lag signature.
      2. Use the signature as context to select a bandit action.
      3. Scale the majority‑vote confidence by σ(expected_reward).

    Returns a list of ProbabilisticLabel objects.
    """
    result: List[ProbabilisticLabel] = []

    for batch in batches:
        doc_id, sig = document_signature(batch)

        # Majority vote (Parent A logic)
        votes = [r.label for r in batch if r.label in (0, 1)]
        if not votes:
            continue
        label = max(set(votes), key=votes.count)
        base_conf = votes.count(label) / len(votes)

        # Bandit context → action
        ba = select_action(sig, actions, algorithm=algorithm)
        # Logistic scaling of expected reward to [0,1]
        scale = 1.0 / (1.0 + math.exp(-ba.expected_reward))

        hybrid_conf = base_conf * scale
        result.append(ProbabilisticLabel(doc_id, label, hybrid_conf))

    return result


def hybrid_select_action(
    doc_batches: List[List[LabelingFunctionResult]],
    actions: List[str],
    algorithm: str = "linucb",
) -> Dict[str, BanditAction]:
    """
    Produce a mapping doc_id → BanditAction where the context for each action
    is the lead‑lag signature of the corresponding document's labeling votes.
    """
    mapping: Dict[str, BanditAction] = {}
    for batch in doc_batches:
        doc_id, sig = document_signature(batch)
        ba = select_action(sig, actions, algorithm=algorithm)
        mapping[doc_id] = ba
    return mapping


def hybrid_update_from_labels(
    hybrid_labels: List[ProbabilisticLabel],
    action_map: Dict[str, BanditAction],
) -> None:
    """
    Feed back the hybrid confidences as rewards to the bandit policy.
    For each document we treat the confidence as a stochastic reward for the
    action selected for that document.
    """
    for pl in hybrid_labels:
        ba = action_map.get(pl.doc_id)
        if ba is None:
            continue
        upd = BanditUpdate(
            context_id=pl.doc_id,
            action_id=ba.action_id,
            reward=pl.confidence,
            propensity=ba.propensity,
        )
        update_policy(upd)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any prior state
    reset_policy()

    # Dummy labeling functions results for three documents
    dummy_batches: List[List[LabelingFunctionResult]] = [
        [
            LabelingFunctionResult("lf_a", "doc1", 1),
            LabelingFunctionResult("lf_b", "doc1", 0),
            LabelingFunctionResult("lf_c", "doc1", 1),
        ],
        [
            LabelingFunctionResult("lf_a", "doc2", 0),
            LabelingFunctionResult("lf_b", "doc2", 0),
            LabelingFunctionResult("lf_c", "doc2", 1),
        ],
        [
            LabelingFunctionResult("lf_a", "doc3", 1),
            LabelingFunctionResult("lf_b", "doc3", 1),
            LabelingFunctionResult("lf_c", "doc3", 1),
        ],
    ]

    actions = ["action_X", "action_Y", "action_Z"]

    # Hybrid aggregation
    hybrid_labels = hybrid_aggregate_labels(dummy_batches, actions, algorithm="linucb")
    print("Hybrid Probabilistic Labels:")
    for pl in hybrid_labels:
        print(asdict(pl))

    # Obtain per‑document actions
    action_map = hybrid_select_action(dummy_batches, actions, algorithm="linucb")
    print("\nSelected Bandit Actions per Document:")
    for doc, act in action_map.items():
        print(doc, asdict(act))

    # Update bandit policy with the hybrid confidences
    hybrid_update_from_labels(hybrid_labels, action_map)

    # Show updated policy statistics
    print("\nPolicy statistics after update:")
    for a, stats in _POLICY.items():
        print(a, {"total_reward": stats[0], "count": stats[1], "mean_reward": _reward(a)})

    # Demonstrate a second round of selection to see learning effect
    second_actions = hybrid_select_action(dummy_batches, actions, algorithm="linucb")
    print("\nSecond round actions (should reflect updated rewards):")
    for doc, act in second_actions.items():
        print(doc, act.action_id, act.expected_reward)