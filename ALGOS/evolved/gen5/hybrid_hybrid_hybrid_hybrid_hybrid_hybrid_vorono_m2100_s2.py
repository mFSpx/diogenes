# DARWIN HAMMER — match 2100, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s3.py (gen4)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_label_foundry_m191_s0.py (gen3)
# born: 2026-05-29T23:40:53Z

"""Hybrid Label‑Bandit & Voronoi‑Morphology Fusion

Parents:
- hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s3.py (labeling + contextual bandit)
- hybrid_hybrid_voronoi_parti_hybrid_label_foundry_m191_s0.py (Voronoi partitioning with Morphology‑based recovery priority)

Mathematical Bridge:
The lead‑lag signature ϕ(x) ∈ ℝ^{2T} produced from binary label votes is interpreted as a
geometric point.  A Morphology instance provides a 2‑D seed (length, width) and a scalar
recovery priority ρ(m).  The Euclidean distance between a document signature (first two
components) and a seed is scaled by (1+ρ(m)), i.e.

    d̃_i = ‖ϕ_{0:2} – (ℓ_i, w_i)‖₂ · (1 + ρ(m_i))

This scaled distance augments the expected reward Ř_i estimated by a linear contextual
bandit (LinUCB).  The hybrid selection score for arm i is

    S_i = Ř_i  – α·d̃_i

where α > 0 balances bandit confidence against Voronoi proximity.  The arm with maximal
S_i supplies both a labeling decision (via its bandit statistics) and a Voronoi region
assignment.  Updates to the bandit parameters are performed with the observed reward,
thereby closing the feedback loop between labeling confidence and spatial partitioning.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – labeling & bandit structures
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
    """Interleave current (lead) and previous (lag) binary labels."""
    if not labels:
        return np.array([], dtype=float)
    lead = np.array(labels, dtype=float)
    lag = np.concatenate(([0.0], lead[:-1]))
    return np.ravel(np.column_stack((lead, lag)))


# ----------------------------------------------------------------------
# Parent B – Voronoi & Morphology structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def recovery_priority(m: Morphology) -> float:
    """
    A scalar expressing how “robust” an endpoint is.
    Larger volume relative to surface area yields higher priority.
    Normalised to keep values roughly in [0, 1].
    """
    vol = m.length * m.width * m.height
    surf = 2 * (m.length * m.width + m.width * m.height + m.height * m.length)
    if surf == 0:
        return 0.0
    raw = vol / surf
    # simple normalisation (max observed in practice assumed ≤1)
    return min(raw, 1.0)


# ----------------------------------------------------------------------
# Hybrid LinUCB bandit where each arm corresponds to a Morphology seed
# ----------------------------------------------------------------------


class HybridBanditArm:
    """
    Linear UCB arm.
    Maintains A (d×d) and b (d) for ridge‑regularised least‑squares.
    """

    def __init__(self, dim: int, alpha: float = 1.0):
        self.dim = dim
        self.alpha = alpha
        self.A = np.identity(dim, dtype=float)
        self.b = np.zeros(dim, dtype=float)

    @property
    def theta(self) -> np.ndarray:
        """Regularised estimator θ = A^{-1} b."""
        return np.linalg.solve(self.A, self.b)

    def expected_reward(self, context: np.ndarray) -> float:
        """Linear estimate of reward for given context vector."""
        return float(self.theta.dot(context))

    def update(self, reward: float, context: np.ndarray) -> None:
        """Standard LinUCB update."""
        self.A += np.outer(context, context)
        self.b += reward * context


# Global bandit registry – one arm per morphology seed
_BANDIT_ARMS: List[HybridBanditArm] = []


def _init_bandit_arms(morphologies: List[Morphology], context_dim: int, alpha: float = 1.0) -> None:
    """Create a bandit arm for each morphology (seed)."""
    global _BANDIT_ARMS
    _BANDIT_ARMS = [HybridBanditArm(context_dim, alpha) for _ in morphologies]


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def hybrid_select_action(
    signature: np.ndarray,
    morphologies: List[Morphology],
    alpha_balance: float = 0.5,
) -> int:
    """
    Choose an arm (seed) by combining bandit expected reward with Voronoi distance.

    Score_i = Ř_i  – α_balance * d̃_i
    where d̃_i = euclidean_distance(sig_2d, (ℓ_i, w_i)) * (1 + ρ(m_i))

    Returns the index of the selected morphology/arm.
    """
    if not _BANDIT_ARMS:
        _init_bandit_arms(morphologies, signature.shape[0])

    sig_2d = (float(signature[0]), float(signature[1])) if signature.size >= 2 else (0.0, 0.0)

    best_score = -math.inf
    best_idx = -1

    for idx, (arm, morph) in enumerate(zip(_BANDIT_ARMS, morphologies)):
        reward_est = arm.expected_reward(signature)
        dist = euclidean_distance(sig_2d, (morph.length, morph.width))
        scaled_dist = dist * (1.0 + recovery_priority(morph))
        score = reward_est - alpha_balance * scaled_dist
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def hybrid_update_arm(
    arm_index: int,
    reward: float,
    signature: np.ndarray,
) -> None:
    """Update the LinUCB statistics of the selected arm."""
    if not (0 <= arm_index < len(_BANDIT_ARMS)):
        raise IndexError("Invalid arm index for hybrid update.")
    _BANDIT_ARMS[arm_index].update(reward, signature)


def hybrid_assign_documents(
    doc_labels: Dict[str, List[int]],
    morphologies: List[Morphology],
    alpha_balance: float = 0.5,
) -> Dict[int, List[str]]:
    """
    For each document:
      1. Build its lead‑lag signature.
      2. Select a morphology/arm using the hybrid score.
      3. Emit a probabilistic label whose confidence is scaled by the arm's
         expected reward (sigmoid transformed).
      4. Update the arm with a synthetic reward (1 if majority label == 1 else 0).

    Returns a mapping from arm index to list of document IDs assigned to that Voronoi region.
    """
    # Ensure bandit arms are ready
    any_signature = next(iter(doc_labels.values()), [])
    dim = 2 * len(any_signature) if any_signature else 2
    _init_bandit_arms(morphologies, dim)

    assignments: Dict[int, List[str]] = {i: [] for i in range(len(morphologies))}

    for doc_id, labels in doc_labels.items():
        signature = lead_lag_transform(labels)
        arm_idx = hybrid_select_action(signature, morphologies, alpha_balance)

        # Compute hybrid confidence
        arm = _BANDIT_ARMS[arm_idx]
        reward_est = arm.expected_reward(signature)
        confidence = 1.0 / (1.0 + math.exp(-reward_est))  # sigmoid

        # Majority vote for label (simple baseline)
        majority_label = int(sum(labels) >= (len(labels) / 2))

        # Emit (could be stored/returned; here just for illustration)
        _ = ProbabilisticLabel(doc_id=doc_id, label=majority_label, confidence=confidence)

        # Synthetic reward for learning: correct if majority label == 1
        synthetic_reward = float(majority_label)
        hybrid_update_arm(arm_idx, synthetic_reward, signature)

        assignments[arm_idx].append(doc_id)

    return assignments


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic dataset
    morphologies = [
        Morphology(length=1.0, width=2.0, height=0.5),
        Morphology(length=3.0, width=1.5, height=0.8),
    ]

    # Each document gets a short binary label sequence
    doc_labels = {
        "doc_A": [1, 0, 1, 1],
        "doc_B": [0, 0, 1, 0],
        "doc_C": [1, 1, 1, 1],
        "doc_D": [0, 1, 0, 0],
    }

    assignments = hybrid_assign_documents(doc_labels, morphologies, alpha_balance=0.3)

    for arm_idx, docs in assignments.items():
        print(f"Arm {arm_idx} (Morphology {morphologies[arm_idx]}) -> Documents: {docs}")

    # Demonstrate that bandit parameters have been updated
    for i, arm in enumerate(_BANDIT_ARMS):
        print(f"\nArm {i} theta after updates: {arm.theta}")