# DARWIN HAMMER — match 3066, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s3.py (gen3)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_rlct_g_m989_s0.py (gen3)
# born: 2026-05-29T23:47:35Z

"""Hybrid algorithm combining:
- Parent A: Hybrid Decision Hygiene & Shannon Entropy with Ternary Lens Audit (hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s3.py)
- Parent B: Hybrid Krampus Brainmap Bandit Router & RLCT Grokking Pheromone Inference (hybrid_hybrid_krampus_brain_hybrid_hybrid_rlct_g_m989_s0.py)

Mathematical bridge:
1. Parent A produces an expected edge‑length vector **E** ∈ ℝⁿ (expected value of graph edge lengths).
2. Parent B maintains a set of bandit actions with propensities **p** ∈ ℝᵐ and a pheromone distribution **ϕ** over surface keys.
3. The hybrid maps **E** onto the bandit space by a linear projection **W** (size m×n) derived from feature‑count statistics of Parent A.
   The weighted propensities become p′ = softmax(W·E).
4. The Shannon entropy H(ϕ) of the pheromone distribution modulates the hygiene scores **h** from Parent A:
   h′ = h * (1 + H(ϕ)/log|ϕ|), providing an information‑theoretic adjustment.

The resulting system jointly updates hygiene scores, bandit propensities, and pheromone signals in a single probabilistic loop.
"""

import sys
import math
import random
import pathlib
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

# ------------------------------------------------------------
# Data structures (simplified extracts from both parents)
# ------------------------------------------------------------

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

class PheromoneSystem:
    """Mimics Parent B's pheromone handling."""
    def __init__(self):
        self.pheromone_signals: dict[str, dict[str, float]] = {}

    def calculate_pheromone_signal(self, surface_key: str, signal_kind: str,
                                   signal_value: float, half_life_seconds: float) -> float:
        """Return decayed pheromone value (no real time tracking, half‑life placeholder)."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        if signal_kind not in self.pheromone_signals[surface_key]:
            self.pheromone_signals[surface_key][signal_kind] = signal_value
        # elapsed_time is mocked as 0 for deterministic tests
        elapsed_time = 0.0
        decayed = self.pheromone_signals[surface_key][signal_kind] * \
                  math.pow(0.5, elapsed_time / half_life_seconds)
        return decayed

    def update_pheromone_signal(self, surface_key: str, signal_kind: str,
                                signal_value: float) -> None:
        """Directly set a pheromone value."""
        if surface_key not in self.pheromone_signals:
            self.pheromone_signals[surface_key] = {}
        self.pheromone_signals[surface_key][signal_kind] = signal_value

# ------------------------------------------------------------
# Core hybrid functions
# ------------------------------------------------------------

def expected_edge_lengths(edge_lengths: dict[tuple[str, str], list[float]]) -> np.ndarray:
    """
    Parent A computes the expected value of each edge length.
    Input: mapping edge -> list of sampled lengths.
    Output: 1‑D array of expectations ordered by sorted edge keys.
    """
    sorted_keys = sorted(edge_lengths.keys())
    expectations = [float(np.mean(edge_lengths[k])) for k in sorted_keys]
    return np.array(expectations, dtype=float)


def project_to_bandit_space(expectations: np.ndarray,
                            feature_counts: dict[str, int],
                            action_ids: list[str]) -> np.ndarray:
    """
    Linear projection from expectation space (size n) to bandit space (size m).
    The projection matrix W is built from feature_counts:
        W[i, j] = count(feature_i) / (j+1)
    Returns the raw (pre‑softmax) propensity vector.
    """
    n = expectations.shape[0]
    m = len(action_ids)
    # Build W
    W = np.zeros((m, n), dtype=float)
    feature_list = list(feature_counts.keys())
    for i, aid in enumerate(action_ids):
        for j in range(n):
            feat = feature_list[j % len(feature_list)]
            W[i, j] = feature_counts[feat] / (j + 1.0)
    raw = W @ expectations
    return raw


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def update_bandit_actions(raw_propensities: np.ndarray,
                          actions: list[BanditAction]) -> list[BanditAction]:
    """
    Convert raw propensities to probabilities via softmax and create
    updated BanditAction objects preserving other fields.
    """
    probs = softmax(raw_propensities)
    updated = []
    for act, p in zip(actions, probs):
        updated.append(BanditAction(
            action_id=act.action_id,
            propensity=p,
            expected_reward=act.expected_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm
        ))
    return updated


def pheromone_entropy(pheromone_system: PheromoneSystem) -> float:
    """
    Compute Shannon entropy of the flattened pheromone distribution.
    """
    values = []
    for surface in pheromone_system.pheromone_signals.values():
        values.extend(surface.values())
    if not values:
        return 0.0
    total = sum(values)
    probs = np.array([v / total for v in values if v > 0])
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def adjust_hygiene_scores(hygiene_scores: dict[str, float],
                          entropy: float) -> dict[str, float]:
    """
    Scale each hygiene score by (1 + H / log K) where K is number of scores.
    """
    K = max(len(hygiene_scores), 1)
    factor = 1.0 + entropy / math.log(K + 1e-9)
    return {k: v * factor for k, v in hygiene_scores.items()}


def hybrid_step(edge_lengths: dict[tuple[str, str], list[float]],
                feature_counts: dict[str, int],
                hygiene_scores: dict[str, float],
                bandit_actions: list[BanditAction],
                pheromone_system: PheromoneSystem) -> tuple[
                    dict[str, float],
                    list[BanditAction],
                    PheromoneSystem]:
    """
    Perform one hybrid iteration:
    1. Compute expected edge lengths (Parent A).
    2. Project to bandit propensities (bridge).
    3. Update bandit actions (Parent B).
    4. Compute pheromone entropy and adjust hygiene scores (bridge).
    5. Optionally decay pheromones (demonstration purpose).
    Returns updated hygiene_scores, bandit_actions, pheromone_system.
    """
    # 1. Expected edge lengths
    exp_lengths = expected_edge_lengths(edge_lengths)

    # 2. Linear projection to bandit space
    raw_props = project_to_bandit_space(exp_lengths, feature_counts,
                                        [a.action_id for a in bandit_actions])

    # 3. Update bandit actions
    updated_actions = update_bandit_actions(raw_props, bandit_actions)

    # 4. Entropy‑driven hygiene adjustment
    ent = pheromone_entropy(pheromone_system)
    adjusted_hygiene = adjust_hygiene_scores(hygiene_scores, ent)

    # 5. Simple pheromone decay (half‑life of 60 seconds, mocked elapsed=0)
    for sk in list(pheromone_system.pheromone_signals.keys()):
        for kind in list(pheromone_system.pheromone_signals[sk].keys()):
            val = pheromone_system.pheromone_signals[sk][kind]
            decayed = pheromone_system.calculate_pheromone_signal(
                sk, kind, val, half_life_seconds=60.0)
            pheromone_system.update_pheromone_signal(sk, kind, decayed)

    return adjusted_hygiene, updated_actions, pheromone_system


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic data for deterministic execution
    edge_lengths_example = {
        ("A", "B"): [1.0, 1.2, 0.9],
        ("B", "C"): [2.0, 2.1],
        ("C", "D"): [0.5, 0.6, 0.55]
    }

    feature_counts_example = {"evidence": 3, "plan": 2, "support": 5}

    hygiene_scores_example = {"evidence": 0.8, "plan": 0.6, "support": 0.9}

    bandit_actions_example = [
        BanditAction("act1", 0.0, 1.0, 0.1, "A"),
        BanditAction("act2", 0.0, 0.8, 0.2, "B"),
        BanditAction("act3", 0.0, 0.5, 0.3, "C")
    ]

    pheromone_system_example = PheromoneSystem()
    pheromone_system_example.update_pheromone_signal("node1", "signalA", 1.0)
    pheromone_system_example.update_pheromone_signal("node2", "signalB", 2.0)

    updated_hygiene, updated_actions, updated_pheromone = hybrid_step(
        edge_lengths=edge_lengths_example,
        feature_counts=feature_counts_example,
        hygiene_scores=hygiene_scores_example,
        bandit_actions=bandit_actions_example,
        pheromone_system=pheromone_system_example
    )

    print("Adjusted Hygiene Scores:")
    for k, v in updated_hygiene.items():
        print(f"  {k}: {v:.4f}")

    print("\nUpdated Bandit Actions (propensities):")
    for a in updated_actions:
        print(f"  {a.action_id}: {a.propensity:.4f}")

    print("\nPheromone Signals after decay:")
    for sk, kinds in updated_pheromone.pheromone_signals.items():
        for kind, val in kinds.items():
            print(f"  {sk}:{kind} = {val:.4f}")