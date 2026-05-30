# DARWIN HAMMER — match 2706, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_cockpit_metri_m287_s4.py (gen3)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s1.py (gen5)
# born: 2026-05-29T23:43:47Z

"""Hybrid Bandit‑Router + Gini‑RBF‑Tropical System
Parents:
* hybrid_hybrid_bandit_router_hybrid_cockpit_metri_m287_s4.py (temperature‑aware bandit with honesty‑weighted pheromones)
* hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s1.py (Gini‑guided Hoeffding splits with RBF similarity and tropical max‑plus algebra)

Mathematical bridge:
Both parents expose a *scalar gain* that modulates a base quantity.
- Parent A provides `G(T,H) = A(T)·H` (temperature activity × honesty weight) which scales the
  context norm used in the UCB exploration term and the pheromone decay factor.
- Parent B supplies a *structural gain* `S = max⁺(log P)` where `log P` is the matrix of
  RBF‑derived log‑probabilities between feature vectors; the tropical max‑plus
  operator (`max⁺`) yields a vector that can weight the Gini‑based decision‑hygiene
  score.

The hybrid algorithm multiplies the two gains, `Γ = G(T,H)·S_i`, producing an
arm‑specific amplification factor that simultaneously respects environmental
temperature, source honesty, data‑distribution inequality (Gini) and relational
similarity (RBF‑tropical). The resulting `Γ` modulates both the UCB exploration
bonus and the pheromone decay, yielding a unified exploration‑exploitation
mechanism."""
from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence, Hashable, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------


def temperature_activity(T: float, A0: float = 1.0, E: float = 0.65, T_opt: float = 25.0) -> float:
    """
    Schoolfield‐type temperature activity gate.
    Returns A(T) ∈ [0,1] where the activity peaks at T_opt.
    """
    if T <= -273.15:
        return 0.0
    # Simple Arrhenius‑like formulation with optimal temperature scaling
    kelvin = T + 273.15
    opt_kelvin = T_opt + 273.15
    activity = A0 * math.exp(-E / kelvin) * (kelvin / opt_kelvin)
    return max(0.0, min(1.0, activity))


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Honesty weight H ∈ [0,1] based on evidence‑coverage.
    """
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (base‑e)."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return -float(np.sum(p * np.log(p)))


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient for a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def rbf_similarity(a: np.ndarray, b: np.ndarray, epsilon: float = 1.0) -> float:
    """Radial Basis Function similarity between two feature vectors."""
    diff = a - b
    r = np.linalg.norm(diff)
    return math.exp(-((epsilon * r) ** 2))


def similarity_matrix(features: Dict[Hashable, Sequence[float]]) -> Tuple[np.ndarray, List[Hashable]]:
    """
    Build a symmetric matrix of RBF similarities for a dict {node: feature_vec}.
    Returns (matrix, ordered_node_list).
    """
    nodes = list(features.keys())
    n = len(nodes)
    mat = np.zeros((n, n), dtype=float)
    vecs = [np.asarray(features[node], dtype=float) for node in nodes]
    for i in range(n):
        for j in range(i, n):
            sim = rbf_similarity(vecs[i], vecs[j])
            mat[i, j] = mat[j, i] = sim
    return mat, nodes


def tropical_max_plus(log_matrix: np.ndarray) -> np.ndarray:
    """
    Tropical max‑plus multiplication of a log‑probability matrix with a
    vector of ones (log‑1 = 0). The result is the row‑wise max of the matrix.
    """
    # In max‑plus algebra, addition becomes max and multiplication becomes +.
    # Multiplying by a vector of zeros (log 1) therefore yields max across rows.
    return np.max(log_matrix, axis=1)


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass
class ArmStats:
    count: int = 0
    total_reward: float = 0.0
    pheromone: float = 1.0  # start with unit pheromone


@dataclass
class HybridBandit:
    arms: Dict[Hashable, ArmStats] = field(default_factory=dict)
    temperature: float = 25.0
    claims_with_evidence: int = 0
    total_claims_emitted: int = 1
    recent_rewards: List[float] = field(default_factory=list)  # for Gini
    features: Dict[Hashable, Sequence[float]] = field(default_factory=dict)  # node → feature vec

    # hyper‑parameters
    exploration_coef: float = 2.0
    pheromone_decay: float = 0.9
    entropy_reg: float = 0.01
    lambda_hygiene: float = 0.5

    def _global_counts(self) -> int:
        return sum(a.count for a in self.arms.values()) + 1  # avoid zero

    def _context_norm(self, context: Sequence[float]) -> float:
        return math.sqrt(sum(x * x for x in context))

    # ------------------------------------------------------------------
    # Hybrid operations
    # ------------------------------------------------------------------

    def hybrid_gain(self) -> float:
        """Scalar gain G = A(T) * H."""
        A = temperature_activity(self.temperature)
        H = anti_slop_ratio(self.claims_with_evidence, self.total_claims_emitted)
        return A * H

    def structural_gain_vector(self) -> np.ndarray:
        """
        Compute S_i = tropical_max_plus(log P) for each arm i.
        P is the RBF similarity matrix built from arm feature vectors.
        """
        if not self.features:
            # fallback to ones
            return np.ones(len(self.arms))
        sim_mat, order = similarity_matrix(self.features)
        # Convert similarities to log‑probabilities (add a small epsilon to avoid log(0))
        eps = 1e-12
        log_prob = np.log(sim_mat + eps)
        s_vec = tropical_max_plus(log_prob)
        # Align with arm ordering
        arm_list = list(self.arms.keys())
        # If ordering differs, map accordingly
        mapping = {node: idx for idx, node in enumerate(order)}
        result = np.ones(len(arm_list))
        for i, arm in enumerate(arm_list):
            if arm in mapping:
                result[i] = s_vec[mapping[arm]]
        return result

    def hybrid_select_action(self, context: Sequence[float]) -> Hashable:
        """
        Select an arm using a temperature‑ and honesty‑aware UCB term that is
        further weighted by the structural gain (tropical max‑plus) and a
        Gini‑based hygiene bonus.
        """
        G = self.hybrid_gain()
        ctx_norm = self._context_norm(context)
        total_counts = self._global_counts()
        arm_list = list(self.arms.keys())
        structural_gain = self.structural_gain_vector()

        # Compute Gini hygiene score from recent rewards
        hygiene_score = gini_coefficient(self.recent_rewards) if self.recent_rewards else 0.0

        best_score = -float('inf')
        chosen = None

        for idx, arm in enumerate(arm_list):
            stats = self.arms[arm]
            avg_reward = stats.total_reward / stats.count if stats.count > 0 else 0.0
            # Classic UCB exploration term
            exploration = (
                math.sqrt(
                    (self.exploration_coef * math.log(total_counts)) / (stats.count + 1e-9)
                )
                * G
                * ctx_norm
            )
            # Hybrid score = avg + exploration + λ·hygiene·structural_gain
            hybrid_score = (
                avg_reward
                + exploration
                + self.lambda_hygiene * hygiene_score * structural_gain[idx]
            )
            if hybrid_score > best_score:
                best_score = hybrid_score
                chosen = arm

        return chosen

    def hybrid_update_policy(
        self,
        arm: Hashable,
        reward: float,
        context: Sequence[float],
    ) -> None:
        """
        Update reward statistics, decay pheromone with the scalar gain G,
        and apply entropy regularisation on the pheromone distribution.
        """
        # ----- reward stats -----
        stats = self.arms.setdefault(arm, ArmStats())
        stats.count += 1
        stats.total_reward += reward

        # ----- recent rewards for Gini -----
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

        # ----- pheromone update -----
        G = self.hybrid_gain()
        decay = self.pheromone_decay ** G  # stronger decay when G is large
        stats.pheromone *= decay

        # Normalise pheromones across all arms
        total_pher = sum(a.pheromone for a in self.arms.values())
        if total_pher > 0:
            for a in self.arms.values():
                a.pheromone /= total_pher

        # ----- entropy regularisation -----
        pher_vec = np.array([a.pheromone for a in self.arms.values()], dtype=float)
        ent = entropy(pher_vec)
        # Push pheromones towards higher entropy (more exploration) by a tiny amount
        for a in self.arms.values():
            a.pheromone += self.entropy_reg * (ent / len(self.arms))

        # Re‑normalise after regularisation
        total_pher = sum(a.pheromone for a in self.arms.values())
        for a in self.arms.values():
            a.pheromone /= total_pher

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def add_arm(self, arm: Hashable, feature_vec: Sequence[float] | None = None) -> None:
        """Register a new arm and optionally its feature vector."""
        self.arms.setdefault(arm, ArmStats())
        if feature_vec is not None:
            self.features[arm] = feature_vec


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a hybrid bandit with three arms
    hb = HybridBandit(temperature=30.0, claims_with_evidence=8, total_claims_emitted=10)

    # Dummy 2‑D feature vectors for tropical similarity
    hb.add_arm("A", feature_vec=[0.1, 0.9])
    hb.add_arm("B", feature_vec=[0.5, 0.5])
    hb.add_arm("C", feature_vec=[0.9, 0.1])

    # Simulated context vector
    ctx = [0.3, 0.7]

    # Run a few selection‑update cycles
    for step in range(20):
        chosen = hb.hybrid_select_action(ctx)
        # Simulate a stochastic reward (higher for arm B)
        reward = random.gauss(mu=1.0 if chosen == "B" else 0.5, sigma=0.1)
        hb.hybrid_update_policy(chosen, reward, ctx)

    # Print final arm statistics
    for arm, stats in hb.arms.items():
        print(
            f"Arm {arm}: count={stats.count}, avg_reward={stats.total_reward / max(1, stats.count):.3f}, "
            f"pheromone={stats.pheromone:.3f}"
        )