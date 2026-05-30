# DARWIN HAMMER — match 4923, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_hybrid_hy_m1295_s0.py (gen6)
# born: 2026-05-29T23:58:49Z

"""
Hybrid Algorithm: Pheromone‑Bandit VRAM Scheduler + Path‑Signature Ternary Binding

Parents:
- hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (VRAM‑Bandit Scheduler with pheromone decay)
- hybrid_hybrid_hybrid_hybrid_hybrid_hdc_hybrid_hy_m1295_s0.py (Path‑Signature + NLMS‑Graph‑Tree Fusion + Ternary Vector Binding)

Mathematical Bridge:
The pheromone level distribution 𝛑(t) is treated as a probability vector over actions.
Its Kullback‑Leibler divergence KL(𝛑‖𝜈) (where 𝜈 is a uniform prior) is added to the classic Upper‑Confidence‑Bound
UCB_i = μ_i + α·√(ln T / n_i) of the multi‑armed bandit.  The resulting hybrid metric
M_i = UCB_i + λ·KL(𝛑‖𝜈) is used as a weighting factor for allocating VRAM.

Simultaneously, each candidate allocation is evaluated by mapping a multivariate path
to its level‑1 and level‑2 signatures (S₁, S₂).  A ternary command vector τ (∈{‑1,0,1}^D) is projected
onto the signature space via a random binding matrix B∈ℝ^{D×d}.  The bound cost
C_i = τ·B·S₁ + τ·B·diag(S₂) provides a data‑driven reward estimate that feeds μ_i
in the bandit update.  Thus the pheromone‑bandit decision loop is tightly coupled with
the signature‑based cost estimation, yielding a unified hybrid allocation system.
"""

import math
import random
import sys
import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Constants & Global Helpers
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path(".")
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768
DEFAULT_BASE_MODEL_MB = 1800
DEFAULT_ADAPTER_MB = 128
DEFAULT_EMBEDDING_MB = 384

TERNARY_DIMS = 12          # Dimension of ternary command vector
PROJECTION_SEED = 42       # Seed for reproducible binding matrix
UCB_CONFIDENCE = 2.0       # Exploration coefficient α
KL_WEIGHT = 0.5            # λ in the hybrid metric
EPS = 1e-12                # Numerical stability


# ----------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # Current selection probability
    expected_reward: float     # μ_i
    confidence_bound: float    # √(ln T / n_i)
    count: int                 # n_i
    algorithm: str = "HybridBandit"


# ----------------------------------------------------------------------
# Core Mathematical Primitives
# ----------------------------------------------------------------------
def pheromone_decay(levels: np.ndarray, decay_rate: float, dt: float) -> np.ndarray:
    """Exponential decay of pheromone levels."""
    decayed = levels * np.exp(-decay_rate * dt)
    # Renormalize to a probability distribution
    total = decayed.sum()
    return decayed / (total + EPS)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q) for discrete distributions."""
    p_safe = np.clip(p, EPS, 1.0)
    q_safe = np.clip(q, EPS, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe))


def ucb_metric(action: BanditAction, total_steps: int) -> float:
    """Classic Upper‑Confidence‑Bound term."""
    if action.count == 0:
        # Encourage exploration of never‑tried arms
        return float("inf")
    return action.expected_reward + UCB_CONFIDENCE * math.sqrt(math.log(total_steps + 1) / (action.count + EPS))


def hybrid_metric(action: BanditAction, pheromone: np.ndarray, total_steps: int) -> float:
    """Hybrid metric M_i = UCB_i + λ·KL(π‖uniform)."""
    uniform = np.full_like(pheromone, 1.0 / len(pheromone))
    kl = kl_divergence(pheromone, uniform)
    return ucb_metric(action, total_steps) + KL_WEIGHT * kl


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag augmentation of a multivariate path."""
    n, d = path.shape
    augmented = np.zeros((n, d + 1))
    augmented[:, :d] = path
    # Cumulative Euclidean distance as the extra dimension
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    augmented[1:, d] = np.cumsum(diffs)
    return augmented


def compute_signatures(augmented_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Level‑1 (sum) and Level‑2 (pairwise outer) signatures."""
    level1 = np.sum(augmented_path, axis=0)
    n = len(augmented_path)
    d = augmented_path.shape[1]
    level2 = np.zeros((d, d))
    for i in range(n - 1):
        level2 += np.outer(augmented_path[i], augmented_path[i + 1])
    return level1, level2


def ternary_vector(raw_command: str, normalized_intent: str, context: str) -> np.ndarray:
    """Deterministic ternary vector from a command envelope."""
    payload = f"{raw_command}|{normalized_intent}|{context}"
    # Simple hash‑to‑int conversion
    h = abs(hash(payload))
    rng = np.random.default_rng(h % (2**32))
    vec = rng.integers(-1, 2, size=TERNARY_DIMS)  # values in {-1,0,1}
    return vec.astype(float)


def binding_matrix(input_dim: int, output_dim: int) -> np.ndarray:
    """Random binding matrix B ∈ ℝ^{input_dim × output_dim}."""
    rng = np.random.default_rng(PROJECTION_SEED)
    return rng.normal(loc=0.0, scale=1.0, size=(input_dim, output_dim))


def signature_cost(ternary_vec: np.ndarray,
                   level1: np.ndarray,
                   level2: np.ndarray,
                   B: np.ndarray) -> float:
    """
    Bind ternary vector to signatures:
    C = τ·B·S₁ + τ·B·diag(S₂)
    """
    bound1 = ternary_vec @ B @ level1
    diag_s2 = np.diag(level2)
    bound2 = ternary_vec @ B @ diag_s2
    return float(bound1 + bound2)


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def update_bandit(actions: List[BanditAction],
                  rewards: List[float]) -> List[BanditAction]:
    """Update expected rewards and confidence bounds for each arm."""
    total_steps = sum(a.count for a in actions) + len(actions)  # include current round
    updated = []
    for act, r in zip(actions, rewards):
        new_count = act.count + 1
        new_mu = ((act.expected_reward * act.count) + r) / new_count
        new_conf = UCB_CONFIDENCE * math.sqrt(math.log(total_steps + 1) / new_count)
        updated.append(BanditAction(
            action_id=act.action_id,
            propensity=act.propensity,  # unchanged here; recomputed later
            expected_reward=new_mu,
            confidence_bound=new_conf,
            count=new_count,
            algorithm=act.algorithm
        ))
    return updated


def select_action(actions: List[BanditAction],
                  pheromone: np.ndarray,
                  total_steps: int) -> BanditAction:
    """Select the arm with the highest hybrid metric."""
    metrics = [hybrid_metric(a, pheromone, total_steps) for a in actions]
    idx = int(np.argmax(metrics))
    return actions[idx]


def allocate_vram(vram_budget: int,
                  bandit_actions: List[BanditAction],
                  path: np.ndarray,
                  raw_command: str,
                  normalized_intent: str,
                  context: str,
                  pheromone_levels: np.ndarray,
                  decay_rate: float = 0.1,
                  dt: float = 1.0) -> VramSlotPlan:
    """
    Unified allocation routine:
    1. Decay pheromones.
    2. Compute path signatures.
    3. Generate ternary command vector.
    4. Bind signatures → cost → reward estimate.
    5. Update bandit with reward.
    6. Choose action via hybrid metric.
    7. Return a VramSlotPlan reflecting the chosen action.
    """
    # 1. Pheromone dynamics
    pheromone = pheromone_decay(pheromone_levels, decay_rate, dt)

    # 2. Path signatures
    aug_path = lead_lag_transform(path)
    lvl1, lvl2 = compute_signatures(aug_path)

    # 3. Ternary vector
    tau = ternary_vector(raw_command, normalized_intent, context)

    # 4. Binding cost → reward (higher cost => lower reward)
    B = binding_matrix(TERNARY_DIMS, lvl1.shape[0])
    cost = signature_cost(tau, lvl1, lvl2, B)
    reward_estimate = max(0.0, 1.0 - (cost / (np.linalg.norm(tau) + EPS)))  # normalized to [0,1]

    # 5. Bandit update
    bandit_actions = update_bandit(bandit_actions, [reward_estimate])

    # 6. Selection using hybrid metric
    total_steps = sum(a.count for a in bandit_actions)
    chosen = select_action(bandit_actions, pheromone, total_steps)

    # 7. Derive VRAM allocation (simple proportional split)
    proportion = chosen.propensity if chosen.propensity > 0 else 1.0 / len(bandit_actions)
    allocated_mb = int(proportion * vram_budget)

    plan = VramSlotPlan(
        artifact_id=chosen.action_id,
        artifact_kind="model_component",
        action="allocate_vram",
        estimated_mb=min(allocated_mb, vram_budget),
        reason="Hybrid pheromone‑bandit + signature cost decision",
        detail={
            "reward_estimate": reward_estimate,
            "cost": cost,
            "pheromone_level": float(pheromone[np.where(np.array([a.action_id for a in bandit_actions]) == chosen.action_id)[0][0]]),
            "signature_level1_norm": float(np.linalg.norm(lvl1)),
            "signature_level2_norm": float(np.linalg.norm(lvl2))
        }
    )
    return plan


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy VRAM budget
    budget = DEFAULT_BUDGET_MB - DEFAULT_RESERVE_MB

    # Initialise three dummy bandit actions
    actions = [
        BanditAction(action_id="model_A", propensity=0.4, expected_reward=0.2, confidence_bound=0.0, count=5),
        BanditAction(action_id="model_B", propensity=0.35, expected_reward=0.3, confidence_bound=0.0, count=3),
        BanditAction(action_id="model_C", propensity=0.25, expected_reward=0.1, confidence_bound=0.0, count=2)
    ]

    # Random multivariate path (5 timesteps, 3 dimensions)
    rng = np.random.default_rng(0)
    sample_path = rng.normal(size=(5, 3))

    # Command envelope
    raw_cmd = "generate_image"
    intent = "high_quality"
    ctx = "user_session_42"

    # Initial pheromone levels (uniform)
    pheromone_vec = np.array([1.0 / len(actions)] * len(actions))

    plan = allocate_vram(
        vram_budget=budget,
        bandit_actions=actions,
        path=sample_path,
        raw_command=raw_cmd,
        normalized_intent=intent,
        context=ctx,
        pheromone_levels=pheromone_vec,
        decay_rate=0.05,
        dt=1.0
    )

    print("Allocation Plan:")
    print(json.dumps(plan.as_dict(), indent=2))