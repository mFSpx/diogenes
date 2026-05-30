# DARWIN HAMMER — match 129, survivor 2
# gen: 1
# parent_a: krampus_brainmap.py (gen0)
# parent_b: bandit_router.py (gen0)
# born: 2026-05-29T23:27:00Z

"""Hybrid Krampus-Bandit Module

This module fuses the *Krampus brain‑map* feature extraction (Parent A) with the
*contextual bandit* routing logic (Parent B).  The mathematical bridge is the
feature dictionary produced by ``extract_master_vector`` which is treated as a
real‑valued context vector **x ∈ ℝᵈ**.  For each possible action we maintain a
linear model

    θ_a = A_a⁻¹ b_a

with A_a ∈ ℝ^{d×d} (regularised Gram matrix) and b_a ∈ ℝ^{d}.  The LinUCB
upper‑confidence bound for action *a* is

    UCB_a(x) = θ_a·x + α·√(xᵀ A_a⁻¹ x)

where α > 0 controls exploration.  The hybrid selector therefore combines the
feature extraction pipeline of Krampus with the LinUCB (or Thompson / ε‑greedy)
decision rule of the bandit router, yielding a single unified system.

Only the Python standard library and NumPy are used.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – feature extraction (simplified stub for demonstration)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Placeholder feature extractor – in practice this calls the real
    krampus_stickers functions.  Here we synthesize deterministic pseudo‑features.
    """
    if not text.strip():
        return {}
    # deterministic pseudo‑features based on hash of words
    words = text.split()
    base = sum(hash(w) for w in words) % 1000
    return {
        "operator_visceral_ratio": (base % 10) / 10.0,
        "operator_tech_ratio": ((base // 10) % 10) / 10.0,
        "operator_legal_osint_ratio": ((base // 100) % 10) / 10.0,
        "operator_ledger_density": ((base // 1000) % 10) / 10.0,
        "operator_recursion_score": ((base // 2) % 5) / 5.0,
        "operator_directive_ratio": ((base // 3) % 7) / 7.0,
        "operator_target_density": ((base // 5) % 9) / 9.0,
        "psyche_forensic_shield_ratio": ((base // 7) % 8) / 8.0,
        "psyche_poetic_entropy": ((base // 11) % 6) / 6.0,
        "psyche_dissociative_index": ((base // 13) % 4) / 4.0,
        "psyche_wrath_velocity": ((base // 17) % 3) / 3.0,
        "resilience_bureaucratic_weaponization_index": ((base // 19) % 5) / 5.0,
        "resilience_resource_exhaustion_metric": ((base // 23) % 6) / 6.0,
        "resilience_swarm_orchestration_density": ((base // 29) % 7) / 7.0,
        "resilience_logic_crucifixion_index": ((base // 31) % 8) / 8.0,
        "resilience_conspiracy_grounding_ratio": ((base // 37) % 9) / 9.0,
        "resilience_chaotic_good_tax": ((base // 41) % 10) / 10.0,
        "rainmaker_corporate_grit_tension": ((base // 43) % 5) / 5.0,
        "rainmaker_countdown_density": ((base // 47) % 4) / 4.0,
        "rainmaker_asset_structuring_weight": ((base // 53) % 3) / 3.0,
        "rainmaker_pitch_formatting_ratio": ((base // 59) % 2) / 2.0,
    }

def extract_master_vector(text: str) -> Dict[str, float]:
    """Human‑readable 20+‑dimensional vector for downstream models."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get("resilience_logic_crucifixion_index", 0.0),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
    }

# ----------------------------------------------------------------------
# Parent B – bandit data structures
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

# ----------------------------------------------------------------------
# Hybrid internal state
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # simple reward stats: [total, count]
_A_MATRICES: Dict[str, np.ndarray] = {}       # per‑action A = I + Σ xxᵀ
_B_VECTORS: Dict[str, np.ndarray] = {}        # per‑action b = Σ r·x
_DIM: int = len(extract_master_vector("dummy"))  # dimensionality of context

def reset_hybrid_state() -> None:
    """Clear all stored statistics."""
    _POLICY.clear()
    _A_MATRICES.clear()
    _B_VECTORS.clear()

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def _initialize_action_if_needed(action: str) -> None:
    """Lazy initialisation of linear‑model structures for a new action."""
    if action not in _A_MATRICES:
        _A_MATRICES[action] = np.identity(_DIM, dtype=float)
        _B_VECTORS[action] = np.zeros(_DIM, dtype=float)

def _vector_from_context(context: Dict[str, float]) -> np.ndarray:
    """Convert a dict context to a fixed‑order NumPy column vector."""
    # deterministic ordering based on sorted keys ensures reproducibility
    ordered_keys = sorted(context.keys())
    vec = np.array([float(context[k]) for k in ordered_keys], dtype=float)
    if vec.size != _DIM:
        # pad or truncate to match expected dimension
        if vec.size < _DIM:
            vec = np.pad(vec, (0, _DIM - vec.size))
        else:
            vec = vec[:_DIM]
    return vec.reshape(-1, 1)  # column vector

def select_action_hybrid(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    alpha: float = 1.0,
    seed: int | str | None = 7,
) -> BanditAction:
    """Select an action using the fused Krampus‑Bandit logic.

    - ``context`` is the feature dict from ``extract_master_vector``.
    - ``algorithm`` can be ``linucb``, ``thompson`` or ``epsilon_greedy``.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Simple reward estimate used by non‑LinUCB algorithms
    def simple_reward(a: str) -> float:
        total, cnt = _POLICY.get(a, [0.0, 0.0])
        return total / cnt if cnt else 0.0

    # LinUCB score
    x = _vector_from_context(context)  # (d,1)
    def linucb_score(a: str) -> float:
        _initialize_action_if_needed(a)
        A_inv = np.linalg.inv(_A_MATRICES[a])
        theta = A_inv @ _B_VECTORS[a].reshape(-1, 1)  # (d,1)
        exploit = float(theta.T @ x)
        explore = alpha * math.sqrt(float(x.T @ A_inv @ x))
        return exploit + explore

    # Thompson sampling score (Beta approximation on linear reward)
    def thompson_score(a: str) -> float:
        _initialize_action_if_needed(a)
        A_inv = np.linalg.inv(_A_MATRICES[a])
        theta = A_inv @ _B_VECTORS[a].reshape(-1, 1)
        # Sample from normal approximation N(theta, sigma^2 A^{-1})
        sigma = math.sqrt(alpha)  # scaling factor for variance
        sample = float(theta.T @ x + sigma * rng.gauss(0, 1) * math.sqrt(float(x.T @ A_inv @ x)))
        return sample

    # Decision logic
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
        score = simple_reward(chosen)
    elif algorithm == "thompson":
        chosen = max(actions, key=thompson_score)
        score = simple_reward(chosen)
    else:  # default to LinUCB
        chosen = max(actions, key=linucb_score)
        score = simple_reward(chosen)

    propensity = 1.0 / len(actions)
    confidence = alpha / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=score,
        confidence_bound=confidence,
        algorithm=algorithm,
    )

def update_hybrid_policy(updates: List[BanditUpdate], context_lookup: Dict[str, Dict[str, float]]) -> None:
    """Update both the simple reward statistics and the linear models.

    ``context_lookup`` maps ``context_id`` to the original feature dict that
    produced the context vector for that interaction.
    """
    for u in updates:
        # Simple reward aggregation (Parent B)
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

        # Linear model update (LinUCB core)
        ctx = context_lookup.get(u.context_id, {})
        if not ctx:
            continue  # cannot update linear model without features
        x = _vector_from_context(ctx)  # (d,1)
        _initialize_action_if_needed(u.action_id)
        A = _A_MATRICES[u.action_id]
        b = _B_VECTORS[u.action_id]

        A += x @ x.T  # rank‑1 update
        b += float(u.reward) * x.ravel()

        _A_MATRICES[u.action_id] = A
        _B_VECTORS[u.action_id] = b

def get_action_estimates(context: Dict[str, float], actions: List[str]) -> Dict[str, float]:
    """Return the current LinUCB estimate (without exploration term) for each action."""
    x = _vector_from_context(context)
    estimates = {}
    for a in actions:
        _initialize_action_if_needed(a)
        A_inv = np.linalg.inv(_A_MATRICES[a])
        theta = A_inv @ _B_VECTORS[a].reshape(-1, 1)
        estimates[a] = float(theta.T @ x)
    return estimates

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text input → context vector
    sample_text = "The quick brown fox jumps over the lazy dog."
    context_vec = extract_master_vector(sample_text)

    # Define a toy action set
    actions = ["alpha", "bravo", "charlie"]

    # Reset any prior state
    reset_hybrid_state()

    # Choose an action using the hybrid selector
    chosen = select_action_hybrid(context_vec, actions, algorithm="linucb", alpha=0.5)
    print("Chosen action:", chosen)

    # Simulate a reward (e.g., 1 for success, 0 for failure)
    reward = 1.0 if chosen.action_id == "alpha" else 0.0

    # Update the policy with the observed reward
    upd = BanditUpdate(
        context_id="run1",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    update_hybrid_policy([upd], {"run1": context_vec})

    # Show updated estimates
    est = get_action_estimates(context_vec, actions)
    print("Post‑update LinUCB estimates:", est)

    sys.exit(0)