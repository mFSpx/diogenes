# DARWIN HAMMER — match 1831, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_ternary_lens__m953_s1.py (gen3)
# parent_b: hybrid_cockpit_metrics_hybrid_hybrid_ternar_m229_s2.py (gen3)
# born: 2026-05-29T23:39:17Z

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Global bandit policy store (Parent A)
_POLICY: Dict[str, List[float]] = {}  # action_id -> [total_reward, count]

def reset_policy() -> None:
    """Clear the bandit policy."""
    _POLICY.clear()

def _average_reward(action_id: str) -> float:
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return total / cnt if cnt else 0.0

def _action_count(action_id: str) -> int:
    return int(_POLICY.get(action_id, [0.0, 0.0])[1])

def update_policy(updates: List[Dict[str, Any]]) -> None:
    """Batch update of the bandit policy."""
    for u in updates:
        stats = _POLICY.setdefault(u["action_id"], [0.0, 0.0])
        stats[0] += float(u["reward"])
        stats[1] += 1.0

def select_action(
    actions: List[str],
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> str:
    """
    ε‑greedy selection over the given actions.
    With probability ε a random action is chosen (exploration);
    otherwise the action with highest average reward is chosen (exploitation).
    """
    rng = rng or random.Random()
    if rng.random() < epsilon:
        return rng.choice(actions)
    # exploitation – pick best known action
    best_action = max(actions, key=_average_reward)
    return best_action

# ----------------------------------------------------------------------
# Audit quality metrics (Parent B)

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of evidenced claims, clipped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed‑ok items, clipped to [0,1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    """Simple linear debt metric."""
    return float(max(0, exports_missing_audit_step))

def compute_quality_factor(audit_context: Dict[str, int]) -> float:
    """
    Composite quality factor Q used throughout the hybrid algorithm.
    Q ∈ (0, 1] – never zero to keep gradients alive.
    """
    q = (
        anti_slop_ratio(
            audit_context.get("claims_with_evidence", 0),
            audit_context.get("total_claims_emitted", 0),
        )
        * cockpit_honesty(
            audit_context.get("displayed_ok", 0),
            audit_context.get("unknown_displayed_as_ok", 0),
        )
        / (1.0 + audit_debt(audit_context.get("exports_missing_audit_step", 0)))
    )
    # Guard against pathological zero; enforce a tiny floor.
    return max(1e-6, q)

# ----------------------------------------------------------------------
# Vector utilities (Parent B)

Vector = Sequence[float]

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity scaled to [0,1]. Returns 0 for zero vectors."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = float(np.dot(a_arr, b_arr))
    sim = dot / (norm_a * norm_b)
    # Clip to [0,1] – negative similarities are treated as 0 (orthogonal)
    return max(0.0, min(1.0, sim))

def social_interaction(
    x: Vector,
    personal_best: Vector,
    g_best: Vector,
    inertia: float,
    c1: float,
    c2: float,
    rng: random.Random,
) -> List[float]:
    """
    PSO velocity update with inertia and acceleration coefficients.
    new_velocity = inertia * v + c1 * r1 * (personal_best - x) + c2 * r2 * (g_best - x)
    The incoming velocity v is assumed to be zero for simplicity (stateless update).
    """
    r1 = rng.random()
    r2 = rng.random()
    x_arr = np.asarray(x, dtype=float)
    p_arr = np.asarray(personal_best, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)

    cognitive = c1 * r1 * (p_arr - x_arr)
    social = c2 * r2 * (g_arr - x_arr)
    vel = inertia * np.zeros_like(x_arr) + cognitive + social
    return vel.tolist()

# ----------------------------------------------------------------------
# Hybrid components

def compute_pruning_probability(
    iteration: int,
    max_iterations: int,
    base_rate: float,
    quality_factor: float,
    min_rate: float = 0.01,
) -> float:
    """
    Exponential decay schedule modulated by the audit‑derived quality factor.
    Guarantees a non‑zero floor (min_rate) to keep occasional exploration.
    """
    if max_iterations <= 0:
        return min_rate
    decay = math.exp(-5.0 * iteration / max_iterations)  # fast early decay
    prob = base_rate * decay * quality_factor
    prob = max(min_rate, min(1.0, prob))
    return prob

def evaluate_candidate(
    candidate: Vector,
    reference: Vector,
    audit_context: Dict[str, int],
) -> float:
    """
    Reward = cosine_similarity(candidate, reference) * Q
    """
    similarity = cosine_similarity(candidate, reference)
    q = compute_quality_factor(audit_context)
    return similarity * q

def _random_vector(
    dim: int,
    low: float = -1.0,
    high: float = 1.0,
    rng: random.Random | None = None,
) -> List[float]:
    rng = rng or random.Random()
    return [rng.uniform(low, high) for _ in range(dim)]

def hybrid_step(
    particles: List[Vector],
    personal_bests: List[Vector],
    g_best: Vector,
    iteration: int,
    max_iterations: int,
    base_prune_rate: float,
    audit_context: Dict[str, int],
    reference: Vector,
    actions: List[str],
    action_hyperparams: Dict[str, Tuple[float, float]],
    bounds: Tuple[float, float] = (-1.0, 1.0),
    epsilon: float = 0.1,
    seed: int | str | None = None,
) -> Tuple[List[Vector], List[Vector], List[float], Dict[str, Any]]:
    """
    One hybrid iteration with deeper integration:
    1. Compute quality factor Q.
    2. Derive inertia weight w from Q and iteration (higher Q → higher inertia).
    3. For each particle:
       a. Select a bandit action (defines c1, c2).
       b. Update velocity via PSO formula, scaled by w.
       c. Apply pruning probability; pruned particles are re‑initialized.
       d. Evaluate reward, update personal best, and record bandit feedback.
    4. Update global best.
    5. Return updated structures and a summary dict.
    """
    rng = random.Random(seed)

    # 1. Quality factor
    Q = compute_quality_factor(audit_context)

    # 2. Inertia weight – linearly decreasing, modulated by Q
    w_max, w_min = 0.9, 0.4
    w = w_max - (w_max - w_min) * (iteration / max_iterations)
    w = w * Q  # higher quality → more momentum

    # 3. Pruning probability
    prune_p = compute_pruning_probability(
        iteration, max_iterations, base_prune_rate, Q
    )

    policy_updates: List[Dict[str, Any]] = []
    new_particles: List[Vector] = []
    new_personal_bests: List[Vector] = []

    # Helper for bounds
    low, high = bounds
    dim = len(reference)

    for idx, (x, p_best) in enumerate(zip(particles, personal_bests)):
        # 3a. Bandit action selection
        action_id = select_action(actions, epsilon=epsilon, rng=rng)
        c1, c2 = action_hyperparams.get(action_id, (1.5, 1.5))

        # 3b. Velocity / position update
        vel = social_interaction(
            x,
            p_best,
            g_best,
            inertia=w,
            c1=c1,
            c2=c2,
            rng=rng,
        )
        x_new_arr = np.asarray(x, dtype=float) + np.asarray(vel, dtype=float)

        # Enforce bounds
        x_new_arr = np.clip(x_new_arr, low, high)

        # 3c. Pruning decision
        if rng.random() < prune_p:
            # Re‑initialize instead of collapsing to zero to retain diversity
            x_new_arr = np.asarray(_random_vector(dim, low, high, rng), dtype=float)

        x_new = x_new_arr.tolist()

        # 3d. Reward evaluation
        reward = evaluate_candidate(
            candidate=x_new,
            reference=reference,
            audit_context=audit_context,
        )

        # Update personal best if improvement
        p_best_reward = evaluate_candidate(
            candidate=p_best,
            reference=reference,
            audit_context=audit_context,
        )
        if reward > p_best_reward:
            new_personal_bests.append(x_new)
        else:
            new_personal_bests.append(p_best)

        new_particles.append(x_new)

        # Record bandit feedback
        policy_updates.append({"action_id": action_id, "reward": reward})

    # 4. Global best update
    rewards = [
        evaluate_candidate(c, reference, audit_context) for c in new_particles
    ]
    best_idx = int(np.argmax(rewards))
    new_g_best = new_particles[best_idx]

    # 5. Apply bandit updates
    update_policy(policy_updates)

    summary = {
        "iteration": iteration,
        "quality_factor": Q,
        "prune_probability": prune_p,
        "inertia_weight": w,
        "global_best_reward": float(rewards[best_idx]),
        "policy_updates": len(policy_updates),
    }

    return new_particles, new_personal_bests, new_g_best, summary

# ----------------------------------------------------------------------
# Example usage (can be removed or adapted by the caller)

if __name__ == "__main__":
    # Simple demo with synthetic data
    DIM = 5
    POP_SIZE = 20
    MAX_ITER = 100
    BASE_PRUNE = 0.3

    # Define actions and their PSO hyper‑parameters
    ACTIONS = ["aggressive", "balanced", "conservative"]
    ACTION_HYPER = {
        "aggressive": (2.0, 2.0),   # high cognitive & social coefficients
        "balanced": (1.5, 1.5),
        "conservative": (1.0, 1.0),  # more exploration
    }

    rng = random.Random(42)
    particles = [_random_vector(DIM, -1.0, 1.0, rng) for _ in range(POP_SIZE)]
    personal_bests = particles.copy()
    g_best = max(particles, key=lambda v: evaluate_candidate(v, np.zeros(DIM), {}))

    audit_ctx = {
        "claims_with_evidence": 80,
        "total_claims_emitted": 100,
        "displayed_ok": 70,
        "unknown_displayed_as_ok": 10,
        "exports_missing_audit_step": 2,
    }

    reference_vec = np.ones(DIM).tolist()

    for it in range(MAX_ITER):
        particles, personal_bests, g_best, info = hybrid_step(
            particles=particles,
            personal_bests=personal_bests,
            g_best=g_best,
            iteration=it,
            max_iterations=MAX_ITER,
            base_prune_rate=BASE_PRUNE,
            audit_context=audit_ctx,
            reference=reference_vec,
            actions=ACTIONS,
            action_hyperparams=ACTION_HYPER,
            bounds=(-1.0, 1.0),
            epsilon=0.1,
            seed=it,  # deterministic per iteration
        )
        if it % 20 == 0:
            print(
                f"Iter {it:03d} | Global reward: {info['global_best_reward']:.4f} "
                f"| Q={info['quality_factor']:.3f} | prune={info['prune_probability']:.3f}"
            )