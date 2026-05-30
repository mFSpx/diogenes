# DARWIN HAMMER — match 1831, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_ternary_lens__m953_s1.py (gen3)
# parent_b: hybrid_cockpit_metrics_hybrid_hybrid_ternar_m229_s2.py (gen3)
# born: 2026-05-29T23:39:17Z

"""Hybrid Algorithm combining DARWIN HAMMER parent A and B.

Parent A contributes a multi‑armed bandit policy (reward accumulation,
action counts, and selection) together with a decreasing‑rate pruning
schedule. Parent B contributes audit‑derived quality metrics
(anti_slop_ratio, cockpit_honesty, audit_debt) and a particle‑swarm‑like
social interaction update for candidate vectors.

Mathematical bridge:
- Each action corresponds to a lens‑candidate vector **x**.
- The expected reward of an action is weighted by the audit quality
  factor **Q = anti_slop_ratio * cockpit_honesty / (1 + audit_debt)**.
- The bandit policy is updated with **r = SSIM(x) * Q** where SSIM is
  approximated by cosine similarity.
- Pruning probability follows a decreasing schedule **p = base *
  (1 - iter / max_iter)** and is further modulated by **Q**.
- Social interaction moves vectors toward personal best **p_best** and
  global best **g_best** using PSO‑style velocity updates; the step size
  is scaled by the same pruning probability, thus fusing both parent
  topologies into a single adaptive system.
"""

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Global policy store (parent A)
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear the bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list[dict[str, Any]]) -> None:
    """Apply a batch of (action_id, reward) updates to the bandit policy."""
    for u in updates:
        stats = _POLICY.setdefault(u["action_id"], [0.0, 0.0])
        stats[0] += float(u["reward"])
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Audit quality metrics (parent B)

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of evidenced claims, clipped to [0,1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed‑ok items, clipped to [0,1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    """Simple linear debt metric."""
    return float(max(0, exports_missing_audit_step))

# ----------------------------------------------------------------------
# Vector utilities (parent B)

Vector = Sequence[float]

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return cosine similarity in [0,1] (0 = orthogonal, 1 = identical)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return 0.0
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))

def social_interaction(
    x: Vector,
    personal_best: Vector,
    g_best: Vector,
    k: int = 1,
    r1: float | None = None,
    r2: float | None = None,
    seed: int | str | None = None,
) -> list[float]:
    """
    PSO‑style velocity update.
    new_velocity = k * (r1 * (personal_best - x) + r2 * (g_best - x))

    r1 and r2 are random coefficients in [0,1]; if omitted they are
    drawn from a reproducible RNG seeded by ``seed``.
    """
    rng = random.Random(seed)
    r1 = r1 if r1 is not None else rng.random()
    r2 = r2 if r2 is not None else rng.random()
    x_arr = np.asarray(x, dtype=float)
    p_arr = np.asarray(personal_best, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)
    vel = k * (r1 * (p_arr - x_arr) + r2 * (g_arr - x_arr))
    return vel.tolist()

# ----------------------------------------------------------------------
# Hybrid components

def compute_pruning_probability(
    iteration: int,
    max_iterations: int,
    base_rate: float,
    quality_factor: float,
) -> float:
    """
    Decreasing schedule multiplied by an audit‑derived quality factor.
    p = base_rate * (1 - iter / max) * quality_factor
    Result is clipped to [0,1].
    """
    if max_iterations <= 0:
        return 0.0
    decay = 1.0 - iteration / max_iterations
    prob = base_rate * decay * quality_factor
    return max(0.0, min(1.0, prob))

def evaluate_candidate(
    candidate: Vector,
    reference: Vector,
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    exports_missing_audit_step: int,
) -> float:
    """
    Compute a reward for a candidate vector.
    - SSIM proxy: cosine similarity with a reference vector.
    - Quality factor Q = anti_slop_ratio * cockpit_honesty / (1 + audit_debt)
    Reward = similarity * Q
    """
    similarity = cosine_similarity(candidate, reference)
    q = (
        anti_slop_ratio(claims_with_evidence, total_claims_emitted)
        * cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
        / (1.0 + audit_debt(exports_missing_audit_step))
    )
    return similarity * q

def hybrid_step(
    particles: list[Vector],
    personal_bests: list[Vector],
    g_best: Vector,
    iteration: int,
    max_iterations: int,
    base_prune_rate: float,
    audit_context: dict[str, int],
    reference: Vector,
    seed: int | str | None = None,
) -> tuple[list[Vector], list[Vector], dict[str, Any]]:
    """
    One hybrid iteration:
    1. Compute quality factor Q from audit_context.
    2. Determine pruning probability.
    3. For each particle:
       a. Update velocity via social_interaction().
       b. Apply pruning: with probability p the particle is zero‑ed (simulating removal).
       c. Evaluate reward and possibly update personal best.
    4. Update global best.
    5. Record bandit policy updates.
    Returns updated particles, personal bests, and a summary dict.
    """
    # 1. Quality factor Q
    Q = (
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

    # 2. Pruning probability
    prune_p = compute_pruning_probability(
        iteration, max_iterations, base_prune_rate, Q
    )

    rng = random.Random(seed)

    policy_updates: list[dict[str, Any]] = []
    new_particles: list[Vector] = []
    new_personal_bests: list[Vector] = []

    for idx, (x, p_best) in enumerate(zip(particles, personal_bests)):
        # 3a. Velocity update
        vel = social_interaction(
            x,
            p_best,
            g_best,
            k=1,
            seed=rng.randint(0, 2**32 - 1),
        )
        # Apply velocity (simple Euler step)
        x_new = (np.asarray(x, dtype=float) + np.asarray(vel, dtype=float)).tolist()

        # 3b. Pruning decision
        if rng.random() < prune_p:
            x_new = [0.0 for _ in x_new]  # pruned particle collapses to zero

        # 3c. Reward evaluation
        reward = evaluate_candidate(
            candidate=x_new,
            reference=reference,
            claims_with_evidence=audit_context.get("claims_with_evidence", 0),
            total_claims_emitted=audit_context.get("total_claims_emitted", 0),
            displayed_ok=audit_context.get("displayed_ok", 0),
            unknown_displayed_as_ok=audit_context.get("unknown_displayed_as_ok", 0),
            exports_missing_audit_step=audit_context.get("exports_missing_audit_step", 0),
        )

        # Update personal best if reward improves
        current_best_reward = evaluate_candidate(
            candidate=p_best,
            reference=reference,
            claims_with_evidence=audit_context.get("claims_with_evidence", 0),
            total_claims_emitted=audit_context.get("total_claims_emitted", 0),
            displayed_ok=audit_context.get("displayed_ok", 0),
            unknown_displayed_as_ok=audit_context.get("unknown_displayed_as_ok", 0),
            exports_missing_audit_step=audit_context.get("exports_missing_audit_step", 0),
        )
        if reward > current_best_reward:
            p_best_new = x_new
        else:
            p_best_new = p_best

        new_particles.append(x_new)
        new_personal_bests.append(p_best_new)

        # Bandit policy update (action_id is particle index)
        policy_updates.append(
            {"action_id": f"particle_{idx}", "reward": reward}
        )

    # 4. Global best selection
    rewards = [
        evaluate_candidate(
            candidate=pb,
            reference=reference,
            claims_with_evidence=audit_context.get("claims_with_evidence", 0),
            total_claims_emitted=audit_context.get("total_claims_emitted", 0),
            displayed_ok=audit_context.get("displayed_ok", 0),
            unknown_displayed_as_ok=audit_context.get("unknown_displayed_as_ok", 0),
            exports_missing_audit_step=audit_context.get("exports_missing_audit_step", 0),
        )
        for pb in new_personal_bests
    ]
    best_idx = int(np.argmax(rewards))
    new_g_best = new_personal_bests[best_idx]

    # 5. Apply bandit updates
    update_policy(policy_updates)

    summary = {
        "iteration": iteration,
        "prune_probability": prune_p,
        "quality_factor": Q,
        "global_best_index": best_idx,
        "global_best_reward": rewards[best_idx],
    }

    return new_particles, new_personal_bests, summary

# ----------------------------------------------------------------------
# Utility for parsing JSON context (parent A)

def parse_context(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

# ----------------------------------------------------------------------
# Smoke test

if __name__ == "__main__":
    # deterministic seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # Create dummy particles (5 particles, 3‑dimensional)
    particles = [np.random.rand(3).tolist() for _ in range(5)]
    personal_bests = [p.copy() for p in particles]
    g_best = particles[0]

    # Audit context example
    audit_ctx = {
        "claims_with_evidence": 80,
        "total_claims_emitted": 100,
        "displayed_ok": 70,
        "unknown_displayed_as_ok": 10,
        "exports_missing_audit_step": 2,
    }

    # Reference vector for SSIM proxy
    reference = np.random.rand(3).tolist()

    max_iters = 10
    base_prune_rate = 0.3

    for it in range(max_iters):
        particles, personal_bests, info = hybrid_step(
            particles=particles,
            personal_bests=personal_bests,
            g_best=g_best,
            iteration=it,
            max_iterations=max_iters,
            base_prune_rate=base_prune_rate,
            audit_context=audit_ctx,
            reference=reference,
            seed=SEED + it,
        )
        # Update global best for next iteration
        g_best = personal_bests[info["global_best_index"]]

        print(
            f"Iter {it:02d} | prune_p={info['prune_probability']:.3f} | "
            f"Q={info['quality_factor']:.3f} | best_reward={info['global_best_reward']:.3f}"
        )

    print("\nFinal bandit policy statistics:")
    for action, (total, count) in _POLICY.items():
        avg = total / count if count else 0.0
        print(f"{action}: avg_reward={avg:.4f} count={int(count)}")