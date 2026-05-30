# DARWIN HAMMER — match 2759, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s0.py (gen6)
# born: 2026-05-29T23:45:35Z

"""Hybrid Regret‑Weighted Bandit with Temperature‑Driven Curvature via Count‑Min Sketch.

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_regret_m746_s0.py (Regret‑Weighted Ternary‑Decision Analyzer + propensity bandit)
- hybrid_hybrid_hybrid_hybrid_privacy_sketc_m2009_s0.py (Schoolfield temperature function + Count‑Min Sketch)

Mathematical bridge:
The sketch provides an online estimate of the ratio ρ = (unique actions observed) / (total actions observed).
We use ρ as a *temperature* τ = τ₀·(1+ρ) that scales:
1. the learning‑rate α in the Ollivier‑Ricci curvature proxy (τ‑dependent α = 0.1·τ).
2. the regret‑weighted probability p_i ∝ exp(−regret_i·τ) used by the bandit selector.

Thus a single scalar τ couples the privacy‑preserving frequency estimator with both the curvature computation
and the regret‑weighted action selection, yielding a unified hybrid system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared immutable data structures
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
# Global stores (policy statistics and sketch)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id -> [cum_reward, count]
_SKETCH: Dict[Tuple[int, int], int] = {}     # (depth,row) -> count
_UNIQUE_ACTIONS: set = set()
_TOTAL_UPDATES: int = 0
_SKETCH_DEPTH: int = 4
_SKETCH_WIDTH: int = 256
_RANDOM_SEEDS: List[int] = [random.randint(0, 2**31 - 1) for _ in range(_SKETCH_DEPTH)]

def reset_all() -> None:
    """Clear policy statistics and the Count‑Min sketch."""
    _POLICY.clear()
    _SKETCH.clear()
    _UNIQUE_ACTIONS.clear()
    global _TOTAL_UPDATES
    _TOTAL_UPDATES = 0

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch implementation
# ----------------------------------------------------------------------
def _hash(item: str, i: int) -> int:
    """Deterministic hash for row i of the sketch."""
    rnd = random.Random(_RANDOM_SEEDS[i])
    # combine builtin hash with row‑specific seed
    return (hash(item) ^ rnd.getrandbits(64)) % _SKETCH_WIDTH

def sketch_update(action_id: str) -> None:
    """Insert an action identifier into the sketch."""
    global _TOTAL_UPDATES
    for i in range(_SKETCH_DEPTH):
        idx = _hash(action_id, i)
        _SKETCH[(i, idx)] = _SKETCH.get((i, idx), 0) + 1
    _UNIQUE_ACTIONS.add(action_id)
    _TOTAL_UPDATES += 1

def sketch_estimate(action_id: str) -> int:
    """Estimate the frequency of *action_id* using the sketch."""
    return min(_SKETCH.get((i, _hash(action_id, i)), 0) for i in range(_SKETCH_DEPTH))

def sketch_ratio() -> float:
    """Return the estimated unique‑to‑total action ratio ρ."""
    if _TOTAL_UPDATES == 0:
        return 0.0
    return len(_UNIQUE_ACTIONS) / _TOTAL_UPDATES

# ----------------------------------------------------------------------
# Temperature computation
# ----------------------------------------------------------------------
BASE_TEMPERATURE: float = 1.0  # τ₀

def compute_temperature() -> float:
    """Temperature τ = τ₀·(1+ρ) where ρ is the sketch ratio."""
    ρ = sketch_ratio()
    return BASE_TEMPERATURE * (1.0 + ρ)

# ----------------------------------------------------------------------
# Regret‑Weighted probability utilities
# ----------------------------------------------------------------------
def _reward(action: str) -> float:
    """Mean reward for *action* (0 if never seen)."""
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    return total / cnt if cnt else 0.0

def _regret(action: str, best_reward: float) -> float:
    """Regret = best_reward − expected_reward."""
    return best_reward - _reward(action)

def update_policy(update: BanditUpdate) -> None:
    """Incorporate a new BanditUpdate into policy statistics and sketch."""
    # policy statistics
    total, cnt = _POLICY.get(update.action_id, [0.0, 0.0])
    _POLICY[update.action_id] = [total + update.reward, cnt + 1]
    # sketch
    sketch_update(update.action_id)

# ----------------------------------------------------------------------
# Curvature proxy (Ollivier‑Ricci like) using temperature‑scaled learning rate
# ----------------------------------------------------------------------
def compute_curvature(adj_matrix: np.ndarray, temperature: float) -> np.ndarray:
    """
    Approximate node‑wise curvature.

    For each node i:
        curvature[i] = α·Σ_j A_ij   with α = 0.1·temperature
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix must be a square 2‑D array")
    α = 0.1 * temperature
    # sum of incident edge weights per node
    incident = adj_matrix.sum(axis=1)
    return α * incident

# ----------------------------------------------------------------------
# Hybrid action selector
# ----------------------------------------------------------------------
def select_hybrid_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "regret_weighted",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a regret‑weighted distribution that is
    temperature‑scaled, then optionally fall back to epsilon‑greedy.

    The returned BanditAction contains the propensity (probability) and
    a confidence bound derived from the curvature vector.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # temperature from sketch
    τ = compute_temperature()

    # compute best mean reward for regret calculation
    best = max(_reward(a) for a in actions)

    # regret‑weighted probabilities p_i ∝ exp(−regret_i·τ)
    weights = []
    for a in actions:
        r = _regret(a, best)
        w = math.exp(-r * τ)
        weights.append(w)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # epsilon‑greedy fallback
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_idx = rng.randrange(len(actions))
    else:
        # sample according to the computed distribution
        cumulative = 0.0
        draw = rng.random()
        for idx, p in enumerate(probs):
            cumulative += p
            if draw <= cumulative:
                chosen_idx = idx
                break
        else:
            chosen_idx = len(actions) - 1  # numerical safety

    chosen_action = actions[chosen_idx]
    propensity = probs[chosen_idx]

    # confidence bound: use curvature as a proxy for uncertainty
    # (larger curvature → larger bound)
    # For demo purposes we compute curvature on a dummy adjacency where
    # each action is a node connected to every other with weight 1.
    n = len(actions)
    dummy_adj = np.ones((n, n)) - np.eye(n)
    curvature_vec = compute_curvature(dummy_adj, τ)
    conf_bound = float(curvature_vec[chosen_idx])

    return BanditAction(
        action_id=chosen_action,
        propensity=propensity,
        expected_reward=_reward(chosen_action),
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )

# ----------------------------------------------------------------------
# Example utility that runs a full update‑select cycle
# ----------------------------------------------------------------------
def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    context_id: str,
    algorithm: str = "regret_weighted",
    seed: int | str | None = 7,
) -> BanditAction:
    """
    1. Select an action.
    2. Simulate a stochastic reward (for demo purposes).
    3. Update policy & sketch.
    4. Return the selected BanditAction.
    """
    selected = select_hybrid_action(context, actions, algorithm, seed=seed)
    # Simulated reward: reward = 1.0 if action id ends with an even digit else 0.0
    reward = 1.0 if selected.action_id[-1].isdigit() and int(selected.action_id[-1]) % 2 == 0 else 0.0
    update = BanditUpdate(
        context_id=context_id,
        action_id=selected.action_id,
        reward=reward,
        propensity=selected.propensity,
    )
    update_policy(update)
    return selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_all()
    # Dummy context (feature vector)
    ctx = {"feature1": 0.3, "feature2": 1.2}
    # Action pool
    act_pool = [f"action_{i}" for i in range(6)]

    # Run a few hybrid steps
    for t in range(10):
        ba = hybrid_step(ctx, act_pool, context_id=f"step_{t}")
        print(
            f"Step {t}: chosen={ba.action_id}, prop={ba.propensity:.3f}, "
            f"exp_reward={ba.expected_reward:.3f}, conf={ba.confidence_bound:.3f}"
        )

    # Demonstrate curvature on a random graph
    rng = np.random.default_rng(42)
    adj = rng.integers(0, 2, size=(len(act_pool), len(act_pool)))
    np.fill_diagonal(adj, 0)  # no self‑loops
    τ = compute_temperature()
    curv = compute_curvature(adj, τ)
    print(f"\nTemperature τ={τ:.3f}")
    print("Curvature vector:", curv)