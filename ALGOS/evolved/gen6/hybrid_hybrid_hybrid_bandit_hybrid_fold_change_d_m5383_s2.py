# DARWIN HAMMER — match 5383, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_geomet_m1176_s0.py (gen5)
# parent_b: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2.py (gen3)
# born: 2026-05-30T00:01:33Z

"""
Hybrid Bandit–Geometric–Fold‑Change Algorithm
Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_geomet_m1176_s0.py (Bandit + Clifford geometric algebra)
- hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s2.py (Fold‑change detection + log‑count store factor)

Mathematical Bridge
------------------
Each BanditAction is lifted to a simple multivector **M = s + v**, where *s* is the
scalar part (the action propensity) and *v* is a vector encoding the expected reward
and confidence bound.  A contextual feature vector *c* is similarly lifted to a
multivector **C = 0 + c**.  The geometric product **M ⋅ C** yields a multivector whose
scalar part is  

    ⟨M C⟩₀ = s·0 + v·c = v·c                     (1)

i.e. the Euclidean dot‑product between the action‑vector and the context‑vector.
This scalar is the *geometric interaction score*.

The fold‑change detection module supplies a log‑count ratio  

    λ = log( N_a + 1 ) - log( N_ref + 1 )          (2)

where *N_a* is the selection count of the candidate action and *N_ref* is a
reference count (here the mean count over all actions).  Equation (2) quantifies
how many “effective activations” the action has relative to the population.

The hybrid propensity fed to the bandit decision rule is therefore

    π̂_a = ε · ⟨M C⟩₀ · λ + (1‑ε)·π_a               (3)

with ε∈[0,1] a blending coefficient and π_a the original propensity stored in
the policy.  The store variable **S** evolves with simple inflow/outflow dynamics
and modulates ε dynamically:

    ε = σ( S ) = 1 / (1 + exp(‑α·(S‑β)))           (4)

where α,β are tunable constants and σ is the logistic function.

The three public functions below implement:
1. `hybrid_select_action` – computes (3) for every candidate and selects by
   epsilon‑greedy.
2. `hybrid_fold_change_detection_series` – runs fold‑change detection on a
   time‑series while biasing the detection threshold with the geometric‑bandit
   score.
3. `hybrid_update` – updates the bandit policy, the store, and recomputes counts.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical in both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # baseline propensity π_a
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float          # propensity used at selection time

# Global policy storage: action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the global bandit statistics."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Mean reward for an action."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Selection count for an action."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: Iterable[BanditUpdate]) -> None:
    """Incorporate a batch of BanditUpdate objects into the global policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Simple multivector utilities (scalar + vector only)
# ----------------------------------------------------------------------
def encode_action_mv(action: BanditAction, dim: int) -> Tuple[float, np.ndarray]:
    """
    Lift a BanditAction to a multivector (scalar, vector) of dimension `dim`.
    * scalar part   = action.propensity
    * vector part   = [expected_reward, confidence_bound, 0, …] padded to `dim`
    """
    scalar = float(action.propensity)
    vec = np.zeros(dim, dtype=float)
    vec[0] = float(action.expected_reward)
    if dim > 1:
        vec[1] = float(action.confidence_bound)
    return scalar, vec

def encode_context_mv(context: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Lift a raw context vector to a multivector.
    * scalar part = 0.0 (no intrinsic propensity)
    * vector part = context (must be 1‑D)
    """
    return 0.0, np.asarray(context, dtype=float)

def geometric_scalar(action_mv: Tuple[float, np.ndarray],
                    context_mv: Tuple[float, np.ndarray]) -> float:
    """
    Return the scalar part ⟨A C⟩₀ of the geometric product.
    For scalar+vector multivectors this reduces to the dot product of the vectors.
    """
    _, a_vec = action_mv
    _, c_vec = context_mv
    # Ensure same dimensionality
    if a_vec.shape != c_vec.shape:
        raise ValueError("Action and context multivectors must share dimensionality")
    return float(np.dot(a_vec, c_vec))

# ----------------------------------------------------------------------
# Fold‑change detection utilities (from parent B)
# ----------------------------------------------------------------------
_EPS = 1e-12

def fold_change(u: float, x: float) -> float:
    """
    Classic fold‑change: u / max(|x|, ε)
    """
    return u / max(abs(x), _EPS)

def fold_change_detection_series(series: List[float]) -> List[float]:
    """
    Apply fold‑change detection to a numeric series.
    Returns the list of fold‑change values (len = len(series)-1).
    """
    if len(series) < 2:
        return []
    changes = []
    prev = series[0]
    for cur in series[1:]:
        changes.append(fold_change(cur, prev))
        prev = cur
    return changes

# ----------------------------------------------------------------------
# Store dynamics (from parent A, simplified)
# ----------------------------------------------------------------------
def update_store(store: float,
                 inflow: List[float],
                 outflow: List[float],
                 decay: float = 0.01) -> float:
    """
    Simple store update:
        S_{t+1} = (1‑decay)·S_t + Σ inflow – Σ outflow
    """
    net = sum(inflow) - sum(outflow)
    return (1.0 - decay) * store + net

def logistic(x: float, alpha: float = 1.0, beta: float = 0.0) -> float:
    """Logistic squashing function σ(x) = 1/(1+exp(-α·(x‑β)))."""
    return 1.0 / (1.0 + math.exp(-alpha * (x - beta)))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_select_action(context: np.ndarray,
                         actions: List[BanditAction],
                         store: float,
                         epsilon: float = 0.1,
                         alpha: float = 1.0,
                         beta: float = 0.0) -> Tuple[str, float]:
    """
    Select an action using the hybrid rule.

    Steps
    -----
    1. Compute the geometric interaction score s_a = ⟨M_a C⟩₀ for each action.
    2. Compute the log‑count ratio λ_a = log(N_a+1) – log(N̄+1).
    3. Blend with the store‑driven mixing coefficient ε̂ = σ(store;α,β).
    4. Form the hybrid propensity π̂_a per Eq. (3).
    5. Perform epsilon‑greedy selection using ε (exploration) and π̂_a (exploitation).

    Returns
    -------
    (selected_action_id, hybrid_propensity)
    """
    if not actions:
        raise ValueError("Action list is empty")

    dim = max(2, context.shape[0])  # ensure at least 2 dimensions for vector part
    ctx_mv = encode_context_mv(context)

    # Reference count = mean count over actions (avoid division by zero)
    counts = np.array([_count(a.action_id) for a in actions], dtype=float)
    mean_count = counts.mean() if counts.size else 0.0

    # Store‑driven blending coefficient
    eps_hat = logistic(store, alpha, beta)

    # Compute hybrid propensities
    hybrid_props = []
    for act in actions:
        act_mv = encode_action_mv(act, dim)
        geom_score = geometric_scalar(act_mv, ctx_mv)          # ⟨M C⟩₀
        lam = math.log(_count(act.action_id) + 1.0) - math.log(mean_count + 1.0 + _EPS)
        pi_hat = eps_hat * geom_score * lam + (1.0 - eps_hat) * act.propensity
        hybrid_props.append(pi_hat)

    # Epsilon‑greedy decision
    if random.random() < epsilon:
        chosen_idx = random.randrange(len(actions))
    else:
        # break ties randomly
        max_val = max(hybrid_props)
        candidates = [i for i, v in enumerate(hybrid_props) if abs(v - max_val) < 1e-12]
        chosen_idx = random.choice(candidates)

    chosen_action = actions[chosen_idx]
    return chosen_action.action_id, hybrid_props[chosen_idx]

def hybrid_fold_change_detection_series(series: List[float],
                                         context: np.ndarray,
                                         actions: List[BanditAction],
                                         store: float) -> List[float]:
    """
    Apply fold‑change detection to `series` while modulating the detection
    threshold with the hybrid geometric‑bandit score for the *most relevant*
    action (the one with highest geometric scalar for the given context).

    The returned list contains the raw fold‑change values multiplied by a
    scaling factor γ = 1 + tanh(π̂_max) where π̂_max is the hybrid propensity of
    the best action.
    """
    # 1. Compute geometric scores for each action
    dim = max(2, context.shape[0])
    ctx_mv = encode_context_mv(context)
    scores = []
    for act in actions:
        act_mv = encode_action_mv(act, dim)
        scores.append(geometric_scalar(act_mv, ctx_mv))
    # 2. Identify best action (largest geometric interaction)
    best_idx = int(np.argmax(scores))
    best_action = actions[best_idx]

    # 3. Compute its hybrid propensity (no epsilon exploration here)
    eps_hat = logistic(store)
    lam = math.log(_count(best_action.action_id) + 1.0) - math.log(
        (np.mean([_count(a.action_id) for a in actions]) + 1.0)
    )
    act_mv = encode_action_mv(best_action, dim)
    geom_score = geometric_scalar(act_mv, ctx_mv)
    pi_hat = eps_hat * geom_score * lam + (1.0 - eps_hat) * best_action.propensity

    # 4. Scale factor
    gamma = 1.0 + math.tanh(pi_hat)

    # 5. Fold‑change detection on the raw series and apply scaling
    raw_fc = fold_change_detection_series(series)
    return [gamma * fc for fc in raw_fc]

def hybrid_update(store: float,
                 inflow: List[float],
                 outflow: List[float],
                 updates: List[BanditUpdate]) -> float:
    """
    Perform a full hybrid update:
    - Apply bandit policy updates.
    - Update the store with given inflow/outflow.
    - Return the new store value.
    """
    update_policy(updates)
    new_store = update_store(store, inflow, outflow)
    return new_store

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny action set
    actions = [
        BanditAction("A", propensity=0.2, expected_reward=1.0, confidence_bound=0.5),
        BanditAction("B", propensity=0.5, expected_reward=0.8, confidence_bound=0.3),
        BanditAction("C", propensity=0.3, expected_reward=0.6, confidence_bound=0.4),
    ]

    # Simulated context (2‑dimensional)
    ctx = np.array([0.7, -0.2])

    # Initial store
    store_val = 0.0

    # Select an action
    sel_id, hybrid_pi = hybrid_select_action(ctx, actions, store_val, epsilon=0.2)
    print(f"Selected action: {sel_id}, hybrid propensity: {hybrid_pi:.4f}")

    # Simulate receiving a reward and perform an update
    reward = random.uniform(0, 1)
    upd = BanditUpdate(context_id="ctx1", action_id=sel_id, reward=reward,
                      propensity=hybrid_pi)
    store_val = hybrid_update(store_val, inflow=[reward], outflow=[0.1], updates=[upd])
    print(f"Updated store: {store_val:.4f}")

    # Fold‑change detection on a synthetic series
    series = [0.5, 0.55, 0.60, 0.58, 0.70, 0.65]
    fc_scaled = hybrid_fold_change_detection_series(series, ctx, actions, store_val)
    print("Scaled fold‑change series:", ["{:.4f}".format(v) for v in fc_scaled])