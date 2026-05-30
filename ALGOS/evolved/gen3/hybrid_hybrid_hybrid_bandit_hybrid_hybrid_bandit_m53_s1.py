# DARWIN HAMMER — match 53, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s1.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_koopman_operator_m64_s2.py (gen2)
# born: 2026-05-29T23:25:31Z

"""
Hybrid Bandit‑Koopman‑Linear Fusion

Parents:
* hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s1.py (Bandit router + TTT‑Linear core)
* hybrid_hybrid_bandit_router_koopman_operator_m64_s2.py (Bandit router + Koopman operator)

Mathematical Bridge
-------------------
1. **Bandit statistics** – For each action *a* we keep empirical mean
   μₐ(t) and count Nₐ(t).  The store Sₜ evolves with observed reward:
   Sₜ₊₁ = Sₜ + reward – cost (cost is taken as 0 here).

2. **Propensity vector** – A contextual‑UCB style score
   sₐ(t)= μₐ(t) + η‖c‖·cₐ(t)  with
   cₐ(t)= (1+Sₜ/(Sₜ+1))/√(1+Nₐ(t)).
   Propensities are the soft‑max of sₐ(t):
   pₐ(t)=exp(sₐ)/∑_b exp(s_b).

3. **Linear (TTT‑Linear) core** – A weight matrix **W** maps the propensity
   vector **p(t)** to a transformed reward estimate
   r̂ᶫₐ(t)= (W p(t))ₐ.

4. **Koopman forecast** – Empirical mean vectors μₜ are stored.
   A Koopman operator **K** is fitted by least‑squares
   K = Y X⁺ where X=[μ₀,…,μ_{T‑1}], Y=[μ₁,…,μ_T].
   The h‑step forecast is μ̂ᵏₜ = Kʰ μₜ.

5. **Hybrid index** – The final action index combines the three
   components with a mixing factor α∈[0,1]:
   Uₐ(t)= α·μ̂ᵏₐ(t) + (1‑α)·r̂ᶫₐ(t) + η‖c‖·cₐ(t).

The algorithm therefore fuses:
* the exploration term from the Bandit‑Router (store‑scaled confidence),
* the linear transformation of propensity scores (TTT‑Linear),
* the future‑reward forecast from the Koopman operator.

The implementation below provides a fully functional hybrid system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Global state (mirrors both parents)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: float = 0.0                           # scalar store influencing confidence
_MEAN_HISTORY: List[np.ndarray] = []         # list of μ vectors over time
_W: np.ndarray = np.array([])                # linear weight matrix (A×A)
_ETA: float = 1.0                            # exploration scaling
_ALPHA: float = 0.5                           # mixing between Koopman and linear core
_RIDGE_LAMBDA: float = 1e-3                    # regularisation for linear weight update
_KOOPMAN_HORIZON: int = 1                     # steps ahead for Koopman forecast


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observed outcome."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _reset_globals(num_actions: int) -> None:
    """Initialise/clear all global structures for a fresh run."""
    global _POLICY, _STORE, _MEAN_HISTORY, _W
    _POLICY.clear()
    _STORE = 0.0
    _MEAN_HISTORY.clear()
    _W = np.eye(num_actions)  # start with identity mapping


def _reward(action: str) -> float:
    """Empirical mean reward μₐ."""
    total, cnt = _POLICY.get(action, [0.0, 0.0])
    return total / cnt if cnt > 0 else 0.0


def _count(action: str) -> int:
    """Number of times action has been taken."""
    return int(_POLICY.get(action, [0.0, 0.0])[1])


def _context_norm(context: Dict[str, float]) -> float:
    """Euclidean norm of the context feature vector."""
    return math.sqrt(sum(v * v for v in context.values()))


def _confidence_multiplier(store: float, count: int) -> float:
    """Store‑dependent confidence term cₐ(t)."""
    return (1.0 + store / (store + 1.0)) / math.sqrt(1.0 + count)


def _mean_vector(actions: List[str]) -> np.ndarray:
    """Return the μ vector ordered according to *actions*."""
    return np.array([_reward(a) for a in actions], dtype=float)


def _fit_koopman_operator(history: List[np.ndarray]) -> np.ndarray:
    """Least‑squares fit of K given a sequence of μ vectors."""
    if len(history) < 2:
        dim = history[0].shape[0] if history else 1
        return np.eye(dim)
    X = np.column_stack(history[:-1])      # shape (A, T‑1)
    Y = np.column_stack(history[1:])       # shape (A, T‑1)
    # Solve K X ≈ Y → K = Y X⁺
    X_pinv = np.linalg.pinv(X)
    K = Y @ X_pinv
    return K


def _forecast_koopman(K: np.ndarray, mu: np.ndarray, horizon: int) -> np.ndarray:
    """Compute Kʰ μ."""
    if horizon <= 0:
        return mu
    return np.linalg.matrix_power(K, horizon) @ mu


def _update_linear_weights(W: np.ndarray,
                           propensities: List[np.ndarray],
                           rewards: List[np.ndarray]) -> np.ndarray:
    """
    Ridge‑regression update of the linear mapping W.
    Solve min_W Σ‖W p_i – r_i‖² + λ‖W‖².
    Closed form: W = R Pᵀ (P Pᵀ + λI)⁻¹
    where P = [p₁ … p_n], R = [r₁ … r_n].
    """
    if not propensities:
        return W
    P = np.column_stack(propensities)   # shape (A, n)
    R = np.column_stack(rewards)        # shape (A, n)
    A = P @ P.T + _RIDGE_LAMBDA * np.eye(P.shape[0])
    W_new = R @ P.T @ np.linalg.inv(A)
    return W_new


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def hybrid_reset(actions: List[str]) -> None:
    """Public reset that prepares the hybrid system for *actions*."""
    _reset_globals(len(actions))


def hybrid_select_action(context: Dict[str, float],
                         actions: List[str],
                         algorithm: str = 'hybrid',
                         epsilon: float = 0.1,
                         seed: int | str | None = 7) -> BanditAction:
    """
    Choose an action using the fused index Uₐ(t).

    Returns a BanditAction containing the chosen action, its propensity,
    the hybrid expected reward, and the confidence bound.
    """
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)

    # 1️⃣ Empirical means and counts
    mu = _mean_vector(actions)                     # shape (A,)
    counts = np.array([_count(a) for a in actions], dtype=float)

    # 2️⃣ Store‑scaled confidence term
    c_vec = np.vectorize(_confidence_multiplier)(_STORE, counts)

    # 3️⃣ Raw UCB‑style scores (used for soft‑max propensities)
    ctx_norm = _context_norm(context)
    raw_scores = mu + _ETA * ctx_norm * c_vec

    # 4️⃣ ε‑greedy fallback
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen_idx = rng.randrange(len(actions))
    else:
        # Soft‑max to obtain propensities
        max_raw = np.max(raw_scores)  # for numerical stability
        exp_scores = np.exp(raw_scores - max_raw)
        propensities = exp_scores / exp_scores.sum()
        # Linear core output
        global _W
        lin_out = _W @ propensities

        # Koopman forecast (if enough history)
        if len(_MEAN_HISTORY) >= 2:
            K = _fit_koopman_operator(_MEAN_HISTORY)
            mu_hat = _forecast_koopman(K, mu, _KOOPMAN_HORIZON)
        else:
            mu_hat = mu.copy()

        # Hybrid combination
        hybrid_est = _ALPHA * mu_hat + (1.0 - _ALPHA) * lin_out

        # Final index adds the confidence term again (as in parent B)
        final_index = hybrid_est + _ETA * ctx_norm * c_vec

        chosen_idx = int(np.argmax(final_index))

    chosen_action = actions[chosen_idx]
    chosen_propensity = propensities[chosen_idx] if 'propensities' in locals() else 0.0
    expected_reward = mu[chosen_idx]               # baseline empirical mean
    confidence = c_vec[chosen_idx]

    return BanditAction(
        action_id=chosen_action,
        propensity=chosen_propensity,
        expected_reward=expected_reward,
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def hybrid_step(updates: List[BanditUpdate],
                context: Dict[str, float],
                actions: List[str]) -> None:
    """
    Process a batch of observed rewards:
    * update policy statistics,
    * evolve the store,
    * record the new mean vector,
    * re‑fit linear weights,
    * (Koopman operator is fitted lazily during selection).
    """
    global _STORE, _MEAN_HISTORY, _W

    # --- 1️⃣ Policy update -------------------------------------------------
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0
        # Store evolves with net reward (cost assumed zero)
        _STORE += float(u.reward)

    # --- 2️⃣ Record new mean vector ----------------------------------------
    mu_new = _mean_vector(actions)
    _MEAN_HISTORY.append(mu_new.copy())
    # keep a bounded history to avoid unbounded memory growth
    if len(_MEAN_HISTORY) > 100:
        _MEAN_HISTORY.pop(0)

    # --- 3️⃣ Linear core update --------------------------------------------
    # Build training matrices from the batch:
    prop_batch: List[np.ndarray] = []
    reward_batch: List[np.ndarray] = []
    for u in updates:
        # recompute propensities for the context at the moment of update
        # (using the same logic as in hybrid_select_action)
        counts = np.array([_count(a) for a in actions], dtype=float)
        c_vec = np.vectorize(_confidence_multiplier)(_STORE, counts)
        mu_vec = _mean_vector(actions)
        ctx_norm = _context_norm(context)
        raw = mu_vec + _ETA * ctx_norm * c_vec
        max_raw = np.max(raw)
        exp_scores = np.exp(raw - max_raw)
        prop = exp_scores / exp_scores.sum()
        prop_batch.append(prop)

        # Target reward vector: place the observed reward at the taken action,
        # zeros elsewhere (sparse target for regression)
        r_vec = np.zeros(len(actions))
        idx = actions.index(u.action_id)
        r_vec[idx] = u.reward
        reward_batch.append(r_vec)

    _W = _update_linear_weights(_W, prop_batch, reward_batch)


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------
def hybrid_select_and_update(context: Dict[str, float],
                             actions: List[str],
                             epsilon: float = 0.1,
                             seed: int | str | None = 7) -> BanditAction:
    """
    One‑step convenience: select an action, simulate a reward (here random),
    and feed the observation back into the hybrid system.
    """
    action = hybrid_select_action(context, actions,
                                  algorithm='epsilon_greedy',
                                  epsilon=epsilon,
                                  seed=seed)

    # Simulated stochastic reward (for demo purposes)
    rng = random.Random(seed)
    simulated_reward = rng.gauss(mu=action.expected_reward, sigma=1.0)

    update = BanditUpdate(
        context_id='ctx',
        action_id=action.action_id,
        reward=simulated_reward,
        propensity=action.propensity,
    )
    hybrid_step([update], context, actions)
    return action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small action set
    actions_list = ['a', 'b', 'c']
    hybrid_reset(actions_list)

    # Dummy constant context
    ctx = {'feature1': 1.0, 'feature2': -0.5}

    # Run a few iterations
    for t in range(20):
        act = hybrid_select_and_update(ctx, actions_list, epsilon=0.2, seed=t)
        print(f"Step {t:02d}: chose {act.action_id} (prop={act.propensity:.3f}, "
              f"exp={act.expected_reward:.3f}, conf={act.confidence_bound:.3f})")

    # Final policy summary
    print("\nFinal empirical means:")
    for a in actions_list:
        print(f"  {a}: μ={_reward(a):.3f}, count={_count(a)}")
    print(f"Final store value: {_STORE:.3f}")
    print(f"Linear weight matrix W:\n{_W}")