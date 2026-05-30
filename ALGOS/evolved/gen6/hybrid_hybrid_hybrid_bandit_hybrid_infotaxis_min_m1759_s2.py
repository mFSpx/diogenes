# DARWIN HAMMER — match 1759, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_geomet_m1176_s0.py (gen5)
# parent_b: hybrid_infotaxis_minhash_m63_s1.py (gen1)
# born: 2026-05-29T23:38:40Z

"""Hybrid Bandit‑MinHash‑Entropy Algorithm
Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_geomet_m1176_s0.py (Bandit with Clifford‑algebra feature interaction)
- hybrid_infotaxis_minhash_m63_s1.py (MinHash similarity & entropy‑based infotaxis)

Mathematical Bridge
------------------
A context is represented by a MinHash signature `σ ∈ ℕᵏ`.  Normalising this signature to a
probability vector `p = σ / sum(σ)` yields a discrete distribution over hash buckets.
Each action `a` carries a *bandit* state `(propensity, expected_reward, confidence_bound)`.
We lift both structures into a common Clifford (geometric) algebra by treating the three
bandit scalars as a grade‑1 multivector `v_a ∈ ℝ³` and the normalised MinHash vector as a
grade‑1 multivector `v_c ∈ ℝᵏ`.  The geometric product of two vectors has a scalar part
equal to their inner product:

    ⟨v_a * v_c⟩₀ = v_a · v_c = Σ_i v_a[i] * v_c[i]      (zero‑filled for missing dimensions)

This scalar fuses the bandit propensity with the contextual similarity.  The resulting
score is used by an ε‑greedy policy, while the entropy of the context distribution
`H(p)` modulates the confidence bound (higher entropy → larger uncertainty → larger
exploration term).  The store variable `S` (from the original bandit) scales ε
dynamically.

The implementation below realises this bridge with three core functions:
`multivector_action`, `multivector_context`, `combined_score`, and a high‑level
`hybrid_bandit_step` that performs selection, reward update, and store adaptation.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set
import numpy as np

# ----------------------------------------------------------------------
# Bandit / Store components (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # raw preference weight
    expected_reward: float     # estimated mean reward
    confidence_bound: float    # exploration bonus
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# Simple in‑memory policy statistics: total reward and count per action
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# Store dynamics (single scalar that influences exploration)
def update_store(store: float, inflow: List[float], outflow: List[float]) -> float:
    """Simple reservoir update: new_store = store + Σ inflow – Σ outflow."""
    return store + sum(inflow) - sum(outflow)

# ----------------------------------------------------------------------
# MinHash & Entropy components (Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

# ----------------------------------------------------------------------
# Hybrid core – geometric algebra bridge
# ----------------------------------------------------------------------
def multivector_action(action: BanditAction) -> np.ndarray:
    """
    Lift a BanditAction into a 3‑dimensional grade‑1 multivector:
        v_a = [propensity, expected_reward, confidence_bound]
    """
    return np.array([action.propensity, action.expected_reward, action.confidence_bound], dtype=float)

def multivector_context(tokens: Iterable[str], k: int = 128) -> np.ndarray:
    """
    Produce a normalised MinHash‑derived multivector.
    The raw signature values are turned into a probability vector by
    subtracting the minimum (to avoid negative values) and normalising.
    """
    sig = np.array(signature(tokens, k), dtype=float)
    # Shift to make all entries non‑negative (MinHash values are already non‑negative)
    shifted = sig - sig.min()
    if shifted.sum() == 0:
        # Degenerate case: all hash values equal → uniform distribution
        probs = np.full_like(shifted, 1.0 / len(shifted))
    else:
        probs = shifted / shifted.sum()
    return probs  # shape (k,)

def combined_score(action_mv: np.ndarray, context_mv: np.ndarray) -> float:
    """
    Scalar part of the geometric product between action and context multivectors.
    Since the vectors have different dimensions, we pad the shorter one with zeros.
    """
    if action_mv.shape[0] < context_mv.shape[0]:
        pad = np.zeros(context_mv.shape[0] - action_mv.shape[0])
        action_mv = np.concatenate([action_mv, pad])
    elif context_mv.shape[0] < action_mv.shape[0]:
        pad = np.zeros(action_mv.shape[0] - context_mv.shape[0])
        context_mv = np.concatenate([context_mv, pad])
    return float(np.dot(action_mv, context_mv))

def compute_confidence(action: BanditAction, context_tokens: Iterable[str], k: int = 128) -> float:
    """
    Adjust the action's confidence bound using the entropy of the context distribution.
    Higher entropy → larger bound (more uncertainty → more exploration).
    """
    ctx_mv = multivector_context(context_tokens, k)
    ctx_entropy = entropy(ctx_mv.tolist())
    # Simple linear scaling: base confidence + λ·entropy
    λ = 0.5
    return action.confidence_bound + λ * ctx_entropy

def epsilon_from_store(store: float) -> float:
    """Map store value to an ε in [0.05, 0.5] using a sigmoid‑like transform."""
    # Clip store to avoid extreme values
    clipped = max(min(store, 10.0), -10.0)
    return 0.05 + 0.45 / (1.0 + math.exp(-clipped))

def select_action(
    actions: List[BanditAction],
    context_tokens: Iterable[str],
    store: float,
    k: int = 128
) -> Tuple[BanditAction, float]:
    """
    ε‑greedy selection based on combined scores.
    Returns the chosen action and the ε used.
    """
    ε = epsilon_from_store(store)
    if random.random() < ε:
        chosen = random.choice(actions)
        return chosen, ε

    # Compute scores for all actions
    ctx_mv = multivector_context(context_tokens, k)
    scores = []
    for a in actions:
        act_mv = multivector_action(a)
        # Incorporate entropy‑adjusted confidence
        adj_conf = compute_confidence(a, context_tokens, k)
        act_mv[2] = adj_conf  # replace confidence component
        score = combined_score(act_mv, ctx_mv)
        scores.append(score)

    # Choose argmax
    max_idx = int(np.argmax(scores))
    return actions[max_idx], ε

def hybrid_bandit_step(
    actions: List[BanditAction],
    context_id: str,
    context_tokens: Iterable[str],
    store: float,
    reward_fn,
    k: int = 128
) -> Tuple[BanditAction, float, float]:
    """
    Perform one hybrid bandit interaction:
        1. Select an action using the fused geometric score.
        2. Observe reward via `reward_fn(action_id)`.
        3. Update policy statistics and the store.
    Returns (chosen_action, observed_reward, updated_store).
    """
    chosen, ε = select_action(actions, context_tokens, store, k)

    # Obtain reward (external black‑box)
    reward = float(reward_fn(chosen.action_id))

    # Update bandit statistics
    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    update_policy([update])

    # Store dynamics: inflow proportional to reward, outflow proportional to ε
    inflow = [reward]
    outflow = [ε]
    new_store = update_store(store, inflow, outflow)

    return chosen, reward, new_store

# ----------------------------------------------------------------------
# Simple reward stub for testing
# ----------------------------------------------------------------------
def _dummy_reward_factory(mapping: Dict[str, float]):
    """Creates a deterministic reward function from a dict."""
    def reward_fn(action_id: str) -> float:
        return mapping.get(action_id, 0.0)
    return reward_fn

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define two actions with arbitrary initial statistics
    actions = [
        BanditAction(action_id="A", propensity=0.6, expected_reward=1.0, confidence_bound=0.2),
        BanditAction(action_id="B", propensity=0.4, expected_reward=0.8, confidence_bound=0.3),
    ]

    # Context textual data
    context_text = "the quick brown fox jumps over the lazy dog"
    tokens = context_text.lower().split()

    # Initial store (neutral)
    store = 0.0

    # Deterministic rewards for reproducibility
    reward_map = {"A": 1.0, "B": 0.0}
    reward_fn = _dummy_reward_factory(reward_map)

    # Run a few steps
    for step in range(5):
        chosen, reward, store = hybrid_bandit_step(
            actions,
            context_id=f"ctx_{step}",
            context_tokens=tokens,
            store=store,
            reward_fn=reward_fn,
            k=64,
        )
        print(f"Step {step}: chose {chosen.action_id}, reward={reward:.2f}, store={store:.2f}")

    # Show final policy statistics
    print("\nPolicy statistics:")
    for aid, (total, cnt) in _POLICY.items():
        print(f"  Action {aid}: total_reward={total:.2f}, count={cnt:.0f}, avg={_reward(aid):.2f}")