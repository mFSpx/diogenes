# DARWIN HAMMER — match 3126, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_fracti_hybrid_hybrid_regret_m1714_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1931_s3.py (gen6)
# born: 2026-05-29T23:47:57Z

"""Hybrid Algorithm integrating:
- Parent A: hyperdimensional MinHash signatures + NLMS adaptive filtering.
- Parent B: regret‑bandit scoring + MinHash similarity‑modulated updates.

Mathematical bridge:
Both parents expose a MinHash similarity measure.  In this fusion the similarity
`s = Sim(sig_i, sig_ref)` is used (i) to scale the NLMS step‑size μ → μ·(1+s)
and (ii) to weight the regret‑bandit score contribution.  The hidden state
vector `h` (from the NLMS filter) is concatenated with a ternary hygiene vector
`t ∈ {‑1,0,1}^d` to form a hybrid representation `v = [h; t]`.  The adaptive
update of the weight matrix `W` therefore follows

    e   = d – v·Wᵀ
    μ′  = μ·(1 + s)                     # similarity‑scaled step size
    W←  W + (μ′/(ε+‖v‖²))·e·v

and the regret‑bandit score for each action `a` is

    r_a = (E_a – E_ref)·(1 + s_a)

where `E` denotes the expected value of the action and `s_a` the MinHash
similarity to a reference action.  The three core functions below implement
this combined dynamics.""" 

import sys
import math
import random
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import List, Iterable, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash using Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> np.ndarray:
    """Compute a MinHash signature for a set of tokens."""
    if num_perm <= 0:
        raise ValueError("num_perm must be positive")
    # initialise with max 64‑bit value
    sig = np.full(num_perm, (1 << 64) - 1, dtype=np.uint64)
    for token in tokens:
        for i in range(num_perm):
            h = _hash(i, token)
            if h < sig[i]:
                sig[i] = h
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Average agreement between two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    return float(np.mean(sig1 == sig2))


def ternary_vector(dim: int, seed: int = None) -> np.ndarray:
    """Generate a random ternary vector (elements ∈ {‑1,0,1})."""
    rng = random.Random(seed)
    return np.array([rng.choice((-1, 0, 1)) for _ in range(dim)], dtype=np.int8)


# ----------------------------------------------------------------------
# Domain data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Descriptor for an action used in regret‑bandit scoring."""
    token: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def nlms_step(
    weights: np.ndarray,
    input_vec: np.ndarray,
    desired: float,
    base_mu: float = 0.1,
    epsilon: float = 1e-8,
    similarity: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single Normalised LMS update.
    The step‑size is scaled by (1 + similarity) where similarity ∈ [0,1].
    Returns the updated weight vector and the instantaneous error.
    """
    # ensure column vector for matrix ops
    x = input_vec.reshape(-1, 1)                     # (d,1)
    w = weights.reshape(1, -1)                       # (1,d)
    y = float(w @ x)                                 # scalar prediction
    e = desired - y                                  # error
    mu_eff = base_mu * (1.0 + similarity)            # similarity‑scaled step
    norm_factor = epsilon + float(x.T @ x)           # scalar denominator
    w_new = w + (mu_eff / norm_factor) * e * x.T     # (1,d)
    return w_new.ravel(), e


def regret_bandit_scores(
    actions: List[MathAction],
    reference: MathAction,
    num_perm: int = 128,
) -> np.ndarray:
    """
    Compute regret‑bandit scores for a list of actions.
    Score for action a:
        r_a = (E_a – E_ref) * (1 + s_a)
    where s_a is the MinHash similarity between the token of a and the token of the reference.
    """
    ref_sig = minhash_signature([reference.token], num_perm)
    scores = np.empty(len(actions), dtype=np.float64)
    for i, act in enumerate(actions):
        act_sig = minhash_signature([act.token], num_perm)
        sim = minhash_similarity(act_sig, ref_sig)   # ∈ [0,1]
        scores[i] = (act.expected_value - reference.expected_value) * (1.0 + sim)
    return scores


def hybrid_update(
    actions: List[MathAction],
    hidden_state: np.ndarray,
    ternary_dim: int,
    weights: np.ndarray,
    desired_output: float,
    base_mu: float = 0.1,
    num_perm: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a full hybrid iteration:
      1. Build a ternary hygiene vector.
      2. Concatenate hidden state with ternary vector → hybrid representation v.
      3. Compute MinHash similarity of each action token to a reference (first action).
      4. Use the average similarity as the scaling factor for NLMS.
      5. Update the NLMS weights with the scaled step‑size.
      6. Return updated weights and regret‑bandit scores.
    """
    if not actions:
        raise ValueError("actions list must not be empty")
    # 1. ternary vector
    t_vec = ternary_vector(ternary_dim, seed=42)
    # 2. hybrid representation
    v = np.concatenate([hidden_state.astype(np.float64), t_vec.astype(np.float64)])
    # 3. similarities to the first action (reference)
    ref_action = actions[0]
    sims = np.empty(len(actions), dtype=np.float64)
    ref_sig = minhash_signature([ref_action.token], num_perm)
    for i, act in enumerate(actions):
        act_sig = minhash_signature([act.token], num_perm)
        sims[i] = minhash_similarity(act_sig, ref_sig)
    # 4. use mean similarity to modulate NLMS step size
    mean_sim = float(np.mean(sims))
    # 5. NLMS weight update
    weights, _ = nlms_step(
        weights,
        v,
        desired_output,
        base_mu=base_mu,
        similarity=mean_sim,
    )
    # 6. regret‑bandit scores
    scores = regret_bandit_scores(actions, ref_action, num_perm=num_perm)
    return weights, scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic scenario
    actions = [
        MathAction(token="alpha", expected_value=1.2),
        MathAction(token="beta", expected_value=0.8),
        MathAction(token="gamma", expected_value=1.5),
    ]
    hidden = np.random.randn(5)          # hidden state dimension 5
    dim_ternary = 4                       # ternary hygiene dimension
    init_weights = np.zeros(hidden.size + dim_ternary)  # start from zero
    desired = 0.0                         # target for NLMS (e.g., zero error)

    new_weights, bandit_scores = hybrid_update(
        actions=actions,
        hidden_state=hidden,
        ternary_dim=dim_ternary,
        weights=init_weights,
        desired_output=desired,
        base_mu=0.05,
        num_perm=64,
    )

    print("Updated weights:", new_weights)
    print("Regret‑bandit scores:", bandit_scores)
    sys.exit(0)