# DARWIN HAMMER — match 13, survivor 2
# gen: 2
# parent_a: regret_engine.py (gen0)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s0.py (gen1)
# born: 2026-05-29T23:22:31Z

"""Hybrid Regret-Weighted Liquid Time‑Constant MinHash (RW‑LTC‑MH) Module.

Parent A: regret_engine.py – computes a softmax‑like regret‑weighted
distribution over actions using expected value, cost, risk and
counter‑factual outcomes.

Parent B: hybrid_liquid_time_constant_minhash_m10_s0.py – implements a
continuous‑time recurrent network (LTC) whose dynamics are modulated by
a MinHash similarity between the hidden state and a set of reference
inputs.

Mathematical bridge:
Each action is projected into the LTC hidden‑state space by feeding a
compact feature vector (expected value, cost, risk) as the LTC input.
The resulting hidden state `h` is used as an *augmented value*
`v̂ = (EV – cost – risk) + w·h`, where `w` is a learnable weight vector.
Thus the LTC supplies a context‑dependent correction to the raw
expected value before the regret‑weighted softmax is applied.  The
MinHash similarity computed inside the LTC continues to influence the
network’s effective time‑constant, linking the two parent dynamics
into a single unified system.

The module provides:
* `compute_regret_weighted_strategy_hybrid` – hybrid strategy computation.
* `ltc_forward` – LTC dynamics with MinHash similarity.
* `hybrid_forward` – runs LTC over a sequence while updating action
  probabilities at each step using the hybrid regret weighting.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple
import numpy as np

# ---------- Parent A structures ----------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ---------- Parent B utilities ----------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos_mask = x >= 0
    z = np.empty_like(x, dtype=float)
    z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    z[~pos_mask] = np.exp(x[~pos_mask]) / (1.0 + np.exp(x[~pos_mask]))
    return z

# ---------- LTC core ----------
def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    reference_inputs: List[np.ndarray],
    k: int = 128,
) -> np.ndarray:
    """Compute the modulation term f(x,I) = σ(W·[x;I]+b) * similarity."""
    concat = np.concatenate([x, I], axis=0)                # (hidden+input,)
    hidden_hash = signature([str(val) for val in x], k=k)  # MinHash of hidden state
    ref_sigs = [signature([str(val) for val in r], k=k) for r in reference_inputs]
    sims = [similarity(hidden_hash, rs) for rs in ref_sigs]
    similarity_value = float(np.mean(sims)) if sims else 1.0
    return sigmoid(W @ concat + b) * similarity_value

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: Dict[str, np.ndarray],
    reference_inputs: List[np.ndarray],
    dt: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """One Euler integration step of the LTC."""
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b, reference_inputs)               # (hidden,)
    dx_dt = -(1.0 / tau + f_val) * x + f_val * A               # (hidden,)
    x_new = x + dt * dx_dt

    # Effective system time‑constant (scalar) derived from vector f_val
    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def ltc_forward(
    I_seq: np.ndarray,
    params: Dict[str, np.ndarray],
    reference_inputs: List[np.ndarray],
    x0: np.ndarray | None = None,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run LTC over an input sequence, returning hidden states and τ_sys."""
    T, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim), dtype=float)
    tau_seq = np.empty(T, dtype=float)

    for t in range(T):
        x, tau_sys = ltc_step(x, I_seq[t], params, reference_inputs, dt=dt)
        X[t] = x
        tau_seq[t] = tau_sys

    return X, tau_seq

# ---------- Hybrid functions ----------
def action_feature_vector(action: MathAction) -> np.ndarray:
    """Compact feature vector fed to the LTC for a single action."""
    return np.array([action.expected_value, action.cost, action.risk], dtype=float)

def compute_action_hidden(
    action: MathAction,
    params: Dict[str, np.ndarray],
    reference_inputs: List[np.ndarray],
) -> np.ndarray:
    """Project an action into the LTC hidden space via a single step."""
    I = action_feature_vector(action)
    hidden_dim = params["A"].shape[0]
    x0 = np.zeros(hidden_dim, dtype=float)
    h, _ = ltc_step(x0, I, params, reference_inputs, dt=0.1)
    return h

def compute_regret_weighted_strategy_hybrid(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    params: Dict[str, np.ndarray],
    reference_inputs: List[np.ndarray],
    weight_vector: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Hybrid regret‑weighted strategy.
    For each action we obtain an LTC hidden state `h`, compute an
    augmented value `v̂ = (EV – cost – risk) + w·h + cf`,
    then apply a softmax over the regret‑adjusted values.
    """
    if not actions:
        return {}

    # Prepare counter‑factual adjustments
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}

    # Determine weight vector size
    hidden_dim = params["A"].shape[0]
    if weight_vector is None:
        rng = np.random.default_rng(0)
        weight_vector = rng.standard_normal(hidden_dim)

    # Compute augmented values
    vals: Dict[str, float] = {}
    for a in actions:
        h = compute_action_hidden(a, params, reference_inputs)          # (hidden,)
        base = a.expected_value - a.cost - a.risk
        aug = base + float(weight_vector @ h) + cf.get(a.id, 0.0)
        vals[a.id] = aug

    # Regret‑weighted softmax (log‑sum‑exp trick)
    best = max(vals.values())
    w_exp = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w_exp.values()) or 1.0
    return {k: v / total for k, v in w_exp.items()}

def hybrid_forward(
    I_seq: np.ndarray,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    params: Dict[str, np.ndarray],
    reference_inputs: List[np.ndarray],
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Run the LTC over a sequence while recomputing the hybrid regret‑weighted
    strategy after each time step.
    Returns hidden states, τ_sys sequence, and the list of probability
    distributions over actions.
    """
    T, _ = I_seq.shape
    hidden_dim = params["A"].shape[0]
    x = np.zeros(hidden_dim, dtype=float)

    hidden_states = np.empty((T, hidden_dim), dtype=float)
    tau_seq = np.empty(T, dtype=float)
    prob_history: List[Dict[str, float]] = []

    # Fixed weight vector for the whole run
    rng = np.random.default_rng(42)
    w_vec = rng.standard_normal(hidden_dim)

    for t in range(T):
        x, tau_sys = ltc_step(x, I_seq[t], params, reference_inputs, dt=dt)
        hidden_states[t] = x
        tau_seq[t] = tau_sys

        # Update action probabilities using the current hidden state as context
        probs = compute_regret_weighted_strategy_hybrid(
            actions,
            counterfactuals,
            params,
            reference_inputs,
            weight_vector=w_vec,
        )
        prob_history.append(probs)

    return hidden_states, tau_seq, prob_history

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Seed for reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)

    # Define a small set of actions
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="C", expected_value=6.0, cost=0.5, risk=2.0),
    ]

    # Counter‑factuals (optional)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=1.0, probability=0.8),
        MathCounterfactual(action_id="C", outcome_value=-2.0, probability=0.5),
    ]

    # LTC parameters
    hidden_dim = 6
    input_dim = 3                     # matches action_feature_vector length
    tau = 1.5
    W = np.random.rand(hidden_dim, hidden_dim + input_dim)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)
    params = {"W": W, "b": b, "tau": tau, "A": A}

    # Reference inputs for MinHash similarity (randomly generated)
    reference_inputs = [np.random.rand(input_dim) for _ in range(5)]

    # Hybrid strategy without a time sequence
    hybrid_probs = compute_regret_weighted_strategy_hybrid(
        actions, counterfactuals, params, reference_inputs
    )
    print("Hybrid regret‑weighted probabilities:", hybrid_probs)

    # Run LTC forward on a random input sequence and observe evolving probabilities
    T = 20
    I_seq = np.random.rand(T, input_dim)
    hidden_states, tau_seq, prob_history = hybrid_forward(
        I_seq, actions, counterfactuals, params, reference_inputs
    )
    print("Hidden states shape:", hidden_states.shape)
    print("τ_sys sequence shape:", tau_seq.shape)
    print("First 3 probability distributions:", prob_history[:3])