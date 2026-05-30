# DARWIN HAMMER — match 13, survivor 0
# gen: 2
# parent_a: regret_engine.py (gen0)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s0.py (gen1)
# born: 2026-05-29T23:22:31Z

#!/usr/bin/env python3
"""Hybrid Regret-Weighted Liquid Time-Constant MinHash (RW-LTC-MH) Networks.

This module integrates the Regret-Weighted strategy from regret_engine.py with the Hybrid Liquid Time-Constant MinHash Networks from hybrid_liquid_time_constant_minhash_m10_s0.py.
The mathematical bridge between these two structures lies in the application of MinHash to the hidden state of the Regret-Weighted strategy, effectively projecting the strategy's decision-making process onto a discrete, hash-based space.
The governing equation of the Regret-Weighted strategy remains unchanged, but the network function now incorporates a MinHash-based similarity metric between the current input and a set of reference inputs, modulating the synaptic drive term in the strategy.

"""

import numpy as np
from dataclasses import dataclass
from typing import Iterable
import hashlib
import math
import random
import sys
import pathlib

@dataclass(frozen=True)
class MathAction:
    id: str; expected_value: float; cost: float=0.0; risk: float=0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str; outcome_value: float; probability: float=1.0

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def signature(tokens: Iterable[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> dict[str,float]:
    if not actions: return {}
    cf={c.action_id:c.outcome_value*c.probability for c in counterfactuals}
    vals={a.id:a.expected_value-a.cost-a.risk+cf.get(a.id,0.0) for a in actions}
    best=max(vals.values()); w={k:math.exp(v-best) for k,v in vals.items()}; total=sum(w.values()) or 1.0
    return {k:v/total for k,v in w.items()}

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    reference_inputs: list[np.ndarray],
    k: int = 128,
) -> np.ndarray:
    concat = np.concatenate([x, I], axis=0)
    hidden_state_hash = signature([str(val) for val in x], k=k)
    similarity_values = [similarity(hidden_state_hash, signature([str(val) for val in ref_input], k=k)) for ref_input in reference_inputs]
    similarity_value = np.mean(similarity_values)
    return sigmoid(W @ concat + b) * similarity_value

def regret_weighted_ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    reference_inputs: list[np.ndarray],
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]
    
    strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    f_val = ltc_f(x, I, W, b, reference_inputs)

    dx_dt = -(1.0 / tau + f_val) * x + f_val * A

    x_new = x + dt * dx_dt

    tau_sys_vec = tau / (1.0 + tau * f_val)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys

def regret_weighted_ltc_forward(
    I_seq: np.ndarray,
    params: dict,
    reference_inputs: list[np.ndarray],
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
    x0: np.ndarray | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    T, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim))
    tau_sys_seq = np.empty(T)

    for t in range(T):
        x, tau_sys = regret_weighted_ltc_step(x, I_seq[t], params, reference_inputs, actions, counterfactuals, dt=dt)
        X[t] = x
        tau_sys_seq[t] = tau_sys

    return X, tau_sys_seq

def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value-a.cost-a.risk), a.id))

if __name__ == "__main__":
    hidden_dim = 10
    input_dim = 5
    tau = 1.0
    seed = 0
    np.random.seed(seed)
    W = np.random.rand(hidden_dim, hidden_dim + input_dim)
    b = np.random.rand(hidden_dim)
    A = np.random.rand(hidden_dim)
    params = {"W": W, "b": b, "tau": tau, "A": A}
    I_seq = np.random.rand(100, input_dim)
    reference_inputs = [np.random.rand(input_dim) for _ in range(10)]
    actions = [MathAction(str(i), np.random.rand(), np.random.rand(), np.random.rand()) for i in range(10)]
    counterfactuals = [MathCounterfactual(str(i), np.random.rand(), np.random.rand()) for i in range(10)]
    X, tau_sys_seq = regret_weighted_ltc_forward(I_seq, params, reference_inputs, actions, counterfactuals)
    print(X.shape, tau_sys_seq.shape)