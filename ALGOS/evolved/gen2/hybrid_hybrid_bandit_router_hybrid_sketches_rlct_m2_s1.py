# DARWIN HAMMER — match 2, survivor 1
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s4.py (gen1)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:22:19Z

"""
Module for the hybrid bandit-router and sketch-RLCT algorithm.

This module combines the bandit-router algorithm from 'hybrid_bandit_router_honeybee_store_m9_s4.py'
and the sketch-RLCT algorithm from 'hybrid_sketches_rlct_grokking_m5_s1.py' by finding a mathematical 
interface between their structures. The bandit-router algorithm uses a Count-Min sketch to estimate the 
empirical log-likelihood sum, while the sketch-RLCT algorithm uses a HyperLogLog estimate of distinct 
tokens to provide a cheap proxy for the effective number of activation patterns that influences the 
RLCT λ. The combined quantities feed the free-energy asymptotic and the RLCT regression.

The mathematical bridge between the two algorithms is the use of log-count statistics. The bandit-router 
algorithm uses the log-count statistics to estimate the expected reward of each action, while the 
sketch-RLCT algorithm uses the log-count statistics to estimate the empirical log-likelihood sum and 
the effective number of activation patterns.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from collections import defaultdict

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

def count_min_sketch(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % width
            table[d][idx] += 1
    return table

def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Very lightweight distinct‑count estimator (exact for small sets)."""
    return len(set(items))

def minhash_lsh_index(docs: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """MinHash‑based LSH bucket index."""
    buckets: defaultdict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min(
            (
                hashlib.sha1(s.encode()).hexdigest()[:6]
                for s in shingles
            ),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    gamma: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    store_factor = 1.0 + store / (store + 1.0)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        def sample(a: str) -> float:
            r = _reward(a)
            n = _count(a)
            a_param = 1.0 + max(0.0, r) * store_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * store_factor
            return rng.betavariate(a_param, b_param)

        chosen = max(actions, key=sample)
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0

        def ucb_score(a: str) -> float:
            mean = _reward(a)
            cnt = _count(a)
            conf = store_factor / math.sqrt(1.0 + cnt)
            return mean + eta * scale * conf + gamma * store_factor

        chosen = max(actions, key=ucb_score)

    prop = 1.0 / len(actions)
    exp_reward = _reward(chosen)
    conf_bound = store_factor / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=prop,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )

def estimate_rlct_from_losses(
    train_losses_per_n: Iterable[float], n_values: Iterable[int]
) -> float:
    """Linear regression of log(loss) vs log(log(n))."""
    losses = np.asarray(list(train_losses_per_n), dtype=np.float64)
    ns = np.asarray(list(n_values), dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

def build_hybrid_sketch(
    docs: Dict[str, Set[str]],
    width: int = 128,
    depth: int = 5,
) -> Tuple[List[List[int]], int, Dict[str, List[str]]]:
    """
    Construct the three sketch structures from a corpus.

    Returns
    -------
    cms : Count‑Min sketch of all tokens (flattened across documents)
    hll_est : HyperLogLog estimate of distinct tokens
    lsh   : MinHash LSH index mapping bucket keys to document IDs
    """
    # Flatten all tokens for the count‑min sketch and HLL
    all_tokens = (token for shingles in docs.values() for token in shingles)
    cms = count_min_sketch(all_tokens, width=width, depth=depth)

    # Re‑iterate tokens for distinct count 
    hll_est = hyperloglog_cardinality(all_tokens)

    # MinHash LSH bucket index
    lsh = minhash_lsh_index(docs)

    return cms, hll_est, lsh

def hybrid_bandit_step(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    true_reward_fn,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    gamma: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float, float]:
    action = hybrid_select_action(context, actions, store, algorithm, epsilon, eta, gamma, seed)
    reward = true_reward_fn(action.action_id)
    update_policy([BanditUpdate(context.get("id", ""), action.action_id, reward, action.propensity)])
    new_store, _ = update_store(store, [reward], [0.0], alpha, beta, dt)
    return action, reward, new_store

def hybrid_rlct_estimate(
    cms: List[List[int]],
    hll_est: int,
    lsh: Dict[str, List[str]],
    train_losses_per_n: Iterable[float],
    n_values: Iterable[int],
) -> float:
    # Estimate RLCT from losses
    rlct_est = estimate_rlct_from_losses(train_losses_per_n, n_values)

    # Use the sketch-based loss curve and evaluate the asymptotic free energy
    # For simplicity, assume the loss curve is a linear function of the log-counts
    log_counts = [math.log(sum(row)) for row in cms]
    loss_curve = np.polyfit(log_counts, train_losses_per_n, 1)

    # Evaluate the asymptotic free energy
    n = hll_est
    L0 = loss_curve[0]
    lambda_rlct = rlct_est
    m = 1
    free_energy = n * L0 + lambda_rlct * math.log(n) - (m - 1) * math.log(math.log(n))

    return free_energy

if __name__ == "__main__":
    # Test the hybrid bandit step
    context = {"id": "context1", "feature1": 0.5, "feature2": 0.3}
    actions = ["action1", "action2"]
    store = 0.0
    def true_reward_fn(action_id: str) -> float:
        return np.random.uniform(0, 1)
    action, reward, new_store = hybrid_bandit_step(context, actions, store, true_reward_fn)
    print(action, reward, new_store)

    # Test the hybrid RLCT estimate
    docs = {"doc1": {"token1", "token2"}, "doc2": {"token2", "token3"}}
    cms, hll_est, lsh = build_hybrid_sketch(docs)
    train_losses_per_n = [0.1, 0.2, 0.3]
    n_values = [10, 20, 30]
    free_energy = hybrid_rlct_estimate(cms, hll_est, lsh, train_losses_per_n, n_values)
    print(free_energy)