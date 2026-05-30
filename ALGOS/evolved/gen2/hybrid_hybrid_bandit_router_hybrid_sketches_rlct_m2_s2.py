# DARWIN HAMMER — match 2, survivor 2
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s4.py (gen1)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:22:19Z

"""Hybrid Bandit‑Sketch RLCT Module

Parents
-------
- **Parent A**: ``hybrid_bandit_router_honeybee_store_m9_s4.py`` – a contextual
  multi‑armed bandit with a “store” that accumulates reward and influences the
  confidence bound via a simple scaling factor.
- **Parent B**: ``hybrid_sketches_rlct_grokking_m5_s1.py`` – sketch primitives
  (Count‑Min, HyperLogLog, MinHash) together with singular‑learning‑theory
  utilities (BIC, WAIC, RLCT estimation, free‑energy asymptotics).

Mathematical Bridge
-------------------
Both families manipulate **log‑count statistics**:

* The bandit’s *store* is a cumulative (linear) function of received rewards.
  If we treat the reward sequence as a data set, the **log‑likelihood**
  contribution of that data can be approximated by a Count‑Min sketch of the
  reward frequencies.  The sketch therefore supplies a cheap estimate of the
  empirical mean reward `\hat{r}` and its variance, which replaces the naïve
  averaging used in Parent A.

* The RLCT (real log‑canonical threshold) formulas of Parent B contain a term
  `λ·log(n)`.  The effective sample size `n` is the number of **distinct
  contexts** observed by the bandit.  A HyperLogLog sketch provides a fast
  estimate of that cardinality, yielding a data‑driven λ that can be fed back
  into the store dynamics.

The hybrid algorithm therefore:

1. **Sketches** the reward stream per action with a Count‑Min sketch.
2. **Estimates** the number of distinct contexts with a HyperLogLog sketch.
3. **Derives** an RLCT estimate from the loss curve (negative reward) using the
   regression routine of Parent B.
4. **Injects** the RLCT‑derived term into the store update and the confidence
   bound used for action selection.

The result is a unified system where exploration‑exploitation balances are
guided by both statistical sketching and singular‑learning‑theory asymptotics."""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Callable

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Sketch primitives (adapted from Parent B)
# ----------------------------------------------------------------------


def count_min_sketch(
    items: Iterable[int], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch for integer‑valued items (e.g. discretised rewards)."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % width
            table[d][idx] += 1
    return table


def count_min_query(sketch: List[List[int]], item: int) -> int:
    """Return the minimum count estimate for *item*."""
    return min(
        sketch[d][int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % len(sketch[d])]
        for d in range(len(sketch))
    )


def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Lightweight distinct‑count estimator (exact for small sets)."""
    return len(set(items))


# ----------------------------------------------------------------------
# Singular‑learning‑theory utilities (adapted from Parent B)
# ----------------------------------------------------------------------


def estimate_rlct_from_losses(
    train_losses_per_n: Iterable[float], n_values: Iterable[int]
) -> float:
    """Linear regression of log(loss) vs log(log(n))."""
    losses = np.asarray(list(train_losses_per_n), dtype=np.float64)
    ns = np.asarray(list(n_values), dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def free_energy_asymptotic(
    n: float, L0: float, lambda_rlct: float, m: int = 1
) -> float:
    """Asymptotic Bayesian free energy."""
    return n * L0 + lambda_rlct * math.log(n) - (m - 1) * math.log(math.log(n))


# ----------------------------------------------------------------------
# Bandit core (adapted from Parent A) – now driven by sketches & RLCT
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


# Global structures ----------------------------------------------------


_POLICY: Dict[str, Tuple[float, int]] = {}  # action_id -> (cumulative_reward, count)
_REWARD_SKETCHES: Dict[str, List[List[int]]] = {}  # action_id -> Count‑Min sketch
_CONTEXT_SEEN: Set[str] = set()  # distinct context identifiers for HLL


def reset_policy() -> None:
    _POLICY.clear()
    _REWARD_SKETCHES.clear()
    _CONTEXT_SEEN.clear()


def _reward(action: str) -> float:
    total, cnt = _POLICY.get(action, (0.0, 0))
    return total / cnt if cnt else 0.0


def _count(action: str) -> int:
    return _POLICY.get(action, (0.0, 0))[1]


def _update_sketch(action: str, reward: float, bins: int = 100) -> None:
    """Discretise *reward* into *bins* and update the Count‑Min sketch."""
    # Map reward ∈ [0,1] → integer bin
    bin_id = int(min(max(reward, 0.0), 1.0) * (bins - 1))
    sketch = _REWARD_SKETCHES.setdefault(action, count_min_sketch([], width=128, depth=5))
    # Increment the sketch in‑place
    for d in range(len(sketch)):
        idx = int(hashlib.sha256(f"{d}:{bin_id}".encode()).hexdigest(), 16) % len(sketch[d])
        sketch[d][idx] += 1


def update_policy(updates: List[BanditUpdate]) -> None:
    """Update both the exact statistics and the sketches."""
    for u in updates:
        # exact stats
        total, cnt = _POLICY.get(u.action_id, (0.0, 0))
        _POLICY[u.action_id] = (total + u.reward, cnt + 1)

        # sketch update
        _update_sketch(u.action_id, u.reward)

        # record distinct context id for HLL estimate
        _CONTEXT_SEEN.add(u.context_id)


def estimate_expected_reward_via_sketch(action: str, bins: int = 100) -> float:
    """Use the Count‑Min sketch to approximate the mean reward for *action*."""
    sketch = _REWARD_SKETCHES.get(action)
    if not sketch:
        return 0.0
    # Reconstruct an approximate histogram from the sketch by querying each bin
    width = len(sketch[0])
    counts = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        # query the sketch for this bin
        counts[b] = count_min_query(sketch, b)
    total_counts = counts.sum()
    if total_counts == 0:
        return 0.0
    # bin centre as reward approximation
    bin_centres = (np.arange(bins) + 0.5) / bins
    return float((counts * bin_centres).sum() / total_counts)


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
    """Select an action using a UCB‑like score that now incorporates:
    - sketch‑based reward estimate,
    - RLCT‑derived term via distinct‑context count,
    - the original store factor.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Store‑derived scaling (same as Parent A)
    store_factor = 1.0 + store / (store + 1.0)

    # RLCT‑derived scaling: λ̂ ≈ log(distinct_contexts)
    distinct = hyperloglog_cardinality(_CONTEXT_SEEN) or 1
    lambda_hat = math.log(distinct)

    # Composite scaling that blends both influences
    composite_factor = store_factor + 0.05 * lambda_hat

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Thompson sampling using sketch‑based beta approximation
        def sample(a: str) -> float:
            r = estimate_expected_reward_via_sketch(a)
            n = _count(a) or 1
            a_param = 1.0 + max(0.0, r) * composite_factor
            b_param = 1.0 + max(0.0, 1.0 - r) * composite_factor
            return rng.betavariate(a_param, b_param)

        chosen = max(actions, key=sample)
    else:
        # LinUCB‑style score
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0

        def ucb_score(a: str) -> float:
            mean = estimate_expected_reward_via_sketch(a)
            cnt = _count(a)
            conf = composite_factor / math.sqrt(1.0 + cnt)
            return mean + eta * scale * conf + gamma * composite_factor

        chosen = max(actions, key=ucb_score)

    prop = 1.0 / len(actions)
    exp_reward = estimate_expected_reward_via_sketch(chosen)
    conf_bound = composite_factor / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=prop,
        expected_reward=exp_reward,
        confidence_bound=conf_bound,
        algorithm=algorithm,
    )


def hybrid_step(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    true_reward_fn: Callable[[str], float],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    eta: float = 0.1,
    gamma: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
    seed: int | str | None = 7,
) -> Tuple[BanditAction, float, float]:
    """One interaction step:
    1. select an action,
    2. observe reward,
    3. update policy & sketches,
    4. update the store using an RLCT‑aware term.
    """
    action = hybrid_select_action(
        context, actions, store, algorithm, epsilon, eta, gamma, seed
    )
    reward = true_reward_fn(action.action_id)

    update_policy(
        [
            BanditUpdate(
                context_id=context.get("id", ""),
                action_id=action.action_id,
                reward=reward,
                propensity=action.propensity,
            )
        ]
    )

    # RLCT term derived from loss curve (negative reward) – we keep a tiny history
    # for the regression.  For simplicity we use the cumulative store as a proxy
    # for sample size n.
    n = max(1, int(store + 1))
    lambda_rlct = estimate_rlct_from_losses([max(1e-9, -reward)], [n])

    # Store update now includes the RLCT contribution
    delta = alpha * reward - beta * lambda_rlct
    new_store = max(0.0, store + dt * delta)

    return action, reward, new_store


def hybrid_rlct_estimate(
    loss_history: List[float],
    n_history: List[int],
) -> float:
    """Public wrapper that returns an RLCT estimate from accumulated losses."""
    return estimate_rlct_from_losses(loss_history, n_history)


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


def _example_true_reward(action_id: str) -> float:
    """Synthetic reward: higher for actions ending with an odd digit."""
    base = {"action1": 0.7, "action2": 0.4, "action3": 0.9}
    return base.get(action_id, random.random())


def main() -> None:
    random.seed(0)
    np.random.seed(0)

    context = {"id": "ctx0", "f1": 0.2, "f2": 0.8}
    actions = ["action1", "action2", "action3"]
    store = 0.0

    loss_hist: List[float] = []
    n_hist: List[int] = []

    for t in range(15):
        act, rew, store = hybrid_step(
            context,
            actions,
            store,
            _example_true_reward,
            algorithm="linucb",
            epsilon=0.2,
            eta=0.2,
            gamma=0.05,
            seed=t,
        )
        loss_hist.append(-rew)  # negative reward as loss
        n_hist.append(t + 1)

        # Occasionally print diagnostics
        if (t + 1) % 5 == 0:
            rlct = hybrid_rlct_estimate(loss_hist, n_hist)
            print(
                f"t={t+1:2d} | action={act.action_id} reward={rew:.3f} store={store:.3f} "
                f"RLCT≈{rlct:.3f}"
            )

    # Final sanity check: the sketches should contain entries for each action
    for a in actions:
        est = estimate_expected_reward_via_sketch(a)
        print(f"Sketch‑based reward estimate for {a}: {est:.3f}")


if __name__ == "__main__":
    main()