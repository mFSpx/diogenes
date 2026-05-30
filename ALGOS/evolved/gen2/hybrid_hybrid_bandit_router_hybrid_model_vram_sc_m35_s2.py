# DARWIN HAMMER — match 35, survivor 2
# gen: 2
# parent_a: hybrid_bandit_router_honeybee_store_m9_s1.py (gen1)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (gen1)
# born: 2026-05-29T23:23:32Z

"""Hybrid Bandit‑TTT Scheduler

This module fuses two distinct parents:

* **Parent A** – a contextual multi‑armed bandit router that emits a
  `BanditAction` containing a *propensity* (interpreted as an inflow rate)
  and a *confidence_bound* (interpreted as an outflow rate).

* **Parent B** – a linear “TTT” (transform‑then‑transform) model that updates
  a weight matrix `W` using gradient descent and provides a VRAM‑size
  estimator for that matrix.

The mathematical bridge is the **store equation** of the honeybee
primitive:


Δstore = α·Σ(inflow) – β·Σ(outflow)
storeₜ₊₁ = max(0, storeₜ + Δstore·dt)


We treat the bandit‐produced `propensity` as an **inflow** of virtual
memory and the `confidence_bound` as an **outflow**.  Simultaneously the
bandit action modulates the learning rate of the TTT update:


η = η₀·(1 + propensity)


Thus a single `BanditAction` drives both the *resource‑store* dynamics
and the *matrix‑learning* dynamics, yielding a unified hybrid system."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Bandit core (Parent A)
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


_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}          # virtual VRAM store per key


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action and return its BanditAction descriptor."""
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta‑Bernoulli posterior with pseudo‑counts derived from rewards
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a)),
                1 + max(0, 1 - _reward(a)),
            ),
        )
    else:  # linucb‑style surrogate
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=_reward(chosen),
        confidence_bound=confidence,
        algorithm=algorithm,
    )


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate observed rewards into the bandit statistics."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Store dynamics (bridge between parents)
# ----------------------------------------------------------------------
def update_vram_store(
    key: str,
    inflow: float,
    outflow: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    """
    Update the virtual VRAM store for ``key`` using the bandit‑derived
    inflow/outflow rates.
    Returns (new_store, delta).
    """
    store = _STORE.get(key, 0.0)
    delta = alpha * inflow - beta * outflow
    new_store = max(0.0, store + dt * delta)
    _STORE[key] = new_store
    return new_store, delta


# ----------------------------------------------------------------------
# TTT linear core (Parent B)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float = 0.01, target: np.ndarray | None = None) -> np.ndarray:
    """One gradient‑descent step of the TTT model."""
    g = ttt_grad(W, x, target)
    return W - eta * g


def weight_matrix_memory_mb(W: np.ndarray, dtype: np.dtype = np.float64) -> int:
    """Memory footprint of a weight matrix in megabytes."""
    element_bytes = np.dtype(dtype).itemsize
    total_bytes = W.size * element_bytes
    return int(round(total_bytes / (1024 * 1024), 0))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_ttt_step(
    W: np.ndarray,
    x: np.ndarray,
    action: BanditAction,
    base_eta: float = 0.01,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform a TTT step where the learning rate is scaled by the bandit
    propensity: η = base_eta·(1 + propensity).
    """
    eta = base_eta * (1.0 + action.propensity)
    return ttt_step(W, x, eta=eta, target=target)


def hybrid_process_sequence(
    context: Dict[str, float],
    actions: List[str],
    x_seq: np.ndarray,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    base_eta: float = 0.01,
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    End‑to‑end hybrid run:

    1. Select a bandit action.
    2. Update the virtual VRAM store using its propensity/outflow.
    3. Run the TTT sequence on ``x_seq`` with a learning‑rate modulated
       by the same action.
    4. Return (hidden_states, final_W, new_store, store_delta).
    """
    # 1. Bandit decision
    bandit_action = select_action(context, actions, algorithm, epsilon, seed)

    # 2. Store update (use first action id as key for simplicity)
    store_key = actions[0]
    new_store, delta = update_vram_store(
        store_key,
        inflow=bandit_action.propensity,
        outflow=bandit_action.confidence_bound,
        alpha=alpha,
        beta=beta,
        dt=dt,
    )

    # 3. TTT sequence with adaptive learning rate
    # Initialise weight matrix once (size compatible with x_seq)
    d_in = x_seq.shape[1]
    W = init_ttt(d_in, seed=seed if isinstance(seed, int) else 0)

    T = x_seq.shape[0]
    H = np.empty((T, W.shape[0]), dtype=float)

    for t in range(T):
        # each step uses the same bandit action (could be refreshed per step)
        W = hybrid_ttt_step(W, x_seq[t], bandit_action, base_eta=base_eta)
        H[t] = W @ x_seq[t]

    return H, W, new_store, delta


def report_vram_plan(W: np.ndarray, reserve_mb: int = 768) -> Dict[str, Any]:
    """
    Produce a simple VRAM‑allocation plan based on the matrix size and the
    current virtual store. Mirrors Parent B's ``plan_ttt_residency`` but
    stays self‑contained.
    """
    needed = weight_matrix_memory_mb(W)
    available = DEFAULT_BUDGET_MB - reserve_mb
    status = "fit" if needed <= available else "exceed"
    return {
        "matrix_mb": needed,
        "available_mb": available,
        "status": status,
        "store_mb": _STORE.get("action1", 0.0),
    }


# ----------------------------------------------------------------------
# Constants used by the VRAM plan helper
# ----------------------------------------------------------------------
DEFAULT_BUDGET_MB = int(
    os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096") if (os := __import__("os")) else "4096"
)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise
    reset_policy()
    actions = ["action1", "action2", "action3"]
    context = {"feature_a": 0.7, "feature_b": 0.2}
    # Random input sequence: 5 timesteps, 4‑dimensional
    rng = np.random.default_rng(42)
    x_seq = rng.standard_normal((5, 4))

    # Run hybrid pipeline
    H, W_final, store, delta = hybrid_process_sequence(
        context,
        actions,
        x_seq,
        algorithm="linucb",
        epsilon=0.05,
        seed=123,
        base_eta=0.02,
        alpha=1.2,
        beta=0.8,
        dt=1.0,
    )

    # Report
    plan = report_vram_plan(W_final)
    print("Hidden states (H):\n", H)
    print("\nFinal weight matrix shape:", W_final.shape)
    print("\nVirtual VRAM store:", store, "(Δ =", delta, ")")
    print("\nVRAM plan:", plan)
    sys.exit(0)