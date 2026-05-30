# DARWIN HAMMER — match 4408, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1216_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_korpus_text_h_m802_s0.py (gen5)
# born: 2026-05-29T23:55:24Z

"""Hybrid Algorithm: Fusion of Bandit Store Update (Parent A) and Sinusoidal Weekday Allocation (Parent B)

Mathematical Bridge
-------------------
Parent A provides a *Bandit‑Router* whose store state ``S`` is updated by a
linear combination of observed rewards and a *signature vector* ϕ∈ℝⁿ
derived from stylometry features.  
Parent B supplies a *weekday‑based sinusoidal weight vector* w(d)∈ℝⁿ
that rotates over the seven days and a VRAM‑aware GPU selector.

The hybrid unifies these structures by projecting the signature ϕ onto the
weekday simplex via an element‑wise product

    a = normalize( w(d) ⊙ ϕ )                (1)

where ⊙ denotes the Hadamard product and *normalize* scales the result to a
row‑stochastic vector.  Vector ``a`` simultaneously encodes (i) the
resource‑allocation preference of the day (Parent B) and (ii) the
task‑specific importance extracted by the signature (Parent A).

Equation (1) supplies the *allocation vector* used in two places:

* **Store update** – the bandit store level ``S`` is driven by the weighted
  reward ``r`` using ``a`` as a gain factor.
* **GPU selection** – the allocation weights bias the choice of GPUs that
  satisfy the VRAM budget.

The following module implements this fused dynamics together with three
core functions that illustrate the hybrid operation."""


import sys
import math
import random
import pathlib
import datetime as dt
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared by both parents)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768


# ----------------------------------------------------------------------
# Parent A data structures (Bandit‑Router)
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
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0
    alpha: float = 0.1          # learning‑rate like factor
    last_update: float = 0.0    # timestamp (seconds since epoch)


# ----------------------------------------------------------------------
# Parent B utilities (weekday sinusoid & VRAM selector)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> List[Dict[str, Any]]:
    """
    Select GPUs that have sufficient VRAM to meet the budget and reserve
    requirements.
    """
    selected = [
        gpu for gpu in gpus if gpu.get("memory.free", 0) >= budget_mb + reserve_mb
    ]
    return selected


# ----------------------------------------------------------------------
# Hybrid primitives
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used for reproducible signatures."""
    data = seed.to_bytes(8, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def generate_signature(context_id: str, dim: int) -> np.ndarray:
    """
    Produce a pseudo‑signature vector ϕ∈ℝᵈ for a given ``context_id``.
    The implementation uses a deterministic hash to obtain reproducible
    components in the interval (0, 1].
    """
    rng = np.random.default_rng(_hash(0xBEEF, context_id) & MAX64)
    phi = rng.random(dim) + 0.1          # avoid exact zeros
    return phi / phi.sum()               # row‑stochastic


def compute_allocation(
    context_id: str,
    dow: int,
    gpus: List[Dict[str, Any]],
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Fuse Parent A's signature with Parent B's weekday weight vector.
    Returns the allocation vector ``a`` (Eq. 1) and the list of VRAM‑eligible GPUs.
    """
    w = weekday_weight_vector(GROUPS, dow)          # shape (n,)
    phi = generate_signature(context_id, len(GROUPS))  # shape (n,)
    a_raw = w * phi                                 # Hadamard product
    a = a_raw / a_raw.sum()                         # ensure stochasticity
    selected_gpus = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    return a, selected_gpus


def bandit_store_update(
    store: StoreState,
    upd: BanditUpdate,
    allocation: np.ndarray,
) -> StoreState:
    """
    Update the honeybee store using the observed reward and the hybrid
    allocation vector.  The update follows a damped exponential moving average:

        S_{t+1} = (1 - α·⟨a⟩)·S_t + α·⟨a⟩·r

    where ⟨a⟩ denotes the mean of the allocation vector.
    """
    mean_a = float(allocation.mean())
    lr = store.alpha * mean_a
    new_level = (1.0 - lr) * store.level + lr * upd.reward
    store.level = new_level
    store.last_update = dt.datetime.utcnow().timestamp()
    return store


def select_action(
    actions: Sequence[BanditAction],
    allocation: np.ndarray,
) -> BanditAction:
    """
    Choose an action by weighting its ``expected_reward`` with the allocation
    vector.  The action receiving the largest weighted score is returned.
    """
    if not actions:
        raise ValueError("no actions to select from")
    # Broadcast allocation across actions (len(actions) may differ from len(GROUPS))
    # We map groups -> actions by simple round‑robin for demonstration.
    n_groups = allocation.shape[0]
    weights = np.zeros(len(actions))
    for i, act in enumerate(actions):
        grp_idx = i % n_groups
        weights[i] = act.expected_reward * allocation[grp_idx]
    best_idx = int(np.argmax(weights))
    return actions[best_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy GPU inventory
    dummy_gpus = [
        {"id": "gpu0", "memory.free": 8192},
        {"id": "gpu1", "memory.free": 2048},
        {"id": "gpu2", "memory.free": 12288},
    ]

    # Current weekday
    today = dt.date.today()
    dow = doomsday(today.year, today.month, today.day)

    # Compute hybrid allocation
    alloc_vec, eligible = compute_allocation(
        context_id="example_context",
        dow=dow,
        gpus=dummy_gpus,
    )
    print("Allocation vector (per group):", np.round(alloc_vec, 4))
    print("Eligible GPUs:", [g["id"] for g in eligible])

    # Initialise store
    store = StoreState(level=0.5, alpha=0.2)

    # Simulate a bandit observation
    upd = BanditUpdate(
        context_id="example_context",
        action_id="act_42",
        reward=1.7,
        propensity=0.3,
    )
    store = bandit_store_update(store, upd, alloc_vec)
    print("Updated store state:", asdict(store))

    # Define a set of candidate actions
    actions = [
        BanditAction("act_1", 0.2, 0.5, 0.1, "algA"),
        BanditAction("act_2", 0.3, 0.8, 0.15, "algA"),
        BanditAction("act_3", 0.5, 0.3, 0.05, "algA"),
        BanditAction("act_4", 0.1, 0.9, 0.2, "algA"),
    ]

    chosen = select_action(actions, alloc_vec)
    print("Chosen action:", asdict(chosen))