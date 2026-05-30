# DARWIN HAMMER — match 4913, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s1.py (gen2)
# born: 2026-05-29T23:58:43Z

"""Hybrid Allocation with Bandit‑CMS Bridge
This module fuses the core topology of:

* **Parent A** – `hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py`
  (weekday‑dependent weight vector and deterministic/LLM split).

* **Parent B** – `hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s1.py`
  (Count‑Min Sketch estimator feeding a bandit selector).

**Mathematical bridge**
The bridge is the *propensity* vector produced by Parent A (a probability
distribution over groups).  The bandit in Parent B needs an estimate of the
relative importance of each action; we supply that estimate by feeding the
Count‑Min Sketch cardinality of the action stream per group into the bandit.
Thus the bandit’s expected reward for a group is the estimated unique‑action
count, while its propensity comes from the weekday‑weight vector.  The selected
group receives the deterministic share of resources; the remaining share is
distributed proportionally to the original weight vector.

The result is a single allocation routine that respects temporal weighting,
privacy‑preserving cardinality estimation, and adaptive bandit‑driven
prioritisation.
"""

import datetime as dt
import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants and tiny helpers (from Parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round to six decimal places – identical to Parent A."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Day‑of‑week used by Parent A (0 = Monday … 6 = Sunday)."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Generate the sinusoidal weight vector (Parent A)."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (from Parent B)
# ----------------------------------------------------------------------
def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Deterministic hashes for CMS rows."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


def count_min_sketch(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> np.ndarray:
    """
    Build a Count‑Min Sketch matrix for the supplied items.
    Returns a (depth, width) integer matrix.
    """
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        indices = _cms_hash(item, depth, width)
        for d, idx in enumerate(indices):
            cms[d, idx] += 1
    return cms


def estimate_cardinality(cms: np.ndarray) -> float:
    """
    Rough cardinality estimator based on the CMS.
    We use the average of the number of non‑zero counters per row,
    scaled by the total width to approximate unique items.
    """
    depth, width = cms.shape
    nonzero_per_row = (cms > 0).sum(axis=1)
    avg_filled = nonzero_per_row.mean()
    # Scale to full width to obtain an estimate of distinct keys.
    return (avg_filled / width) * width  # simplifies to avg_filled


# ----------------------------------------------------------------------
# Bandit structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridCMSBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: dict[str, List[float]] = {}  # action_id -> [cumulative_reward, count]


def reset_policy() -> None:
    """Clear the global policy store."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate observed rewards into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def _reward(action_id: str) -> float:
    """Mean reward for an action (0 if never seen)."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def _confidence(action_id: str, total_selections: int) -> float:
    """UCB‑style confidence term."""
    _, n = _POLICY.get(action_id, [0.0, 0.0])
    if n == 0:
        return float("inf")  # force exploration
    return math.sqrt(2.0 * math.log(max(total_selections, 1)) / n)


def build_bandit_actions(
    groups: Sequence[str],
    propensities: np.ndarray,
    cardinalities: Dict[str, float],
) -> List[BanditAction]:
    """
    Create a BanditAction for each group.
    Expected reward = estimated unique‑action count (cardinality).
    Confidence bound follows a simple UCB formula.
    """
    actions: List[BanditAction] = []
    total_selections = sum(_POLICY.get(g, [0.0, 0.0])[1] for g in groups) + 1
    for i, grp in enumerate(groups):
        prop = _pct(propensities[i])
        reward_est = _pct(cardinalities.get(grp, 0.0))
        conf = _pct(_confidence(grp, total_selections))
        actions.append(
            BanditAction(
                action_id=grp,
                propensity=prop,
                expected_reward=reward_est,
                confidence_bound=conf,
            )
        )
    return actions


def select_group_via_bandit(actions: List[BanditAction]) -> str:
    """
    Choose the group with the highest Upper Confidence Bound:
    UCB = expected_reward + confidence_bound.
    """
    best = max(actions, key=lambda a: a.expected_reward + a.confidence_bound)
    return best.action_id


# ----------------------------------------------------------------------
# Hybrid allocation routine (core fusion)
# ----------------------------------------------------------------------
def allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    action_stream: Iterable[Tuple[str, str]] = (),
) -> Dict[str, Any]:
    """
    Perform the fused allocation.

    * `total_units` – total resource pool.
    * `date` – calendar date (determines weekday weight).
    * `deterministic_target_pct` – percentage of the pool reserved for the
      deterministic (bandit‑chosen) group.
    * `groups` – list of group identifiers.
    * `action_stream` – iterable of (group, action_id) pairs representing
      observed actions for the CMS estimator.

    Returns a dictionary with per‑group allocation details.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    # ------------------------------------------------------------------
    # 1️⃣ Baseline LLM share via weekday weight vector (Parent A)
    # ------------------------------------------------------------------
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)

    # Baseline LLM allocation per group
    llm_per_group = llm_units * weight_vec
    allocation: Dict[str, Dict[str, Any]] = {
        grp: {
            "llm_units": _pct(llm_per_group[i]),
            "llm_share_pct": _pct(weight_vec[i] * 100.0),
            "weekday_weight": _pct(weight_vec[i]),
            "deterministic_units": 0.0,  # to be filled after bandit decision
        }
        for i, grp in enumerate(groups)
    }

    # ------------------------------------------------------------------
    # 2️⃣ Build a CMS per group from the action stream (Parent B)
    # ------------------------------------------------------------------
    actions_by_group: Dict[str, List[str]] = {g: [] for g in groups}
    for grp, act_id in action_stream:
        if grp in actions_by_group:
            actions_by_group[grp].append(act_id)

    cardinalities: Dict[str, float] = {}
    for grp, acts in actions_by_group.items():
        if acts:
            cms = count_min_sketch(acts)
            cardinalities[grp] = estimate_cardinality(cms)
        else:
            cardinalities[grp] = 0.0

    # ------------------------------------------------------------------
    # 3️⃣ Bandit decision using CMS‑derived rewards (Parent B)
    # ------------------------------------------------------------------
    bandit_actions = build_bandit_actions(groups, weight_vec, cardinalities)
    chosen_group = select_group_via_bandit(bandit_actions)

    # Record deterministic allocation
    allocation[chosen_group]["deterministic_units"] = _pct(deterministic_units)

    # ------------------------------------------------------------------
    # 4️⃣ Update the policy with a synthetic reward (for demonstration)
    # ------------------------------------------------------------------
    # Reward is proportional to the cardinality we just observed.
    synthetic_reward = cardinalities.get(chosen_group, 0.0)
    update_policy(
        [
            BanditUpdate(
                context_id="hybrid_allocation",
                action_id=chosen_group,
                reward=synthetic_reward,
                propensity=allocation[chosen_group]["weekday_weight"],
            )
        ]
    )

    # ------------------------------------------------------------------
    # 5️⃣ Final aggregation
    # ------------------------------------------------------------------
    for grp in groups:
        total = allocation[grp]["llm_units"] + allocation[grp]["deterministic_units"]
        allocation[grp]["total_units"] = _pct(total)

    # Summary payload
    result = {
        "date": date.isoformat(),
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "weekday": dow,
        "chosen_group": chosen_group,
        "per_group": allocation,
    }
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple reproducible test
    random.seed(42)

    # Generate a synthetic action stream: 200 actions uniformly across groups,
    # each action_id is a random UUID‑like string.
    def gen_action_id() -> str:
        return f"act_{random.getrandbits(64):016x}"

    stream = [
        (random.choice(GROUPS), gen_action_id()) for _ in range(200)
    ]

    today = dt.date.today()
    out = allocate_hybrid(
        total_units=1_000.0,
        date=today,
        deterministic_target_pct=85.0,
        groups=GROUPS,
        action_stream=stream,
    )

    # Pretty‑print the result (no external libraries)
    for key, val in out.items():
        if key == "per_group":
            print("\nPer‑group allocation:")
            for grp, details in val.items():
                print(f"  {grp}: {details}")
        else:
            print(f"{key}: {val}")

    # Ensure that the policy has been updated
    print("\nPolicy snapshot:")
    for act, stats in _POLICY.items():
        print(f"  {act}: cumulative_reward={stats[0]}, selections={stats[1]}")