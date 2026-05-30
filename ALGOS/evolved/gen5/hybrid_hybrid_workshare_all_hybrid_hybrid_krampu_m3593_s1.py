# DARWIN HAMMER — match 3593, survivor 1
# gen: 5
# parent_a: hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s1.py (gen4)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_hybrid_fisher_m1097_s4.py (gen4)
# born: 2026-05-29T23:50:56Z

"""Hybrid Workshare‑Bandit‑Pheromone Algorithm
------------------------------------------------
Parent A: `hybrid_workshare_allocator_hybrid_hybrid_hybrid_m1490_s1.py`
    – deterministic workshare allocation producing per‑group resource budgets.
Parent B: `hybrid_hybrid_krampus_brain_hybrid_hybrid_fisher_m1097_s4.py`
    – high‑dimensional pheromone entries that decay over time and are weighted
      by Fisher information and entropy.

Mathematical Bridge
-------------------
The bridge is the **resource‑scaled update rule** that appears in both
parents:

* In the bandit primitive the store (or value) update is  
  `V_{t+1} = V_t + η·(r_t – V_t)`, where `η` is a learning‑rate.
* In the pheromone model the signal decays multiplicatively by a factor
  `δ = 0.5^{Δt/τ}` (half‑life decay).

We fuse them by letting the learning‑rate `η` be a *function of the
workshare allocation* (more deterministic budget → slower learning) and
by applying the decay factor `δ` **before** the bandit update.  The
resulting hybrid update for a pheromone entry `e` is


e.signal_value ← e.signal_value * δ
e.signal_value ← e.signal_value + η·(reward – e.signal_value)


The hybrid decision stage then combines:

* the scaled signal values,
* Fisher‑information weights derived from recent rewards,
* an entropy‑based information‑density metric
  `I(e) = w·s·(‑log(s))` where `s` is the normalized signal and `w` the
  Fisher weight.

The entry with maximal `I(e)` is selected as the action for the current
context.
"""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone
import uuid
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Workshare core (Parent A)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def allocate_workshare(*, total_units: float,
                       deterministic_target_pct: float = 90.0,
                       groups: Tuple[str, ...] = GROUPS) -> Dict[str, Any]:
    """Return a dict describing deterministic and LLM workshare per group."""
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "lanes": lanes,
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
    }

# ----------------------------------------------------------------------
# Pheromone core (Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int = 300):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key          # e.g. group name
        self.signal_kind = signal_kind          # e.g. "reward_estimate"
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since the last call to apply_decay()."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)

class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PheromoneStore, cls).__new__(cls)
            cls._instance._entries = []
        return cls._instance

    def add(self, entry: PheromoneEntry) -> None:
        self._entries.append(entry)

    def all(self) -> List[PheromoneEntry]:
        return list(self._entries)

    def decay_all(self) -> None:
        for e in self._entries:
            e.apply_decay()

# ----------------------------------------------------------------------
# Fisher information utilities (bridge component)
# ----------------------------------------------------------------------
def _approx_fisher_information(rewards: np.ndarray) -> np.ndarray:
    """
    Approximate Fisher information for a set of scalar rewards.
    For a scalar parameter θ with likelihood p(r|θ) ∝ exp(-(r-θ)^2 / (2σ^2)),
    the Fisher information reduces to 1/σ^2.  We estimate σ^2 as the sample
    variance of the rewards and return the same value for each reward.
    """
    if rewards.size == 0:
        return np.ones_like(rewards)
    var = np.var(rewards, ddof=1)
    if var <= 0:
        var = 1e-9
    return np.full_like(rewards, 1.0 / var, dtype=float)

def compute_fisher_weights(rewards: List[float]) -> np.ndarray:
    """Public wrapper returning a weight per reward."""
    arr = np.asarray(rewards, dtype=float)
    return _approx_fisher_information(arr)

# ----------------------------------------------------------------------
# Hybrid update rule (core fusion)
# ----------------------------------------------------------------------
def hybrid_update(entry: PheromoneEntry,
                 reward: float,
                 workshare: Dict[str, Any],
                 eta_base: float = 0.2) -> None:
    """
    Apply decay then a bandit‑style stochastic update.
    The learning rate η is scaled by the deterministic share:
        η = η_base * (deterministic_units / total_units)
    """
    # 1. decay according to half‑life
    entry.apply_decay()

    # 2. compute scaled learning rate
    det_units = workshare.get("deterministic_units", 0.0)
    tot_units = workshare.get("total_units", 1.0)
    eta = eta_base * (det_units / tot_units)

    # 3. bandit update (store equation)
    entry.signal_value = entry.signal_value + eta * (reward - entry.signal_value)

# ----------------------------------------------------------------------
# Decision layer – entropy‑weighted Fisher scoring
# ----------------------------------------------------------------------
def hybrid_select_action(store: PheromoneStore,
                         workshare: Dict[str, Any],
                         recent_rewards: List[float]) -> PheromoneEntry | None:
    """
    Choose the pheromone entry that maximizes the information‑density metric:

        I(e) = w_e * s_e * (-log(s_e))

    where:
        s_e = normalized signal value across all entries,
        w_e = Fisher weight derived from recent rewards belonging to the same group.

    Returns None if the store is empty.
    """
    entries = store.all()
    if not entries:
        return None

    # Decay everything once more to keep freshness
    store.decay_all()

    # Gather raw signals
    raw_signals = np.array([e.signal_value for e in entries], dtype=float)
    # Guard against all zeros
    if np.allclose(raw_signals, 0):
        normalized = np.full_like(raw_signals, 1.0 / len(raw_signals))
    else:
        normalized = raw_signals / raw_signals.sum()

    # Fisher weights per group – we map recent rewards to groups by round‑robin
    # for simplicity (real implementation would have a proper context mapping).
    group_list = [e.surface_key for e in entries]
    unique_groups = list(dict.fromkeys(group_list))  # preserve order
    # Distribute rewards across groups
    rewards_per_group: Dict[str, List[float]] = {g: [] for g in unique_groups}
    for idx, r in enumerate(recent_rewards):
        grp = unique_groups[idx % len(unique_groups)]
        rewards_per_group[grp].append(r)

    # Compute Fisher weight for each group
    fisher_weights: Dict[str, float] = {}
    for grp, rlist in rewards_per_group.items():
        if rlist:
            w = compute_fisher_weights(rlist).mean()
        else:
            w = 1.0  # neutral weight if no recent data
        fisher_weights[grp] = w

    # Build the information‑density score
    info_scores = []
    for i, e in enumerate(entries):
        s = normalized[i]
        w = fisher_weights.get(e.surface_key, 1.0)
        # Avoid log(0) by clipping
        s_clipped = max(s, 1e-12)
        info = w * s_clipped * (-math.log(s_clipped))
        info_scores.append(info)

    # Select entry with maximal score
    best_idx = int(np.argmax(info_scores))
    return entries[best_idx]

# ----------------------------------------------------------------------
# Demonstration helpers
# ----------------------------------------------------------------------
def simulate_rewards(n: int) -> List[float]:
    """Generate n synthetic rewards drawn from a normal distribution."""
    return [random.gauss(0.5, 0.15) for _ in range(n)]

def populate_store(store: PheromoneStore,
                   workshare: Dict[str, Any],
                   n_per_group: int = 3) -> None:
    """Create a few pheromone entries for each group with random initial signals."""
    for lane in workshare["lanes"]:
        group = lane["group"]
        for _ in range(n_per_group):
            init_signal = random.uniform(0.0, 1.0)
            entry = PheromoneEntry(
                surface_key=group,
                signal_kind="reward_estimate",
                signal_value=init_signal,
                half_life_seconds=300
            )
            store.add(entry)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Allocate workshare
    ws = allocate_workshare(total_units=100.0, deterministic_target_pct=80.0)

    # 2. Initialise pheromone store and populate
    store = PheromoneStore()
    populate_store(store, ws, n_per_group=4)

    # 3. Simulate a batch of rewards and perform hybrid updates
    rewards_batch = simulate_rewards(20)
    # Apply updates round‑robin across entries
    entries = store.all()
    for i, reward in enumerate(rewards_batch):
        entry = entries[i % len(entries)]
        hybrid_update(entry, reward, ws, eta_base=0.25)

    # 4. Choose an action using the hybrid decision rule
    recent = simulate_rewards(12)          # recent reward history for Fisher weighting
    chosen = hybrid_select_action(store, ws, recent)

    # 5. Print summary
    print("Workshare allocation:")
    for lane in ws["lanes"]:
        print(f"  {lane['group']}: {lane['llm_units']} LLM units")
    print("\nChosen pheromone entry:")
    if chosen:
        print(f"  UUID: {chosen.uuid}")
        print(f"  Group: {chosen.surface_key}")
        print(f"  Signal value (post‑update): {chosen.signal_value:.6f}")
    else:
        print("  No entry selected.")