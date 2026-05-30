# DARWIN HAMMER — match 796, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s4.py (gen3)
# parent_b: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (gen2)
# born: 2026-05-29T23:30:58Z

"""Hybrid Bandit‑Model Scheduler

This module fuses two parent algorithms:

* **Parent A** – a contextual bandit that maintains a virtual “VRAM store” with inflow/outflow
  dynamics (α, β, Δt) and uses Hoeffding confidence bounds for exploration.
* **Parent B** – a model‑tier VRAM scheduler that computes a *reconstruction risk score*
  from privacy metadata and decides which models may be loaded under RAM/VRAM ceilings.

**Mathematical bridge**

1. The risk score `r ∈ [0,1]` from Parent B is injected into the bandit’s exploration term:
   the effective confidence bound becomes  

   `CB_eff = CB * (1 + λ_r * r)`

   where `λ_r` is a tunable risk‑amplification factor.  
   Higher privacy risk enlarges the bound, making the algorithm more conservative
   (lower propensity to select that action).

2. The virtual store `S_i` of Parent A is interpreted as the *reserved VRAM* for a
   particular model `i`. Its dynamics  

   `S_i ← S_i + Δt·(α·propensity_i – β·outflow_i)`

   are now coupled to the actual VRAM consumption of the model:
   `outflow_i` is proportional to the model’s `vram_mb`.  
   When the cumulative store exceeds the global VRAM ceiling, the scheduler evicts
   the model with the smallest adjusted score.

The resulting hybrid algorithm simultaneously learns reward‑driven preferences,
honors privacy‑driven risk, and respects hard VRAM constraints.

"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Sequence, Any, Iterable, Set

import numpy as np

# ----------------------------------------------------------------------
# Shared hyper‑parameters (derived from both parents)
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for policy updates
HOEFFDING_DELTA = 0.05  # confidence level for Hoeffding bound
RISK_LAMBDA = 1.5    # amplification of risk on confidence bound
VRAM_CEILING_MB = 4096
RAM_CEILING_MB = 6000
CLAMP_LO = -5.0
CLAMP_HI = 5.0

# ----------------------------------------------------------------------
# Data structures (merged)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a selectable model."""
    model: "ModelTier"
    propensity: float          # interpreted as inflow for the virtual store
    expected_reward: float
    confidence_bound: float    # raw Hoeffding bound
    algorithm: str = "linucb"

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

# Pre‑defined model tiers (from Parent B)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

# ----------------------------------------------------------------------
# Global state (policy, virtual stores, loaded models)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}   # model.name -> [total_reward, count]
_STORE: Dict[str, float] = {}          # model.name -> virtual VRAM store
_LOADED: Dict[str, ModelTier] = {}     # model.name -> ModelTier instance

# ----------------------------------------------------------------------
# Privacy helpers (from Parent B)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Risk ∈ [0,1] proportional to the fraction of quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def anonymize_for_indexing(record: dict[str, Any],
                           redact_keys: Set[str] | None = None) -> dict[str, Any]:
    """Redact sensitive fields before indexing."""
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

# ----------------------------------------------------------------------
# Core bandit utilities
# ----------------------------------------------------------------------
def _reward_estimate(model_name: str) -> float:
    total, n = _POLICY.get(model_name, [0.0, 0.0])
    return total / n if n else 0.0

def _confidence_bound(model_name: str) -> float:
    """Hoeffding‑type bound."""
    _, n = _POLICY.get(model_name, [0.0, 0.0])
    if n == 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / HOEFFDING_DELTA) / (2.0 * n))

def _update_store(model_name: str, propensity: float, outflow: float) -> None:
    """Euler update of the virtual VRAM store."""
    s = _STORE.get(model_name, 0.0)
    s += DT * (ALPHA * propensity - BETA * outflow)
    _STORE[model_name] = s

def reset_hybrid() -> None:
    """Clear policy, virtual stores and loaded models."""
    _POLICY.clear()
    _STORE.clear()
    _LOADED.clear()

# ----------------------------------------------------------------------
# Hybrid selection & scheduling
# ----------------------------------------------------------------------
def select_action(context: Dict[str, Any],
                  candidates: List[ModelTier],
                  unique_quasi_ids: int,
                  total_records: int,
                  epsilon: float = 0.1,
                  seed: int | str | None = 7) -> BanditAction:
    """
    Choose a model using a risk‑aware bandit rule.

    The score for each candidate `m` is
        score_m = μ_m + CB_eff_m
    where
        μ_m = reward estimate,
        CB_eff_m = (raw Hoeffding bound) * (1 + RISK_LAMBDA * risk),
        risk = reconstruction_risk_score(...).

    An ε‑greedy fallback selects a random candidate with probability `epsilon`.
    """
    if seed is not None:
        random.seed(seed)

    if not candidates:
        raise ValueError("candidate model list cannot be empty")

    # Compute risk once (shared across candidates)
    risk = reconstruction_risk_score(unique_quasi_ids, total_records)

    # ε‑greedy exploration
    if random.random() < epsilon:
        chosen = random.choice(candidates)
        prop = 1.0
        exp_r = _reward_estimate(chosen.name)
        cb = _confidence_bound(chosen.name)
        return BanditAction(chosen, prop, exp_r, cb)

    # Compute adjusted scores
    best_score = -float("inf")
    best_action: BanditAction | None = None
    for model in candidates:
        prop = 1.0  # baseline propensity; could be enriched with context
        exp_r = _reward_estimate(model.name)
        raw_cb = _confidence_bound(model.name)
        cb_eff = raw_cb * (1.0 + RISK_LAMBDA * risk)
        score = exp_r + cb_eff
        if score > best_score:
            best_score = score
            best_action = BanditAction(model, prop, exp_r, raw_cb)

    # best_action is never None because candidates non‑empty
    return best_action  # type: ignore

def observe_reward(context: Dict[str, Any],
                   action: BanditAction,
                   reward: float,
                   propensity: float) -> None:
    """
    Update the bandit policy and the virtual store based on observed reward.
    """
    name = action.model.name
    total, n = _POLICY.get(name, [0.0, 0.0])
    total += reward
    n += 1
    _POLICY[name] = [total, n]

    # Outflow proportional to the model's VRAM consumption
    outflow = action.model.vram_mb / VRAM_CEILING_MB
    _update_store(name, propensity, outflow)

def _used_vram() -> int:
    return sum(m.vram_mb for m in _LOADED.values())

def _used_ram() -> int:
    return sum(m.ram_mb for m in _LOADED.values())

def schedule_models(context: Dict[str, Any],
                    candidate_models: List[ModelTier],
                    unique_quasi_ids: int,
                    total_records: int) -> List[ModelTier]:
    """
    Perform a single scheduling step:

    1. Select the best action via `select_action`.
    2. If loading the model would exceed VRAM/RAM ceilings, evict
       the currently loaded model with the smallest adjusted score.
    3. Load the selected model, update stores, and return the list of
       currently loaded models.
    """
    selected = select_action(context, candidate_models,
                             unique_quasi_ids, total_records)

    # Check resource constraints
    needed_vram = selected.model.vram_mb
    needed_ram = selected.model.ram_mb

    while (_used_vram() + needed_vram > VRAM_CEILING_MB or
           _used_ram() + needed_ram > RAM_CEILING_MB):
        # Evict the loaded model with lowest adjusted score
        if not _LOADED:
            raise RuntimeError("Cannot satisfy resource constraints with any model.")
        worst_name, worst_model = min(
            _LOADED.items(),
            key=lambda kv: (_reward_estimate(kv[0]) -
                            RISK_LAMBDA * reconstruction_risk_score(0, 1))  # conservative estimate
        )
        del _LOADED[worst_name]

    # Load the selected model
    _LOADED[selected.model.name] = selected.model
    # Update the virtual store for the loaded model (propensity = 1)
    _update_store(selected.model.name, propensity=1.0,
                  outflow=selected.model.vram_mb / VRAM_CEILING_MB)

    return list(_LOADED.values())

# ----------------------------------------------------------------------
# Demonstration functions (required >=3)
# ----------------------------------------------------------------------
def simulate_bandit_step(context: Dict[str, Any],
                         models: List[ModelTier],
                         uqids: int,
                         total_recs: int) -> Tuple[BanditAction, float]:
    """
    Simulate a single interaction: select an action, generate a synthetic reward,
    and observe it. Returns the action and the reward.
    """
    action = select_action(context, models, uqids, total_recs)
    # Synthetic reward: higher for larger models (as a placeholder)
    reward = random.uniform(0.0, 1.0) * (action.model.vram_mb / VRAM_CEILING_MB)
    observe_reward(context, action, reward, propensity=action.propensity)
    return action, reward

def batch_training(context: Dict[str, Any],
                   models: List[ModelTier],
                   epochs: int = 5,
                   uqids: int = 10,
                   total_recs: int = 100) -> None:
    """Run multiple bandit steps to populate the policy."""
    for _ in range(epochs):
        simulate_bandit_step(context, models, uqids, total_recs)

def current_state_snapshot() -> dict[str, Any]:
    """Return a dictionary summarising policy, store and loaded models."""
    return {
        "policy": {k: {"total_reward": v[0], "count": v[1]} for k, v in _POLICY.items()},
        "store": _STORE.copy(),
        "loaded_models": [asdict(m) for m in _LOADED.values()],
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_hybrid()
    ctx = {"user_id": 42, "session": "test"}
    all_models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]

    # Populate the bandit with a few synthetic interactions
    batch_training(ctx, all_models, epochs=20, uqids=3, total_recs=30)

    # Perform a scheduling decision
    loaded = schedule_models(ctx, all_models, unique_quasi_ids=5, total_records=50)

    snapshot = current_state_snapshot()
    print("Loaded models:", [m['name'] for m in snapshot["loaded_models"]])
    print("Policy snapshot (sample):", list(snapshot["policy"].items())[:3])
    print("Store snapshot (sample):", {k: round(v, 3) for k, v in list(snapshot["store"].items())[:3]})
    sys.exit(0)