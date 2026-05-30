# DARWIN HAMMER — match 102, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s4.py (gen3)
# parent_b: hybrid_cockpit_metrics_hybrid_pheromone_inf_m64_s1.py (gen2)
# born: 2026-05-29T23:26:46Z

"""Hybrid Regret-Pheromone Router

This module fuses the core mathematics of two parent algorithms:

* **Parent A – Hybrid Regret-Weighted Bandit with Honeybee Store and MinHash Bridge**  
  The regret-weighted value of each action is computed from expected value, cost, risk and counterfactual outcomes.  
  A MinHash signature of the action's token set provides a similarity metric that can modulate those values.

* **Parent B – Hybrid Cockpit-Pheromone Metrics**  
  A cockpit honesty/evidence-coverage metrics are fused with the surface-pheromone / infotaxis entropy search.

The mathematical bridge is established by interpreting the cockpit metrics as prior probabilities 
that weight pheromone signals and entropy calculations. These probabilities are used to scale 
the regret-weighted utility before it enters the bandit's soft-max. The resulting utility drives 
both the action selection (propensity) and the store update (inflow proportional to the chosen action's propensity).
"""

import sys
import pathlib
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict
import numpy as np

# Data structures
@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret-weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"

@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy (not used directly in the demo)."""
    context_id: str
    action_id: str
    reward: float

# Cockpit metrics (Parent B)
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Fraction of claims that have supporting evidence."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
        claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known to be OK."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def audit_debt(exports_missing_audit_step: int) -> float:
    """Raw count of missing audit steps (non-negative)."""
    return float(max(0, exports_missing_audit_step))

# MinHash (Parent A)
def minhash(tokens: Tuple[str, ...], seed: int = 42) -> int:
    """Compute MinHash signature."""
    m = hashlib.md5()
    for token in sorted(tokens):
        m.update(token.encode())
    return int(m.hexdigest(), 16) % (2**32)

# Regret-weighted utility (Parent A)
def regret_weighted_utility(action: MathAction, 
                             counterfactuals: List[MathCounterfactual], 
                             similarity: float) -> float:
    """Compute regret-weighted utility."""
    expected_value = action.expected_value
    cost = action.cost
    risk = action.risk
    regret = 0.0
    for cf in counterfactuals:
        if cf.action_id == action.id:
            regret += cf.outcome_value * cf.probability
    utility = expected_value - cost - risk - regret
    return utility * similarity

# Hybrid operation
def hybrid_regret_pheromone(action: MathAction, 
                             counterfactuals: List[MathCounterfactual], 
                             claims_with_evidence: int, 
                             total_claims_emitted: int, 
                             displayed_ok: int, 
                             unknown_displayed_as_ok: int) -> BanditAction:
    """Compute hybrid regret-pheromone action."""
    # Compute cockpit metrics
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    
    # Compute MinHash similarity
    similarity = minhash(action.tokens) / (2**32)
    
    # Compute regret-weighted utility
    utility = regret_weighted_utility(action, counterfactuals, similarity * anti_slop * honesty)
    
    # Compute propensity (soft-max)
    propensity = math.exp(utility) / (1 + math.exp(utility))
    
    # Compute expected reward and confidence bound
    expected_reward = action.expected_value * propensity
    confidence_bound = math.sqrt(propensity * (1 - propensity))
    
    return BanditAction(action.id, propensity, expected_reward, confidence_bound)

def calculate_pheromone_signal(base_signal: float,
                               half_life_seconds: float,
                               elapsed_seconds: float) -> float:
    """Exponential decay of a pheromone signal."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be > 0")
    decay = math.pow(0.5, elapsed_seconds / half_life_seconds)
    return base_signal * decay

def hybrid_pheromone_update(action: BanditAction, 
                             base_signal: float, 
                             half_life_seconds: float, 
                             elapsed_seconds: float) -> float:
    """Compute hybrid pheromone update."""
    pheromone_signal = calculate_pheromone_signal(base_signal, half_life_seconds, elapsed_seconds)
    return pheromone_signal * action.propensity

if __name__ == "__main__":
    # Create a test action
    action = MathAction("test_action", ("token1", "token2"), 10.0)
    counterfactuals = [MathCounterfactual("test_action", 5.0)]
    claims_with_evidence = 10
    total_claims_emitted = 10
    displayed_ok = 5
    unknown_displayed_as_ok = 0
    
    # Compute hybrid regret-pheromone action
    bandit_action = hybrid_regret_pheromone(action, counterfactuals, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok)
    print(bandit_action)
    
    # Compute hybrid pheromone update
    base_signal = 1.0
    half_life_seconds = 10.0
    elapsed_seconds = 5.0
    pheromone_update = hybrid_pheromone_update(bandit_action, base_signal, half_life_seconds, elapsed_seconds)
    print(pheromone_update)