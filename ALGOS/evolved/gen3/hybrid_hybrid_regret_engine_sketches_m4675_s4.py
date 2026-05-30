# DARWIN HAMMER — match 4675, survivor 4
# gen: 3
# parent_a: hybrid_regret_engine_hybrid_doomsday_cale_m19_s4.py (gen2)
# parent_b: sketches.py (gen0)
# born: 2026-05-29T23:57:21Z

"""
Hybrid Regret-Weighted MinHash LSH Calendar
Parent A: hybrid_regret_engine_hybrid_doomsday_cale_m19_s4.py - computes a softmax-like probability distribution 
           over actions using regret-adjusted expected values and evaluates their inequality with the Gini coefficient.
Parent B: sketches.py - provides MinHash LSH indexing for similarity search.

Mathematical bridge:
    • The regret engine outputs a normalized probability vector **p** over a set of actions.
    • MinHash LSH generates a compact representation of sets (e.g., actions) for efficient similarity search.
    • By treating each action as a set of its constituent parts (e.g., expected values, costs, risks), 
      we can apply MinHash LSH to the output of the regret engine, effectively fusing the two 
      operations. The hybrid system (i) builds a regret-weighted strategy from action values 
      and (ii) generates a compact, similarity-preserving representation of those strategies 
      using MinHash LSH.

"""

from __future__ import annotations
import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable
import hashlib

# ----------------------------------------------------------------------
# Data structures 
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Core algorithms from the two parents
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: list[MathAction],
    counterfactuals: list[MathCounterfactual],
) -> dict[str, float]:
    # Compute regret-adjusted expected values
    regret_values = {}
    for action in actions:
        regret = 0
        for cf in counterfactuals:
            if cf.action_id == action.id:
                regret += cf.outcome_value * cf.probability
        regret_values[action.id] = action.expected_value - regret

    # Normalize to get probability distribution
    total_regret = sum(regret_values.values())
    probabilities = {action_id: regret / total_regret for action_id, regret in regret_values.items()}
    return probabilities

def gini_coefficient(probabilities: dict[str, float]) -> float:
    # Calculate Gini coefficient
    n = len(probabilities)
    index = 1
    gini = 0
    for p in probabilities.values():
        gini += (2 * index - n - 1) * p
        index += 1
    return gini / (n * (n - 1))

def minhash_lsh_index(actions: dict[str, set[str]]) -> dict[str, list[str]]:
    buckets = defaultdict(list)
    for action_id, parts in actions.items():
        key = min((hashlib.sha1(part.encode()).hexdigest()[:6] for part in parts), default='empty')
        buckets[key].append(action_id)
    return dict(buckets)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_regret_minhash_lsh(actions: list[MathAction], 
                                counterfactuals: list[MathCounterfactual]) -> dict[str, list[str]]:
    probabilities = compute_regret_weighted_strategy(actions, counterfactuals)
    gini = gini_coefficient(probabilities)

    # Convert actions to sets of their constituent parts
    action_parts = {}
    for action in actions:
        parts = {f"EV:{action.expected_value}", f"C:{action.cost}", f"R:{action.risk}"}
        action_parts[action.id] = parts

    # Apply MinHash LSH
    return minhash_lsh_index(action_parts)

def get_similar_actions(hybrid_output: dict[str, list[str]], action_id: str) -> list[str]:
    similar_actions = []
    for key, actions in hybrid_output.items():
        if action_id in actions:
            similar_actions.extend(actions)
    return similar_actions

def evaluate_strategy(actions: list[MathAction], 
                      counterfactuals: list[MathCounterfactual], 
                      similar_action_ids: list[str]) -> float:
    similar_actions = [action for action in actions if action.id in similar_action_ids]
    similar_probabilities = compute_regret_weighted_strategy(similar_actions, counterfactuals)
    return gini_coefficient(similar_probabilities)

if __name__ == "__main__":
    actions = [
        MathAction("action1", 10.0, 1.0, 0.5),
        MathAction("action2", 20.0, 2.0, 1.0),
        MathAction("action3", 30.0, 3.0, 1.5),
    ]

    counterfactuals = [
        MathCounterfactual("action1", 5.0),
        MathCounterfactual("action2", 10.0),
        MathCounterfactual("action3", 15.0),
    ]

    hybrid_output = hybrid_regret_minhash_lsh(actions, counterfactuals)
    similar_actions = get_similar_actions(hybrid_output, "action1")
    strategy_evaluation = evaluate_strategy(actions, counterfactuals, similar_actions)
    print(strategy_evaluation)