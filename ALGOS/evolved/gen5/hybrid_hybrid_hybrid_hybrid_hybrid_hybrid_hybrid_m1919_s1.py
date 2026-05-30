# DARWIN HAMMER — match 1919, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_model__m860_s2.py (gen3)
# born: 2026-05-29T23:39:46Z

import numpy as np
import math
import random
import sys
import pathlib

# Types
Node = object
Graph = dict[Node, set[Node]]

# Constants
TERNARY_DIMS = 12

def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def compute_regret_weighted_strategy(
    actions: list, counterfactuals: list
) -> dict[str, float]:
    if not actions:
        return {}
    exp_map = {a['id']: a['expected_value'] for a in actions}
    regrets = {a['id']: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf['action_id'] not in exp_map:
            continue
        diff = cf['outcome_value'] - exp_map[cf['action_id']]
        regrets[cf['action_id']] += diff * cf['probability']

    max_r = max(regrets.values())
    exp_vals = {aid: math.exp(r - max_r) for aid, r in regrets.items()}
    z = sum(exp_vals.values())
    probs = {aid: v / z for aid, v in exp_vals.items()}
    return probs

def hoeffding_bound(n: int, epsilon: float, delta: float) -> float:
    """Hoeffding bound for the given number of samples and confidence level."""
    return math.sqrt((math.log(1.0 / delta) * n) / (2.0 * epsilon**2))

def acceptance_probability(beta: float, temperature: float) -> float:
    """Compute the acceptance probability."""
    return 1.0 / (1.0 + math.exp(beta / temperature))

def krampus_curvature(graph: Graph, temperature: float) -> dict[Node, float]:
    """Compute the Krampus-Ollivier-Ricci curvature of the graph."""
    curvatures = {}
    for node in graph:
        neighbors = graph[node]
        if len(neighbors) < 2:
            curvatures[node] = 0.0
            continue
        beta = 1.0
        acceptance_prob = acceptance_probability(beta, temperature)
        curvatures[node] = 1.0 - acceptance_prob
    return curvatures

def regret_weighted_hoeffding(actions: list, counterfactuals: list, temperature: float) -> dict[str, float]:
    """Compute the regret-weighted strategy using the Hoeffding bound."""
    exp_map = {a['id']: a['expected_value'] for a in actions}
    regrets = {a['id']: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf['action_id'] not in exp_map:
            continue
        diff = cf['outcome_value'] - exp_map[cf['action_id']]
        regrets[cf['action_id']] += diff * cf['probability']

    max_r = max(regrets.values())
    exp_vals = {aid: math.exp(r - max_r) for aid, r in regrets.items()}
    z = sum(exp_vals.values())
    probs = {aid: v / z for aid, v in exp_vals.items()}
    hoeffding_ubs = {aid: hoeffding_bound(len(counterfactuals), 0.1, 0.1) for aid in probs}
    weighted_probs = {aid: probs[aid] * (1.0 - hoeffding_ubs[aid] / (1 + hoeffding_ubs[aid])) for aid in probs}
    return weighted_probs

def hybrid_strategy(graph: Graph, actions: list, counterfactuals: list, temperature: float) -> dict[str, float]:
    """Compute the hybrid strategy using the Krampus-Ollivier-Ricci curvature and regret-weighted strategy."""
    krampus_curvatures = krampus_curvature(graph, temperature)
    regret_weighted = regret_weighted_hoeffding(actions, counterfactuals, temperature)
    weighted_probs = {}
    for node, children in graph.items():
        node_curvature = krampus_curvatures[node]
        for child in children:
            if child not in regret_weighted:
                continue
            weighted_probs[child] = node_curvature * regret_weighted[child]
    return weighted_probs

if __name__ == "__main__":
    graph = {1: {2, 3}, 2: {4, 5}, 3: {6, 7}, 4: {8, 9}, 5: {10, 11}, 6: {12, 13}, 7: {14, 15}}
    actions = [{'id': 1, 'expected_value': 1.0}, {'id': 2, 'expected_value': 2.0}]
    counterfactuals = [{'action_id': 1, 'outcome_value': 3.0, 'probability': 0.5}, {'action_id': 2, 'outcome_value': 4.0, 'probability': 0.5}]
    temperature = 1.0
    strategy = hybrid_strategy(graph, actions, counterfactuals, temperature)
    print(strategy)