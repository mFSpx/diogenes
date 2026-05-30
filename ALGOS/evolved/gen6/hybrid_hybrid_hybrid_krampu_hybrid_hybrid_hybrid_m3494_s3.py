# DARWIN HAMMER — match 3494, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_capybara_opti_m2678_s1.py (gen5)
# born: 2026-05-29T23:50:28Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np

Point = Tuple[float, float]
Edge = Tuple[str, str]

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "ImprovedHybridKrampusCapybara"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    feature_vec: np.ndarray

def gaussian_kernel(x: np.ndarray, x_prime: np.ndarray, epsilon: float = 1.0) -> float:
    return np.exp(-epsilon**2 * np.linalg.norm(x - x_prime)**2)

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    adj = {}
    edge_len = {}
    root_dist = {}

    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = np.linalg.norm(np.array(nodes[a]) - np.array(nodes[b]))
        edge_len[(b, a)] = np.linalg.norm(np.array(nodes[b]) - np.array(nodes[a]))

    for node in nodes:
        root_dist[node] = np.linalg.norm(np.array(nodes[node]) - np.array(nodes[root]))

    return adj, edge_len, root_dist

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    return delta_max * np.exp(-alpha * (t / t_max))

def update_policy(update: BanditUpdate) -> None:
    context_id = update.context_id
    action_id = update.action_id
    reward = update.reward
    feature_vec = update.feature_vec

    kernel_weights = []
    for i, context in enumerate(contexts.values()):
        kernel_weight = gaussian_kernel(feature_vec, context[1])
        kernel_weights.append((context[0], kernel_weight))
    kernel_weights = np.array(kernel_weights)

    global policy
    policy[context_id] = (action_id, reward, kernel_weights)

def select_action(context_id: str) -> BanditAction:
    policy_context = policy.get(context_id)

    if policy_context is None:
        return BanditAction(action_id="default", propensity=0.0, expected_reward=0.0, confidence_bound=0.0)

    kernel_weights = policy_context[2]
    expected_reward = np.sum([weight * list(contexts.values())[i][2] for i, (context_id, weight) in enumerate(kernel_weights)])
    confidence_bound = np.sqrt(np.sum([weight**2 for context_id, weight in kernel_weights]))

    action_id = policy_context[0]
    propensity = np.random.rand()
    return BanditAction(action_id=action_id, propensity=propensity, expected_reward=expected_reward, confidence_bound=confidence_bound)

def extract_full_features(context_id: str) -> np.ndarray:
    context = contexts.get(context_id)

    if context is None:
        return np.zeros(10)

    feature_vec = context[1]
    full_features = np.concatenate((feature_vec, np.random.rand(5)))
    return full_features

def integrate_contexts(contexts: Dict[str, Tuple[str, np.ndarray, float]]) -> Dict[str, Tuple[str, np.ndarray, float]]:
    integrated_contexts = {}
    for context_id, (action_id, feature_vec, reward) in contexts.items():
        integrated_feature_vec = np.concatenate((feature_vec, np.random.rand(5)))
        integrated_contexts[context_id] = (action_id, integrated_feature_vec, reward)
    return integrated_contexts

policy = {}
contexts = {}

if __name__ == "__main__":
    context_id = "context1"
    action_id = "action1"
    reward = 10.0
    feature_vec = np.random.rand(10)
    contexts[context_id] = (action_id, feature_vec, reward)
    update = BanditUpdate(context_id, action_id, reward, 0.0, feature_vec)
    update_policy(update)

    context_id = "context1"
    action = select_action(context_id)
    print(action)

    context_id = "context1"
    full_features = extract_full_features(context_id)
    print(full_features)

    integrated_contexts = integrate_contexts(contexts)
    print(integrated_contexts)