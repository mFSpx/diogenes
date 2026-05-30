# DARWIN HAMMER — match 2706, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_cockpit_metri_m287_s4.py (gen3)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s1.py (gen5)
# born: 2026-05-29T23:43:47Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence, Hashable, Iterable
import numpy as np

def temperature_activity(T: float, T_opt: float = 310.0, k: float = 0.05) -> float:
    return 1.0 / (1.0 + math.exp(-k * (T - T_opt)))

def honesty_weight(claims_with_evidence: int, total_claims_emitted: int) -> float:
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def hybrid_gain(T: float, claims_with_evidence: int, total_claims_emitted: int) -> float:
    return temperature_activity(T) * honesty_weight(claims_with_evidence, total_claims_emitted)

def entropy(prob_dist: Iterable[float]) -> float:
    probs = np.array(list(prob_dist), dtype=float)
    total = probs.sum()
    if total == 0.0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += (2 * i - n - 1) * x
    return cumulative / (n * sum(xs))

def rbf_similarity(a: Sequence[float], b: Sequence[float], epsilon: float = 1.0) -> float:
    if len(a) != len(b):
        raise ValueError("Feature vectors must have equal length")
    r2 = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.exp(-epsilon * r2)

def tropical_max_plus(log_probs: Sequence[float], additive: Sequence[float]) -> float:
    if len(log_probs) != len(additive):
        raise ValueError("Sequences must be of equal length")
    return max(lp + ad for lp, ad in zip(log_probs, additive))

Node = Hashable
FeatureVec = Sequence[float]

@dataclass
class HybridPolicy:
    pheromone: Dict[Node, float] = field(default_factory=dict)
    features: Dict[Node, FeatureVec] = field(default_factory=dict)

    def normalize_pheromone(self) -> None:
        total = sum(self.pheromone.values())
        if total == 0.0:
            n = len(self.pheromone)
            if n == 0:
                return
            uniform = 1.0 / n
            for k in self.pheromone:
                self.pheromone[k] = uniform
        else:
            for k in self.pheromone:
                self.pheromone[k] /= total

def hybrid_compute_gain(T: float, claims_with_evidence: int, total_claims_emitted: int) -> float:
    return hybrid_gain(T, claims_with_evidence, total_claims_emitted)

def hybrid_node_scores(policy: HybridPolicy, context: FeatureVec, T: float, claims_with_evidence: int, total_claims_emitted: int) -> Dict[Node, float]:
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)
    λ = 1.0 - G
    policy.normalize_pheromone()
    pher_vals = np.array([policy.pheromone.get(n, 0.0) for n in policy.features], dtype=float)
    safe_vals = np.where(pher_vals > 0, pher_vals, 1e-12)
    log_probs = np.log(safe_vals)
    scores: Dict[Node, float] = {}
    for idx, (node, feat) in enumerate(policy.features.items()):
        sim = rbf_similarity(context, feat) * G
        additive = np.array([rbf_similarity(context, f) * G for f in policy.features.values()])
        score = tropical_max_plus(log_probs, additive)
        scores[node] = score + sim  # Include the similarity in the score
    return scores

def hybrid_update_policy(policy: HybridPolicy, selected_node: Node, reward: float, T: float, claims_with_evidence: int, total_claims_emitted: int, decay_rate: float = 0.05) -> None:
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)
    λ = 1.0 - G
    policy.normalize_pheromone()
    pher_vals = np.array([policy.pheromone.get(n, 0.0) for n in policy.features], dtype=float)
    safe_vals = np.where(pher_vals > 0, pher_vals, 1e-12)
    log_probs = np.log(safe_vals)
    for node in policy.features:
        if node == selected_node:
            policy.pheromone[node] = policy.pheromone[node] * (1 - decay_rate) + reward * G
        else:
            policy.pheromone[node] = policy.pheromone[node] * (1 - decay_rate)
    policy.normalize_pheromone()

def hybrid_regularization(policy: HybridPolicy, T: float, claims_with_evidence: int, total_claims_emitted: int) -> float:
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)
    λ = 1.0 - G
    policy.normalize_pheromone()
    pher_vals = np.array([policy.pheromone.get(n, 0.0) for n in policy.features], dtype=float)
    safe_vals = np.where(pher_vals > 0, pher_vals, 1e-12)
    log_probs = np.log(safe_vals)
    reg = G * entropy(pher_vals) + λ * gini_coefficient(pher_vals)
    return reg

def main():
    # Example usage
    policy = HybridPolicy()
    policy.pheromone = {'A': 0.5, 'B': 0.5}
    policy.features = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    context = [1, 2, 3]
    T = 310.0
    claims_with_evidence = 10
    total_claims_emitted = 20
    scores = hybrid_node_scores(policy, context, T, claims_with_evidence, total_claims_emitted)
    print(scores)

if __name__ == "__main__":
    main()