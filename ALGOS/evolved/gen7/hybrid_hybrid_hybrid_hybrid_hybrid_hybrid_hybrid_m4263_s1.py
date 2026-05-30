# DARWIN HAMMER — match 4263, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1605_s0.py (gen6)
# born: 2026-05-29T23:54:39Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable
from collections import defaultdict

@dataclass(frozen=True)
class Endpoint:
    morphology: List[float]

@dataclass(frozen=True)
class FractionalHealthScore:
    score: float
    weights: List[float]

@dataclass(frozen=True)
class LabelingFunctionResult: 
    lf_name: str 
    doc_id: str 
    label: int 

@dataclass(frozen=True)
class ProbabilisticLabel: 
    doc_id: str 
    label: int 
    confidence: float 

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

_POLICY: dict = defaultdict(lambda: [0.0, 0.0])

def shannon_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Lists must have equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def decision_hygiene_entropy(feature_counts: List[int]) -> float:
    return shannon_entropy(feature_counts)

def compute_health_score(endpoint: Endpoint, feature_counts: List[int]) -> FractionalHealthScore:
    decision_hygiene = decision_hygiene_entropy(feature_counts)
    score = np.dot(endpoint.morphology, [decision_hygiene] * len(endpoint.morphology))
    weights = [1.0] * len(endpoint.morphology)
    return FractionalHealthScore(score, weights)

def labeling_function(name: str|None=None): 
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__ 
        return fn 
    return deco 

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    labels = []
    for batch in batches:
        doc_id = batch[0].doc_id
        label_counts = defaultdict(int)
        for result in batch:
            label_counts[result.label] += 1
        max_label = max(label_counts, key=label_counts.get)
        confidence = label_counts[max_label] / len(batch)
        labels.append(ProbabilisticLabel(doc_id, max_label, confidence))
    return labels

def bandit_update(action: BanditAction, update: BanditUpdate) -> None:
    _POLICY[action.action_id][0] += update.reward
    _POLICY[action.action_id][1] += 1

def kl_divergence(p: List[float], q: List[float]) -> float:
    total = sum(p)
    if total == 0:
        return 0.0
    divergence = 0.0
    for i in range(len(p)):
        prob_p = p[i] / total
        prob_q = q[i] / sum(q)
        if prob_p > 0:
            divergence += prob_p * math.log2(prob_p / prob_q)
    return divergence

def improved_compute_health_score(endpoint: Endpoint, feature_counts: List[int], prior_counts: List[int]) -> FractionalHealthScore:
    decision_hygiene = decision_hygiene_entropy(feature_counts)
    prior_hygiene = decision_hygiene_entropy(prior_counts)
    kl_div = kl_divergence(feature_counts, prior_counts)
    score = np.dot(endpoint.morphology, [decision_hygiene, kl_div] * len(endpoint.morphology))
    weights = [1.0, 0.5] * len(endpoint.morphology)
    return FractionalHealthScore(score, weights)

def improved_bandit_update(action: BanditAction, update: BanditUpdate) -> None:
    _POLICY[action.action_id][0] += update.reward * action.propensity
    _POLICY[action.action_id][1] += 1

if __name__ == "__main__":
    endpoint = Endpoint([0.5, 0.5])
    feature_counts = [1, 2]
    prior_counts = [2, 3]
    health_score = improved_compute_health_score(endpoint, feature_counts, prior_counts)
    print(health_score)

    batches = [[LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc1", 1)],
               [LabelingFunctionResult("lf1", "doc2", 0), LabelingFunctionResult("lf2", "doc2", 0)]]
    labels = aggregate_labels(batches)
    for label in labels:
        print(label)

    action = BanditAction("action1", 0.5, 0.0, 0.0, "algorithm1")
    update = BanditUpdate("context1", "action1", 1.0, 0.5)
    improved_bandit_update(action, update)
    print(_POLICY[action.action_id])