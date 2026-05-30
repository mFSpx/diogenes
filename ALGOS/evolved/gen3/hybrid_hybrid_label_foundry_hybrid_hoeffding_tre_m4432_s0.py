# DARWIN HAMMER — match 4432, survivor 0
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s1.py (gen1)
# born: 2026-05-29T23:55:33Z

"""
Hybrid Algorithm: Fusing Label Foundry and Hoeffding Tree with Gini Coefficient

This module mathematically fuses the core topologies of hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (Parent A)
and hybrid_hoeffding_tree_gini_coefficient_m13_s1.py (Parent B). The bridge between the two parents lies in the concept
of confidence and uncertainty. Parent A produces a confidence value for each document via vote majority, while Parent B
uses the Hoeffding bound and Gini coefficient to measure uncertainty in the data distribution.

The hybrid algorithm integrates these concepts by using the confidence values from Parent A as input to the Hoeffding bound
calculation in Parent B, and incorporating the Gini coefficient as a regularization term to adaptively adjust the decision-making
criteria. Specifically, the hybrid algorithm uses the confidence values to compute a weighted Hoeffding bound, which is then
used to inform the splitting decision in the Hoeffding tree.

Mathematically, the hybrid algorithm can be represented as:

    c_hybrid = c · (1 - gini_coeff)
    eps_hybrid = hoeffding_bound_with_gini(r, delta, n, gini_coeff * c_hybrid)

where c is the confidence value from Parent A, gini_coeff is the Gini coefficient from Parent B, and eps_hybrid is the weighted
Hoeffding bound used to inform the splitting decision.

"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List
from collections import defaultdict

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]

def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A‑logic: majority vote with confidence = proportion of votes."""
    votes: dict = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out = []
    for doc_id, labels in votes.items():
        label = np.argmax(np.bincount(labels))
        confidence = labels.count(label) / len(labels)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out

def hoeffding_bound_with_gini(r: float, delta: float, n: int, gini_coeff: float) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    
    # Regularize the Hoeffding bound with the Gini coefficient
    regularization_term = gini_coeff * math.pi / 6
    return math.sqrt((r * r * math.log(1.0 / delta) + regularization_term) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split_with_gini_and_confidence(best_gain: float, second_best_gain: float, r: float, delta: float, n: int, 
                                          tie_threshold: float = 0.05, gini_coeff: float = 0.5, 
                                          confidence: float = 1.0) -> SplitDecision:
    # Compute weighted Hoeffding bound using confidence
    eps = hoeffding_bound_with_gini(r, delta, n, gini_coeff * confidence)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)

def hybrid_label_hoeffding(batches: List[List[LabelingFunctionResult]], best_gain: float, second_best_gain: float, 
                           r: float, delta: float, n: int, tie_threshold: float = 0.05, 
                           gini_coeff: float = 0.5) -> SplitDecision:
    # Aggregate labels and compute confidence
    labels = aggregate_labels(batches)
    confidence = np.mean([label.confidence for label in labels])
    
    # Compute split decision using weighted Hoeffding bound
    return should_split_with_gini_and_confidence(best_gain, second_best_gain, r, delta, n, 
                                                  tie_threshold, gini_coeff, confidence)

if __name__ == "__main__":
    batches = [[LabelingFunctionResult("lf1", "doc1", 1), LabelingFunctionResult("lf2", "doc1", 1)],
               [LabelingFunctionResult("lf1", "doc2", 0), LabelingFunctionResult("lf2", "doc2", 0)]]
    best_gain = 0.8
    second_best_gain = 0.4
    r = 0.5
    delta = 0.1
    n = 100
    decision = hybrid_label_hoeffding(batches, best_gain, second_best_gain, r, delta, n)
    print(decision)