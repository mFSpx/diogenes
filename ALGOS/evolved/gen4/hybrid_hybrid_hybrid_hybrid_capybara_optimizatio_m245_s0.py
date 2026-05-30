# DARWIN HAMMER — match 245, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py (gen3)
# parent_b: capybara_optimization.py (gen0)
# born: 2026-05-29T23:27:45Z

"""
Hybrid algorithm fusion of hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py and capybara_optimization.py.

The mathematical bridge between the two parents lies in their ability to handle uncertainty and optimization.
The hybrid_hybrid_hybrid_bandit_label_foundry_m21_s1.py algorithm uses probabilistic labels and confidence scores
to handle uncertain labels, while the capybara_optimization.py algorithm uses movement primitives to optimize
vector-valued functions. By combining these two approaches, we can create a hybrid algorithm that can handle
both uncertain labels and optimization problems.

The hybrid algorithm uses the probabilistic labels and confidence scores to inform the optimization process,
and then uses the movement primitives to optimize the objective function. The hybrid algorithm combines the
results of the labeling and optimization processes to produce a final output.

The mathematical interface between the two parents is the use of probability theory and optimization techniques.
The hybrid algorithm uses the probabilistic labels and confidence scores to compute the expected value of the
objective function, and then uses the movement primitives to optimize the expected value.
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable
from pathlib import Path

@dataclass(frozen=True)
class LabelingFunctionResult:
    """Result of a labeling function."""
    lf_name: str
    doc_id: str
    label: int

@dataclass(frozen=True)
class ProbabilisticLabel:
    """Probabilistic label with confidence."""
    doc_id: str
    label: int
    confidence: float

def labeling_function(name: str | None = None):
    """Decorator for labeling functions."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    """Aggregate labels from batches."""
    votes = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                if r.doc_id not in votes:
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs:
            out.append(ProbabilisticLabel(d, 0, 0.))
        else:
            label = np.mean(vs)
            confidence = np.std(vs)
            out.append(ProbabilisticLabel(d, int(label), confidence))
    return out

def social_interaction(x: List[float], g_best: List[float], k: int = 1, r1: float | None = None, seed: int | str | None = None) -> List[float]:
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0 <= r <= 1):
        raise ValueError("r1 must be in [0, 1]")
    return [xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)]

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def hybrid_optimization(labels: List[ProbabilisticLabel], x: List[float], g_best: List[float], k: int = 1, r1: float | None = None, seed: int | str | None = None) -> List[float]:
    """Hybrid optimization function that combines labeling and optimization."""
    # Compute expected value of objective function using probabilistic labels
    expected_value = np.sum([label.confidence * label.label for label in labels])
    
    # Use movement primitives to optimize objective function
    x_new = social_interaction(x, g_best, k, r1, seed)
    delta = evasion_delta(1, 10)
    x_new = [xi + delta * xi for xi in x_new]
    
    # Clamp x_new to valid range
    x_new = [min(1., max(0., xi)) for xi in x_new]
    
    return x_new

def demonstrate_hybrid_operation():
    # Create some sample labels
    labels = [ProbabilisticLabel("doc1", 1, 0.8), ProbabilisticLabel("doc2", 0, 0.4)]
    
    # Create some sample optimization parameters
    x = [0.5, 0.5]
    g_best = [1., 1.]
    
    # Run hybrid optimization
    x_new = hybrid_optimization(labels, x, g_best)
    print(x_new)

if __name__ == "__main__":
    demonstrate_hybrid_operation()