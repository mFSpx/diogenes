# DARWIN HAMMER — match 5184, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1988_s1.py (gen5)
# parent_b: hybrid_hoeffding_tree_gini_coefficient_m13_s5.py (gen1)
# born: 2026-05-30T00:00:26Z

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Callable, Any
from collections import Counter
from pathlib import Path

# Global configuration
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          
ALPHA: float = 5.0             
LAMBDA: float = 0.7            
MINHASH_K: int = 64            
MAX64: int = (1 << 64) - 1     
SEED: int = 12345              

random.seed(SEED)
np.random.seed(SEED)

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound for a range‑bounded random variable."""
    if r <= 0 or not (0.0 < delta < 1.0) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float,
                 second_best_gain: float,
                 r: float,
                 delta: float,
                 n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    """Decide whether to split a node using a Hoeffding bound that is
    adaptively scaled by the current liquid‑time constant."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = ("gap_exceeds_bound" if gap > eps
              else ("tie_threshold" if eps < tie_threshold
                    else "wait"))
    return SplitDecision(split, eps, gap, reason)

def tropical_max_plus(v1: float, v2: float) -> float:
    """Tropical max-plus operation."""
    return max(v1, v2)

def weighted_voting(gains: List[float], weights: List[float]) -> float:
    """Weighted voting system using tropical max-plus operation."""
    return max([g * w for g, w in zip(gains, weights)])

def hybrid_hoeffding_tropical(gains: List[float], 
                             weights: List[float], 
                             r: float, 
                             delta: float, 
                             n: int) -> SplitDecision:
    """Hybrid Hoeffding-tropical max-plus decision rule."""
    sorted_gains = sorted(enumerate(gains), key=lambda x: x[1], reverse=True)
    best_gain = weighted_voting([gains[i] for i, _ in sorted_gains], weights)
    second_best_gain = weighted_voting([gains[i] for i, _ in sorted_gains[1:]], weights)
    return should_split(best_gain, second_best_gain, r, delta, n)

def improved_hybrid_hoeffding_tropical(gains: List[float], 
                                       weights: List[float], 
                                       r: float, 
                                       delta: float, 
                                       n: int) -> SplitDecision:
    """Improved hybrid Hoeffding-tropical max-plus decision rule."""
    sorted_gains = sorted(enumerate(gains), key=lambda x: x[1], reverse=True)
    best_gain = weighted_voting([gains[i] for i, _ in sorted_gains], weights)
    second_best_gain = weighted_voting([gains[i] for i, _ in sorted_gains[1:]], weights)
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < 0.05
    reason = ("gap_exceeds_bound" if gap > eps
              else ("tie_threshold" if eps < 0.05
                    else "wait"))
    return SplitDecision(split, eps, gap, reason)

if __name__ == "__main__":
    gains = [0.5, 0.3, 0.2]
    weights = [0.4, 0.3, 0.3]
    r = 0.5
    delta = 0.1
    n = 100
    decision = improved_hybrid_hoeffding_tropical(gains, weights, r, delta, n)
    print(decision)