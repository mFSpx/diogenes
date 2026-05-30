# DARWIN HAMMER — match 2753, survivor 8
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hard_t_m1426_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hdc_serpentin_m1033_s0.py (gen3)
# born: 2026-05-29T23:45:48Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Clifford algebra helpers (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)
                n -= 2
                i = -1  
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    if not blade_a:
        return blade_b, 1
    if not blade_b:
        return blade_a, 1
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(mv_a: Dict[FrozenSet[int], float],
                      mv_b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def multivector_norm(mv: Dict[FrozenSet[int], float]) -> float:
    return math.sqrt(sum(c * c for c in mv.values()))


def point_to_multivector(coords: Tuple[float, ...]) -> Dict[FrozenSet[int], float]:
    mv: Dict[FrozenSet[int], float] = {}
    for idx, val in enumerate(coords, start=1):
        if abs(val) > 1e-12:
            mv[frozenset({idx})] = val
    return mv


def geometric_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    mv1 = point_to_multivector(p1)
    mv2 = point_to_multivector(p2)
    gp = geometric_product(mv1, mv2)          
    return multivector_norm(gp)


# ----------------------------------------------------------------------
# Sphericity & Morphology (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float = 1.0


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    return (length * width * height) ** (1.0 / 3.0) / length


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    def __init__(self, width: int = 64, depth: int = 4, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("Width and depth must be positive integers.")
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def _hash(item: str, depth: int, width: int) -> List[int]:
        return [
            int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            for d in range(depth)
        ]

    def update(self, item: str, inc: int = 1) -> None:
        indices = self._hash(item, self.depth, self.width)
        for d, idx in enumerate(indices):
            self.tables[d, idx] += inc

    def estimate(self, item: str) -> int:
        indices = self._hash(item, self.depth, self.width)
        return int(min(self.tables[d, idx] for d, idx in enumerate(indices)))


# ----------------------------------------------------------------------
# Bandit structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "HybridGeometricCMSBandit"


def _ucb_score(action: BanditAction, total_counts: int, exploration_coef: float = 2.0) -> float:
    if action.propensity == 0:
        return float('inf')
    avg_reward = action.expected_reward / action.propensity
    bonus = exploration_coef * math.sqrt(math.log(max(total_counts, 1)) / action.propensity)
    return avg_reward + bonus


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def update_cms_with_distance(cms: CountMinSketch,
                             distance: float,
                             scale: float = 1.0) -> None:
    token = f"{int(distance * scale):08d}"
    cms.update(token)


def compute_novelty_estimate(cms: CountMinSketch,
                             distance: float,
                             scale: float = 1.0) -> float:
    token = f"{int(distance * scale):08d}"
    count = cms.estimate(token)
    return 1.0 / (1.0 + count)


def select_action_via_bandit(actions: List[BanditAction], cms: CountMinSketch, total_counts: int) -> str:
    best_score = float('-inf')
    best_action = None
    for action in actions:
        distance = float(action.action_id)
        novelty = compute_novelty_estimate(cms, distance)
        score = _ucb_score(action, total_counts)
        if score > best_score:
            best_score = score
            best_action = action.action_id
    return best_action


def main():
    # Example usage:
    cms = CountMinSketch(width=128, depth=8)
    actions = [BanditAction(action_id=str(i)) for i in range(10)]
    total_counts = 100
    best_action = select_action_via_bandit(actions, cms, total_counts)
    print(f"Best action: {best_action}")


if __name__ == "__main__":
    main()