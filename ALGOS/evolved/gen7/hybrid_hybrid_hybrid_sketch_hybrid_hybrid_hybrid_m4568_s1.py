# DARWIN HAMMER — match 4568, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_sketch_m983_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2381_s4.py (gen6)
# born: 2026-05-29T23:56:34Z

"""Hybrid Sketch–Geometric Bandit Module

Parents:
- hybrid_hybrid_sketches_rlct_hybrid_hybrid_sketch_m983_s1.py (count‑min sketch,
  HyperLogLog, bandit policy, VRAM budgeting)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2381_s4.py (Clifford algebra,
  multivector representation, geometric product)

Mathematical bridge:
The count‑min sketch table is interpreted as a grade‑1 multivector whose basis
blades correspond to the (depth, column) pair of each hash counter.  A sketch
therefore becomes a vector in a Clifford algebra ℝ^{d·w}.  The geometric
product of two such multivectors yields higher‑grade blades that encode joint
frequency information of two item sets.  The scalar part of the product (grade‑0)
acts as a fused reward estimate, while the magnitude (norm) of the multivector
relates to the HyperLogLog cardinality estimate and is used as a confidence
bound.  VRAM budgeting is driven by the number of non‑zero components of the
resulting multivector (each component occupies 8 bytes as a double).

This module provides a unified API that:
1. builds a multivector from a count‑min sketch,
2. combines sketches geometrically,
3. produces bandit actions whose expected reward and confidence bound are
   derived from the algebraic fusion and respect a VRAM budget.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Bandit & VRAM utilities (Parent A)
# ----------------------------------------------------------------------
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

@dataclass
class VRAMBudget:
    budget_mb: int
    reserve_mb: int
    used_mb: int = 0

    def remaining_mb(self) -> int:
        return max(0, self.budget_mb - self.used_mb - self.reserve_mb)

    def allocate(self, size_bytes: int) -> bool:
        size_mb = (size_bytes + (1 << 20) - 1) >> 20
        if size_mb <= self.remaining_mb():
            self.used_mb += size_mb
            return True
        return False

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def hyper_log_log(items: List[str]) -> int:
    """Simple HyperLogLog cardinality estimator (parent A)."""
    m = 0
    for item in items:
        h = int(hashlib.sha256(item.encode()).hexdigest(), 16)
        w = (h & 0xFFFFFFFF) % (2**32)
        m = max(m, _rho(w))
    return 2 ** m

def _rho(w: int) -> int:
    # position of leftmost 1-bit (1‑based)
    return math.floor(math.log2((w ^ (w >> 1)) + 1))

# ----------------------------------------------------------------------
# Clifford algebra utilities (Parent B)
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
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Grade‑wise sparse representation of a Clifford algebra element."""
    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def __add__(self, other: "Multivector") -> "Multivector":
        comp = self.components.copy()
        for k, v in other.components.items():
            comp[k] = comp.get(k, 0.0) + v
        return Multivector(comp, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({k: v * scalar for k, v in self.components.items()}, self.n)

    __rmul__ = __mul__

    def copy(self) -> "Multivector":
        return Multivector(self.components.copy(), self.n)

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.components.values()))

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "0"
        terms = []
        for k, v in sorted(self.components.items(), key=lambda kv: (len(kv[0]), kv[0])):
            basis = "e" + "".join(str(idx) for idx in sorted(k)) if k else ""
            terms.append(f"{v:.3g}{'*' if basis else ''}{basis}")
        return " + ".join(terms)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Full geometric product (grade‑mixing)."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            new_blade, sign = _multiply_blades(blade_a, blade_b)
            result[new_blade] = result.get(new_blade, 0.0) + sign * coeff_a * coeff_b
    return Multivector(result, a.n)


# ----------------------------------------------------------------------
# Hybrid operations – the mathematical fusion
# ----------------------------------------------------------------------
def sketch_to_multivector(items: List[str], width: int = 64, depth: int = 4) -> Multivector:
    """
    Build a grade‑1 multivector from a count‑min sketch.
    Each (depth, column) pair maps to a unique basis blade index:
        idx = depth * width + column
    The blade coefficient equals the counter value.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            col = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][col] += 1

    components: Dict[FrozenSet[int], float] = {}
    for d, row in enumerate(table):
        for c, count in enumerate(row):
            if count:
                idx = d * width + c
                components[frozenset({idx})] = float(count)

    return Multivector(components, n=depth * width)


def combine_sketches_geometric(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """
    Fuse two sketches via the geometric product.
    The resulting multivector contains joint frequency information.
    """
    return geometric_product(mv_a, mv_b)


def bandit_actions_from_context(
    context_items: List[str],
    budget: VRAMBudget,
    width: int = 64,
    depth: int = 4,
    algorithm_name: str = "HybridSketchGeometric"
) -> List[BanditAction]:
    """
    Produce a list of BanditAction objects for a given context.
    Steps:
    1. Create a multivector sketch of the context.
    2. Estimate cardinality with HyperLogLog (used as uncertainty scale).
    3. Check VRAM budget; each component costs 8 bytes.
    4. For each distinct action_id observed in the sketch (derived from basis idx),
       compute expected reward from the global policy and a confidence bound that
       grows with the multivector norm and HLL estimate.
    """
    # 1. Sketch → multivector
    mv = sketch_to_multivector(context_items, width, depth)

    # 2. HyperLogLog cardinality (uncertainty proxy)
    hll_est = hyper_log_log(context_items)

    # 3. VRAM consumption check
    required_bytes = len(mv.components) * 8  # double precision per component
    if not budget.allocate(required_bytes):
        raise MemoryError(f"Insufficient VRAM: need {required_bytes} bytes, "
                          f"{budget.remaining_mb()} MiB remaining.")

    # 4. Build actions
    actions: List[BanditAction] = []
    norm = mv.norm()
    for blade, coeff in mv.components.items():
        # Derive a deterministic action identifier from the blade index set
        action_id = "a_" + "_".join(str(i) for i in sorted(blade))
        propensity = min(1.0, coeff / (sum(mv.components.values()) + 1e-9))
        expected = _reward(action_id)  # policy may be empty → 0.0
        # Confidence bound combines sketch norm and HLL estimate
        conf = (norm / (len(mv.components) + 1)) * math.log2(hll_est + 1)
        actions.append(
            BanditAction(
                action_id=action_id,
                propensity=propensity,
                expected_reward=expected,
                confidence_bound=conf,
                algorithm=algorithm_name,
            )
        )
    return actions


# ----------------------------------------------------------------------
# Additional helper demonstrating the full pipeline
# ----------------------------------------------------------------------
def fused_reward_estimate(context_a: List[str], context_b: List[str]) -> float:
    """
    Compute a scalar reward estimate by:
    - converting each context to a multivector sketch,
    - geometrically combining them,
    - extracting the scalar (grade‑0) part.
    If the scalar part is zero (common for pure grade‑1 inputs),
    fall back to the norm as a proxy.
    """
    mv_a = sketch_to_multivector(context_a)
    mv_b = sketch_to_multivector(context_b)
    combined = combine_sketches_geometric(mv_a, mv_b)
    scalar = combined.scalar_part()
    return scalar if abs(scalar) > 1e-9 else combined.norm()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed
    random.seed(0)

    # Dummy contexts
    ctx1 = [f"user{i}" for i in range(10)]
    ctx2 = [f"item{j}" for j in range(5)]

    # VRAM budget of 1 MiB with 0 reserve
    vram = VRAMBudget(budget_mb=1, reserve_mb=0)

    # Generate actions for ctx1
    actions = bandit_actions_from_context(ctx1, vram)
    print("Generated actions (first context):")
    for act in actions[:5]:
        print(act)

    # Compute fused reward estimate between two contexts
    est = fused_reward_estimate(ctx1, ctx2)
    print("\nFused scalar reward estimate between ctx1 and ctx2:", est)

    # Demonstrate policy update
    updates = [
        BanditUpdate(context_id="c1", action_id=act.action_id, reward=random.random(), propensity=act.propensity)
        for act in actions[:3]
    ]
    update_policy(updates)
    print("\nPolicy after updates:", _POLICY)