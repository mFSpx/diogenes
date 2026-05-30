# DARWIN HAMMER — match 3845, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py (gen3)
# parent_b: hybrid_hybrid_caputo_fracti_hybrid_hybrid_regret_m880_s2.py (gen4)
# born: 2026-05-29T23:51:57Z

"""
Hybrid Fusion of:
- ``hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py`` (Clifford geometric product for resource allocation)
- ``hybrid_hybrid_caputo_fracti_hybrid_hybrid_regret_m880_s2.py`` (Caputo fractional memory kernel and regret‑entropy metric)

Mathematical Bridge
------------------
Both parents operate on a **temporal sequence of updates**.  
Parent A treats each update as a multivector ΔR and combines it with the
current allocation R using the **Clifford geometric product**.  
Parent B supplies a **Caputo fractional kernel** ϕ(t;α) that yields a set of
weights wₖ = ϕ(T‑1‑k;α)/Σⱼϕ(T‑1‑j;α) and a regret‑driven **ternary symbol**
sₖ = sign(Δcₖ ‑ c_min) ∈ {‑1,0,+1}, whose weighted entropy is added to the
cost.

The hybrid algorithm therefore:
1. Computes fractional‑kernel weights wₖ for the whole update horizon.
2. Uses the geometric product to propagate the multivector allocation,
   weighting each product by wₖ.
3. Evaluates a scalar “cost” from each multivector update, derives a
   ternary regret symbol, and forms the weighted entropy term.
4. Returns a **hybrid metric**  

   H = Σₖ wₖ·costₖ + λ·H_entropy ,

   together with the weighted geometric‑product allocation.

The code below implements this fusion with three public functions that
demonstrate the combined behaviour.
"""

import math
import random
import sys
import pathlib
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Clifford‑algebra helpers (Parent A)
# ---------------------------------------------------------------------------

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Duplicate indices cancel (e² = 0) and flip the sign accordingly.
    """
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
                # duplicate basis vector squares to zero → remove both
                lst.pop(j)
                lst.pop(j)  # the element that shifted into position j
                n -= 2
                i -= 1  # stay on the same outer iteration
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades using the exterior‑algebra sign rule.
    Returns (result_blade, sign). The scalar part is represented by an empty
    frozenset.
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign

def geometric_product(
    A: Dict[FrozenSet[int], float],
    B: Dict[FrozenSet[int], float]
) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors represented as dicts
    {blade: coefficient}. Returns a new multivector dict.
    """
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in A.items():
        for blade_b, coeff_b in B.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-12}

def multivector_norm(M: Dict[FrozenSet[int], float]) -> float:
    """L1‑norm of a multivector (sum of absolute scalar coefficients)."""
    return sum(abs(c) for c in M.values())

# ---------------------------------------------------------------------------
# Caputo fractional kernel & entropy (Parent B)
# ---------------------------------------------------------------------------

def caputo_weights(T: int, alpha: float) -> np.ndarray:
    """Return normalized Caputo‑type weights wₖ for k = 0 … T‑1.
    The kernel φ(k;α) = (k+1)^{α‑1} reproduces the power‑law memory
    used in the original parent.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    ks = np.arange(T)  # 0 … T‑1
    phi = (ks + 1) ** (alpha - 1.0)
    total = phi.sum()
    if total == 0:
        raise RuntimeError("Kernel sum vanished")
    return phi / total

def ternary_symbol(value: float, eps: float = 1e-12) -> int:
    """Map a real value to a ternary symbol:
    -1 if value < -eps, 0 if |value| ≤ eps, +1 if value > eps.
    """
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0

def weighted_entropy(symbols: List[int], weights: np.ndarray) -> float:
    """Shannon entropy of the weighted ternary distribution."""
    if len(symbols) != len(weights):
        raise ValueError("Length mismatch")
    # accumulate weighted frequencies
    freq = { -1: 0.0, 0: 0.0, 1: 0.0 }
    for s, w in zip(symbols, weights):
        freq[s] += w
    # normalize (should already sum to 1, but guard against round‑off)
    total = sum(freq.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for p in freq.values():
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

# ---------------------------------------------------------------------------
# Hybrid core (fusion of both parents)
# ---------------------------------------------------------------------------

def hybrid_allocation(
    R0: Dict[FrozenSet[int], float],
    updates: List[Dict[FrozenSet[int], float]],
    alpha: float = 0.7,
    lam: float = 1.0
) -> Tuple[Dict[FrozenSet[int], float], float]:
    """
    Perform a fractional‑memory weighted allocation.

    Parameters
    ----------
    R0 : multivector
        Initial resource‑allocation multivector.
    updates : list of multivectors
        Sequential updates ΔR₀ … ΔR_{T‑1}.
    alpha : float
        Order of the Caputo kernel (0 < α ≤ 1). Controls memory depth.
    lam : float
        Weight of the entropy term in the hybrid metric.

    Returns
    -------
    R_final : multivector
        Weighted geometric‑product accumulation of the updates.
    H : float
        Hybrid metric = weighted cost + λ·entropy.
    """
    T = len(updates)
    if T == 0:
        return R0.copy(), 0.0

    # 1. fractional weights
    w = caputo_weights(T, alpha)  # shape (T,)

    # 2. compute scalar costs from each update (L1 norm)
    costs = np.array([multivector_norm(u) for u in updates])

    # 3. ternary regret symbols (relative to minimal cost seen so far)
    min_cost = costs.min()
    symbols = [ternary_symbol(c - min_cost) for c in costs]

    # 4. weighted entropy term
    H_entropy = weighted_entropy(symbols, w)

    # 5. weighted cost term
    weighted_cost = float(np.dot(w, costs))

    # 6. hybrid metric
    H = weighted_cost + lam * H_entropy

    # 7. weighted geometric product accumulation
    R = R0.copy()
    for wk, Δ in zip(w, updates):
        # apply geometric product, then blend with current allocation
        prod = geometric_product(R, Δ)
        # weighted blend: R ← (1‑wk)·R + wk·prod
        R = {blade: (1 - wk) * R.get(blade, 0.0) + wk * prod.get(blade, 0.0)
             for blade in set(R) | set(prod)}
    return R, H

def random_multivector(dim: int, max_grade: int = 2) -> Dict[FrozenSet[int], float]:
    """
    Generate a random multivector in a Clifford algebra of given dimension.
    Only blades up to `max_grade` are populated.
    """
    mv: Dict[FrozenSet[int], float] = {}
    for grade in range(max_grade + 1):
        # number of possible blades of this grade = C(dim, grade)
        if grade == 0:
            # scalar part
            mv[frozenset()] = random.uniform(-1.0, 1.0)
        else:
            # sample a few random blades of this grade
            for _ in range(random.randint(0, dim)):
                blade = frozenset(random.sample(range(dim), grade))
                mv[blade] = random.uniform(-1.0, 1.0)
    # prune zeros
    return {b: c for b, c in mv.items() if abs(c) > 1e-12}

def demo_hybrid():
    """Simple demonstration that runs the hybrid allocation on synthetic data."""
    dim = 4  # 4‑dimensional space → 2⁴‑dim algebra
    R0 = random_multivector(dim)
    updates = [random_multivector(dim) for _ in range(5)]
    R_final, H = hybrid_allocation(R0, updates, alpha=0.8, lam=0.5)

    print("Initial multivector (scalar part only):", R0.get(frozenset(), 0.0))
    print("Final multivector (scalar part only):  ", R_final.get(frozenset(), 0.0))
    print("Hybrid metric H:", H)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_hybrid()