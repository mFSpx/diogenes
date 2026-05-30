# DARWIN HAMMER — match 1669, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s0.py (gen4)
# born: 2026-05-29T23:38:09Z

import math
import random
import sys
import pathlib
import numpy as np
from typing import Tuple, List, Dict, Any

class Multivector:
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector({k: v for k, v in result.items() if abs(v) > 1e-15}, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        neg = Multivector({k: -v for k, v in other.components.items()}, other.n)
        return self.__add__(neg)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({k: scalar * v for k, v in self.components.items()}, self.n)

    def __truediv__(self, scalar: float) -> "Multivector":
        return Multivector({k: v / scalar for k, v in self.components.items()}, self.n)

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l
            ) ** order * (
                1
                if (x <= t[i + 1])
                else (
                    (x - t[i + 1]) / denom_r
                ) ** order
            )
            B_new[:, i] = term_l
        B = np.hstack((B, B_new))
    return B

def lead_lag_multivector_transform(path):
    """
    Apply lead-lag transform to path signature and represent as multivector.
    """
    path_sig = lead_lag_transform(path)
    multivector = Multivector({frozenset(): 0.0}, len(path))
    for i in range(len(path_sig)):
        multivector.components[frozenset([i])] = path_sig[i, 0]
        multivector.components[frozenset([i + 1])] = path_sig[i, 1]
    return multivector

def hybrid_update_rule(multivector, path, k=3):
    """
    Hybrid update rule combining flux-based conductance update primitive with radial-basis surrogate model.
    """
    B = bspline_basis(path, np.linspace(0, 1, 100), k)
    multivector = multivector * B[0, :]
    return multivector

def hybrid_path_signature_update_rule(path, k=3):
    """
    Hybrid update rule for path signature, combining path signature-weighted sum with radial-basis surrogate model.
    """
    multivector = lead_lag_multivector_transform(path)
    multivector = hybrid_update_rule(multivector, path, k)
    return multivector

if __name__ == "__main__":
    np.random.seed(0)
    path = np.random.rand(10, 2)
    multivector = hybrid_path_signature_update_rule(path)
    print(multivector)