# DARWIN HAMMER — match 1064, survivor 0
# gen: 5
# parent_a: hybrid_ssim_hybrid_hybrid_hybrid_m134_s3.py (gen4)
# parent_b: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# born: 2026-05-29T23:32:34Z

import math
import random
import sys
import pathlib
from typing import Sequence, Dict, Tuple, FrozenSet
import numpy as np

class Multivector:
    """Simple Euclidean Clifford algebra up to grade 2."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        # Remove near‑zero components for cleanliness
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }
        self.n = int(n)  # dimension of the underlying vector space

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, value in other.components.items():
            if blade not in result:
                result[blade] = value
            else:
                result[blade] += value
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        result = {}
        for blade1, value1 in self.components.items():
            for blade2, value2 in other.components.items():
                new_blade = frozenset(blade1.union(blade2))
                if new_blade not in result:
                    result[new_blade] = value1 * value2
                else:
                    result[new_blade] += value1 * value2
        return Multivector(result, self.n)

def stats_to_multivector(seq: Sequence[float]) -> Multivector:
    """Convert a 1‑D sequence into a multivector of moments."""
    mean = np.mean(seq)
    variance = np.var(seq)
    covariance = 0
    multivector_components = {frozenset(): mean}
    multivector_components[frozenset([0])]= variance
    multivector_components[frozenset([0,1])] = covariance
    return Multivector(multivector_components, 2)

def geometric_ssim(x: Sequence[float], y: Sequence[float]) -> float:
    """Hybrid similarity using the geometric product of the moment multivectors."""
    multivector_x = stats_to_multivector(x)
    multivector_y = stats_to_multivector(y)
    product = multivector_x * multivector_y
    return product.scalar_part()

def sketch_to_multivector(sketch: Sequence[int]) -> Multivector:
    """Convert a count-min sketch into a multivector."""
    mean = np.mean(sketch)
    variance = np.var(sketch)
    covariance = 0
    multivector_components = {frozenset(): mean}
    multivector_components[frozenset([0])]= variance
    multivector_components[frozenset([0,1])] = covariance
    return Multivector(multivector_components, 2)

def estimate_rlct_from_losses(train_losses_per_n, n_values):
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

def hybrid_ssim_rlct(data, width=64, depth=4):
    sketch = count_min_sketch(data, width, depth)
    flat_sketch = [item for sublist in sketch for item in sublist]
    losses = [item for item in flat_sketch if item > 0]
    n_values = [i+1 for i in range(len(losses))]
    if len(losses) > 1:
        rlct = estimate_rlct_from_losses(losses, n_values)
    else:
        rlct = 0
    multivector_sketch = sketch_to_multivector(flat_sketch)
    return geometric_ssim(data, data) + rlct

def basic_ssim(x: Sequence[float], y: Sequence[float]) -> float:
    """Classic SSIM."""
    return np.mean((x - np.mean(x)) * (y - np.mean(y))) / np.sqrt(np.var(x) * np.var(y))

def main():
    data = np.random.rand(100)
    print(hybrid_ssim_rlct(data))
    print(basic_ssim(data, data))

if __name__ == "__main__":
    main()