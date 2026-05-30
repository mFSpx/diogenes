# DARWIN HAMMER — match 2864, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1407_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m2091_s0.py (gen4)
# born: 2026-05-29T23:46:24Z

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * max(height, 1e-6))

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def compute_morphology_vector(m: Morphology) -> np.ndarray:
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rti = righting_time_index(m)
    rp = recovery_priority(m)
    return np.array([sph, flat, rti, rp])

class Multivector:
    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        return {b: v for b, v in self.components.items() if len(b) == k}

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return tuple(sorted(lst)), sign
    return tuple(sorted(lst)), sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return result, sign

def multivector_from_morphology_vector(vector: np.ndarray) -> Multivector:
    components = {}
    for i, value in enumerate(vector):
        components[tuple([i])] = value
    return Multivector(components, len(vector))

def geometric_product(multivector_a: Multivector, multivector_b: Multivector) -> Multivector:
    components = {}
    for blade_a, value_a in multivector_a.components.items():
        for blade_b, value_b in multivector_b.components.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            if blade in components:
                components[blade] += sign * value_a * value_b
            else:
                components[blade] = sign * value_a * value_b
    return Multivector(components, multivector_a.n)

def hybrid_recovery_priority(m: Morphology) -> float:
    vector = compute_morphology_vector(m)
    multivector = multivector_from_morphology_vector(vector)
    multivector_squared = geometric_product(multivector, multivector)
    updated_vector = np.array([multivector_squared.components.get(tuple([i]), 0) for i in range(len(vector))])
    updated_morphology = Morphology(m.length, m.width, m.height, m.mass)
    updated_morphology_vector = compute_morphology_vector(updated_morphology)
    return recovery_priority(Morphology(updated_morphology.length, updated_morphology.width, updated_morphology.height, m.mass))

def hybrid_righting_time_index(m: Morphology) -> float:
    vector = compute_morphology_vector(m)
    multivector = multivector_from_morphology_vector(vector)
    multivector_squared = geometric_product(multivector, multivector)
    updated_vector = np.array([multivector_squared.components.get(tuple([i]), 0) for i in range(len(vector))])
    updated_morphology = Morphology(m.length, m.width, m.height, m.mass)
    return righting_time_index(updated_morphology)

if __name__ == "__main__":
    m = Morphology(1.0, 2.0, 3.0, 4.0)
    print(hybrid_recovery_priority(m))
    print(hybrid_righting_time_index(m))