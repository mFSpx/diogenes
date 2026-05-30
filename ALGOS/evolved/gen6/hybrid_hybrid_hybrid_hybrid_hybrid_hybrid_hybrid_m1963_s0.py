# DARWIN HAMMER — match 1963, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s0.py (gen4)
# born: 2026-05-29T23:40:11Z

"""
This module defines a novel hybrid algorithm that mathematically fuses the core 
topologies of two parent algorithms: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m613_s1.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_percyphon_hyb_m1057_s0.py. The exact mathematical bridge 
between their structures lies in the concept of entropy, which is used in both 
algorithms to measure uncertainty or information. In the first algorithm, 
entropy is implicit in the calculation of the pheromone signal decay, while in 
the second algorithm, entropy is related to the morphological characteristics of 
the system. This hybrid algorithm leverages the concept of entropy to integrate 
the governing equations of both parent algorithms, creating a unified system 
that combines the path signature system with pheromone signal decay and 
morphological analysis.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def hybrid_signature_level1(path, morphology):
    path = np.asarray(path, dtype=float)
    si = sphericity_index(morphology.length, morphology.width, morphology.height)
    fi = flatness_index(morphology.length, morphology.width, morphology.height)
    return (path[-1] - path[0]) * si * fi

def hybrid_signature_level2(path, morphology):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    si = sphericity_index(morphology.length, morphology.width, morphology.height)
    fi = flatness_index(morphology.length, morphology.width, morphology.height)
    return (running.T @ increments) * si * fi

class Morphology:
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

if __name__ == "__main__":
    path = np.random.rand(10, 3)
    morphology = Morphology(1.0, 2.0, 3.0, 4.0)
    print(hybrid_signature_level1(path, morphology))
    print(hybrid_signature_level2(path, morphology))