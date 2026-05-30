# DARWIN HAMMER — match 4917, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m706_s2.py (gen5)
# born: 2026-05-29T23:58:52Z

"""
This module fuses two parent algorithms: 
- hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s3.py: Hybrid Rectified Flow and Tropical Gini-KAN System
- hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m706_s2.py: Hybrid Voronoi-Epistemic Partitioning

The mathematical bridge between the two structures is found by integrating 
the straight-line interpolant of Rectified Flow with the epistemic certainty 
handling via CertaintyFlag objects and the weighted distance metric from 
the Voronoi partitioning. This yields a unified system that combines the 
predictive power of the Rectified Flow with the uncertainty handling of 
the epistemic certainty flags and the spatial partitioning of the Voronoi 
algorithm.
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = frozenset(int)          # basis blade represented by a set of indices
Multivector = dict(Blade, float)  # mapping blade → coefficient

# ----------------------------------------------------------------------
# CertaintyFlag class
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable description of epistemic certainty."""
    label: str
    confidence_bps: int               # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: str

# ----------------------------------------------------------------------
# Rectified Flow utilities
# ----------------------------------------------------------------------
def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0.

    Broadcasts t over a leading batch dimension.  If x0 has shape (B, d) and t
    has shape (B,), t is reshaped to (B, 1) 
    """
    t = np.reshape(t, (-1, 1))
    return t * x1 + (1 - t) * x0

def flow_target(x0, x1):
    """Target vector field: v_theta(Z_t, t) = (X_1 - X_0)."""
    return x1 - x0

# ----------------------------------------------------------------------
# Gini coefficient utility
# ----------------------------------------------------------------------
def gini_coefficient(values):
    """Gini impurity of a non-negative numeric collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - (n - 1)) * x for i, x in enumerate(xs)) / (n * sum(xs))

# ----------------------------------------------------------------------
# Weighted distance metric
# ----------------------------------------------------------------------
def weighted_distance(point, site, certainty_flag):
    """Weighted distance metric: d̂(i, p) = ‖p – s_i‖ · (1 – confidence_i / 10_000)."""
    distance = np.linalg.norm(np.array(point) - np.array(site))
    return distance * (1 - certainty_flag.confidence_bps / 10000)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_flow_target(point, site, certainty_flag, x0, x1):
    """Hybrid flow target: combines the Rectified Flow target with the weighted distance metric."""
    return flow_target(x0, x1) * weighted_distance(point, site, certainty_flag)

def hybrid_gini_coefficient(values, certainty_flags):
    """Hybrid Gini coefficient: combines the Gini impurity with the epistemic certainty."""
    return gini_coefficient(values) * sum(certainty_flag.confidence_bps for certainty_flag in certainty_flags) / 10000

def hybrid_partitioning(points, sites, certainty_flags):
    """Hybrid partitioning: combines the Voronoi partitioning with the epistemic certainty."""
    assignments = []
    for point in points:
        distances = [weighted_distance(point, site, certainty_flag) for site, certainty_flag in zip(sites, certainty_flags)]
        assignments.append(np.argmin(distances))
    return assignments

if __name__ == "__main__":
    points = [(1, 2), (3, 4), (5, 6)]
    sites = [(0, 0), (2, 2), (4, 4)]
    certainty_flags = [CertaintyFlag("FACT", 5000, "HIGH"), CertaintyFlag("PROBABLE", 3000, "MEDIUM"), CertaintyFlag("POSSIBLE", 1000, "LOW")]
    x0 = np.array([1, 2])
    x1 = np.array([3, 4])
    
    print(hybrid_flow_target(points[0], sites[0], certainty_flags[0], x0, x1))
    print(hybrid_gini_coefficient([1, 2, 3], certainty_flags))
    print(hybrid_partitioning(points, sites, certainty_flags))