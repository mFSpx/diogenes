# DARWIN HAMMER — match 4917, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m706_s2.py (gen5)
# born: 2026-05-29T23:58:52Z

"""
Hybrid Rectified Flow, Voronoi, and Tropical Gini-KAN System

This module fuses the Hybrid Rectified Flow and Tropical Gini-KAN algorithm 
(hybrid_hybrid_rectified_flo_hybrid_hybrid_gini_c_m1244_s3.py) with the 
Hybrid Voronoi-Epistemic Partitioning algorithm (hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m706_s2.py).
The mathematical bridge between the two structures is found by integrating 
the straight-line interpolant of Rectified Flow with the weighted Voronoi 
distance and the Gini-coefficient driven impurity measure with a radial-basis 
similarity matrix built from perceptual hashes.

The Rectified Flow's straight-line interpolant is used to generate input 
features for the Voronoi partitioning, which are then used to predict the 
output vector field of the Rectified Flow. The Gini coefficient provides a 
scalar impurity measure for a candidate split, which is then used to compute 
a hybrid split score that couples impurity, tropical belief and similarity 
in a single unified criterion.
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
# Rectified Flow and KAN utilities
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
# Voronoi and epistemic certainty helpers (Hybrid Voronoi-Epistemic Partitioning)
# ----------------------------------------------------------------------

def weighted_distance(p, s_i, confidence_i):
    """Weighted Euclidean distance: d̂(i, p) = ‖p – s_i‖ · (1 – confidence_i / 10_000)."""
    return np.linalg.norm(p - s_i) * (1 - confidence_i / 10000)

def voronoi_assignment(points, sites, confidences):
    """Assign points to the site with the minimum weighted distance."""
    return np.argmin([weighted_distance(p, s_i, c_i) for p, s_i, c_i in zip(points, sites, confidences)], axis=1)

# ----------------------------------------------------------------------
# Tropical Gini and RBF similarity utilities
# ----------------------------------------------------------------------

def gini_coefficient(values):
    """Gini impurity of a non-negative numeric collection."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) / (n - 1) for i, x in enumerate(xs, 1))

def rbf_similarity(hash1, hash2):
    """Radial-basis similarity matrix built from perceptual hashes."""
    return np.exp(-np.linalg.norm(hash1 - hash2))

# ----------------------------------------------------------------------
# Hybrid Rectified Flow and Voronoi utilities
# ----------------------------------------------------------------------

def hybrid_interpolant(x0, x1, t, sites, confidences):
    """Hybrid straight-line interpolant and Voronoi assignment."""
    interpolant_value = interpolant(x0, x1, t)
    assignment = voronoi_assignment(interpolant_value, sites, confidences)
    return interpolant_value, assignment

def hybrid_flow_target(x0, x1, t, sites, confidences):
    """Hybrid target vector field and Voronoi assignment."""
    interpolant_value, assignment = hybrid_interpolant(x0, x1, t, sites, confidences)
    return flow_target(x0, x1), assignment

# ----------------------------------------------------------------------
# Hybrid Tropical Gini and Voronoi utilities
# ----------------------------------------------------------------------

def hybrid_gini_score(values, assignment):
    """Hybrid Gini score and Voronoi assignment."""
    gini_value = gini_coefficient(values)
    voronoi_value = np.mean([weighted_distance(p, s_i, c_i) for p, s_i, c_i in zip(values, assignment, sites)])
    return gini_value + voronoi_value

def hybrid_split_score(values, assignment, sites, confidences):
    """Hybrid split score that couples impurity, tropical belief and similarity."""
    gini_value = gini_coefficient(values)
    voronoi_value = np.mean([weighted_distance(p, s_i, c_i) for p, s_i, c_i in zip(values, assignment, confidences)])
    rbf_value = np.mean(rbf_similarity(hash1, hash2) for hash1, hash2 in zip(assignment, sites))
    return gini_value + voronoi_value + rbf_value

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    sites = np.random.rand(10, 2)
    confidences = np.random.rand(10)
    values = np.random.rand(100, 2)
    hash1 = np.random.rand(100)
    hash2 = np.random.rand(100)
    t = np.random.rand(100)
    x0 = np.random.rand(100, 2)
    x1 = np.random.rand(100, 2)
    
    interpolant_value, assignment = hybrid_interpolant(x0, x1, t, sites, confidences)
    flow_target_value, assignment = hybrid_flow_target(x0, x1, t, sites, confidences)
    gini_value = gini_coefficient(values)
    voronoi_value = np.mean([weighted_distance(p, s_i, c_i) for p, s_i, c_i in zip(values, assignment, confidences)])
    rbf_value = np.mean(rbf_similarity(hash1, hash2) for hash1, hash2 in zip(assignment, sites))
    hybrid_score = hybrid_split_score(values, assignment, sites, confidences)
    
    print("Hybrid interpolant value:", interpolant_value)
    print("Hybrid flow target value:", flow_target_value)
    print("Gini value:", gini_value)
    print("Voronoi value:", voronoi_value)
    print("RBF value:", rbf_value)
    print("Hybrid score:", hybrid_score)