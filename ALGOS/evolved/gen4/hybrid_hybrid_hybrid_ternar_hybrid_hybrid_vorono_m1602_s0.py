# DARWIN HAMMER — match 1602, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s0.py (gen3)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s2.py (gen2)
# born: 2026-05-29T23:37:39Z

"""
Hybrid algorithm merging the Bayesian edge-prior update and ternary-router style graph handling 
from `hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s0.py` (Parent A) 
with the Voronoi-partition and thermal-morphology circuit breaking from 
`hybrid_hybrid_voronoi_parti_hybrid_endpoint_circ_m314_s2.py` (Parent B).

Mathematical bridge:
* The Bayesian edge-prior update from Parent A supplies the likelihood *ℓ* 
  and false-positive rate *α* used in the circuit-breaker failure threshold *τ* 
  of Parent B.
* The Voronoi partition from Parent B groups nodes in the graph from Parent A.
* The morphology (sphericity *σ*) from Parent B scales the failure threshold *τ*.
"""

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic I/O helpers
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def update_edge_priors(priors: np.ndarray, likelihood: float, evidence: np.ndarray, alpha: float) -> np.ndarray:
    """Update edge priors using Bayes' rule."""
    return (priors * likelihood * evidence) / (priors * likelihood * evidence + (1 - priors) * alpha)

def compute_material_cost(lengths: np.ndarray, priors: np.ndarray) -> float:
    """Compute material cost as the dot product of lengths and priors."""
    return np.dot(lengths, priors)

# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15  # K
    t_high: float = 307.15  # K
    delta_h_low: float = -45_000.0
    delta_h_high: float = 20_000.0

def schoolfield_activity(params: SchoolfieldParams, temperature: float) -> float:
    """Compute activity using the Schoolfield model."""
    t = temperature
    return (params.rho_25 * 
            math.exp(params.delta_h_activation * (1 / params.t_low - 1 / t)) / 
            (1 + math.exp(params.delta_h_low * (1 / params.t_low - 1 / t)) + 
             math.exp(params.delta_h_high * (1 / params.t_high - 1 / t))))

def circuit_breaker_threshold(activity: float, sphericity: float, base_threshold: int) -> int:
    """Compute circuit-breaker failure threshold."""
    return max(1, int(base_threshold * (1 - activity) * sphericity))

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_voronoi_bayes(points: List[Tuple[float, float]], 
                          seeds: List[Tuple[float, float]], 
                          priors: np.ndarray, 
                          likelihood: float, 
                          evidence: np.ndarray, 
                          alpha: float, 
                          schoolfield_params: SchoolfieldParams, 
                          base_threshold: int) -> Dict[int, Tuple[float, int]]:
    """Assign points to Voronoi regions and compute circuit-breaker thresholds."""
    regions = assign(points, seeds)
    thresholds = {}
    for i, region in regions.items():
        temperature = schoolfield_params.t_low + (schoolfield_params.t_high - schoolfield_params.t_low) * random.random()
        activity = schoolfield_activity(schoolfield_params, temperature)
        sphericity = 0.5 + 0.5 * random.random()  # placeholder morphology
        threshold = circuit_breaker_threshold(activity, sphericity, base_threshold)
        updated_priors = update_edge_priors(priors, likelihood, evidence, alpha)
        material_cost = compute_material_cost(np.array([distance(seeds[i], p) for p in region]), updated_priors)
        thresholds[i] = (material_cost, threshold)
    return thresholds

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Voronoi assignment of *points* to the nearest *seeds*."""
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2)]
    seeds = [(0, 0), (2, 2)]
    priors = np.array([0.5, 0.5])
    likelihood = 1.0
    evidence = np.array([1.0, 1.0])
    alpha = 0.1
    schoolfield_params = SchoolfieldParams()
    base_threshold = 10
    thresholds = hybrid_voronoi_bayes(points, seeds, priors, likelihood, evidence, alpha, schoolfield_params, base_threshold)
    print(thresholds)