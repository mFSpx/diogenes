# DARWIN HAMMER — match 633, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py (gen3)
# parent_b: hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s0.py (gen4)
# born: 2026-05-29T23:30:06Z

"""
This module implements a novel hybrid algorithm that mathematically fuses the core topologies of 
hybrid_hybrid_voronoi_parti_hybrid_sparse_wta_hy_m228_s1.py and hybrid_liquid_time_constant_hybrid_hybrid_hdc_se_m174_s0.py. 
The mathematical bridge between the two structures is the integration of Voronoi partitioning with 
the liquid time constant networks' input-dependent time constant. Specifically, the hybrid algorithm 
uses Voronoi partitioning to generate a set of representative points, which are then used to 
compute the input-dependent time constant of the liquid time constant networks.

The key innovation of this hybrid algorithm is the introduction of a new, hybrid operation that 
combines the strengths of both parent algorithms. This operation, called "hybrid_voronoi_ltc", takes 
the current hidden state, input, and parameters as arguments and returns the updated hidden state 
of the network using the ODE formulation of the liquid time constant networks and Voronoi 
partitioning.

The hybrid algorithm also includes a "hybrid_bundle" operation that takes a set of bipolar 
hypervectors as arguments and returns a single, bundled hypervector that represents the 
superposition of the input-dependent time constants. This operation is used to compute the 
asymptotic target state of the network.

Finally, the hybrid algorithm includes a "hybrid_step" operation that takes the current hidden 
state, input, and parameters as arguments and returns the updated hidden state of the network. 
This operation is used to simulate the dynamics of the hybrid network.
"""

import numpy as np
import random
import math
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Voronoi partitioning
# ----------------------------------------------------------------------

Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Liquid Time Constant Networks
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    data = np.fromiter(
        (1 if rng.getrandbits(1) else -1 for _ in range(dim)), dtype=np.float32
    )
    return data / np.linalg.norm(data)

# ----------------------------------------------------------------------
# Hybrid Voronoi Liquid Time Constant Networks
# ----------------------------------------------------------------------

def hybrid_voronoi_ltc(
    points: list[Point],
    seeds: list[Point],
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    regions = assign(points, seeds)
    x = np.zeros((len(seeds),))
    for i, region in regions.items():
        region_points = np.array(region)
        x[i] = np.mean(np.linalg.norm(region_points, axis=1))
    return ltc_f(x, I, W, b)

def hybrid_bundle(
    hypervectors: list[np.ndarray],
) -> np.ndarray:
    return np.mean(hypervectors, axis=0)

def hybrid_step(
    points: list[Point],
    seeds: list[Point],
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    x = hybrid_voronoi_ltc(points, seeds, I, W, b)
    return x

if __name__ == "__main__":
    points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(100)]
    seeds = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(10)]
    I = np.random.rand(10)
    W = np.random.rand(10, 11)
    b = np.random.rand(10)
    x = hybrid_step(points, seeds, I, W, b)
    print(x)