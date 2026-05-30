# DARWIN HAMMER — match 3541, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m606_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_physarum_netw_m1368_s1.py (gen6)
# born: 2026-05-29T23:50:42Z

"""Hybrid Privacy‑Fisher‑Sketch‑Physarum Model

Parents:
- hybrid_hybrid_hybrid_privac_hybrid_hybrid_m606_s2.py (privacy‑aware resource matrix,
  Gaussian‑based Fisher weighting and circuit‑breaker)
- hybrid_hybrid_hybrid_sketch_hybrid_physarum_netw_m1368_s1.py (Count‑Min sketch as a
  cellular sheaf, sheaf Laplacian energy, physarum‑inspired conductance dynamics,
  hyperdimensional binding)

Mathematical Bridge
-------------------
Both parents rely on a *scalar quality metric* that can be interpreted as a
“information density”:

* In the privacy‑Fisher side the metric is the Fisher information
  \(\mathcal{F}(θ)=\frac{(\partial_θ g)^2}{g}\) of a Gaussian beam \(g\).
* In the sketch‑physarum side the metric is the sheaf Laplacian energy
  \(E = \sum_{i}\|δ(s_i)\|^2\), i.e. the squared norm of coboundary differences of the
  Count‑Min sketch rows.

We fuse them by **using the Fisher score as a global temperature that scales the
physarum conductance updates derived from the sketch energy**.  Concretely:

1. Build a composite resource matrix **R** (RAM, privacy‑load) – size *m×n*.
2. Build a Count‑Min sketch matrix **S** from a data stream – size *d×w*.
3. Compute the sheaf Laplacian energy vector **e** (length *d*) from **S**.
4. Derive node pressures **p = e / (‖e‖+ε)**.
5. Initialise conductance matrix **C** (same shape as **S**) with hyper‑dimensional
   binding scalars.
6. Update conductances with a physarum rule
   \[
   \dot C = τ·\mathcal{F}(θ)·(p_i - C_{ij}),
   \]
   where \(τ\) is a time step and \(\mathcal{F}(θ)\) is the Fisher score.
7. Weight the resource matrix **R** by the same Fisher score to obtain a
   privacy‑aware load **L = \mathcal{F}(θ)·R**.
8. A circuit‑breaker opens if \(\mathcal{F}(θ) > \text{threshold}\).

The resulting pipeline produces a single scalar decision (breaker state) while
simultaneously adapting the sketch‑driven physarum network under privacy‑aware
load constraints.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Fisher utilities (Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam g(θ;center,width)."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0,
                 eps: float = 1e-12) -> float:
    """Fisher information of the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Resource matrix construction (Parent A)
# ----------------------------------------------------------------------
def build_resource_matrix(resources: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Construct a composite resource matrix R.

    ``resources`` maps a resource name to a tuple (ram_gb, privacy_load).
    The matrix has shape (len(resources), 2) where column 0 is RAM and column 1
    is privacy load.
    """
    rows = []
    for name, (ram, priv) in resources.items():
        rows.append([ram, priv])
    return np.array(rows, dtype=float)


# ----------------------------------------------------------------------
# Count‑Min sketch as a cellular sheaf (Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(stream: List[int], depth: int, width: int) -> np.ndarray:
    """
    Build a Count‑Min sketch matrix S of shape (depth, width) from an integer stream.
    Each row corresponds to a distinct hash function.
    """
    sketch = np.zeros((depth, width), dtype=int)
    for item in stream:
        for i in range(depth):
            h = int(hashlib.sha256(f"{item}_{i}".encode()).hexdigest(), 16)
            idx = h % width
            sketch[i, idx] += 1
    return sketch


def sheaf_laplacian_energy(sketch: np.ndarray) -> np.ndarray:
    """
    Compute the sheaf Laplacian energy per row.

    For each row we treat consecutive columns as edges and compute the squared
    differences (the coboundary).  The returned vector has length ``sketch.shape[0]``.
    """
    diffs = np.diff(sketch, axis=1)               # shape (depth, width-1)
    energy = np.sum(diffs ** 2, axis=1)           # one energy per depth
    return energy.astype(float)


# ----------------------------------------------------------------------
# Hyperdimensional utilities (Parent B)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: int | str | None = None) -> List[int]:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: List[int], b: List[int]) -> List[int]:
    """Binding (component‑wise multiplication) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[List[int]]) -> List[int]:
    """Superposition (element‑wise sum) of hypervectors."""
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    # Binarise by sign
    return [1 if s >= 0 else -1 for s in sums]


# ----------------------------------------------------------------------
# Physarum‑inspired conductance dynamics (Parent B) modulated by Fisher (Parent A)
# ----------------------------------------------------------------------
def initialise_conductance(sketch: np.ndarray, dim: int = 10000) -> np.ndarray:
    """
    Initialise a conductance matrix C with the same shape as ``sketch``.
    Each entry is a scalar derived from a hyperdimensional binding of a
    symbol (row index) and a symbol (column index).  The binding result is
    reduced to a float via the mean of its components.
    """
    depth, width = sketch.shape
    C = np.empty((depth, width), dtype=float)
    for i in range(depth):
        row_sym = symbol_vector(f"row_{i}", dim)
        for j in range(width):
            col_sym = symbol_vector(f"col_{j}", dim)
            bound = bind(row_sym, col_sym)
            C[i, j] = np.mean(bound)  # value in [-1, 1]
    # Shift to positive domain for conductance
    C = (C + 1.0) / 2.0 + 0.1  # avoid zero conductance
    return C


def physarum_update(C: np.ndarray,
                    pressures: np.ndarray,
                    theta: float,
                    dt: float = 0.1,
                    eps: float = 1e-12) -> np.ndarray:
    """
    Perform one physarum conductance update step.

    C          – current conductance matrix (depth × width)
    pressures – vector of node pressures (length = depth)
    theta     – parameter for Fisher score (privacy‑load angle)
    dt        – time step
    """
    F = fisher_score(theta)                     # scalar temperature
    depth, width = C.shape
    # Broadcast pressures across columns
    p_mat = pressures[:, None]                  # shape (depth,1)
    dC = dt * F * (p_mat - C)                    # physarum rule
    C_new = C + dC
    # Keep conductances strictly positive
    C_new = np.clip(C_new, eps, None)
    return C_new


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_weighted_load(resources: Dict[str, Tuple[float, float]],
                        theta: float) -> np.ndarray:
    """
    Compute the privacy‑aware weighted load L = F(θ)·R.

    Returns a matrix of the same shape as the resource matrix.
    """
    R = build_resource_matrix(resources)
    F = fisher_score(theta)
    return F * R


def hybrid_conductance_step(sketch: np.ndarray,
                            C: np.ndarray,
                            theta: float,
                            dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    One hybrid step that:
    1. Computes sheaf Laplacian energy → pressures.
    2. Updates conductance matrix C using physarum dynamics scaled by Fisher.
    Returns the updated conductance matrix and the pressure vector.
    """
    energy = sheaf_laplacian_energy(sketch)          # shape (depth,)
    norm = np.linalg.norm(energy) + 1e-12
    pressures = energy / norm                       # normalized pressures
    C_new = physarum_update(C, pressures, theta, dt)
    return C_new, pressures


def circuit_breaker(theta: float, threshold: float = 0.5) -> bool:
    """
    Simple endpoint circuit breaker driven by Fisher score.
    Returns True if the breaker opens (i.e., F(θ) exceeds threshold).
    """
    return fisher_score(theta) > threshold


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy resource specification
    resources_example = {
        "cpu": (32.0, 0.2),
        "gpu": (16.0, 0.5),
        "ssd": (4.0, 0.1)
    }

    # Parameter theta (e.g., privacy angle)
    theta_val = 0.3

    # 1. Compute weighted load
    load = hybrid_weighted_load(resources_example, theta_val)
    print("Weighted load matrix L:\n", load)

    # 2. Build a sketch from a synthetic integer stream
    stream = [random.randint(0, 1000) for _ in range(500)]
    depth, width = 5, 50
    sketch = count_min_sketch(stream, depth, width)
    print("\nSketch shape:", sketch.shape)

    # 3. Initialise conductance matrix
    C = initialise_conductance(sketch, dim=256)   # smaller dim for speed
    print("\nInitial conductance stats: min {:.3f}, max {:.3f}".format(C.min(), C.max()))

    # 4. Perform a hybrid physarum step
    C, pressures = hybrid_conductance_step(sketch, C, theta_val, dt=0.05)
    print("\nPressures:", pressures)
    print("Conductance after update: min {:.3f}, max {:.3f}".format(C.min(), C.max()))

    # 5. Check circuit breaker
    if circuit_breaker(theta_val, threshold=0.4):
        print("\nCircuit breaker OPENED (Fisher score exceeded threshold).")
    else:
        print("\nCircuit breaker CLOSED (system stable).")