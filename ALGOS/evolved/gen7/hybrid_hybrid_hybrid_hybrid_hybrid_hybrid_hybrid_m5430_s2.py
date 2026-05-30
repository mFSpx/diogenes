# DARWIN HAMMER — match 5430, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1316_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_tempor_hybrid_hybrid_doomsd_m676_s0.py (gen3)
# born: 2026-05-30T00:01:48Z

"""Hybrid Algorithm: Physarum‑Sheaf + Temporal‑Gini/Doomsday Fusion
===================================================================

This module merges the core topologies of two parent algorithms:

* **Parent A** – a Physarum‑Sheaf framework that uses MinHash/L​SH, Count‑Min
  sketches and Bayesian updates to drive information‑transport gain `α`.
* **Parent B** – a temporal‑spatial toolkit that extracts a Gini coefficient
  from scalar distributions and a Doomsday‑derived weekday to generate a
  symbolic hypervector.

**Mathematical bridge**

1. The *Gini coefficient* `G` (Parent B) rescales the information‑transport
   gain `α` of the Physarum‑Sheaf update (Parent A):
   `α = α₀ · (1 + G)`, where `α₀` is a base gain.
2. The *weekday* obtained by the Doomsday algorithm (Parent B) is turned into
   a deterministic binary hypervector `h`.  This hypervector seeds the
   MinHash hash functions used in the Count‑Min sketch and the LSH index,
   providing a unified random seed that couples the temporal and spatial
   components.

The three public functions below demonstrate the hybrid operation:
`gini_coefficient`, `doomsday_hypervector`, and `hybrid_physarum_step`.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import datetime as dt

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]          # (x, y) coordinates
Edge = Tuple[int, int]               # (source_index, target_index)
DocID = str
Shingle = str

# ----------------------------------------------------------------------
# Parent‑A utilities (adapted)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 32‑bit hash using a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=4).digest(), "big")

def count_min_sketch(items: Iterable[Any], width: int = 64, depth: int = 4, seed_vec: List[int] = None):
    """Count‑Min sketch with optional seeded hash functions."""
    table = [[0] * width for _ in range(depth)]
    seeds = seed_vec if seed_vec is not None else [random.randint(0, 2**31) for _ in range(depth)]
    for item in items:
        for d, s in enumerate(seeds):
            idx = _hash(s, str(item)) % width
            table[d][idx] += 1
    return table, seeds

def minhash_lsh_index(docs: Dict[DocID, Iterable[Shingle]], seed_vec: List[int] = None) -> Dict[str, List[DocID]]:
    """LSH index based on MinHash signatures; seeds derived from Doomsday hypervector."""
    if seed_vec is None:
        seed_vec = [random.randint(0, 2**31) for _ in range(3)]
    buckets: Dict[str, List[DocID]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        mins = []
        for s in seed_vec:
            min_hash = min(_hash(s, sh) for sh in shingles) if shingles else 0
            mins.append(min_hash)
        # combine the three minima into a short key
        key = hashlib.sha1("".join(map(str, mins)).encode()).hexdigest()[:6]
        buckets[key].append(doc_id)
    return dict(buckets)

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    if marginal <= 0:
        return 0.0
    return (likelihood * prior) / marginal

# ----------------------------------------------------------------------
# Parent‑B utilities (adapted)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)

def doomsday_numpy(date: dt.date) -> int:
    """Weekday (0=Sunday … 6=Saturday) via Doomsday algorithm."""
    t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    y = date.year
    m = date.month
    d = date.day
    if m < 3:
        y -= 1
        m += 10
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + d) % 7

def doomsday_hypervector(date: dt.date, dim: int = 128) -> np.ndarray:
    """Deterministic binary hypervector seeded by the Doomsday weekday."""
    wd = doomsday_numpy(date)                     # 0‑6
    rng = np.random.default_rng(seed=wd)         # reproducible per weekday
    return rng.integers(0, 2, size=dim, dtype=np.int8)

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_physarum_step(
    points: List[Point],
    edges: List[Edge],
    docs: Dict[DocID, List[Shingle]],
    scalar_values: List[float],
    date: dt.date,
    base_alpha: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, List[DocID]]]:
    """
    Perform a single Physarum‑Sheaf update where:
      * `α` is scaled by the Gini coefficient of `scalar_values`.
      * MinHash/Count‑Min seeds come from the Doomsday hypervector of `date`.
      * Edge conductances are returned as a numpy array.
    """
    # 1️⃣  Temporal bridge – Gini rescales α
    G = gini_coefficient(scalar_values)
    alpha = base_alpha * (1.0 + G)          # α ∈ (0, 2·base_alpha)

    # 2️⃣  Temporal bridge – Doomsday hypervector → seeds
    hv = doomsday_hypervector(date, dim=3)  # three 0/1 seeds → convert to ints
    seed_vec = [int(b) for b in hv[:3]]

    # 3️⃣  Spatial bridge – Count‑Min sketch of edge identifiers
    edge_ids = [f"{u}-{v}" for u, v in edges]
    cms_table, cms_seeds = count_min_sketch(edge_ids, seed_vec=seed_vec)

    # 4️⃣  LSH index for document similarity (used later for Bayesian update)
    lsh_index = minhash_lsh_index(docs, seed_vec=seed_vec)

    # 5️⃣  Initialise conductances (positive)
    conduct = np.ones(len(edges), dtype=float)

    # 6️⃣  Physarum‑Sheaf update (simplified)
    for i, (u, v) in enumerate(edges):
        # geometric flow proportional to distance and α
        dist = length(points[u], points[v]) + 1e-9
        flow = alpha / dist

        # sketch‑derived weight: lower sketch count → higher uncertainty → lower flow
        sketch_count = sum(cms_table[d][cms_seeds[d] % len(cms_table[d])] for d in range(len(cms_table)))
        weight = 1.0 / (1.0 + sketch_count)

        # Bayesian adjustment using a dummy likelihood derived from document overlap
        docs_u = set(docs.get(str(u), []))
        docs_v = set(docs.get(str(v), []))
        overlap = len(docs_u & docs_v) / max(1, len(docs_u | docs_v))
        prior = conduct[i]
        likelihood = overlap
        marginal = bayes_marginal(prior, likelihood, false_positive=0.1)
        posterior = bayes_update(prior, likelihood, marginal)

        # final conductance update
        conduct[i] = max(0.0, conduct[i] + flow * weight * posterior)

    return conduct, lsh_index

def hybrid_gini_alpha_demo(scalar_vals: List[float]) -> float:
    """Return the α scaling factor derived from Gini – a tiny pure‑math demo."""
    G = gini_coefficient(scalar_vals)
    return 0.5 * (1.0 + G)

def hybrid_doomsday_hv_demo(date: dt.date) -> np.ndarray:
    """Return the Doomsday hypervector – a pure‑temporal demo."""
    return doomsday_hypervector(date, dim=64)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic geometry
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    eds = [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3)]

    # Dummy documents (shingles are just stringified coordinates)
    docs = {
        "0": [f"{pts[0][0]:.1f},{pts[0][1]:.1f}"],
        "1": [f"{pts[1][0]:.1f},{pts[1][1]:.1f}"],
        "2": [f"{pts[2][0]:.1f},{pts[2][1]:.1f}"],
        "3": [f"{pts[3][0]:.1f},{pts[3][1]:.1f}"],
    }

    # Random scalar values for Gini
    scalars = [random.random() for _ in pts]

    # Current date for Doomsday hypervector
    today = dt.date.today()

    conductances, lsh = hybrid_physarum_step(
        points=pts,
        edges=eds,
        docs=docs,
        scalar_values=scalars,
        date=today,
        base_alpha=0.5,
    )

    print("Conductances after hybrid step:", conductances)
    print("LSH index sample (first 3 buckets):", dict(list(lsh.items())[:3]))
    print("α scaling demo:", hybrid_gini_alpha_demo(scalars))
    print("Doomsday hypervector (first 16 bits):", hybrid_doomsday_hv_demo(today)[:16])