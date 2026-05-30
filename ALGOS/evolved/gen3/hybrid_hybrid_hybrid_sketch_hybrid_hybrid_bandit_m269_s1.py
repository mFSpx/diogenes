# DARWIN HAMMER — match 269, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s2.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (gen2)
# born: 2026-05-29T23:28:02Z

"""Hybrid Sketch‑Sheaf & Bandit‑Store Allocator

Parents:
- hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s2.py (sketch‑sheaf RLCT)
- hybrid_hybrid_bandit_router_workshare_allocator_m60_s0.py (bandit router + workshare allocator)

Mathematical Bridge:
Both parents manipulate a *state* that is updated by linear operators.
The sketch‑sheaf side builds a cellular sheaf whose coboundary matrix **B**
produces a residual vector **r = B·s** and a Laplacian **L = Bᵀ·B** whose
energy `E = sᵀ·L·s` measures inconsistency of the sketch reduction.
The bandit‑store side maintains a scalar store level `ℓ` whose dynamics are
driven by an inflow/outflow term `Δ = α·in – β·out`.  

We fuse them by treating the sheaf energy `E` as the *reward* signal that
drives the bandit policy, while the store’s dance signal `d = gain·ℓ/limit`
modulates the *target* allocation used in the RLCT regression.  In this way
the linear algebra of the sheaf directly influences the stochastic
allocation, and the stochastic allocation feeds back into the regression
that estimates the Real Log‑Canonical Threshold (RLCT).

The module therefore provides:
1. `count_min_sheaf` – construct a sheaf from Count‑Min sketches.
2. `hybrid_rlct_via_sheaf` – compute an RLCT using coboundary residual norms.
3. `bandit_allocation_with_sheaf` – bandit UCB allocation where the reward
   is the sheaf Laplacian energy and the store’s dance scales the exploration
   term.
4. `hybrid_info_loss` – a blended information‑loss measure combining RLCT and
   normalized sheaf energy.
"""

import hashlib
import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sheaf utilities (from Parent A)
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf over a simple line graph of hash buckets.

    Each node (bucket) carries a vector space of dimension `depth`.
    Edges connect consecutive buckets and carry identity restriction maps.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node_id -> dim
        self.edges = list(edges)                  # list of (u, v) tuples

    def coboundary_matrix(self) -> np.ndarray:
        """Build the signed incidence matrix B of shape (|E|·d, |V|·d).

        For each edge (u, v) we place +I_d at column block u and -I_d at column
        block v.  d is the common node dimension (all nodes share the same depth).
        """
        d = next(iter(self.node_dims.values()))
        m = len(self.edges) * d
        n = len(self.node_dims) * d
        B = np.zeros((m, n))
        for e_idx, (u, v) in enumerate(self.edges):
            for i in range(d):
                row = e_idx * d + i
                col_u = u * d + i
                col_v = v * d + i
                B[row, col_u] = 1.0
                B[row, col_v] = -1.0
        return B

    def laplacian(self) -> np.ndarray:
        """L = Bᵀ·B."""
        B = self.coboundary_matrix()
        return B.T @ B

# ---------------------------------------------------------------------------
# Store dynamics (from Parent B)
# ---------------------------------------------------------------------------

@dataclass
class StoreState:
    """Honeybee‑style store whose level modulates bandit exploration."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = field(default=0.0, init=False, repr=False)

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply Δ = α·∑in – β·∑out and integrate."""
        self._last_delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * self._last_delta)
        return self.level, self._last_delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        if self.limit == 0:
            return 0.0
        return self.gain * (self._last_delta / self.limit)

# ---------------------------------------------------------------------------
# Bandit primitives (from Parent B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridSheafBandit"

def ucb_score(mean: float, count: int, total: int, c: float = 1.0) -> float:
    """Upper‑Confidence‑Bound score."""
    if count == 0:
        return float('inf')
    return mean + c * math.sqrt(math.log(total + 1) / count)

# ---------------------------------------------------------------------------
# Hybrid functions
# ---------------------------------------------------------------------------

def _hash_functions(depth: int) -> List[Callable[[bytes], int]]:
    """Generate `depth` deterministic hash functions using MD5 with different salts."""
    salts = [f"seed{i}".encode('utf-8') for i in range(depth)]

    def make_hash(salt: bytes):
        def h(x: bytes) -> int:
            return int(hashlib.md5(salt + x).hexdigest(), 16)
        return h

    return [make_hash(s) for s in salts]

def count_min_sheaf(data: List[bytes], depth: int, width: int) -> Tuple[Sheaf, np.ndarray]:
    """
    Build a Count‑Min sketch from `data`, then interpret it as a sheaf.

    Returns
    -------
    sheaf : Sheaf
        Nodes = hash buckets, each with dimension `depth`.
    sections : np.ndarray
        Flattened vector `s` of shape (width*depth,) containing the sketch counts
        for each bucket/dimension.
    """
    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be positive integers")

    hash_fns = _hash_functions(depth)
    sketch = np.zeros((depth, width), dtype=np.float64)

    for item in data:
        for d, h in enumerate(hash_fns):
            idx = h(item) % width
            sketch[d, idx] += 1.0

    # Build sheaf: line graph connecting bucket i to i+1
    node_dims = {i: depth for i in range(width)}
    edges = [(i, i + 1) for i in range(width - 1)]
    sheaf = Sheaf(node_dims, edges)

    # Flatten sections column‑wise (bucket major)
    sections = sketch.T.reshape(-1)  # shape (width*depth,)
    return sheaf, sections

def hybrid_rlct_via_sheaf(sheaf: Sheaf, sections: np.ndarray) -> float:
    """
    Compute a Real Log‑Canonical Threshold (RLCT) from coboundary residual norms.

    Steps
    -----
    1. Compute residual r = B·s.
    2. For each depth level `k` (1..depth) compute the ℓ₂‑norm of the first
       `k·width` entries of r (simulating increasing sketch depth).
    3. Fit log‑log linear regression `log(norm) = a + b·log(k)`.
    4. Return `-b` as the RLCT estimate (positive slope → loss).

    Returns
    -------
    rlct : float
    """
    B = sheaf.coboundary_matrix()
    r = B @ sections  # residual vector

    depth = next(iter(sheaf.node_dims.values()))
    width = len(sheaf.node_dims)

    norms = []
    ks = []
    for k in range(1, depth + 1):
        # residual entries corresponding to first k dimensions per node
        idx_end = k * (len(sheaf.edges) * 1)  # each edge contributes k entries
        # Since B is block‑structured, we can simply take first k·|E| rows
        residual_slice = r[: idx_end]
        norm = np.linalg.norm(residual_slice, 2)
        norms.append(max(norm, 1e-12))  # avoid log(0)
        ks.append(k)

    logk = np.log(ks)
    logy = np.log(norms)
    # Simple least‑squares fit
    A = np.vstack([logk, np.ones_like(logk)]).T
    slope, _ = np.linalg.lstsq(A, logy, rcond=None)[0]
    rlct = -slope
    return rlct

def bandit_allocation_with_sheaf(
    store: StoreState,
    sheaf: Sheaf,
    sections: np.ndarray,
    action_ids: List[str]
) -> List[BanditAction]:
    """
    Perform a single bandit allocation step where the reward is the sheaf
    Laplacian energy.  The store's `dance` signal scales the exploration term
    in the UCB score.

    Parameters
    ----------
    store : StoreState
        Current store; will be updated with inflow/outflow derived from the
        sheaf energy.
    sheaf : Sheaf
        Sheaf structure (provides Laplacian).
    sections : np.ndarray
        Flattened sketch sections.
    action_ids : List[str]
        Identifiers of the available actions.

    Returns
    -------
    actions : List[BanditAction]
        One action per `action_id` with computed propensity and confidence.
    """
    # Compute sheaf energy (reward)
    L = sheaf.laplacian()
    energy = float(sections.T @ (L @ sections))

    # Use energy as positive reward; also feed it as inflow to the store
    inflow = [energy]
    outflow = []  # no explicit outflow in this simple setting
    store.update(inflow, outflow)

    # Bandit statistics (mocked with simple counters)
    total_counts = getattr(store, "_total_counts", 0) + 1
    setattr(store, "_total_counts", total_counts)

    actions = []
    for aid in action_ids:
        # Mock historical data stored on the store object
        count = getattr(store, f"_cnt_{aid}", 0) + 1
        setattr(store, f"_cnt_{aid}", count)

        mean_reward = energy / count  # naive average reward per action
        # Exploration term scaled by dance signal
        c = 1.0 + max(0.0, store.dance)
        score = ucb_score(mean_reward, count, total_counts, c=c)

        actions.append(
            BanditAction(
                action_id=aid,
                propensity=score,
                expected_reward=mean_reward,
                confidence_bound=c * math.sqrt(math.log(total_counts + 1) / count),
            )
        )
    return actions

def hybrid_info_loss(
    rlct: float,
    sheaf: Sheaf,
    sections: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Blend the RLCT estimate with the normalized sheaf Laplacian energy.

    If `normalize` is True, the energy is divided by the total number of
    entries to obtain a per‑component average.

    Returns
    -------
    loss : float
    """
    L = sheaf.laplacian()
    energy = float(sections.T @ (L @ sections))
    if normalize:
        energy /= sections.size
    # Simple convex combination; weight 0.5 each
    return 0.5 * rlct + 0.5 * energy

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data: 1000 random byte strings
    random.seed(42)
    data = [random.randbytes(8) for _ in range(1000)]

    depth = 5
    width = 20

    # Build sheaf from sketch
    sheaf, sect = count_min_sheaf(data, depth, width)

    # Compute hybrid RLCT
    rlct_val = hybrid_rlct_via_sheaf(sheaf, sect)
    print(f"Hybrid RLCT estimate: {rlct_val:.4f}")

    # Initialise store
    store = StoreState(level=2.0, alpha=0.5, beta=0.3, dt=0.1, gain=2.0, limit=5.0)

    # Perform bandit allocation
    actions = bandit_allocation_with_sheaf(
        store,
        sheaf,
        sect,
        action_ids=["cpu", "gpu", "fpga"]
    )
    for a in actions:
        print(f"Action {a.action_id}: propensity={a.propensity:.4f}, "
              f"exp_reward={a.expected_reward:.4f}, conf={a.confidence_bound:.4f}")

    # Compute blended information loss
    loss = hybrid_info_loss(rlct_val, sheaf, sect)
    print(f"Hybrid information loss: {loss:.6f}")