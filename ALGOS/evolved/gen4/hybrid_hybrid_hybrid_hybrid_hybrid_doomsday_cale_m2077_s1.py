# DARWIN HAMMER — match 2077, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_bandit_m269_s0.py (gen3)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py (gen1)
# born: 2026-05-29T23:40:47Z

import numpy as np
import math
import random
import sys
import pathlib
import datetime as dt
from typing import Dict, List, Tuple, Iterable

class Sheaf:
    """Cellular sheaf over a finite graph.

    node_dims: mapping node_id → dimension (here always 1)
    edges: list of (u, v) tuples, undirected for Laplacian construction
    """

    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node_id -> int
        self.edges = list(edge_list)              # list of (u, v) tuples

    def compute_laplacian(self) -> np.ndarray:
        """Return the (symmetric) sheaf Laplacian L = δᵀδ.

        For a graph with unit edge weights and 1‑dimensional node spaces,
        L coincides with the ordinary combinatorial Laplacian.
        """
        n = len(self.node_dims)
        L = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            # increment degrees
            L[u, u] += 1.0
            L[v, v] += 1.0
            # off‑diagonal entries
            L[u, v] -= 1.0
            L[v, u] -= 1.0
        return L


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index (0=Monday … 6=Sunday) for a Gregorian date."""
    return (dt.date(year, month, day).weekday() + 1) % 7  # shift to match parent B's convention


def weekday_distribution(year: int, month: int, num_days: int) -> np.ndarray:
    """Count occurrences of each weekday in the month segment."""
    counts = np.zeros(7, dtype=float)
    for day in range(1, num_days + 1):
        wd = doomsday(year, month, day)
        counts[wd] += 1.0
    return counts


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative vector."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def weekday_graph_sheaf(year: int, month: int, num_days: int) -> Sheaf:
    """Construct a 7‑node sheaf whose edges follow the chronological order of weekdays.

    Nodes 0‑6 correspond to Monday‑Sunday.
    An edge (i, (i+1) mod 7) encodes the natural adjacency of consecutive days.
    The graph is independent of the particular month, but we keep the signature
    for API symmetry with the Gini side.
    """
    node_dims = {i: 1 for i in range(7)}
    edges = [(i, (i + 1) % 7) for i in range(7)]  # circular chain
    return Sheaf(node_dims, edges)


def sheaf_laplacian_energy(sheaf: Sheaf) -> float:
    """Compute the Frobenius norm squared of the sheaf Laplacian (∥L∥_F²).

    This scalar measures the “energy” of the topological coupling.
    """
    L = sheaf.compute_laplacian()
    u, s, vh = np.linalg.svd(L, full_matrices=False)
    return np.sum(s ** 2)


def hybrid_gini_metric(year: int, month: int, num_days: int) -> float:
    """Combine the weekday Gini coefficient with sheaf Laplacian energy.

    H = G · exp(E_norm) where E_norm = E / (7 * max(E)).
    """
    # statistical part
    counts = weekday_distribution(year, month, num_days)
    G = gini_coefficient(counts)

    # topological part
    sheaf = weekday_graph_sheaf(year, month, num_days)
    E = sheaf_laplacian_energy(sheaf)
    E_norm = E / (7 * np.max([E, 1e-8]))  # numerical stability

    return G * math.exp(E_norm)


def hybrid_resource_allocation(
    year: int,
    month: int,
    num_days: int,
    total_resource: float
) -> Dict[int, float]:
    """Allocate a scalar resource across weekdays using the hybrid metric.

    The base allocation is proportional to weekday counts.
    The hybrid factor (exp(E_norm)) uniformly scales the allocation,
    preserving the total while embedding topological information.
    Returns a mapping weekday → allocated amount.
    """
    counts = weekday_distribution(year, month, num_days)
    if counts.sum() == 0:
        raise ValueError("No days to allocate over")
    base_share = counts / counts.sum()  # probabilities per weekday

    sheaf = weekday_graph_sheaf(year, month, num_days)
    E = sheaf_laplacian_energy(sheaf)
    E_norm = E / (7 * np.max([E, 1e-8]))  # numerical stability
    scale = math.exp(E_norm)

    allocation = {wd: float(total_resource) * share * scale for wd, share in enumerate(base_share)}
    # renormalize to exactly total_resource (numerical safety)
    total_alloc = sum(allocation.values())
    if total_alloc != 0:
        factor = total_resource / total_alloc
        for wd in allocation:
            allocation[wd] *= factor
    return allocation


if __name__ == "__main__":
    # Example parameters
    YEAR = 2023
    MONTH = 9
    NUM_DAYS = 30
    TOTAL_RESOURCE = 1000.0

    # Compute hybrid metric
    h_metric = hybrid_gini_metric(YEAR, MONTH, NUM_DAYS)
    print(f"Hybrid Gini metric (G·exp(E/7)) = {h_metric:.6f}")

    # Perform resource allocation
    alloc = hybrid_resource_allocation(YEAR, MONTH, NUM_DAYS, TOTAL_RESOURCE)
    print("Resource allocation per weekday (0=Mon … 6=Sun):")
    for wd in range(7):
        print(f"  Weekday {wd}: {alloc[wd]:.2f}")

    # Verify that allocation sums to TOTAL_RESOURCE (within tolerance)
    total = sum(alloc.values())
    print(f"Total allocated: {total:.2f} (target {TOTAL_RESOURCE})")