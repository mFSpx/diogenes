# DARWIN HAMMER — match 4896, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py (gen5)
# born: 2026-05-29T23:58:35Z

"""
Module hybrid_fusion.py: Fusing 
hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py and 
hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py.
The mathematical bridge between the two parent algorithms lies in the combination of 
geometric product from parent A and Fisher information from parent B, 
effectively fusing their core topologies.

Parents:
- hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_minimu_m1187_s1.py 
- hybrid_hybrid_fisher_locali_hybrid_hybrid_geomet_m412_s1.py
"""

import numpy as np
import math
from typing import Dict, FrozenSet, Tuple
import random
import sys
import pathlib
from collections.abc import Iterable
from datetime import date
import bisect

def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    Sort indices and cancel pairs of equal indices (which square to +1).
    Returns the sorted tuple of remaining indices and the sign (+1 / -1)
    induced by the swaps needed to bring the original list into the sorted order.
    """
    # Count occurrences
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1

    # Cancel even occurrences
    remaining = [i for i, c in counts.items() if c % 2 == 1]
    # The remaining list may contain duplicates (odd count >1) – keep one copy per odd count
    # Build the list respecting original multiplicities (odd only)
    cleaned = []
    for i in indices:
        if counts[i] % 2 == 1:
            cleaned.append(i)
            counts[i] = 0  # ensure we keep only one copy
    # Sort while tracking sign via bubble‑sort swaps
    lst = list(cleaned)
    sign = 1
    for i in range(len(lst)):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return tuple(lst), sign
    return tuple(lst), sign


def _multiply_blades(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(tuple(combined))
    return result, sign


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hybrid_fusion(blade_a: Tuple[int, ...], blade_b: Tuple[int, ...], theta: float, center: float, width: float) -> Tuple[Tuple[int, ...], float]:
    """
    Fuses the geometric product from parent A with Fisher information from parent B.

    Parameters:
    blade_a (Tuple[int, ...]): First basis blade.
    blade_b (Tuple[int, ...]): Second basis blade.
    theta (float): Angle for Fisher information.
    center (float): Center for Fisher information.
    width (float): Width for Fisher information.

    Returns:
    Tuple[Tuple[int, ...], float]: Fused blade and Fisher score.
    """
    result, sign = _multiply_blades(blade_a, blade_b)
    fisher = fisher_score(theta, center, width)
    return result, fisher * sign


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(nodes: Iterable[Tuple[float, float]], edges: Iterable[Tuple[int, int]], root: int) -> Dict:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    metrics : dict mapping node → metrics
    """
    adj = {n: [] for n in nodes}
    edge_len = {}
    for u, v in edges:
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        adj[u].append(v)
    dist = {root: 0}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in dist:
                dist[neighbor] = dist[node] + edge_len[(node, neighbor)]
                queue.append(neighbor)
    metrics = {n: {'dist': dist[n], 'neighbors': adj[n]} for n in nodes}
    return metrics


if __name__ == "__main__":
    blade_a = (1, 2)
    blade_b = (3, 4)
    theta = 0.5
    center = 0.0
    width = 1.0
    result, fisher = hybrid_fusion(blade_a, blade_b, theta, center, width)
    print(f"Fused blade: {result}, Fisher score: {fisher}")

    nodes = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    edges = [(0, 1), (1, 2)]
    root = 0
    metrics = tree_metrics(nodes, edges, root)
    print(metrics)