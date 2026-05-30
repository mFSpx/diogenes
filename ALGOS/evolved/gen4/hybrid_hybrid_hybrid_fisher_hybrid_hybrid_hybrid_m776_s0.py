# DARWIN HAMMER — match 776, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s4.py (gen3)
# born: 2026-05-29T23:30:48Z

"""
This module implements the Hybrid Infotaxis-MinHash-Minimum Cost Tree VRAM Scheduler algorithm, 
combining the entropy-driven decision logic of Infotaxis with the set-similarity machinery of MinHash, 
the information density scoring of Fisher localization, and the expected VRAM consumption of a model component tree.
The mathematical bridge lies in the concept of information density, which is used to determine the best action
in Infotaxis, and the expected VRAM consumption in the VRAM Scheduler.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
import hashlib
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[
    dict[str, list[str]],
    dict[tuple[str, str], float],
    dict[str, float],
]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adjacency : dict[str, list[str]]
        Adjacency matrix.
    edge_lengths : dict[tuple[str, str], float]
        Edge lengths.
    distances : dict[str, float]
        Root-to-node distances.
    """
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    edge_lengths: dict[tuple[str, str], float] = {}
    distances: dict[str, float] = {node: float('inf') for node in nodes}
    distances[root] = 0.0

    for edge in edges:
        a, b = edge
        adjacency[a].append(b)
        adjacency[b].append(a)
        edge_lengths[edge] = length(nodes[a], nodes[b])

    for node in nodes:
        for neighbor in adjacency[node]:
            if distances[node] + edge_lengths[(node, neighbor)] < distances[neighbor]:
                distances[neighbor] = distances[node] + edge_lengths[(node, neighbor)]

    return adjacency, edge_lengths, distances

def hybrid_infotaxis_minhash_vram_scheduler(
    theta: float,
    center: float,
    width: float,
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
    eps: float = 1e-12,
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Hybrid Infotaxis-MinHash-Minimum Cost Tree VRAM Scheduler.

    Returns
    -------
    expected_vram : float
        Expected VRAM consumption.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    fisher_info = (derivative * derivative) / intensity

    adjacency, edge_lengths, distances = tree_metrics(nodes, edges, root)

    expected_vram = 0.0
    for node in nodes:
        if node != root:
            expected_vram += fisher_info * distances[node]

    ssim_value = ssim(
        [x for x, _ in nodes.values()],
        [y for _, y in nodes.values()],
        dynamic_range,
        k1,
        k2,
    )

    return expected_vram + ssim_value

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def ssim(x: list[float], y: list[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    return ((2 * mx * my + k1 * dynamic_range) * (2 * mx * my + k2 * dynamic_range)) / (mx ** 2 + my ** 2 + k1 * dynamic_range + k2 * dynamic_range)

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 1.0),
        'C': (2.0, 2.0),
        'D': (3.0, 3.0),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
    root = 'A'

    theta = 1.0
    center = 1.0
    width = 1.0
    eps = 1e-12
    dynamic_range = 255.0
    k1 = 0.01
    k2 = 0.03

    expected_vram = hybrid_infotaxis_minhash_vram_scheduler(
        theta,
        center,
        width,
        nodes,
        edges,
        root,
        eps,
        dynamic_range,
        k1,
        k2,
    )
    print(expected_vram)