#!/usr/bin/env python3
"""Minimum-cost tree scoring for length/path trade-offs."""
from __future__ import annotations
import math
Point = tuple[float, float]
Edge = tuple[str, str]


def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())
