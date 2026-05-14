#!/usr/bin/env python3
"""Pheromone-weighted link selection primitives."""
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class EdgeSignal:
    item: str
    pheromone: float
    utility: float = 0.0
    cost: float = 0.0


def edge_value(utility: float, cost: float) -> float:
    return utility - cost


def evaporate(strength: float, reinforcement: float = 0.0, phi: float = 0.05, dt: float = 1.0) -> float:
    if phi < 0 or dt < 0:
        raise ValueError('phi and dt must be non-negative')
    return max(0.0, strength * math.exp(-phi * dt) + reinforcement)


def probabilities(edges: Iterable[EdgeSignal], curiosity: float = 0.05, gamma: float = 1.5) -> list[tuple[str, float]]:
    if curiosity < 0:
        raise ValueError('curiosity must be non-negative')
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    es = list(edges)
    if not es:
        return []
    weights=[]
    for e in es:
        signal=max(0.0, e.pheromone + edge_value(e.utility, e.cost))
        weights.append((curiosity + signal) ** gamma)
    total=sum(weights)
    if total <= 0:
        p=1.0/len(es)
        return [(e.item,p) for e in es]
    return [(e.item,w/total) for e,w in zip(es,weights)]


def top_k(edges: Iterable[EdgeSignal], k: int, curiosity: float = 0.05, gamma: float = 1.5) -> list[tuple[str, float]]:
    if k <= 0:
        return []
    return sorted(probabilities(edges, curiosity, gamma), key=lambda x: (-x[1], x[0]))[:k]


def sample(edges: Iterable[EdgeSignal], k: int, curiosity: float = 0.05, gamma: float = 1.5, seed: int | str | None = None) -> list[str]:
    probs=probabilities(edges, curiosity, gamma)
    rng=random.Random(seed)
    selected=[]
    pool=probs[:]
    for _ in range(min(k, len(pool))):
        r=rng.random(); acc=0.0
        for idx,(item,p) in enumerate(pool):
            acc += p
            if r <= acc:
                selected.append(item)
                pool.pop(idx)
                total=sum(x[1] for x in pool)
                pool=[(x[0], x[1]/total) for x in pool] if total else pool
                break
    return selected
