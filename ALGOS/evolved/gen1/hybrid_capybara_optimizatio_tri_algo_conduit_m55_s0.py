# DARWIN HAMMER — match 55, survivor 0
# gen: 1
# parent_a: capybara_optimization.py (gen0)
# parent_b: tri_algo_conduit.py (gen0)
# born: 2026-05-29T23:23:56Z

"""Hybrid algorithm fusing the Capybara Optimization Algorithm and the Tri-algo Conduit.

The mathematical bridge between the two structures is the concept of signal processing and optimization. 
The Capybara Optimization Algorithm uses social interaction and evasion strategies to optimize the movement of agents, 
while the Tri-algo Conduit uses signal scores and recovery priorities to decide whether to stay dormant, burst, or recover. 
In this hybrid algorithm, we integrate the governing equations of both parents by using the signal scores from the Tri-algo Conduit 
to influence the social interaction and evasion strategies in the Capybara Optimization Algorithm.

This integration allows the hybrid algorithm to adapt to changing environments and optimize the movement of agents based on signal scores.
"""

import numpy as np
import math
import random
import sys
import pathlib

Vector = np.ndarray

def social_interaction(x: Vector, g_best: Vector, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0 <= r <= 1):
        raise ValueError("r1 must be in [0, 1]")
    return np.array([xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)])

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def predator_evasion(x: Vector, delta: float, r2: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    rng = random.Random(seed)
    r = rng.random() if r2 is None else r2
    if not (0 <= r <= 1):
        raise ValueError("r2 must be in [0, 1]")
    step = (2 * r - 1) * delta
    return np.array([xi + step * xi for xi in x])

def signal_scores(data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0) -> tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0, 0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0, 0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(chunk) / 8.0

def shannon_entropy(data: bytes) -> float:
    frequency = {}
    for byte in data:
        frequency[byte] = frequency.get(byte, 0) + 1
    entropy = 0.0
    for count in frequency.values():
        probability = count / len(data)
        entropy -= probability * math.log2(probability)
    return entropy

def hybrid_optimization(x: Vector, g_best: Vector, data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0) -> np.ndarray:
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    delta = evasion_delta(1, 10, delta_max=signal)
    return social_interaction(x, g_best, k=1, r1=signal, seed=42)

def hybrid_evasion(x: Vector, g_best: Vector, data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0) -> np.ndarray:
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    delta = evasion_delta(1, 10, delta_max=signal)
    return predator_evasion(x, delta, r2=signal, seed=42)

def main():
    x = np.array([1.0, 2.0, 3.0])
    g_best = np.array([4.0, 5.0, 6.0])
    data = b"Hello, World!"
    status_code = 200
    mime = "text/plain"
    keyword_hits = 2
    structural_links = 1
    print(hybrid_optimization(x, g_best, data, status_code, mime, keyword_hits, structural_links))
    print(hybrid_evasion(x, g_best, data, status_code, mime, keyword_hits, structural_links))

if __name__ == "__main__":
    main()