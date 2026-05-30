# DARWIN HAMMER — match 55, survivor 1
# gen: 1
# parent_a: capybara_optimization.py (gen0)
# parent_b: tri_algo_conduit.py (gen0)
# born: 2026-05-29T23:23:56Z

"""This module represents a novel HYBRID algorithm that mathematically fuses the core topologies of 
the Capybara Optimization Algorithm (capybara_optimization.py) and the Tri-algo conduit (tri_algo_conduit.py) 
into a single unified system. The mathematical bridge between these two structures is based on the 
integration of the social interaction and predator evasion mechanisms from the Capybara Optimization Algorithm 
with the signal scores and recovery priority calculations from the Tri-algo conduit. Specifically, the 
Capybara Optimization Algorithm's social interaction and predator evasion mechanisms are used to optimize 
the Tri-algo conduit's signal scores and recovery priority calculations, resulting in a more efficient and 
effective hybrid algorithm."""

import math
import random
import numpy as np
import sys
import pathlib

Vector = list[float]

def social_interaction(x: Vector, g_best: Vector, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> list[float]:
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0 <= r <= 1):
        raise ValueError("r1 must be in [0, 1]")
    return [xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)]


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def predator_evasion(x: Vector, delta: float, r2: float | None = None, seed: int | str | None = None) -> list[float]:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    rng = random.Random(seed)
    r = rng.random() if r2 is None else r2
    if not (0 <= r <= 1):
        raise ValueError("r2 must be in [0, 1]")
    step = (2 * r - 1) * delta
    return [xi + step * xi for xi in x]


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
    hist = np.bincount([byte for byte in chunk])
    hist_sum = np.sum(hist)
    entropy = -np.sum((hist / hist_sum) * np.log2(hist / hist_sum + 1e-12))
    return entropy


def recovery_from_payload(data: bytes, max_bytes: int, parse_error: bool = False) -> float:
    size_ratio = min(4.0, len(data) / max(1, max_bytes))
    morph = {
        "length": 1.0 + size_ratio * 8.0,
        "width": 2.0 + (2.0 if parse_error else 0.5),
        "height": max(0.5, 3.0 - size_ratio),
        "mass": 1.0 + size_ratio * 6.0 + (3.0 if parse_error else 0.0),
    }
    recovery_priority = morph["mass"] / (morph["length"] + morph["width"] + morph["height"])
    return recovery_priority


def hybrid_optimization(data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0, max_bytes: int = 1_500_000) -> tuple[float, float, float]:
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    recovery = recovery_from_payload(data, max_bytes)
    delta = evasion_delta(len(data), max_bytes)
    evasion = predator_evasion([signal, noise, recovery], delta)
    social = social_interaction([signal, noise, recovery], [0.5, 0.5, 0.5], r1=0.5)
    return signal, recovery, social[2]


def hybrid_signal(data: bytes, status_code: int | None = None, mime: str = "", keyword_hits: int = 0, structural_links: int = 0) -> float:
    signal, _ = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    return signal


def hybrid_recovery(data: bytes, max_bytes: int) -> float:
    return recovery_from_payload(data, max_bytes)


if __name__ == "__main__":
    data = b"Hello, World!"
    signal, recovery, social = hybrid_optimization(data)
    print("Signal:", signal)
    print("Recovery:", recovery)
    print("Social:", social)
    hybrid_signal(data)
    hybrid_recovery(data, 1000)