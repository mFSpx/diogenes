# DARWIN HAMMER — match 3596, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_diffus_m1445_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1963_s2.py (gen6)
# born: 2026-05-29T23:50:53Z

import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import exp, log
from random import random
from sys import exit
from pathlib import Path
from typing import Any, Dict

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)
    running = path[:-1] - path[0]
    return running.T @ increments

def calculate_entropy(signature: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(signature)
    entropy = -sum([e * log(e) for e in eigenvalues if e > 0])
    return entropy

def calculate_hybrid_risk_score(m: Morphology, path: np.ndarray, alpha: float = 0.5) -> float:
    recovery_p = recovery_priority(m)
    signature = signature_level2(path)
    entropy = calculate_entropy(signature)
    return alpha * recovery_p + (1 - alpha) * entropy

def main():
    morphology = Morphology(1.0, 2.0, 3.0, 10.0)
    path = np.random.rand(10, 2)
    risk_score = calculate_hybrid_risk_score(morphology, path)
    print(f"Hybrid risk score: {risk_score}")

if __name__ == "__main__":
    main()