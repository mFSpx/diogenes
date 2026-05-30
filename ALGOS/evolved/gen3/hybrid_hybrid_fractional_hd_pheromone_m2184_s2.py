# DARWIN HAMMER — match 2184, survivor 2
# gen: 3
# parent_a: hybrid_fractional_hdc_hybrid_endpoint_circ_m119_s0.py (gen2)
# parent_b: pheromone.py (gen0)
# born: 2026-05-29T23:41:17Z

from __future__ import annotations
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * math.pi, size=d)
        hv = np.exp(1j * theta).astype(np.complex128)
    elif kind == "bipolar":
        hv = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
    elif kind == "real":
        hv = rng.normal(size=d)
        hv /= np.linalg.norm(hv)  # unit L2 norm
    else:
        raise ValueError(f"Unsupported kind={kind!r}")
    return hv

def fractional_bind(hv_a: np.ndarray, hv_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if hv_a.shape != hv_b.shape:
        raise ValueError("Hypervectors must have the same shape")
    bound = hv_a * np.power(hv_b, alpha)
    return bound

def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-norm vector cannot be normalized")
    return v / norm

@dataclass
class PheromoneSignal:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float
    created_at: str = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    detail: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {"script": "ALGOS/hybrid_pheromone_hdc.py"}

def decay_weight(signal: PheromoneSignal, now_iso: str | None = None) -> float:
    now = datetime.fromisoformat((now_iso or datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
    created = datetime.fromisoformat(signal.created_at.replace('Z', '+00:00'))
    delta_seconds = (now - created).total_seconds()
    if signal.half_life_seconds <= 0:
        return signal.signal_value
    weight = signal.signal_value * math.exp(-delta_seconds / signal.half_life_seconds)
    return max(weight, 0.0)

def log_pheromone(signal: PheromoneSignal, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or Path(__file__).resolve().parents[1] / "05_OUTPUTS" / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"pheromone_{signal.signal_kind}_{timestamp}.json"
    report_path = out_dir / filename
    payload = asdict(signal)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    payload["report_path"] = str(report_path.relative_to(Path(__file__).resolve().parents[1]))
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"REPORT_PATH={report_path}")
    return report_path

def endpoint_geometry_vector(d: int = 10000, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=d)
    return normalize_vector(vec)

def hybrid_health_score(
    hv: np.ndarray,
    geom_vec: np.ndarray,
    signal: PheromoneSignal,
) -> float:
    hv_norm = normalize_vector(hv)
    geom_norm = normalize_vector(geom_vec)
    similarity = np.real(np.vdot(hv_norm, geom_norm))
    weight = decay_weight(signal)
    health = weight * similarity
    return health

def hybrid_bind_and_score(
    dim: int = 10000,
    alpha: float = 0.5,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    hv_a = random_hv(d=dim, kind="complex", seed=seed)
    hv_b = random_hv(d=dim, kind="complex", seed=(seed or random.randint(0, 1000)))
    bound_hv = fractional_bind(hv_a, hv_b, alpha)
    geom_vec = endpoint_geometry_vector(d=dim, seed=seed)
    signal = PheromoneSignal("surface", "kind", 1.0, 3600)
    health = hybrid_health_score(bound_hv, geom_vec, signal)
    return bound_hv, health

def improved_hybrid_bind_and_score(
    dim: int = 10000,
    alpha: float = 0.5,
    seed: int | None = None,
    num_signals: int = 5,
) -> Tuple[np.ndarray, float, list]:
    hv_a = random_hv(d=dim, kind="complex", seed=seed)
    hv_b = random_hv(d=dim, kind="complex", seed=(seed or random.randint(0, 1000)))
    bound_hv = fractional_bind(hv_a, hv_b, alpha)
    geom_vec = endpoint_geometry_vector(d=dim, seed=seed)
    signals = [PheromoneSignal("surface", f"kind_{i}", 1.0, 3600) for i in range(num_signals)]
    health_scores = [hybrid_health_score(bound_hv, geom_vec, signal) for signal in signals]
    return bound_hv, sum(health_scores) / len(health_scores), health_scores

# Example usage:
bound_hv, health, health_scores = improved_hybrid_bind_and_score()
print(f"Bound Hypervector: {bound_hv}")
print(f"Health Score: {health}")
print(f"Health Scores: {health_scores}")