# DARWIN HAMMER — match 1594, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s1.py (gen3)
# parent_b: hybrid_fractional_hdc_counterfactual_effec_m38_s1.py (gen1)
# born: 2026-05-29T23:37:47Z

import datetime as dt
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
DEFAULT_BUDGET_MB = 4096
DEFAULT_RESERVE_MB = 768

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _pct(value: float) -> float:
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def vram_aware_gpu_selection(
    gpus: List[Dict[str, Any]], budget_mb: int, reserve_mb: int
) -> List[Dict[str, Any]]:
    selected = []
    for gpu in gpus:
        if gpu.get("memory.free", 0) >= budget_mb + reserve_mb:
            selected.append(gpu)
    return selected


# ---------------------------------------------------------------------------
# Hyperdimensional primitives
# ---------------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1.0, 1.0], size=d).astype(np.float64)
    vec = rng.normal(size=d)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def unbind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, 0.0)
    return result


def fractional_power(hv: np.ndarray, exponent: float) -> np.ndarray:
    if np.iscomplexobj(hv):
        magnitude = np.abs(hv) ** exponent
        angle = np.angle(hv) * exponent
        return magnitude * np.exp(1j * angle)
    return np.power(hv, exponent)


def bundle(hvs: List[np.ndarray]) -> np.ndarray:
    if not hvs:
        raise ValueError("bundle requires at least one hypervector")
    summed = np.sum(hvs, axis=0)
    norm = np.linalg.norm(summed)
    return summed / norm if norm != 0 else summed


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.vdot(a, b)  
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(dot) / norm) if norm != 0 else 0.0


def cleanup(hv: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    mask = np.abs(hv) >= threshold
    return hv * mask


def encode_sequence(seq: Sequence[int], d: int = 10000, seed: int | None = None) -> np.ndarray:
    symbols = [random_hv(d, seed=seed + i if seed is not None else None) for i in seq]
    return bundle(symbols)


def fractional_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a_fp = fractional_power(a, 1.0 - alpha)
    b_fp = fractional_power(b, alpha)
    return bind(a_fp, b_fp)


# ---------------------------------------------------------------------------
# Hybrid structures
# ---------------------------------------------------------------------------
def generate_group_hvs(groups: Sequence[str], d: int = 10000, seed: int = 0) -> Dict[str, np.ndarray]:
    hvs: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(groups):
        hvs[name] = random_hv(d, kind="complex", seed=seed + idx)
    return hvs


def encode_gpu_hv(gpu: Dict[str, Any], d: int = 10000, base_exponent: float = 1.0) -> np.ndarray:
    free_mb = float(gpu.get("memory.free", 0))
    exponent = 1.0 / (1.0 + math.exp(- (free_mb - 2048) / 512.0))
    exponent = max(0.0, min(1.0, exponent))  
    base_hv = random_hv(d, kind="complex")
    return fractional_power(base_hv, exponent)


def hybrid_allocation_plan(
    groups: Sequence[str], 
    gpus: List[Dict[str, Any]], 
    date: dt.date, 
    d: int = 10000,
    budget_mb: int = DEFAULT_BUDGET_MB,
    reserve_mb: int = DEFAULT_RESERVE_MB
) -> np.ndarray:
    group_hvs = generate_group_hvs(groups, d)
    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)
    selected_gpus = vram_aware_gpu_selection(gpus, budget_mb, reserve_mb)
    gpu_hvs = [encode_gpu_hv(gpu, d) for gpu in selected_gpus]
    weighted_group_hv = bundle([bind(group_hvs[group], weight_vec[i]) for i, group in enumerate(groups)])
    gpu_composite_hv = bundle([bind(gpu_hv, weighted_group_hv) for gpu_hv in gpu_hvs])
    return gpu_composite_hv


def estimate_plan_effect(plan_hv: np.ndarray, reference_hv: np.ndarray) -> float:
    return similarity(plan_hv, reference_hv)


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gpus = [
        {"memory": {"free": 8192}},
        {"memory": {"free": 4096}},
        {"memory": {"free": 16384}}
    ]
    date = dt.date.today()
    plan_hv = hybrid_allocation_plan(GROUPS, gpus, date)
    reference_hv = random_hv()
    effect_estimate = estimate_plan_effect(plan_hv, reference_hv)
    print(f"Estimated effect: {effect_estimate:.4f}")