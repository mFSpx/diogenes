# DARWIN HAMMER — match 4416, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s0.py (gen6)
# born: 2026-05-29T23:55:35Z

"""
This module fuses the core topologies of hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py (parent A)
and hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s0.py (parent B) by integrating the SSIM (Structural Similarity Index Measure) 
from parent A with the lead-lag transform and B-spline basis from parent B. The mathematical bridge is found 
in the use of similarity measures and geometric transformations to analyze physical and logical entities.

Parent A: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s0.py
"""

import numpy as np
import math
import random
import sys
import pathlib
import json
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME_DIR = ROOT / "04_RUNTIME" / "fairyfuse"
DEFAULT_HEARTBEAT = RUNTIME_DIR / "ternary_router_heartbeat.jsonl"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    ssim_map = ((2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)) * ((2 * sigma_xy + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2))
    return np.mean(ssim_map)

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (x - t[i]) / denom_l if denom_l > 0 else 0
            term_r = (t[i + order] - x) / denom_r if denom_r > 0 else 0
            B_new[:, i] = term_l * B[:, i] + term_r * B[:, i + 1]
        B = B_new
    return B

def hybrid_similarity(path: np.ndarray, signal: np.ndarray) -> float:
    lead_lag_path = lead_lag_transform(path)
    b_spline = bspline_basis(signal, np.linspace(0, 1, len(signal)), k=3)
    similarity_map = np.dot(lead_lag_path, b_spline.T)
    return ssim(similarity_map, np.ones_like(similarity_map))

def hybrid_signal_process(signal: np.ndarray, path: np.ndarray) -> np.ndarray:
    b_spline = bspline_basis(signal, np.linspace(0, 1, len(signal)), k=3)
    lead_lag_path = lead_lag_transform(path)
    return np.dot(lead_lag_path, b_spline.T)

def hybrid_structure_analyze(structure: np.ndarray) -> dict[str, float]:
    morphology = {
        "length": np.max(structure),
        "width": np.percentile(structure, 75),
        "height": np.std(structure),
        "mass": np.sum(structure)
    }
    sphericity = (morphology["length"] * morphology["width"] * morphology["height"]) ** (1.0 / 3.0) / np.max([morphology["length"], morphology["width"], morphology["height"]])
    return {
        "sphericity": sphericity,
        "morphology": morphology
    }

if __name__ == "__main__":
    np.random.seed(42)
    signal = np.random.rand(100)
    path = np.random.rand(10, 3)
    similarity = hybrid_similarity(path, signal)
    print(f"Hybrid similarity: {similarity:.4f}")
    processed_signal = hybrid_signal_process(signal, path)
    print(f"Processed signal shape: {processed_signal.shape}")
    structure = np.random.rand(100)
    analysis = hybrid_structure_analyze(structure)
    print(f"Sphericity: {analysis['sphericity']:.4f}")