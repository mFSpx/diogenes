# DARWIN HAMMER — match 4416, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s0.py (gen6)
# born: 2026-05-29T23:55:35Z

import numpy as np
import math
import random
import sys
import pathlib
import json
import os
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

def route_command(text: str, intent: str, context: dict[str, any]) -> dict[str, any]:
    # This function is assumed to be implemented elsewhere
    pass

def route_packet(packet: dict[str, any]) -> dict[str, any]:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }
    route = route_command(text[:4096], intent, context).to_dict()
    route["engine_channel"] = "cpu_fairyfuse_ternary"
    route["outbound_state"] = "draft_only"
    return route

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
            term_l = (t[i + order] - x) / denom_l
            term_r = (x - t[i + order - 1]) / denom_r
            B_new[:, i] = ((term_l * B[:, i] + (order - 1)) / denom_l) + ((term_r * B[:, i + 1] + order) / denom_r)

    return B_new

class Morphology:
    def __init__(self, length: float, width: float, height: float, sphericity: float):
        self.length = length
        self.width = width
        self.height = height
        self.sphericity = sphericity

    def __dict__(self):
        return {"length": self.length, "width": self.width, "height": self.height, "sphericity": self.sphericity}

def hybrid_route(packet: dict[str, any]) -> dict[str, any]:
    route = route_packet(packet)
    if route["text_surface"] is None and packet.get("raw_command") is None and packet.get("text") is None:
        x = np.array([])
        y = np.array([])
    else:
        x = np.asarray(route["text_surface"] or packet.get("raw_command") or packet.get("text") or "").astype(np.float64)
        y = np.asarray(route["text_surface"] or packet.get("raw_command") or packet.get("text") or "").astype(np.float64)
    if len(x) > 0 and len(y) > 0:
        ssim_value = ssim(x, y)
    else:
        ssim_value = np.nan
    grid = np.linspace(0, 1, 10)
    B = bspline_basis(x, grid) if len(x) > 0 else np.array([])
    route["ssim_value"] = ssim_value
    route["bspline_basis"] = B.tolist()
    return route

def hybrid_morphology(length: float, width: float, height: float) -> Morphology:
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    sphericity = geo_mean / max(length, width, height)
    return Morphology(length, width, height, sphericity)

def hybrid_ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    if len(pred) > 0 and len(t) > 0:
        ssim_value = ssim(pred, t)
    else:
        ssim_value = np.nan
    return residual, ssim_value

if __name__ == "__main__":
    packet = {
        "text_surface": "Hello, World!",
        "normalized_intent": "greeting",
        "ontology_terms": ["greeting", "hello"],
        "epistemic_flag": True,
        "payload": {"key": "value"},
    }
    route = hybrid_route(packet)
    print(route)
    morphology = hybrid_morphology(5.0, 3.0, 2.0)
    print(morphology.__dict__())
    W = np.random.rand(10, 10)
    x = np.random.rand(10)
    residual, ssim_value = hybrid_ttt_loss(W, x)
    print(residual, ssim_value)