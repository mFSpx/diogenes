# DARWIN HAMMER — match 4800, survivor 0
# gen: 6
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s2.py (gen5)
# born: 2026-05-29T23:58:04Z

# hybrid_model_vram_scheduler_ttt_linear_m2002_s4.py
# DARWIN HAMMER — match 2002, survivor 4
# gen: 1
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s2.py (gen5)
# born: 2026-05-29T23:50:00Z

"""
This module fuses the mathematical structures of 
hybrid_model_vram_scheduler_ttt_linear_m11_s4.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2002_s2.py. 
The mathematical bridge between these two structures is 
established by introducing a spatial-aware VRAM scheduler 
that uses signal and noise scores from the Possum Filter 
as inputs to learn a mapping between the scores and 
the output of the Hybrid Morphology-SSIM-Hygiene Algorithm, 
enabling it to adapt to changing environments and 
optimize the movement of agents based on signal scores 
while considering spatial-aware privacy risks and 
physical similarity and textual confidence, all within 
the constraints of a limited VRAM budget.
"""

import json
import os
import shutil
import subprocess
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Global constants & helpers (Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def _append_runtime_receipt(receipt: Dict[str, Any], *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")

def _query_nvidia_smi() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"status": "missing", "message": "nvidia-smi not found"}
    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu:index,name,memory.total,memory.used,memory.free,driver_version,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return {"status": "error", "stderr": cp.stderr[-500:]}
    gpus: List[Dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
            }
        )
    return {"gpus": gpus}

# ----------------------------------------------------------------------
# Math bridge: Spatial-aware VRAM Scheduler (Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

@dataclass(frozen=True)
class SpatialAwareVRAMScheduler:
    centers: List[Tuple[float, ...]]
    weights: List

def spatial_aware_vram_scheduler(gpus: List[Dict[str, Any]], vram_budget_mb: int) -> Dict[str, Any]:
    gpu_scores = [sum(g["used_mb"] for g in gpus[:i+1]) for i in range(len(gpus))]
    spatial_scores = [haversine_m((0.0, 0.0), (g["used_mb"] / vram_budget_mb, 0.0)) for g in gpus]
    scores = [gaussian(x, epsilon=0.1) for x in gpu_scores]
    weights = [gaussian(x, epsilon=0.1) for x in spatial_scores]
    A = [[scores[i] * weights[j] for j in range(len(gpus))] for i in range(len(gpus))]
    b = [1.0 for _ in range(len(gpus))]
    assignment = solve_linear(A, b)
    return {"assignment": {i: round(a, 2) for i, a in enumerate(assignment)}}

# ----------------------------------------------------------------------
# Hybrid functions (Parent A & B)
# ----------------------------------------------------------------------
def hybrid_vram_scheduler(gpus: List[Dict[str, Any]], vram_budget_mb: int) -> Dict[str, Any]:
    preemption_scores = [_query_nvidia_smi()["gpus"]]
    spatial_scores = [haversine_m((0.0, 0.0), (g["used_mb"] / vram_budget_mb, 0.0)) for g in gpus]
    scores = [gaussian(x, epsilon=0.1) for x in spatial_scores]
    weights = [gaussian(x, epsilon=0.1) for x in preemption_scores]
    A = [[scores[i] * weights[j] for j in range(len(gpus))] for i in range(len(gpus))]
    b = [1.0 for _ in range(len(gpus))]
    assignment = solve_linear(A, b)
    return {"assignment": {i: round(a, 2) for i, a in enumerate(assignment)}}

def hybrid_preemption(gpus: List[Dict[str, Any]], vram_budget_mb: int) -> Dict[str, Any]:
    preemption_scores = [_query_nvidia_smi()["gpus"]]
    spatial_scores = [haversine_m((0.0, 0.0), (g["used_mb"] / vram_budget_mb, 0.0)) for g in gpus]
    scores = [gaussian(x, epsilon=0.1) for x in spatial_scores]
    weights = [gaussian(x, epsilon=0.1) for x in preemption_scores]
    A = [[scores[i] * weights[j] for j in range(len(gpus))] for i in range(len(gpus))]
    b = [1.0 for _ in range(len(gpus))]
    assignment = solve_linear(A, b)
    return {"assignment": {i: round(a, 2) for i, a in enumerate(assignment)}}

def hybrid_spatial_scores(gpus: List[Dict[str, Any]], vram_budget_mb: int) -> List[float]:
    spatial_scores = [haversine_m((0.0, 0.0), (g["used_mb"] / vram_budget_mb, 0.0)) for g in gpus]
    return [gaussian(x, epsilon=0.1) for x in spatial_scores]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    gpus = _query_nvidia_smi()["gpus"]
    vram_budget_mb = DEFAULT_BUDGET_MB
    print(hybrid_vram_scheduler(gpus, vram_budget_mb))
    print(hybrid_preemption(gpus, vram_budget_mb))
    print(hybrid_spatial_scores(gpus, vram_budget_mb))