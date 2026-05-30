# DARWIN HAMMER — match 5530, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_capybara_opti_hybrid_hybrid_nlms_o_m1337_s0.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s1.py (gen3)
# born: 2026-05-30T00:02:42Z

import math
import random
import numpy as np
import sys
import pathlib
import json
import os
import shutil
import subprocess
import datetime
import argparse
from pathlib import Path
from typing import Any

# Constants & utility helpers
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

def now_z() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def _append_runtime_receipt(receipt: dict[str, Any], *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str) + "\n")

def gpu_memory() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"status": "missing", "message": "nvidia-smi not found"}
    cp = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pstate",
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
    gpus = cp.stdout.strip().splitlines()
    gpu_dict = {}
    for gpu in gpus:
        gpu_id, name, total, used, free, driver_version, pstate = gpu.split(",")
        gpu_dict[name] = {"id": gpu_id, "total": int(total), "used": int(used), "free": int(free), "driver_version": driver_version, "pstate": pstate}
    return gpu_dict

# ----------------------------------------------------------------------
# Social interaction and predator evasion functions (Parent A)
# ----------------------------------------------------------------------
def social_interaction(x: np.ndarray, g_best: np.ndarray, k: int = 1, r1: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if x.shape != g_best.shape:
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    if not (0 <= r <= 1):
        raise ValueError("r1 must be in [0, 1]")
    return x + r * (g_best - k * x)

def predator_evasion(x: np.ndarray, g_worst: np.ndarray, k: int = 1, r2: float | None = None, seed: int | str | None = None) -> np.ndarray:
    if x.shape != g_worst.shape:
        raise ValueError("x and g_worst must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r2 is None else r2
    if not (0 <= r <= 1):
        raise ValueError("r2 must be in [0, 1]")
    return x - r * (g_worst - k * x)

# ----------------------------------------------------------------------
# NLMS adaptive filter functions (Parent B)
# ----------------------------------------------------------------------
def nlms_update(x: np.ndarray, w: np.ndarray, d: np.ndarray, mu: float, k: int) -> np.ndarray:
    if x.shape != w.shape or x.shape != d.shape[:-1]:
        raise ValueError("x, w, and d must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    return w + mu * (d - np.dot(x, w)) / (np.dot(x.T, x) + k)

def similarity_matrix(x: np.ndarray, k: int) -> np.ndarray:
    if len(x.shape) != 2:
        raise ValueError("x must be a 2D vector")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    return np.dot(x, x.T) / (np.linalg.norm(x, axis=1)[:, np.newaxis]**2 + k)

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_update(x: np.ndarray, g_best: np.ndarray, g_worst: np.ndarray, w: np.ndarray, d: np.ndarray, mu: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    adapted_x = social_interaction(x, g_best, k=k)
    adapted_w = nlms_update(adapted_x, w, d, mu, k)
    adapted_similarity = similarity_matrix(adapted_x, k=k)
    return adapted_w, adapted_similarity

def hybrid_predict(x: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.dot(x, w) + d

def hybrid_train(x: np.ndarray, d: np.ndarray, mu: float, k: int, epochs: int, g_best: np.ndarray, g_worst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.random.rand(x.shape[1])
    similarity = similarity_matrix(x, k=k)
    for _ in range(epochs):
        adapted_w, adapted_similarity = hybrid_update(x, g_best, g_worst, w, d, mu, k)
        w = adapted_w
        similarity = adapted_similarity
    return w, similarity

# ----------------------------------------------------------------------
# GPU resource management
# ----------------------------------------------------------------------
def manage_gpu_resources(gpu_dict: dict[str, Any], budget_mb: int, reserve_mb: int) -> dict[str, Any]:
    gpu_dict["budget_mb"] = budget_mb
    gpu_dict["reserve_mb"] = reserve_mb
    return gpu_dict

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid algorithm smoke test")
    parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("-b", "--budget-mb", type=int, default=4096, help="GPU budget in MB")
    parser.add_argument("-r", "--reserve-mb", type=int, default=768, help="GPU reserve in MB")
    args = parser.parse_args()
    gpu_dict = gpu_memory()
    managed_gpu_dict = manage_gpu_resources(gpu_dict, args.budget_mb, args.reserve_mb)
    print(managed_gpu_dict)

    # Additional test
    np.random.seed(0)
    x = np.random.rand(10, 5)
    d = np.random.rand(10)
    g_best = np.random.rand(10, 5)
    g_worst = np.random.rand(10, 5)
    w, similarity = hybrid_train(x, d[:, np.newaxis], 0.1, 1, 100, g_best, g_worst)
    print(w)
    print(similarity)