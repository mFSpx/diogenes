# DARWIN HAMMER — match 868, survivor 1
# gen: 4
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s3.py (gen1)
# parent_b: hybrid_hybrid_ternary_lens__hybrid_hybrid_minimu_m219_s0.py (gen3)
# born: 2026-05-29T23:31:14Z

"""
Hybrid algorithm combining the VRAM scheduler from hybrid_model_vram_scheduler_ttt_linear_m11_s3.py and the bayesian utilities and edge cost computation from hybrid_hybrid_ternary_lens__hybrid_hybrid_minimu_m219_s0.py.

The mathematical bridge between these two algorithms lies in the application of bayesian utilities to the VRAM scheduling process. In the hybrid_model_vram_scheduler_ttt_linear_m11_s3.py, the VRAM allocation is determined by a linear function of the GPU's memory usage. In the hybrid_hybrid_ternary_lens__hybrid_hybrid_minimu_m219_s3.py, bayesian utilities are used to compute the marginal probability P(E) and update the prior probability P(H|E). 

In this hybrid algorithm, we integrate the bayesian utilities into the VRAM scheduling process by using the marginal probability P(E) to modulate the allocation probability p(t) per-candidate. This allows us to incorporate the uncertainty in the classification process into the VRAM allocation schedule.

Authors: [Your Name]

Date: 2026-05-29
"""

import numpy as np
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Tuple

import json

# Constants & utility helpers
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
    """Query a single GPU via nvidia-smi.  Returns a dict with total/used/free MB."""
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
    gpus: list[dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total": int(total),
                "used": int(used),
                "free": int(free),
                "driver": driver,
                "pstate": pstate,
            }
        )
    return {"gpus": gpus}

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not 0.0 <= prior <= 1.0 or not 0.0 <= likelihood <= 1.0 or not 0.0 <= false_positive <= 1.0:
        raise ValueError("All probability arguments must be in [0, 1]")
    return (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * false_positive))

def vram_scheduler(memory: dict[str, Any], budget_mb: int) -> dict[str, Any]:
    gpus = memory["gpus"]
    allocation = {}
    for gpu in gpus:
        free_mb = gpu["free"]
        if free_mb < budget_mb:
            allocation[gpu["name"]] = budget_mb
        else:
            allocation[gpu["name"]] = free_mb
    return allocation

def vram_scheduler_bayesian(memory: dict[str, Any], budget_mb: int, prior: float, likelihood: float, false_positive: float) -> dict[str, Any]:
    gpus = memory["gpus"]
    allocation = {}
    for gpu in gpus:
        free_mb = gpu["free"]
        marginal_prob = bayes_marginal(prior, likelihood, false_positive)
        if free_mb < budget_mb * marginal_prob:
            allocation[gpu["name"]] = int(budget_mb * marginal_prob)
        else:
            allocation[gpu["name"]] = int(free_mb)
    return allocation

def hybrid_scheduler(memory: dict[str, Any], budget_mb: int, prior: float, likelihood: float, false_positive: float) -> dict[str, Any]:
    vram_alloc = vram_scheduler(memory, budget_mb)
    bayesian_vram_alloc = vram_scheduler_bayesian(memory, budget_mb, prior, likelihood, false_positive)
    hybrid_alloc = {}
    for gpu in memory["gpus"]:
        hybrid_alloc[gpu["name"]] = int((vram_alloc[gpu["name"]] + bayesian_vram_alloc[gpu["name"]]) / 2)
    return hybrid_alloc

def gpu_memory_summary() -> None:
    memory = gpu_memory()
    print(json.dumps(memory, indent=4))

def smoke_test() -> None:
    memory = gpu_memory()
    budget_mb = 2048
    prior = 0.8
    likelihood = 0.9
    false_positive = 0.1
    print("VRAM allocation under deterministic scheduling:", vram_scheduler(memory, budget_mb))
    print("VRAM allocation under Bayesian scheduling:", vram_scheduler_bayesian(memory, budget_mb, prior, likelihood, false_positive))
    print("Hybrid VRAM allocation:", hybrid_scheduler(memory, budget_mb, prior, likelihood, false_positive))

if __name__ == "__main__":
    gpu_memory_summary()
    smoke_test()