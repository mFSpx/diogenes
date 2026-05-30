# DARWIN HAMMER — match 3, survivor 4
# gen: 2
# parent_a: hybrid_model_vram_scheduler_ttt_linear_m11_s4.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s0.py (gen1)
# born: 2026-05-29T23:22:36Z

import json
import os
import shutil
import subprocess
import sys
import math
import random
from collections import deque, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

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
                "driver_version": driver,
                "pstate": pstate,
            }
        )
    if not gpus:
        return {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}
    return {"status": "ok", "selected_index": gpus[0]["index"], **gpus[0], "gpus": gpus}

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class VramPlanner:
    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB, reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._artifacts: Dict[str, VramSlotPlan] = {}
        self._last_gpu_query: Dict[str, Any] | None = None
        self._curvature_values: Dict[str, float] = {}

    def _gpu_info(self) -> Dict[str, Any]:
        if self._last_gpu_query is None:
            self._last_gpu_query = _query_nvidia_smi()
        return self._last_gpu_query

    def total_committed_mb(self) -> int:
        return sum(plan.estimated_mb for plan in self._artifacts.values())

    def available_budget_mb(self) -> int:
        gpu_total = self._gpu_info().get("total_mb", self.static_budget_mb)
        effective_budget = min(self.static_budget_mb, gpu_total) - self.reserve_mb
        return max(effective_budget - self.total_committed_mb(), 0)

    def register(self, plan: VramSlotPlan) -> VramSlotPlan:
        self._artifacts[plan.artifact_id] = plan
        receipt = {
            "timestamp": now_z(),
            "event": "register_artifact",
            "artifact_id": plan.artifact_id,
            "action": plan.action,
            "estimated_mb": plan.estimated_mb,
            "available_budget_mb": self.available_budget_mb(),
        }
        _append_runtime_receipt(receipt)
        return plan

    def unregister(self, artifact_id: str) -> None:
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            receipt = {
                "timestamp": now_z(),
                "event": "unregister_artifact",
                "artifact_id": artifact_id,
                "available_budget_mb": self.available_budget_mb(),
            }
            _append_runtime_receipt(receipt)

    def can_accommodate(self, mb: int) -> Tuple[bool, str]:
        if mb <= self.available_budget_mb():
            return True, "sufficient budget"
        else:
            return False, f"need {mb} MB, only {self.available_budget_mb()} MB free"

    def artifact_weights(self) -> Dict[str, float]:
        total = self.total_committed_mb() or 1.0
        return {aid: plan.estimated_mb / total for aid, plan in self._artifacts.items()}

    def adjacency_from_artifacts(self) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {}
        for aid1, plan1 in self._artifacts.items():
            for aid2, plan2 in self._artifacts.items():
                if aid1 != aid2 and plan1.artifact_kind == plan2.artifact_kind:
                    if aid1 not in adjacency:
                        adjacency[aid1] = []
                    if aid2 not in adjacency:
                        adjacency[aid2] = []
                    adjacency[aid1].append(aid2)
                    adjacency[aid2].append(aid1)
        return adjacency

    def compute_curvature(self) -> Dict[str, float]:
        adjacency = self.adjacency_from_artifacts()
        weights = self.artifact_weights()
        curvature_values: Dict[str, float] = {}
        for aid in self._artifacts:
            neighbors = adjacency.get(aid, [])
            weight = weights[aid]
            curvature = 0.0
            for neighbor in neighbors:
                neighbor_weight = weights[neighbor]
                curvature += weight * neighbor_weight
            curvature_values[aid] = curvature
        self._curvature_values = curvature_values
        return curvature_values

    def register_with_curvature(self, plan: VramSlotPlan) -> VramSlotPlan:
        self.register(plan)
        self.compute_curvature()
        return plan

    def get_curvature(self, aid: str) -> float:
        if aid not in self._curvature_values:
            self.compute_curvature()
        return self._curvature_values.get(aid, 0.0)

class HybridVRAMCurvatureScheduler:
    def __init__(self, vram_planner: VramPlanner):
        self.vram_planner = vram_planner

    def schedule(self, aid: str, action: str, estimated_mb: int) -> bool:
        plan = VramSlotPlan(aid, "default", action, estimated_mb, "default", {})
        if self.vram_planner.can_accommodate(estimated_mb)[0]:
            self.vram_planner.register_with_curvature(plan)
            return True
        else:
            return False

    def get_curvature(self, aid: str) -> float:
        return self.vram_planner.get_curvature(aid)

def main():
    vram_planner = VramPlanner()
    scheduler = HybridVRAMCurvatureScheduler(vram_planner)
    plan = VramSlotPlan("aid1", "default", "action1", 100, "reason1", {})
    scheduler.schedule("aid1", "action1", 100)
    print(scheduler.get_curvature("aid1"))

if __name__ == "__main__":
    main()