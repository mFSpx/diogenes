# DARWIN HAMMER — match 11, survivor 4
# gen: 1
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:17:54Z

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

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


# ----------------------------------------------------------------------
# VRAM planning core – deeper integration
# ----------------------------------------------------------------------
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
    """Tracks VRAM usage of multiple artifacts and provides a budget‑aware API."""

    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB, reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._artifacts: Dict[str, VramSlotPlan] = {}
        self._last_gpu_query: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # GPU query caching (avoid repeated nvidia‑smi calls)
    # ------------------------------------------------------------------
    def _gpu_info(self) -> Dict[str, Any]:
        if self._last_gpu_query is None:
            self._last_gpu_query = _query_nvidia_smi()
        return self._last_gpu_query

    # ------------------------------------------------------------------
    # Public budgeting helpers
    # ------------------------------------------------------------------
    def total_committed_mb(self) -> int:
        return sum(plan.estimated_mb for plan in self._artifacts.values())

    def available_budget_mb(self) -> int:
        gpu_total = self._gpu_info().get("total_mb", self.static_budget_mb)
        effective_budget = min(self.static_budget_mb, gpu_total) - self.reserve_mb
        return max(effective_budget - self.total_committed_mb(), 0)

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------
    def register(self, plan: VramSlotPlan) -> VramSlotPlan:
        """Add or replace an artifact plan. Returns the stored plan."""
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

    # ------------------------------------------------------------------
    # Decision helper used by TTT
    # ------------------------------------------------------------------
    def can_accommodate(self, mb: int) -> Tuple[bool, str]:
        """Return (allowed, reason) for a prospective allocation of `mb` megabytes."""
        if mb <= self.available_budget_mb():
            return True, "fits within dynamic VRAM budget"
        return False, f"requires {mb} MB > available {self.available_budget_mb()} MB"


# ----------------------------------------------------------------------
# Memory estimation utilities (Parent B + extensions)
# ----------------------------------------------------------------------
def weight_matrix_memory_mb(W: np.ndarray) -> int:
    """Accurately compute memory footprint of a NumPy matrix in MiB."""
    element_bytes = W.dtype.itemsize
    total_bytes = W.size * element_bytes
    return int(round(total_bytes / (1024 * 1024), 0))


def temporary_buffers_mb(W: np.ndarray, x: np.ndarray) -> int:
    """
    Estimate transient memory used during a single TTT step:
    - gradient (same shape as W)
    - intermediate vectors (x, pred, residual)
    """
    grad_bytes = W.nbytes
    vec_bytes = x.nbytes * 3  # x, pred, residual
    total = grad_bytes + vec_bytes
    return int(round(total / (1024 * 1024), 0))


# ----------------------------------------------------------------------
# TTT core (Parent B) – unchanged semantics, but dtype‑aware
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0, dtype: np.dtype = np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d_out = d_in if d_out is None else d_out
    return (rng.standard_normal((d_out, d_in), dtype=dtype) * scale).astype(dtype)


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)


def ttt_step(W: np.ndarray, x: np.ndarray, eta: float = 0.01, target: np.ndarray | None = None) -> np.ndarray:
    g = ttt_grad(W, x, target)
    return W - eta * g


def ttt_forward(W: np.ndarray, x: np.ndarray, eta: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    W_new = ttt_step(W, x, eta=eta)
    h = W_new @ x
    return h, W_new


# ----------------------------------------------------------------------
# Hybrid utilities – richer residency planning
# ----------------------------------------------------------------------
def plan_ttt_residency(
    planner: VramPlanner,
    W: np.ndarray,
    payload: Dict[str, Any] | None = None,
) -> VramSlotPlan:
    """Create or update a VramSlotPlan for the weight matrix, registering it with `planner`."""
    payload = payload or {}
    w_mb = weight_matrix_memory_mb(W)

    allowed, reason = planner.can_accommodate(w_mb)
    action = "allow" if allowed else "defer"

    detail = {
        "weight_matrix_mb": w_mb,
        "available_budget_mb": planner.available_budget_mb(),
        "reason": reason,
        "payload": payload,
        "gpu_info": planner._gpu_info(),
    }

    plan = VramSlotPlan(
        artifact_id="ttt_weight_matrix",
        artifact_kind="matrix",
        action=action,
        estimated_mb=w_mb,
        reason=reason,
        detail=detail,
    )
    return planner.register(plan)


def _maybe_compress(W: np.ndarray, planner: VramPlanner) -> np.ndarray:
    """
    If the weight matrix cannot fit, attempt a cheap compression:
    - down‑cast to float16 (halves memory)
    - re‑plan and keep if it now fits
    Returns the (potentially) compressed matrix.
    """
    if W.dtype == np.float16:
        return W  # already compressed, nothing else we can do

    compressed = W.astype(np.float16)
    w_mb = weight_matrix_memory_mb(compressed)
    allowed, _ = planner.can_accommodate(w_mb)
    if allowed:
        # Update the planner with the new (smaller) plan
        plan = VramSlotPlan(
            artifact_id="ttt_weight_matrix",
            artifact_kind="matrix",
            action="allow",
            estimated_mb=w_mb,
            reason="compressed to float16 to fit budget",
            detail={"original_dtype": str(W.dtype), "compressed_dtype": str(compressed.dtype)},
        )
        planner.register(plan)
        return compressed
    return W  # fallback – keep original (will be handled by caller)


def ttt_sequence_with_budget(
    x_seq: np.ndarray,
    planner: VramPlanner,
    eta: float = 0.01,
    d_model: int | None = None,
    init_dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTT over a token sequence while continuously respecting VRAM constraints.
    The planner is consulted before each step for both persistent and temporary
    memory usage. If a step would exceed the budget, the weight matrix is
    optionally compressed; if still impossible, the step is skipped.
    Returns the hidden states and the final weight matrix.
    """
    x_seq = np.asarray(x_seq, dtype=float)
    T, d_in = x_seq.shape

    # Initialise weight matrix respecting the planner
    if "ttt_weight_matrix" in planner._artifacts:
        # reuse existing matrix (already registered)
        W = np.array(planner._artifacts["ttt_weight_matrix"].detail.get("matrix_snapshot", []), dtype=init_dtype)
    else:
        d_out = d_model if d_model is not None else d_in
        W = init_ttt(d_in, d_out=d_out, dtype=init_dtype)
        # Register initial plan
        plan = plan_ttt_residency(planner, W, payload={"stage": "initialisation"})
        if plan.action == "defer":
            # Try compression before giving up
            W = _maybe_compress(W, planner)
            plan = plan_ttt_residency(planner, W, payload={"stage": "post‑compression"})
            if plan.action == "defer":
                raise MemoryError(f"Cannot allocate initial TTT weight matrix ({weight_matrix_memory_mb(W)} MB) within budget.")
        # Store a snapshot for future reuse
        planner._artifacts["ttt_weight_matrix"].detail["matrix_snapshot"] = W.tolist()

    d_out = W.shape[0]
    H = np.empty((T, d_out), dtype=float)

    for t in range(T):
        x = x_seq[t].astype(W.dtype, copy=False)

        # Estimate temporary buffer usage for this step
        temp_mb = temporary_buffers_mb(W, x)
        allowed, reason = planner.can_accommodate(temp_mb)
        if not allowed:
            # Attempt to shrink temporary usage by casting to float16 temporarily
            x = x.astype(np.float16, copy=False)
            temp_mb = temporary_buffers_mb(W.astype(np.float16), x)
            allowed, reason = planner.can_accommodate(temp_mb)

        if not allowed:
            # Skip update but still produce a forward pass with current W
            h = W @ x
            H[t] = h
            receipt = {
                "timestamp": now_z(),
                "event": "ttt_step_skipped",
                "step": t,
                "reason": reason,
                "temp_mb": temp_mb,
                "available_budget_mb": planner.available_budget_mb(),
            }
            _append_runtime_receipt(receipt)
            continue

        # Perform the actual update
        h, W = ttt_forward(W, x, eta=eta)
        H[t] = h

        # Update planner with possibly changed matrix size (unlikely but for completeness)
        plan = plan_ttt_residency(planner, W, payload={"step": t})
        if plan.action == "defer":
            # Try compression on‑the‑fly
            W = _maybe_compress(W, planner)
            plan = plan_ttt_residency(planner, W, payload={"step": t, "post_compression": True})
            if plan.action == "defer":
                # If still cannot fit, revert to previous matrix and skip further updates
                receipt = {
                    "timestamp": now_z(),
                    "event": "ttt_fatal_memory",
                    "step": t,
                    "reason": "cannot fit weight matrix even after compression",
                }
                _append_runtime_receipt(receipt)
                raise MemoryError("Out of VRAM during TTT updates.")

        # Store snapshot for possible reuse after the loop
        planner._artifacts["ttt_weight_matrix"].detail["matrix_snapshot"] = W.tolist()

    return H, W


# ----------------------------------------------------------------------
# Example usage (would be removed or guarded in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check when run as script
    planner = VramPlanner()
    seq = np.random.randn(10, 64).astype(np.float32)
    H, W_final = ttt_sequence_with_budget(seq, planner, eta=0.005, d_model=64)
    print("Final hidden shape:", H.shape)
    print("Final weight shape:", W_final.shape)
    print("Remaining VRAM budget (MiB):", planner.available_budget_mb())