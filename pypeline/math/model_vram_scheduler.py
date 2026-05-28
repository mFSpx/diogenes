from __future__ import annotations

"""Advisory VRAM / LoRA preemption planner.

No model weights are imported or loaded here.  The planner estimates whether the
resident DeepSeek/Qwen synthesis model, transient embedding lane, and selected
LoRA adapter cartridges can fit inside a 4GB GTX 1650 budget.  Decisions are
ledgered as advisory governor receipts when a Postgres connection is supplied.
"""

import json
import os
import shutil
import subprocess
import gc
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = ROOT / "04_RUNTIME" / "lora_cartridges"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


DEFAULT_BASE_MODEL_ID = "deepseek-1.5b-indy_reads-reads"
DEFAULT_BASE_MODEL_MB = 1800
DEFAULT_ADAPTER_MB = 128
DEFAULT_EMBEDDING_MB = 384
DEFAULT_BUDGET_MB = _env_int("LUCIDOTA_VRAM_BUDGET_MB", 4096)
DEFAULT_RESERVE_MB = _env_int("LUCIDOTA_VRAM_RESERVE_MB", 768)
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"
DEFAULT_LORA_CONTEXT_TOKENS = _env_int("LUCIDOTA_LORA_CONTEXT_TOKENS", 512)
MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS = _env_int("LUCIDOTA_MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS", 256)
DEEPSEEK_Q4_MODEL = ROOT / "03_VAULT" / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
MAMBA_GRAPH_CACHE_MODEL = ROOT / "03_VAULT" / "models" / "mamba-1.4b-hf-Q2_K.gguf"
DUAL_GPU_DEEPSEEK_MB = 1250
DUAL_GPU_MAMBA_CACHE_MB = 1200
DUAL_GPU_HEADROOM_MB = 1100
FAIRYFUSE_TERNARY_WEIGHTS_MB = 2240


@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePreemptionReceipt:
    receipt_kind: str
    action: str
    status: str
    generated_at: str
    detail: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DualEngineResidencyPlan:
    schema: str
    generated_at: str
    advisory_only: bool
    decision: str
    cpu_fairyfuse: dict[str, Any]
    gpu_q4_deepseek: dict[str, Any]
    outbound_state: str
    policy: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _append_runtime_receipt(receipt: RuntimePreemptionReceipt, *, path: Path | None = None) -> None:
    target = path or (RUNTIME_DIR / "preemption_receipts.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(receipt.as_dict(), ensure_ascii=False, sort_keys=True, default=str) + "\n")


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
    gpus: list[dict[str, Any]] = []
    for line in cp.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append({"index": int(idx), "name": name, "total_mb": int(total), "used_mb": int(used), "free_mb": int(free), "driver_version": driver, "pstate": pstate})
    return {"status": "ok", "selected_index": gpus[0]["index"], **gpus[0], "gpus": gpus} if gpus else {"status": "error", "stdout": cp.stdout[-500:], "stderr": cp.stderr[-500:]}


def _file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0,
    }


def plan_dual_engine_residency(
    payload: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    *,
    include_gpu: bool = True,
) -> dict[str, Any]:
    """Plan the always-on CPU FairyFuse + GPU Q4 DeepSeek residency envelope.

    This is intentionally advisory and side-effect-light: it reads hardware state
    and file receipts but does not import or allocate model weights.
    """
    payload = payload or {}
    state = state or {}
    gpu = gpu_memory() if include_gpu else {"status": "skipped"}
    observed_total = int(gpu.get("total_mb") or DEFAULT_BUDGET_MB) if isinstance(gpu, dict) else DEFAULT_BUDGET_MB
    budget = min(DEFAULT_BUDGET_MB, observed_total) if observed_total else DEFAULT_BUDGET_MB
    resident_gpu_mb = DUAL_GPU_DEEPSEEK_MB + DUAL_GPU_MAMBA_CACHE_MB
    requested_adapters = select_adapters(payload, state)
    adapter_headroom_mb = max(0, budget - resident_gpu_mb - 512)
    token_ceiling = lora_context_token_ceiling(payload, state)
    decision = "allow" if resident_gpu_mb <= budget else "defer"
    cpu_weights_env = os.environ.get("LUCIDOTA_FAIRYFUSE_TERNARY_WEIGHTS") or ""
    cpu_weights = Path(cpu_weights_env).expanduser() if cpu_weights_env else ROOT / "03_VAULT/models/fairyfuse/deepseek_ternary_packed.bin"
    if not cpu_weights.is_absolute():
        cpu_weights = ROOT / cpu_weights
    plan = DualEngineResidencyPlan(
        schema="lucidota.dual_engine_residency_plan.v1",
        generated_at=now_z(),
        advisory_only=True,
        decision=decision,
        cpu_fairyfuse={
            "engine_channel": "cpu_fairyfuse_ternary",
            "always_on": True,
            "runtime": "python_ctypes_mmap",
            "packed_ternary_weights": _file_receipt(cpu_weights),
            "expected_mmap_mb": FAIRYFUSE_TERNARY_WEIGHTS_MB,
            "lazy_page_fault_management": True,
            "lora_adapter_lane": "ternary_quantized_conceptual_adapters",
            "kernel_path": os.environ.get("LUCIDOTA_FAIRYFUSE_TERNARY_KERNEL", ""),
        },
        gpu_q4_deepseek={
            "engine_channel": "gpu_q4_deepseek",
            "always_on": True,
            "runtime": "llama_cpp_cuda_q4_k_m",
            "base_model": _file_receipt(DEEPSEEK_Q4_MODEL),
            "mamba_graph_stream_cache": _file_receipt(MAMBA_GRAPH_CACHE_MODEL),
            "budget_mb": budget,
            "deepseek_resident_mb": DUAL_GPU_DEEPSEEK_MB,
            "mamba_cache_resident_mb": DUAL_GPU_MAMBA_CACHE_MB,
            "estimated_resident_mb": resident_gpu_mb,
            "adapter_headroom_mb": adapter_headroom_mb,
            "can_hot_swap_q4_lora": adapter_headroom_mb > 0,
            "requested_adapter_count": len(requested_adapters),
            "lora_context_token_ceiling": token_ceiling,
            "temporal_mtime_snapshot_ceiling": MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS,
            "context_reaper": "flush llama.cpp slot/KV context after synthesis pass",
            "gpu": gpu,
        },
        outbound_state="draft_only",
        policy="CPU FairyFuse and GPU Q4 lanes are always-on; embeddings flash then evict; outbound text remains draft_only.",
    ).as_dict()
    receipt = RuntimePreemptionReceipt(
        receipt_kind="dual_engine_residency_plan",
        action="plan_dual_engine_residency",
        status=decision,
        generated_at=now_z(),
        detail=plan,
    )
    _append_runtime_receipt(receipt, path=RUNTIME_DIR / "dual_engine_receipts.jsonl")
    return plan


def _int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def load_lora_manifests(root: Path = LORA_DIR) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("manifest_path", str(path.relative_to(ROOT)))
            data.setdefault("local_path", str(path.parent.relative_to(ROOT)))
            out.append(data)
        except Exception:
            continue
    return out


def manifest_adapter_paths(manifest: dict[str, Any]) -> list[Path]:
    """Return concrete local GGUF adapter paths advertised by a LoRA manifest.

    Dataset-only cartridges are valid LUCIDOTA artifacts but are not hot-swappable
    into llama.cpp until a GGUF adapter exists.  This function only returns
    files that are present on disk.
    """
    candidates: list[Path] = []
    for key in ("lora_gguf", "lora_gguf_path", "adapter_gguf", "adapter_gguf_path", "adapter_path", "gguf_path"):
        value = manifest.get(key)
        if not value:
            continue
        p = Path(str(value))
        if not p.is_absolute():
            p = ROOT / p
        if p.exists() and p.is_file():
            candidates.append(p)
    local = manifest.get("local_path") or manifest.get("manifest_path")
    if local:
        base = ROOT / str(local)
        if base.name == "manifest.json":
            base = base.parent
        if base.exists():
            candidates.extend(sorted(base.glob("*.gguf")))
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def select_adapters(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = payload or {}
    state = state or {}
    requested: list[str] = []
    for key in ("adapter_id", "lora_adapter_id"):
        if payload.get(key):
            requested.append(str(payload[key]))
    for key in ("adapter_ids", "lora_adapter_ids", "lora_adapters"):
        value = payload.get(key)
        if isinstance(value, str):
            requested.extend([x.strip() for x in value.split(",") if x.strip()])
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
            requested.extend([str(x) for x in value])

    manifests = load_lora_manifests()
    by_id = {m.get("adapter_id"): m for m in manifests if m.get("adapter_id")}
    selected: list[dict[str, Any]] = []
    for adapter_id in dict.fromkeys(requested):
        selected.append(dict(by_id.get(adapter_id) or {"adapter_id": adapter_id, "status": "requested_missing_manifest"}))

    sha = state.get("digest") or payload.get("source_sha256") or payload.get("sha256")
    if sha:
        for m in manifests:
            if m.get("source_sha256") == sha and m.get("adapter_id") not in {x.get("adapter_id") for x in selected}:
                selected.append(dict(m))

    title_hint = str(payload.get("title") or state.get("source_path_repo_relative") or state.get("source_path_absolute") or "").lower()
    if not selected and title_hint:
        for m in manifests:
            title = str(m.get("title") or m.get("source_path") or "").lower()
            if title and any(tok and tok in title for tok in title_hint.replace("_", " ").replace("-", " ").split()[:8]):
                selected.append(dict(m))
                break
    return selected


def has_mtime_snapshot_tag(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> bool:
    payload = payload or {}
    state = state or {}
    try:
        return "mtime_snapshot_v1" in json.dumps({"payload": payload, "state": state}, ensure_ascii=False, default=str)
    except Exception:
        return False


def lora_context_token_ceiling(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> int:
    return MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS if has_mtime_snapshot_tag(payload, state) else DEFAULT_LORA_CONTEXT_TOKENS


def _slot(action: str, artifact_id: str, artifact_kind: str, mb: int, reason: str, detail: dict[str, Any] | None = None) -> VramSlotPlan:
    return VramSlotPlan(artifact_id=artifact_id, artifact_kind=artifact_kind, action=action, estimated_mb=int(mb), reason=reason, detail=detail or {})


def plan_lora_preemption(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None, *, include_gpu: bool = True) -> dict[str, Any]:
    payload = payload or {}
    state = state or {}
    gpu = gpu_memory() if include_gpu else {"status": "skipped"}
    budget = _int(payload.get("budget_vram_mb") or payload.get("vram_budget_mb"), DEFAULT_BUDGET_MB)
    reserve = _int(payload.get("vram_reserve_mb") or payload.get("kv_cache_reserve_mb"), DEFAULT_RESERVE_MB)
    base_model_id = str(payload.get("model_id") or payload.get("base_model_id") or DEFAULT_BASE_MODEL_ID)
    base_mb = _int(payload.get("base_model_vram_mb") or payload.get("model_vram_mb"), DEFAULT_BASE_MODEL_MB)
    embedding_mb = _int(payload.get("embedding_model_vram_mb"), DEFAULT_EMBEDDING_MB)
    embedding_requested = bool(payload.get("embedding_requested") or payload.get("dynamic_embedding") or payload.get("embedding_flash"))
    adapters = select_adapters(payload, state)
    token_ceiling = lora_context_token_ceiling(payload, state)
    for adapter in adapters:
        paths = manifest_adapter_paths(adapter)
        if paths:
            adapter["hot_swap_ready"] = True
            adapter["adapter_gguf_paths"] = [_rel(p) for p in paths]
        else:
            adapter["hot_swap_ready"] = False
        adapter["effective_max_source_tokens"] = min(_int(adapter.get("max_source_tokens"), DEFAULT_LORA_CONTEXT_TOKENS), token_ceiling)

    observed_total = gpu.get("total_mb") if gpu.get("status") == "ok" else None
    observed_free = gpu.get("free_mb") if gpu.get("status") == "ok" else None
    effective_budget = min(budget, int(observed_total)) if observed_total else budget
    max_resident = max(0, effective_budget - reserve)
    allocatable_now = max(0, (min(effective_budget, int(observed_free)) if observed_free is not None else effective_budget) - reserve)

    slots: list[VramSlotPlan] = []
    resident_used = 0
    if base_mb > max_resident:
        slots.append(_slot("reject_slot", base_model_id, "base_model", base_mb, f"base model needs {base_mb}MB but max resident capacity is {max_resident}MB"))
    elif base_mb <= allocatable_now:
        slots.append(_slot("load_model", base_model_id, "base_model", base_mb, "primary synthesis model fits current allocatable VRAM"))
        resident_used += base_mb
    else:
        slots.append(_slot("defer_model", base_model_id, "base_model", base_mb, f"primary model fits budget but current allocatable VRAM is {allocatable_now}MB"))

    remaining_after_base = max(0, allocatable_now - resident_used)
    for m in adapters:
        adapter_id = str(m.get("adapter_id") or "unknown_adapter")
        mb = _int(m.get("expected_vram_mb") or m.get("adapter_vram_mb"), DEFAULT_ADAPTER_MB)
        combined = base_mb + mb
        if not m.get("hot_swap_ready"):
            action = "defer_adapter"
            reason = "adapter manifest is present but no local GGUF adapter file is available for llama.cpp hot-swap"
        elif mb > max_resident or combined > max_resident:
            action = "reject_slot"
            reason = f"base+adapter estimated {combined}MB exceeds resident capacity {max_resident}MB"
        elif mb <= remaining_after_base and slots[0].action == "load_model":
            action = "attach_adapter"
            reason = "adapter can hot-swap onto resident base model within current VRAM"
            remaining_after_base -= mb
            resident_used += mb
        else:
            action = "defer_adapter"
            reason = f"adapter is valid but should wait for embedding eviction or resident-slot rotation; remaining now {remaining_after_base}MB"
        slots.append(_slot(action, adapter_id, "lora_adapter", mb, reason, {k: m.get(k) for k in ("title", "author", "status", "training_status", "target_model_id", "manifest_path", "local_path") if k in m}))

    if embedding_requested:
        if embedding_mb <= max_resident:
            slots.append(_slot("flash_embedding_then_evict", str(payload.get("embedding_model") or "embedding-mini"), "embedding_model", embedding_mb, "embedding lane is transient and must be evicted before synthesis residency"))
        else:
            slots.append(_slot("reject_slot", str(payload.get("embedding_model") or "embedding-mini"), "embedding_model", embedding_mb, "embedding model alone exceeds max slot capacity"))

    actions = [s.action for s in slots]
    decision = "reject" if "reject_slot" in actions else "defer" if any(a.startswith("defer") for a in actions) else "allow"
    return {
        "schema": "lucidota.model_vram_lora_preemption_plan.v1",
        "generated_at": now_z(),
        "advisory_only": True,
        "decision": decision,
        "budget_vram_mb": budget,
        "effective_budget_mb": effective_budget,
        "reserve_mb": reserve,
        "allocatable_now_mb": allocatable_now,
        "estimated_resident_mb": resident_used,
        "headroom_after_attached_mb": max_resident - resident_used,
        "gpu": gpu,
        "slots": [s.as_dict() for s in slots],
        "adapter_candidates": adapters,
        "lora_context_token_ceiling": token_ceiling,
        "temporal_fragility_policy": {
            "mtime_snapshot_v1_fast_path": token_ceiling == MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS,
            "default_lora_context_tokens": DEFAULT_LORA_CONTEXT_TOKENS,
            "mtime_snapshot_lora_context_tokens": MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS,
            "date_extraction_router": "treelite" if token_ceiling == MTIME_SNAPSHOT_LORA_CONTEXT_TOKENS else "default",
        },
        "dual_engine_residency": plan_dual_engine_residency(payload, state, include_gpu=include_gpu),
        "policy": "no model import; no LoRA load; advisory receipt only; embeddings flash then evict",
    }


def persist_governor_decision(conn: Any, plan: dict[str, Any], *, loadout_id: str = "gtx1650-special-forces-v0") -> str | None:
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        conn.execute("CREATE SCHEMA IF NOT EXISTS lucidota_runtime")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lucidota_runtime.load_governor_decision (
              decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
              loadout_id text NOT NULL,
              target_gpu text NOT NULL DEFAULT '',
              budget_vram_mb integer NOT NULL,
              observed_used_mb integer,
              observed_free_mb integer,
              estimated_required_mb integer NOT NULL,
              headroom_mb integer NOT NULL,
              decision text NOT NULL CHECK (decision IN ('allow','defer','reject')),
              rationale text NOT NULL DEFAULT '',
              detail jsonb NOT NULL DEFAULT '{}'::jsonb,
              gpu_observation jsonb NOT NULL DEFAULT '{}'::jsonb,
              action_plan jsonb NOT NULL DEFAULT '{}'::jsonb,
              created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        gpu = plan.get("gpu") or {}
        row = conn.execute(
            """
            INSERT INTO lucidota_runtime.load_governor_decision(
              loadout_id, target_gpu, budget_vram_mb, observed_used_mb, observed_free_mb,
              estimated_required_mb, headroom_mb, decision, rationale, detail, gpu_observation, action_plan
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
            RETURNING decision_id::text
            """,
            (
                loadout_id,
                str(gpu.get("name") or os.environ.get("LUCIDOTA_TARGET_GPU") or "GTX 1650 4GB"),
                int(plan.get("effective_budget_mb") or plan.get("budget_vram_mb") or DEFAULT_BUDGET_MB),
                gpu.get("used_mb"),
                gpu.get("free_mb"),
                int(plan.get("estimated_resident_mb") or 0),
                int(plan.get("headroom_after_attached_mb") or 0),
                plan.get("decision") or "defer",
                f"decision={plan.get('decision')} resident={plan.get('estimated_resident_mb')}MB reserve={plan.get('reserve_mb')}MB headroom={plan.get('headroom_after_attached_mb')}MB",
                json.dumps(plan, ensure_ascii=False, default=str),
                json.dumps(gpu, ensure_ascii=False, default=str),
                json.dumps(plan, ensure_ascii=False, default=str),
            ),
        ).fetchone()
        return str(row[0]) if row else None
    except Exception:
        return None


def evict_transient_embedding_context(reason: str = "embedding_vector_generation_complete") -> RuntimePreemptionReceipt:
    """Best-effort memory eviction hook for transient embedding lanes.

    This does not kill model daemons or touch the resident synthesis server.  It
    releases Python-side references and, when torch is importable, empties CUDA
    allocator caches after an embedding mini-model has completed its vector pass.
    """
    detail: dict[str, Any] = {"reason": reason, "gc_collected": gc.collect()}
    try:
        import torch  # type: ignore

        detail["torch_imported"] = True
        detail["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            before = int(torch.cuda.memory_allocated())
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
                detail["ipc_collect"] = True
            except Exception as exc:
                detail["ipc_collect_warning"] = f"{type(exc).__name__}: {exc}"
            after = int(torch.cuda.memory_allocated())
            detail["cuda_memory_allocated_before"] = before
            detail["cuda_memory_allocated_after"] = after
    except Exception as exc:
        detail["torch_imported"] = False
        detail["torch_warning"] = f"{type(exc).__name__}: {exc}"
    receipt = RuntimePreemptionReceipt(
        receipt_kind="embedding_eviction",
        action="evict_transient_embedding_context",
        status="completed",
        generated_at=now_z(),
        detail=detail,
    )
    _append_runtime_receipt(receipt)
    return receipt


class EmbeddingFlashEvictionContext:
    """Context manager for on-demand embedding model residency.

    Use this around the actual vector encoder call:

        with EmbeddingFlashEvictionContext(plan) as receipt:
            vectors = encoder(texts)

    `__exit__` always evicts transient Python/CUDA allocator state and records a
    receipt, even when the encoder raises.
    """

    def __init__(self, plan: dict[str, Any] | None = None, *, reason: str = "embedding_flash_lane"):
        self.plan = plan or {}
        self.reason = reason
        self.enter_receipt: RuntimePreemptionReceipt | None = None
        self.exit_receipt: RuntimePreemptionReceipt | None = None

    def __enter__(self) -> RuntimePreemptionReceipt:
        self.enter_receipt = RuntimePreemptionReceipt(
            receipt_kind="embedding_flash",
            action="enter_embedding_flash_lane",
            status="entered",
            generated_at=now_z(),
            detail={"reason": self.reason, "plan_decision": self.plan.get("decision"), "slots": self.plan.get("slots", [])},
        )
        _append_runtime_receipt(self.enter_receipt)
        return self.enter_receipt

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "exception_then_evicted" if exc else "evicted"
        self.exit_receipt = evict_transient_embedding_context(reason=f"{self.reason}:{status}")
        return False


def flash_embed_then_evict(encoder: Any, *args: Any, plan: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    """Run a caller-supplied encoder under mandatory transient-context eviction."""
    with EmbeddingFlashEvictionContext(plan):
        return encoder(*args, **kwargs)


def build_llama_lora_args(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None, *, init_without_apply: bool = True) -> dict[str, Any]:
    """Build llama.cpp server args for LoRA cartridges that have GGUF adapters."""
    payload = payload or {}
    state = state or {}
    adapters = select_adapters(payload, state)
    args: list[str] = []
    selected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for adapter in adapters:
        paths = manifest_adapter_paths(adapter)
        if not paths:
            missing.append({"adapter_id": adapter.get("adapter_id"), "reason": "no local GGUF adapter path in manifest/cartridge"})
            continue
        for path in paths:
            args.extend(["--lora", str(path)])
            selected.append({"adapter_id": adapter.get("adapter_id"), "path": _rel(path), "title": adapter.get("title")})
    if selected and init_without_apply:
        args.append("--lora-init-without-apply")
    return {
        "schema": "lucidota.llama_lora_args.v1",
        "generated_at": now_z(),
        "args": args,
        "selected_adapters": selected,
        "deferred_adapters": missing,
        "init_without_apply": bool(init_without_apply),
    }


def _http_json(method: str, url: str, payload: Any | None = None, timeout: float = 5.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost/control URL supplied by runtime config
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body) if body.strip() else None


def context_reaper_flush(
    *,
    llama_server_url: str | None = None,
    slot_id: int = 0,
    timeout: float = 3.0,
    reason: str = "gpu_synthesis_pass_complete",
) -> dict[str, Any]:
    """Best-effort llama.cpp KV/context reaper.

    It never kills or restarts the resident DeepSeek server.  If the server is
    unavailable or its slot endpoint differs, the receipt is recorded as
    deferred and the caller continues.
    """
    base = (llama_server_url or os.environ.get("LUCIDOTA_LLAMA_SERVER_URL") or "http://127.0.0.1:8080").rstrip("/")
    attempts = [
        ("POST", f"{base}/slots/{slot_id}?action=erase", None),
        ("POST", f"{base}/slots/{slot_id}", {"action": "erase"}),
        ("POST", f"{base}/slot/erase", {"id_slot": slot_id}),
    ]
    detail: dict[str, Any] = {"server_url": base, "slot_id": slot_id, "reason": reason, "attempts": []}
    status = "deferred"
    for method, url, payload in attempts:
        entry = {"method": method, "url": url, "payload": payload}
        try:
            entry["response"] = _http_json(method, url, payload=payload, timeout=timeout)
            entry["ok"] = True
            status = "completed"
            detail["attempts"].append(entry)
            break
        except urllib.error.HTTPError as exc:
            entry["ok"] = False
            entry["warning"] = f"HTTPError:{exc.code}:{exc.reason}"
        except Exception as exc:
            entry["ok"] = False
            entry["warning"] = f"{type(exc).__name__}: {exc}"
        detail["attempts"].append(entry)
    receipt = RuntimePreemptionReceipt(
        receipt_kind="context_reaper",
        action="flush_gpu_kv_context",
        status=status,
        generated_at=now_z(),
        detail=detail,
    )
    _append_runtime_receipt(receipt, path=RUNTIME_DIR / "dual_engine_receipts.jsonl")
    return receipt.as_dict()


def llama_lora_hot_swap(
    *,
    server_url: str | None = None,
    adapter_scales: list[dict[str, Any]] | None = None,
    timeout: float = 5.0,
) -> RuntimePreemptionReceipt:
    """Apply global LoRA scales on a running llama.cpp server.

    llama.cpp exposes `GET /lora-adapters` and `POST /lora-adapters`.  This hook
    only changes adapter scales that were already loaded at server start with
    `--lora`; it does not restart the resident model.
    """
    base = (server_url or os.environ.get("LUCIDOTA_LLAMA_SERVER_URL") or "http://127.0.0.1:8080").rstrip("/")
    payload = adapter_scales or []
    detail: dict[str, Any] = {"server_url": base, "requested_scales": payload}
    status = "completed"
    try:
        before = _http_json("GET", f"{base}/lora-adapters", timeout=timeout)
        detail["before"] = before
        after = _http_json("POST", f"{base}/lora-adapters", payload=payload, timeout=timeout)
        detail["after"] = after
    except Exception as exc:
        status = "deferred"
        detail["warning"] = f"{type(exc).__name__}: {exc}"
        detail["safe_fallback"] = "no resident model restart; request remains draft/advisory"
    receipt = RuntimePreemptionReceipt(
        receipt_kind="lora_hot_swap",
        action="POST /lora-adapters",
        status=status,
        generated_at=now_z(),
        detail=detail,
    )
    _append_runtime_receipt(receipt)
    return receipt


def neo_knows_kung_fu_context(
    payload: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    *,
    server_url: str | None = None,
) -> dict[str, Any]:
    """Return a complete runtime hook packet for DeepSeek+LoRA preemption.

    The packet is safe to persist in workflow state.  If a running llama.cpp
    server and loaded adapter IDs are supplied, it attempts a hot-swap scale
    update; otherwise it records an explicit deferred receipt.
    """
    payload = payload or {}
    state = state or {}
    plan = plan_lora_preemption(payload, state)
    llama_args = build_llama_lora_args(payload, state, init_without_apply=True)
    adapter_scales = payload.get("llama_lora_scales") or payload.get("lora_scales")
    hot_swap_receipt: dict[str, Any] | None = None
    if isinstance(adapter_scales, list):
        hot_swap_receipt = llama_lora_hot_swap(server_url=server_url, adapter_scales=adapter_scales).as_dict()
    elif llama_args["selected_adapters"]:
        receipt = RuntimePreemptionReceipt(
            receipt_kind="lora_hot_swap",
            action="prepare_loaded_adapters",
            status="prepared_not_applied",
            generated_at=now_z(),
            detail={"reason": "adapters have GGUF paths; provide llama_lora_scales with server adapter ids to apply without restart", "llama_args": llama_args},
        )
        _append_runtime_receipt(receipt)
        hot_swap_receipt = receipt.as_dict()
    else:
        receipt = RuntimePreemptionReceipt(
            receipt_kind="lora_hot_swap",
            action="defer_dataset_only_cartridges",
            status="deferred",
            generated_at=now_z(),
            detail={"reason": "selected cartridges are dataset/training manifests without local GGUF adapter files", "llama_args": llama_args},
        )
        _append_runtime_receipt(receipt)
        hot_swap_receipt = receipt.as_dict()
    return {
        "schema": "lucidota.neo_knows_kung_fu_context.v1",
        "generated_at": now_z(),
        "plan": plan,
        "llama_args": llama_args,
        "hot_swap_receipt": hot_swap_receipt,
        "dual_engine_residency": plan.get("dual_engine_residency") or plan_dual_engine_residency(payload, state),
        "context_reaper_hook": "pypeline.math.model_vram_scheduler.context_reaper_flush",
        "embedding_eviction_hook": "pypeline.math.model_vram_scheduler.EmbeddingFlashEvictionContext",
        "policy": "DeepSeek base remains resident; embeddings run in flash lane and evict; LoRA adapters hot-swap only when GGUF adapters are preloaded.",
    }
