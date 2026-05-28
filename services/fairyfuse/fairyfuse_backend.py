#!/usr/bin/env python3
"""FairyFuse resident ternary backend for LUCIDOTA.

Local-only CPU residency layer:
- memory-map packed ternary weights when present;
- invoke an optional fused ternary GEMV shared library through ctypes;
- otherwise keep a deterministic multiplication-free symbolic route alive.

The public ``route_command`` API remains compatible with the original v0 router.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TERNARY_DIMS = 25
BACKEND_NAME = "FAIRYFUSE_V1_RESIDENT_TERNARY"
FAST_PATH_CONTRACT = "Resident CPU ternary route; mmap weights + ctypes fused GEMV when available; symbolic no-matmul fallback otherwise."
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_CANDIDATES = [
    ROOT / "03_VAULT" / "models" / "fairyfuse" / "deepseek_ternary_packed.bin",
    ROOT / "03_VAULT" / "models" / "ternary" / "deepseek_ternary_packed.bin",
    ROOT / "03_VAULT" / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-ternary-packed.bin",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_payload(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    return json.dumps(
        {"raw_command": raw_command, "normalized_intent": normalized_intent, "context": context},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def payload_sha256(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(raw_command, normalized_intent, context).encode()).hexdigest()


def primitive_id(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> int:
    digest = payload_sha256(raw_command, normalized_intent, context)
    return int(digest[:10], 16) % TERNARY_DIMS


def ternary_vector(raw_command: str, normalized_intent: str, context: dict[str, Any], dims: int = TERNARY_DIMS) -> list[int]:
    seed = canonical_payload(raw_command, normalized_intent, context).encode()
    digest = hashlib.sha512(seed).digest()
    return [(digest[idx % len(digest)] % 3) - 1 for idx in range(dims)]


def confidence_bps(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> int:
    score = 3000
    score += min(2500, len(raw_command.strip()) * 20)
    score += 2500 if normalized_intent.strip() else 0
    score += min(1500, len(context) * 300)
    return max(0, min(9900, score))


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def default_weights_path() -> Path | None:
    env = _env_path("LUCIDOTA_FAIRYFUSE_TERNARY_WEIGHTS")
    if env:
        return env
    for candidate in DEFAULT_WEIGHTS_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_WEIGHTS_CANDIDATES[0]


@dataclass(frozen=True)
class FairyFuseRoute:
    schema: str
    backend: str
    status: str
    payload_sha256: str
    primitive_id: int
    ternary_vector: list[int]
    confidence_bps: int
    fast_path_preserved: bool
    fast_path_contract: str
    created_at: str
    runtime_status: dict[str, Any] | None = None
    inference_receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PackedTernaryWeights:
    path: str
    size_bytes: int
    mapped: bool
    status: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FairyFuseResidentEngine:
    """Always-on CPU ternary runtime façade.

    The mmap object is held for the lifetime of the Python process.  If a native
    shared library is configured, calls are passed as raw pointers.  Missing
    weights/kernel keep the resident route alive in symbolic mode instead of
    faking a loaded model.
    """

    def __init__(self, weights_path: Path | None = None, kernel_path: Path | None = None, *, dims: int = TERNARY_DIMS):
        self.weights_path = weights_path or default_weights_path()
        self.kernel_path = kernel_path or _env_path("LUCIDOTA_FAIRYFUSE_TERNARY_KERNEL")
        self.kernel_symbol = os.environ.get("LUCIDOTA_FAIRYFUSE_TERNARY_SYMBOL", "fairyfuse_ternary_gemv_f32")
        self.dims = int(os.environ.get("LUCIDOTA_FAIRYFUSE_DIMS", dims))
        self.rows = int(os.environ.get("LUCIDOTA_FAIRYFUSE_ROWS", TERNARY_DIMS))
        self._fh: Any | None = None
        self._mmap: mmap.mmap | None = None
        self._kernel: Any | None = None
        self._kernel_fn: Any | None = None
        self._loaded_at: str | None = None
        self._load_errors: list[str] = []

    def load(self) -> "FairyFuseResidentEngine":
        if self._loaded_at:
            return self
        self._loaded_at = utc_now()
        self._load_weights()
        self._load_kernel()
        return self

    def _load_weights(self) -> None:
        p = self.weights_path
        if not p:
            self._load_errors.append("weights_path_unset")
            return
        try:
            if not p.exists() or not p.is_file():
                self._load_errors.append(f"weights_missing:{_rel(p)}")
                return
            self._fh = p.open("rb")
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception as exc:
            self._load_errors.append(f"weights_mmap_failed:{type(exc).__name__}:{exc}")
            self._mmap = None

    def _load_kernel(self) -> None:
        p = self.kernel_path
        if not p:
            return
        try:
            if not p.exists() or not p.is_file():
                self._load_errors.append(f"kernel_missing:{_rel(p)}")
                return
            self._kernel = ctypes.CDLL(str(p))
            self._kernel_fn = getattr(self._kernel, self.kernel_symbol)
            self._kernel_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
            self._kernel_fn.restype = None
        except Exception as exc:
            self._load_errors.append(f"kernel_load_failed:{type(exc).__name__}:{exc}")
            self._kernel = None
            self._kernel_fn = None

    @property
    def mmap_loaded(self) -> bool:
        return self._mmap is not None

    @property
    def kernel_loaded(self) -> bool:
        return self._kernel_fn is not None

    def weights_receipt(self) -> PackedTernaryWeights:
        p = self.weights_path
        exists = bool(p and p.exists() and p.is_file())
        return PackedTernaryWeights(
            path=_rel(p) if p else "",
            size_bytes=int(p.stat().st_size) if exists else 0,
            mapped=self.mmap_loaded,
            status="mapped" if self.mmap_loaded else "missing_or_deferred",
            detail={"lazy_page_faults": self.mmap_loaded, "read_only": True, "expected_weight_mb": 2240},
        )

    def status(self) -> dict[str, Any]:
        self.load()
        return {
            "schema": "lucidota.fairyfuse.resident_engine_status.v1",
            "backend": BACKEND_NAME,
            "always_on": True,
            "mode": "ctypes_mmap_kernel" if self.kernel_loaded and self.mmap_loaded else "mmap_symbolic" if self.mmap_loaded else "symbolic_no_weights",
            "fast_path_preserved": True,
            "multiplication_free": True,
            "loaded_at": self._loaded_at,
            "weights": self.weights_receipt().to_dict(),
            "kernel": {"path": _rel(self.kernel_path) if self.kernel_path else "", "symbol": self.kernel_symbol, "loaded": self.kernel_loaded},
            "dims": self.dims,
            "rows": self.rows,
            "load_errors": list(self._load_errors),
        }

    def _fallback_projection(self, vector: Iterable[int | float], *, rows: int | None = None) -> list[int]:
        vec = [int(1 if float(v) > 0 else -1 if float(v) < 0 else 0) for v in vector]
        n_rows = int(rows or self.rows or TERNARY_DIMS)
        out: list[int] = []
        mm = self._mmap
        mlen = len(mm) if mm is not None else 0
        for r in range(n_rows):
            total = 0
            for c, value in enumerate(vec):
                if value == 0:
                    continue
                if mm is not None and mlen:
                    w = (mm[(r * max(1, len(vec)) + c) % mlen] % 3) - 1
                else:
                    w = ((r * 131 + c * 17 + len(vec)) % 3) - 1
                if w > 0:
                    total += value
                elif w < 0:
                    total -= value
            out.append(total)
        return out

    def _kernel_projection(self, vector: Iterable[int | float], *, rows: int | None = None) -> list[float] | None:
        if not (self.kernel_loaded and self.mmap_loaded):
            return None
        try:
            import numpy as np  # type: ignore
            vec = np.ascontiguousarray(list(vector), dtype=np.float32)
            n_rows = int(rows or self.rows or TERNARY_DIMS)
            out = np.zeros((n_rows,), dtype=np.float32)
            weight_ptr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mmap)))  # type: ignore[arg-type]
            self._kernel_fn(weight_ptr, vec.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(n_rows), ctypes.c_uint64(vec.size))
            return [float(x) for x in out.tolist()]
        except (TypeError, BufferError):
            # ACCESS_READ mmap may not expose a writable buffer pointer; keep zero-copy vector path but fall back for weights.
            return None
        except Exception as exc:
            self._load_errors.append(f"kernel_projection_failed:{type(exc).__name__}:{exc}")
            return None

    def infer_vector(self, vector: Iterable[int | float], *, rows: int | None = None) -> dict[str, Any]:
        self.load()
        started = datetime.now(timezone.utc)
        kernel_out = self._kernel_projection(vector, rows=rows)
        if kernel_out is not None:
            projection: list[int | float] = kernel_out
            mode = "ctypes_mmap_kernel"
        else:
            projection = self._fallback_projection(vector, rows=rows)
            mode = "multiplication_free_fallback"
        elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
        return {
            "schema": "lucidota.fairyfuse.ternary_inference_receipt.v1",
            "backend": BACKEND_NAME,
            "mode": mode,
            "projection": projection,
            "rows": len(projection),
            "runtime_ms": round(elapsed_ms, 3),
            "weights_mmap_loaded": self.mmap_loaded,
            "kernel_loaded": self.kernel_loaded,
            "multiplication_free": True,
            "created_at": utc_now(),
        }

    def route(self, raw_command: str, normalized_intent: str, context: dict[str, Any]) -> FairyFuseRoute:
        vec = ternary_vector(raw_command, normalized_intent, context, dims=self.dims)
        receipt = self.infer_vector(vec, rows=min(self.rows, TERNARY_DIMS))
        status = "USABLE_NOW_RESIDENT_TERNARY" if self.mmap_loaded else "USABLE_NOW_SYMBOLIC_BACKEND_WEIGHTS_MISSING"
        return FairyFuseRoute(
            schema="lucidota.fairyfuse.route.v1",
            backend=BACKEND_NAME,
            status=status,
            payload_sha256=payload_sha256(raw_command, normalized_intent, context),
            primitive_id=primitive_id(raw_command, normalized_intent, context),
            ternary_vector=vec,
            confidence_bps=confidence_bps(raw_command, normalized_intent, context),
            fast_path_preserved=True,
            fast_path_contract=FAST_PATH_CONTRACT,
            created_at=utc_now(),
            runtime_status=self.status(),
            inference_receipt=receipt,
        )

    def close(self) -> None:
        try:
            if self._mmap is not None:
                self._mmap.close()
        finally:
            self._mmap = None
            try:
                if self._fh is not None:
                    self._fh.close()
            finally:
                self._fh = None


_RESIDENT_ENGINE: FairyFuseResidentEngine | None = None


def resident_engine_from_env() -> FairyFuseResidentEngine:
    global _RESIDENT_ENGINE
    if _RESIDENT_ENGINE is None:
        _RESIDENT_ENGINE = FairyFuseResidentEngine().load()
    return _RESIDENT_ENGINE


def route_command(raw_command: str, normalized_intent: str, context: dict[str, Any]) -> FairyFuseRoute:
    return resident_engine_from_env().route(raw_command, normalized_intent, context)
