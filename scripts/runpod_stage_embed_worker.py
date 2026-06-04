#!/usr/bin/env python3
"""Stage and locally execute a lightweight RunPod embed worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

try:
    import requests
except Exception:  # pragma: no cover - defensive in minimal environments
    requests = None  # type: ignore


SCHEMA_RUN = "lucidota.runpod_stage_embed_worker.run_plan.v1"
SCHEMA_STAGE = "lucidota.runpod_stage_embed_worker.stage_receipt.v1"
SCHEMA_LOCAL_RECEIPT = "lucidota.runpod_stage_embed_worker.local_run_receipt.v1"
DEFAULT_REMOTE_ROOT = "/workspace/lucidota_ingest_accel"
DEFAULT_REMOTE_WORKER_PATH = f"{DEFAULT_REMOTE_ROOT}/runpod_embed_worker.py"
DEFAULT_INPUT = f"{DEFAULT_REMOTE_ROOT}/chunks_500tok.jsonl"
DEFAULT_OUTPUT = f"{DEFAULT_REMOTE_ROOT}/chunk_embeddings.jsonl"
DEFAULT_RECEIPT = f"{DEFAULT_REMOTE_ROOT}/receipts/chunk_embeddings_receipt.json"
DEFAULT_DIMENSIONS = 16
DEFAULT_MODEL_API_URL_ENV = "RUNPOD_EMBED_MODEL_URL"
DEFAULT_MODEL_ID_ENV = "RUNPOD_EMBED_MODEL_ID"
DEFAULT_MODEL_KEY_ENV = "RUNPOD_EMBED_MODEL_KEY"
DEFAULT_LOCAL_MODEL_ENV = "RUNPOD_EMBED_LOCAL_MODEL"
DEFAULT_LOCAL_DEVICE_ENV = "RUNPOD_EMBED_DEVICE"
DEFAULT_LOCAL_MODEL = ""
DEFAULT_LOCAL_DEVICE = "auto"
DEFAULT_LOCAL_BATCH_SIZE = 16
REQUEST_TIMEOUT = 30


REMOTE_WORKER_SOURCE = r'''#!/usr/bin/env python3
"""RunPod-side chunk embedding worker.

Reads JSONL chunks and writes embeddings JSONL plus a receipt. No DB writes are
performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

REMOTE_ROOT = "/workspace/lucidota_ingest_accel"
DEFAULT_INPUT_JSONL = f"{REMOTE_ROOT}/chunks_500tok.jsonl"
DEFAULT_OUTPUT_JSONL = f"{REMOTE_ROOT}/chunk_embeddings.jsonl"
DEFAULT_RECEIPT_JSON = f"{REMOTE_ROOT}/receipts/chunk_embeddings_receipt.json"
DEFAULT_DIMENSIONS = 16
DEFAULT_MODEL_URL = ""
DEFAULT_MODEL_ID = ""
DEFAULT_MODEL_KEY = ""
DEFAULT_LOCAL_MODEL = ""
DEFAULT_LOCAL_DEVICE = "auto"
LOCAL_BATCH_SIZE = 16


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _coerce_dimensions(values: list[float], dimensions: int) -> list[float]:
    if dimensions <= 0:
        return [round(float(v), 8) for v in values]
    trimmed = [float(v) for v in values[:dimensions]]
    if len(trimmed) < dimensions:
        trimmed.extend([0.0] * (dimensions - len(trimmed)))
    return [round(v, 8) for v in trimmed]


def deterministic_vector(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    if dimensions <= 0:
        return []
    vals: list[float] = []
    for i in range(dimensions):
        h = hashlib.sha256(f"{i}:{text}".encode("utf-8", errors="replace")).digest()
        v = int.from_bytes(h[:4], "big") / float(0xFFFFFFFF)
        vals.append((v * 2.0) - 1.0)
    mag = sqrt(sum(v * v for v in vals))
    if mag == 0:
        return [0.0] * dimensions
    return [round(v / mag, 8) for v in vals]


def _resolve_device(raw_device: str) -> str:
    if raw_device and raw_device.lower() != "auto":
        return raw_device
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def model_embedding(text: str, *, model_url: str, model_id: str, model_key: str, dimensions: int) -> list[float] | None:
    if not (model_url and model_id):
        return None
    endpoint = model_url.rstrip("/") + "/v1/embeddings"
    payload = json.dumps({"model": model_id, "input": [text]}).encode("utf-8")
    headers = {"content-type": "application/json", "user-agent": "runpod-embed-worker/1"}
    if model_key:
        headers["Authorization"] = f"Bearer {model_key}"
    req = urllib.request.Request(endpoint, method="POST", data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            data = body.get("data") or []
            if not data:
                return None
            embedding = data[0].get("embedding") or []
            if not embedding:
                return None
            return _coerce_dimensions([float(v) for v in embedding], dimensions=dimensions)
    except Exception:
        return None


def _row_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("content") or "")


def _row_chunk_id(row: dict[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("chunk_ref") or row.get("id") or "")


def _load_sentence_transformer(local_model: str, device: str) -> tuple[Any, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        return None, f"SENTENCE_TRANSFORMERS_IMPORT_FAILED:{exc}"
    try:
        return SentenceTransformer(local_model, device=device), None
    except Exception as exc:
        return None, f"SENTENCE_TRANSFORMERS_LOAD_FAILED:{exc}"


def _encode_sentence_transformer_batch(
    model: Any,
    texts: list[str],
    *,
    dimensions: int,
    batch_size: int = LOCAL_BATCH_SIZE,
) -> tuple[list[list[float]] | None, str | None]:
    try:
        raw_vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    except Exception as exc:
        return None, f"SENTENCE_TRANSFORMERS_ENCODE_FAILED:{exc}"

    vectors: list[list[float]] = []
    try:
        for raw in raw_vectors:
            if not isinstance(raw, (list, tuple)):
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                else:
                    return None, "SENTENCE_TRANSFORMERS_VECTOR_TYPE_ERROR"
            vectors.append(_coerce_dimensions([float(v) for v in raw], dimensions=dimensions))
        return vectors, None
    except Exception as exc:
        return None, f"SENTENCE_TRANSFORMERS_VECTOR_CAST_FAILED:{exc}"


def _write_row(
    fh,
    *,
    chunk_id: str,
    source_path: str,
    text: str,
    vector: list[float],
    status: str,
    provider: str,
    model: str,
    error: str | None,
) -> None:
    fh.write(
        json.dumps(
            {
                "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                "chunk_id": chunk_id,
                "text_sha256": sha256_text(text),
                "status": status,
                "provider": provider,
                "model": model,
                "dimensions": len(vector),
                "embedding": vector,
                "error": error,
                "source_path": source_path,
                "chunk_text_preview": text[:200],
            },
            sort_keys=True,
        )
        + "\n"
    )


def run_worker(
    *,
    input_path: str,
    output_path: str,
    receipt_path: str,
    dimensions: int,
    model_url: str,
    model_id: str,
    model_key: str,
    local_model: str = DEFAULT_LOCAL_MODEL,
    local_device: str = DEFAULT_LOCAL_DEVICE,
    local_batch_size: int = LOCAL_BATCH_SIZE,
) -> dict[str, Any]:
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    receipt_path = os.fspath(receipt_path)

    seen = 0
    written = 0
    failed = 0
    model_used = False

    use_local_model = bool(local_model)
    local_sentence_transformer = None
    local_model_error: str | None = None
    if use_local_model:
        resolved_device = _resolve_device(local_device)
        local_sentence_transformer, local_model_error = _load_sentence_transformer(local_model, resolved_device)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Path(input_path).open("r", encoding="utf-8") as inp, out.open("w", encoding="utf-8") as out_fh:
        batch_rows: list[tuple[str, str, str]] = []
        batch_texts: list[str] = []

        def flush_local_batch() -> None:
            nonlocal written, model_used
            if not batch_rows:
                return

            if local_sentence_transformer is None:
                effective_error = local_model_error or "LOCAL_MODEL_MISSING_OR_FAILED"
                for chunk_id, source_path, text in batch_rows:
                    vector = deterministic_vector(text, dimensions=dimensions)
                    _write_row(
                        out_fh,
                        chunk_id=chunk_id,
                        source_path=source_path,
                        text=text,
                        vector=vector,
                        status="EMBEDDED",
                        provider="deterministic",
                        model="sha256-placeholder",
                        error=effective_error,
                    )
                    written += 1
                batch_rows.clear()
                batch_texts.clear()
                return

            vectors, err = _encode_sentence_transformer_batch(
                local_sentence_transformer,
                batch_texts,
                dimensions=dimensions,
                batch_size=local_batch_size,
            )
            if vectors is None:
                effective_error = err or "LOCAL_MODEL_MISSING_OR_FAILED"
                for chunk_id, source_path, text in batch_rows:
                    vector = deterministic_vector(text, dimensions=dimensions)
                    _write_row(
                        out_fh,
                        chunk_id=chunk_id,
                        source_path=source_path,
                        text=text,
                        vector=vector,
                        status="EMBEDDED",
                        provider="deterministic",
                        model="sha256-placeholder",
                        error=effective_error,
                    )
                    written += 1
            else:
                for (chunk_id, source_path, text), vector in zip(batch_rows, vectors):
                    _write_row(
                        out_fh,
                        chunk_id=chunk_id,
                        source_path=source_path,
                        text=text,
                        vector=vector,
                        status="EMBEDDED",
                        provider="sentence_transformers",
                        model=local_model,
                        error=None,
                    )
                    written += 1
                    model_used = True

            batch_rows.clear()
            batch_texts.clear()

        for line in inp:
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except Exception:
                failed += 1
                continue

            text = _row_text(row)
            chunk_id = _row_chunk_id(row)
            source_path = str(row.get("source_path", ""))

            if not text.strip():
                _write_row(
                    out_fh,
                    chunk_id=chunk_id,
                    source_path=source_path,
                    text=text,
                    vector=[],
                    status="BLOCKED",
                    provider="blocked",
                    model="",
                    error="SKIPPED_NO_TEXT",
                )
                written += 1
                continue

            if use_local_model:
                batch_rows.append((chunk_id, source_path, text))
                batch_texts.append(text)
                if len(batch_rows) >= local_batch_size:
                    flush_local_batch()
                continue

            embedding = model_embedding(text, model_url=model_url, model_id=model_id, model_key=model_key, dimensions=dimensions)
            if embedding is not None:
                model_used = True
                _write_row(
                    out_fh,
                    chunk_id=chunk_id,
                    source_path=source_path,
                    text=text,
                    vector=embedding,
                    status="EMBEDDED",
                    provider="model",
                    model=model_id,
                    error=None,
                )
            else:
                vector = deterministic_vector(text, dimensions=dimensions)
                _write_row(
                    out_fh,
                    chunk_id=chunk_id,
                    source_path=source_path,
                    text=text,
                    vector=vector,
                    status="EMBEDDED",
                    provider="deterministic",
                    model="sha256-placeholder",
                    error="MODEL_MISSING_OR_FAILED",
                )
            written += 1

        if use_local_model:
            flush_local_batch()

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    output = {
        "schema": "lucidota.runpod_embed_worker.receipt.v1",
        "generated_at": now_z(),
        "status": "PASS" if written else "NO_OUTPUT",
        "input_path": input_path,
        "output_path": output_path,
        "receipt_path": receipt_path,
        "seen": seen,
        "written": written,
        "failed": failed,
        "model_requested": bool(model_url and model_id),
        "model_used": model_used,
        "dimensions": dimensions,
        "output_sha256": digest,
        "output_lines": written,
    }

    r = Path(receipt_path)
    r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    input_path = os.environ.get("RUNPOD_INPUT_JSONL", DEFAULT_INPUT_JSONL)
    output_path = os.environ.get("RUNPOD_OUTPUT_JSONL", DEFAULT_OUTPUT_JSONL)
    receipt_path = os.environ.get("RUNPOD_RECEIPT_JSON", DEFAULT_RECEIPT_JSON)
    model_url = os.environ.get("RUNPOD_EMBED_MODEL_URL", DEFAULT_MODEL_URL)
    model_id = os.environ.get("RUNPOD_EMBED_MODEL_ID", DEFAULT_MODEL_ID)
    model_key = os.environ.get("RUNPOD_EMBED_MODEL_KEY", DEFAULT_MODEL_KEY)
    local_model = os.environ.get("RUNPOD_EMBED_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
    local_device = os.environ.get("RUNPOD_EMBED_DEVICE", DEFAULT_LOCAL_DEVICE)
    dimensions = int(os.environ.get("RUNPOD_EMBED_DIMENSIONS", str(DEFAULT_DIMENSIONS)))

    result = run_worker(
        input_path=input_path,
        output_path=output_path,
        receipt_path=receipt_path,
        dimensions=dimensions,
        model_url=model_url,
        model_id=model_id,
        model_key=model_key,
        local_model=local_model,
        local_device=local_device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''



def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def deterministic_vector(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    if dimensions <= 0:
        return []
    vals: list[float] = []
    for i in range(dimensions):
        h = hashlib.sha256(f"{i}:{text}".encode("utf-8", errors="replace")).digest()
        v = int.from_bytes(h[:4], "big") / float(0xFFFFFFFF)
        vals.append((v * 2.0) - 1.0)
    mag = sum(v * v for v in vals) ** 0.5
    if mag == 0:
        return [0.0 for _ in range(dimensions)]
    return [round(v / mag, 8) for v in vals]


def _coerce_dimensions(values: list[float], dimensions: int) -> list[float]:
    if dimensions <= 0:
        return [round(float(v), 8) for v in values]
    trimmed = [float(v) for v in values[:dimensions]]
    if len(trimmed) < dimensions:
        trimmed.extend([0.0] * (dimensions - len(trimmed)))
    return [round(v, 8) for v in trimmed]


def _resolve_device(raw_device: str) -> str:
    if raw_device and raw_device.lower() != "auto":
        return raw_device
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _model_request(
    *,
    model_url: str,
    model_id: str,
    model_key: str,
    text: str,
    request_post: Callable[..., Any],
    timeout: int = REQUEST_TIMEOUT,
) -> list[float] | None:
    if not (model_url and model_id):
        return None

    headers = {"content-type": "application/json", "user-agent": "runpod-embed-worker/1"}
    if model_key:
        headers["Authorization"] = f"Bearer {model_key}"

    url = model_url.rstrip("/") + "/v1/embeddings"
    payload = {"model": model_id, "input": [text]}
    response = request_post(url, headers=headers, json=payload, timeout=timeout)
    status = int(getattr(response, "status_code", 0) or 0)
    if status != 200:
        return None

    body = getattr(response, "json", lambda: {})()
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    embedding = first.get("embedding") if isinstance(first, dict) else None
    if not isinstance(embedding, list):
        return None
    try:
        return [float(v) for v in embedding]
    except Exception:
        return None


def _load_sentence_transformer(local_model: str, local_device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(local_model, device=local_device)


def _encode_sentence_transformer_batch(
    model: Any,
    texts: list[str],
    *,
    dimensions: int,
    batch_size: int = DEFAULT_LOCAL_BATCH_SIZE,
) -> list[list[float]]:
    raw_vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    vectors: list[list[float]] = []
    for raw in raw_vectors:
        if not isinstance(raw, (list, tuple)):
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            else:
                raise TypeError("SENTENCE_TRANSFORMERS_VECTOR_TYPE_ERROR")
        vectors.append(_coerce_dimensions([float(v) for v in raw], dimensions=dimensions))
    return vectors


def _parse_row(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("chunk_id") or row.get("chunk_ref") or row.get("id") or ""),
        str(row.get("text") or row.get("content") or ""),
        str(row.get("source_path") or ""),
    )


def run_local_worker(
    input_jsonl: Path,
    output_jsonl: Path,
    receipt_path: Path,
    *,
    dimensions: int = DEFAULT_DIMENSIONS,
    model_url: str = "",
    model_id: str = "",
    model_key: str = "",
    local_model: str = "",
    local_device: str = DEFAULT_LOCAL_DEVICE,
    request_post: Callable[..., Any] = None,
    request_timeout: int = REQUEST_TIMEOUT,
    sentence_transformer_ctor: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if request_post is None:
        if requests is None:  # pragma: no cover - import-time guard
            raise RuntimeError("requests_not_available")
        request_post = requests.post

    if sentence_transformer_ctor is None:
        sentence_transformer_ctor = _load_sentence_transformer

    resolved_device = _resolve_device(local_device)
    local_model_requested = bool(local_model)
    local_sentence_transformer: Any | None = None
    local_model_error: str | None = None
    if local_model_requested:
        try:
            local_sentence_transformer = sentence_transformer_ctor(local_model, resolved_device)
        except Exception as exc:
            local_model_error = "LOCAL_MODEL_MISSING_OR_FAILED"

    seen = 0
    embedded = 0
    blocked = 0
    failed = 0
    model_requested = bool(model_url and model_id)
    model_used = False

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8")

    with input_jsonl.open("r", encoding="utf-8") as inp, output_jsonl.open("w", encoding="utf-8") as out:
        batch_rows: list[tuple[str, str, str]] = []
        batch_texts: list[str] = []
        output_lines = 0

        def flush_local_batch() -> None:
            nonlocal embedded, output_lines, model_used
            if not batch_rows:
                return

            if local_sentence_transformer is None:
                error = local_model_error or "LOCAL_MODEL_MISSING_OR_FAILED"
                for chunk_id, source_path, text in batch_rows:
                    vector = deterministic_vector(text, dimensions=dimensions)
                    out.write(
                        json.dumps(
                            {
                                "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                                "chunk_id": chunk_id,
                                "source_path": source_path,
                                "text_sha256": sha256_text(text),
                                "status": "EMBEDDED",
                                "provider": "deterministic",
                                "model": "sha256-placeholder",
                                "dimensions": len(vector),
                                "embedding": vector,
                                "error": error,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    embedded += 1
                    output_lines += 1
                batch_rows.clear()
                batch_texts.clear()
                return

            try:
                vectors = _encode_sentence_transformer_batch(
                    local_sentence_transformer,
                    batch_texts,
                    dimensions=dimensions,
                    batch_size=DEFAULT_LOCAL_BATCH_SIZE,
                )
            except Exception as exc:
                error = f"SENTENCE_TRANSFORMERS_ENCODE_FAILED:{exc}"
                for chunk_id, source_path, text in batch_rows:
                    vector = deterministic_vector(text, dimensions=dimensions)
                    out.write(
                        json.dumps(
                            {
                                "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                                "chunk_id": chunk_id,
                                "source_path": source_path,
                                "text_sha256": sha256_text(text),
                                "status": "EMBEDDED",
                                "provider": "deterministic",
                                "model": "sha256-placeholder",
                                "dimensions": len(vector),
                                "embedding": vector,
                                "error": error,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    embedded += 1
                    output_lines += 1
            else:
                for (chunk_id, source_path, text), vector in zip(batch_rows, vectors):
                    out.write(
                        json.dumps(
                            {
                                "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                                "chunk_id": chunk_id,
                                "source_path": source_path,
                                "text_sha256": sha256_text(text),
                                "status": "EMBEDDED",
                                "provider": "sentence_transformers",
                                "model": local_model,
                                "dimensions": len(vector),
                                "embedding": vector,
                                "error": None,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    embedded += 1
                    output_lines += 1
                    model_used = True

            batch_rows.clear()
            batch_texts.clear()

        for line in inp:
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except Exception:
                failed += 1
                continue

            chunk_id, text, source_path = _parse_row(row if isinstance(row, dict) else {})
            if not text.strip():
                out.write(
                    json.dumps(
                        {
                            "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                            "chunk_id": chunk_id,
                            "source_path": source_path,
                            "text_sha256": sha256_text(text),
                            "status": "BLOCKED",
                            "provider": "blocked",
                            "model": "",
                            "dimensions": 0,
                            "embedding": [],
                            "error": "SKIPPED_NO_TEXT",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                blocked += 1
                output_lines += 1
                continue

            if local_model_requested:
                batch_rows.append((chunk_id, source_path, text))
                batch_texts.append(text)
                if len(batch_rows) >= DEFAULT_LOCAL_BATCH_SIZE:
                    flush_local_batch()
                continue

            vec = None
            if model_requested:
                vec = _model_request(
                    model_url=model_url,
                    model_id=model_id,
                    model_key=model_key,
                    text=text,
                    request_post=request_post,
                    timeout=request_timeout,
                )
                if vec is not None:
                    model_used = True
                    vec = _coerce_dimensions(vec, dimensions=dimensions)
            if vec is None:
                vec = deterministic_vector(text, dimensions=dimensions)
                provider = "deterministic"
                model = "sha256-placeholder"
                error = "MODEL_MISSING_OR_FAILED" if model_requested else None
            else:
                provider = "model"
                model = model_id
                error = None

            out.write(
                json.dumps(
                    {
                        "schema": "lucidota.runpod_embed_worker.embedding_row.v1",
                        "chunk_id": chunk_id,
                        "source_path": source_path,
                        "text_sha256": sha256_text(text),
                        "status": "EMBEDDED",
                        "provider": provider,
                        "model": model,
                        "dimensions": len(vec),
                        "embedding": vec,
                        "error": error,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            embedded += 1
            output_lines += 1

        if local_model_requested:
            flush_local_batch()

    output_sha256 = hashlib.sha256(output_jsonl.read_bytes()).hexdigest() if output_jsonl.exists() else ""
    receipt = {
        "schema": SCHEMA_LOCAL_RECEIPT,
        "generated_at": now_z(),
        "status": "PASS" if output_lines else "NO_INPUT",
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "receipt_path": str(receipt_path),
        "dimensions": dimensions,
        "seen": seen,
        "embedded": embedded,
        "blocked": blocked,
        "failed": failed,
        "model_requested": model_requested,
        "model_used": model_used,
        "output_lines": output_lines,
        "output_sha256": output_sha256,
    }

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _quoted_path(path: str) -> str:
    safe = "/"
    return quote(path.lstrip("/"), safe=safe)


def build_jupyter_contents_url(base_url: str, remote_path: str) -> str:
    return f"{base_url.rstrip('/')}/api/contents/{_quoted_path(remote_path)}"


def build_remote_worker_script() -> str:
    return REMOTE_WORKER_SOURCE


def build_plan(
    jupyter_url: str,
    remote_path: str,
    *,
    input_jsonl: str = DEFAULT_INPUT,
    output_jsonl: str = DEFAULT_OUTPUT,
    receipt_path: str = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    command = (
        "RUNPOD_INPUT_JSONL={inp} RUNPOD_OUTPUT_JSONL={out} RUNPOD_RECEIPT_JSON={receipt} "
        "python3 {worker}"
    ).format(
        worker=shlex.quote(remote_path),
        inp=shlex.quote(input_jsonl),
        out=shlex.quote(output_jsonl),
        receipt=shlex.quote(receipt_path),
    )
    return {
        "schema": SCHEMA_RUN,
        "generated_at": now_z(),
        "jupyter_url": jupyter_url,
        "remote_worker_path": remote_path,
        "remote_command": command,
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "receipt_json": receipt_path,
        "model_env": [
            DEFAULT_MODEL_API_URL_ENV,
            DEFAULT_MODEL_ID_ENV,
            DEFAULT_MODEL_KEY_ENV,
            DEFAULT_LOCAL_MODEL_ENV,
            DEFAULT_LOCAL_DEVICE_ENV,
        ],
        "dolphin_touched": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
    }


def stage_worker(
    *,
    jupyter_url: str,
    jupyter_token: str | None,
    remote_path: str,
    request_put: Callable[..., Any] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    if request_put is None:
        if requests is None:  # pragma: no cover - import-time guard
            raise RuntimeError("requests_not_available")
        request_put = requests.put

    script_text = build_remote_worker_script()
    url = build_jupyter_contents_url(jupyter_url, remote_path)
    headers = {"Content-Type": "application/json"}
    if jupyter_token:
        headers["Authorization"] = f"token {jupyter_token}"

    payload = {
        "type": "file",
        "format": "text",
        "name": Path(remote_path).name,
        "path": str(Path(remote_path).parent).lstrip("/"),
        "content": script_text,
    }
    response = request_put(url, headers=headers, json=payload, timeout=timeout)
    status_code = int(getattr(response, "status_code", 0) or 0)
    return {
        "schema": SCHEMA_STAGE,
        "generated_at": now_z(),
        "action": "stage_remote_worker",
        "status": "PASS" if 200 <= status_code < 300 else "FAIL",
        "jupyter_url": jupyter_url,
        "remote_path": remote_path,
        "request_url": url,
        "request_status": status_code,
        "request_text": str(getattr(response, "text", ""))[:200],
        "uploaded_bytes": len(script_text.encode("utf-8")),
        "remote_receipt_root": DEFAULT_REMOTE_ROOT,
        "dolphin_touched": False,
        "db_writes_performed": False,
        "graph_writes_performed": False,
    }


def load_env_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.jupyter_url:
        args.jupyter_url = os.environ.get("JUPYTER_URL") or os.environ.get("RUNPOD_JUPYTER_URL")
    if not args.jupyter_token:
        args.jupyter_token = os.environ.get("JUPYTER_TOKEN") or os.environ.get("RUNPOD_JUPYTER_TOKEN")
    if not args.model_url:
        args.model_url = os.environ.get(DEFAULT_MODEL_API_URL_ENV)
    if not args.model_id:
        args.model_id = os.environ.get(DEFAULT_MODEL_ID_ENV)
    if not args.model_key:
        args.model_key = os.environ.get(DEFAULT_MODEL_KEY_ENV)
    if not args.local_model:
        args.local_model = os.environ.get(DEFAULT_LOCAL_MODEL_ENV, "")
    if not args.local_device:
        args.local_device = os.environ.get(DEFAULT_LOCAL_DEVICE_ENV, DEFAULT_LOCAL_DEVICE)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="runpod-stage-embed-worker",
        description="Stage/run lightweight RunPod embedding worker on Jupyter and optionally test it locally.",
    )
    ap.add_argument("--action", choices=["stage", "local-run", "build-script"], default="stage")
    ap.add_argument("--jupyter-url", help="RunPod/Jupyter base URL (or env JUPYTER_URL)")
    ap.add_argument("--jupyter-token", help="Jupyter token (or env JUPYTER_TOKEN)")
    ap.add_argument("--remote-script-path", default=DEFAULT_REMOTE_WORKER_PATH, help="Remote path for worker file")
    ap.add_argument("--input-jsonl", default=DEFAULT_INPUT, type=Path, help="Input JSONL path")
    ap.add_argument("--output-jsonl", default=DEFAULT_OUTPUT, type=Path, help="Output JSONL path")
    ap.add_argument("--receipt", default=DEFAULT_RECEIPT, type=Path, help="Output receipt path")
    ap.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    ap.add_argument("--model-url", help="Optional remote model URL (enables model mode only when set)")
    ap.add_argument("--model-id", help="Optional model id")
    ap.add_argument("--model-key", help="Optional model key")
    ap.add_argument("--local-model", default="", help="Optional local sentence-transformers model")
    ap.add_argument("--local-device", default="", help="sentence-transformers device: auto,cpu,cuda")
    ap.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT)
    ap.add_argument("--json", action="store_true", help="Print machine-readable receipt")
    ap.add_argument("--plan", action="store_true", help="Print run plan and dry-run only")
    ap.add_argument("--save-script", type=Path, help="Write worker script to local path instead of only printing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_args(args)

    if args.action == "build-script":
        script = build_remote_worker_script()
        if args.save_script:
            args.save_script.parent.mkdir(parents=True, exist_ok=True)
            args.save_script.write_text(script, encoding="utf-8")
            path = str(args.save_script)
        else:
            path = "stdout"
            print(script)
        receipt = {
            "schema": SCHEMA_RUN,
            "generated_at": now_z(),
            "action": "build_script",
            "remote_path": args.remote_script_path,
            "bytes": len(script.encode("utf-8")),
            "saved": path,
            "dolphin_touched": False,
            "db_writes_performed": False,
            "graph_writes_performed": False,
        }
        if args.json:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    if args.action == "local-run":
        receipt = run_local_worker(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
            receipt_path=args.receipt,
            dimensions=args.dimensions,
            model_url=args.model_url or "",
            model_id=args.model_id or "",
            model_key=args.model_key or "",
            local_model=args.local_model or "",
            local_device=args.local_device or DEFAULT_LOCAL_DEVICE,
        )
        if args.json:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] in {"PASS", "NO_INPUT"} else 1

    if not args.jupyter_url:
        plan = build_plan(args.jupyter_url or "", args.remote_script_path)
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print("missing jupyter URL; pass --jupyter-url or set JUPYTER_URL")
        return 1

    plan = build_plan(args.jupyter_url, args.remote_script_path)
    if args.plan:
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print(json.dumps(plan, sort_keys=True))
        return 0

    result = stage_worker(
        jupyter_url=args.jupyter_url,
        jupyter_token=args.jupyter_token,
        remote_path=args.remote_script_path,
        timeout=args.timeout,
    )

    if args.save_script:
        Path(args.save_script).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_script).write_text(build_remote_worker_script(), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": result["status"], "remote_path": result["remote_path"], "request_status": result["request_status"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
