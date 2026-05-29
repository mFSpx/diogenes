#!/usr/bin/env python3
"""Embedding provider abstraction for corpus ingestion.

Groq is preferred when configured with an embedding-capable model.
When Groq embedding is not configured, the provider can fall back to a
deterministic local vector for degraded operation, but never returns an
empty vector as success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
MODEL_AUDIT_DIR = ROOT / "05_OUTPUTS" / "model_invocation_audits"
DEFAULT_LOCAL_MODEL = "lucidota.deterministic-hash-384"
DEFAULT_DIMENSIONS = 384


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def groq_embedding_config() -> dict[str, Any]:
    api_key = _env("GROQ_API_KEY")
    model = _env("LUCIDOTA_GROQ_EMBEDDING_MODEL", "GROQ_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL")
    base_url = _env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    return {
        "configured": bool(api_key and model),
        "api_key_present": bool(api_key),
        "model": model or "",
        "base_url": base_url,
        "reason": None if api_key and model else "GROQ_EMBEDDING_NOT_CONFIGURED",
    }


def local_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    if dimensions <= 0:
        return []
    vec = [0.0] * dimensions
    tokens = [t for t in re.split(r"\W+", text.lower()) if t]
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="replace")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        weight = 1.0 + (digest[4] / 255.0)
        vec[idx] += weight
    norm = sum(v * v for v in vec) ** 0.5
    if norm:
        vec = [round(v / norm, 8) for v in vec]
    return vec


def _groq_embed(text: str, *, model: str, api_key: str, base_url: str) -> list[float]:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"openai_client_unavailable:{exc}") from exc
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=text)
    data = getattr(response, "data", None) or []
    if not data:
        raise RuntimeError("groq_embeddings_empty_response")
    emb = list(getattr(data[0], "embedding", []) or [])
    if not emb:
        raise RuntimeError("groq_embeddings_empty_vector")
    return [float(x) for x in emb]


def probe() -> dict[str, Any]:
    groq = groq_embedding_config()
    local = {
        "configured": True,
        "model": DEFAULT_LOCAL_MODEL,
        "dimensions": DEFAULT_DIMENSIONS,
        "provider": "local",
    }
    if groq["configured"]:
        verdict = "PASS"
        provider = "groq"
        blocked_gap = None
    else:
        verdict = "BLOCKED"
        provider = "local"
        blocked_gap = "GROQ_EMBEDDING_NOT_CONFIGURED"
    return {
        "schema": "lucidota.embedding_provider.probe.v1",
        "generated_at_utc": now(),
        "verdict": verdict,
        "provider": provider,
        "groq": groq,
        "local": local,
        "blocked_gap": blocked_gap,
    }


@dataclass
class EmbedResult:
    row: dict[str, Any]
    groq_receipt: dict[str, Any] | None = None


def embed_text(text: str, *, source_path: str = "", chunk_id: str = "", prefer_groq: bool = True) -> EmbedResult:
    text = text or ""
    text_sha = sha256_text(text)
    groq = groq_embedding_config()
    created = now()
    if not text.strip():
        row = {
            "schema": "lucidota.embedding.row.v1",
            "source_path": source_path,
            "chunk_id": chunk_id,
            "provider": "blocked",
            "model": "",
            "dimensions": 0,
            "vector": [],
            "text_sha256": text_sha,
            "created_at_utc": created,
            "receipt_ref": None,
            "status": "BLOCKED",
            "error": "SKIPPED_NO_TEXT",
        }
        return EmbedResult(row=row, groq_receipt=None)

    if prefer_groq and groq["configured"]:
        try:
            vector = _groq_embed(text, model=groq["model"], api_key=_env("GROQ_API_KEY"), base_url=groq["base_url"])
            receipt = {
                "schema": "lucidota.model_invocation.groq_embedding.v1",
                "provider": "groq",
                "model": groq["model"],
                "input_count": 1,
                "success_count": 1,
                "failed_count": 0,
                "dimensions": len(vector),
                "canonical_graph_writes": False,
                "external_effects": False,
                "verdict": "PASS",
                "generated_at_utc": created,
                "source_path": source_path,
                "chunk_id": chunk_id,
            }
            receipt_path = MODEL_AUDIT_DIR / f"groq_embedding_{stamp()}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            row = {
                "schema": "lucidota.embedding.row.v1",
                "source_path": source_path,
                "chunk_id": chunk_id,
                "provider": "groq",
                "model": groq["model"],
                "dimensions": len(vector),
                "vector": vector,
                "text_sha256": text_sha,
                "created_at_utc": created,
                "receipt_ref": str(receipt_path.relative_to(ROOT)),
                "status": "EMBEDDED",
                "error": None,
            }
            return EmbedResult(row=row, groq_receipt=receipt)
        except Exception as exc:
            if not prefer_groq:
                raise
            groq_error = repr(exc)
    else:
        groq_error = "GROQ_EMBEDDING_NOT_CONFIGURED"

    local_vector = local_embedding(text)
    if not local_vector:
        row = {
            "schema": "lucidota.embedding.row.v1",
            "source_path": source_path,
            "chunk_id": chunk_id,
            "provider": "blocked",
            "model": "",
            "dimensions": 0,
            "vector": [],
            "text_sha256": text_sha,
            "created_at_utc": created,
            "receipt_ref": None,
            "status": "FAILED",
            "error": groq_error if groq["configured"] else "LOCAL_EMBEDDING_FAILED",
        }
        return EmbedResult(row=row, groq_receipt=None)

    row = {
        "schema": "lucidota.embedding.row.v1",
        "source_path": source_path,
        "chunk_id": chunk_id,
        "provider": "local",
        "model": DEFAULT_LOCAL_MODEL,
        "dimensions": len(local_vector),
        "vector": local_vector,
        "text_sha256": text_sha,
        "created_at_utc": created,
        "receipt_ref": None,
        "status": "EMBEDDED",
        "error": groq_error if groq["configured"] else "GROQ_EMBEDDING_NOT_CONFIGURED",
    }
    return EmbedResult(row=row, groq_receipt=None)


def embed_file(chunks_jsonl: Path, out_jsonl: Path, *, prefer_groq: bool = True) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    groq_receipts: list[str] = []
    stats = {"seen": 0, "embedded_groq": 0, "embedded_local": 0, "blocked": 0, "failed": 0, "skipped": 0}
    for line in chunks_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["seen"] += 1
        chunk = json.loads(line)
        text = str(chunk.get("text") or chunk.get("content") or "")
        result = embed_text(text, source_path=str(chunk.get("source_path") or ""), chunk_id=str(chunk.get("chunk_id") or chunk.get("id") or ""), prefer_groq=prefer_groq)
        rows_out.append(result.row)
        status = result.row.get("status")
        if status == "EMBEDDED" and row.get("provider") == "groq":
            stats["embedded_groq"] += 1
        elif status == "EMBEDDED" and row.get("provider") == "local":
            stats["embedded_local"] += 1
        elif status == "BLOCKED":
            stats["blocked"] += 1
        elif status == "FAILED":
            stats["failed"] += 1
        else:
            stats["skipped"] += 1
        if result.groq_receipt:
            groq_receipts.append(result.row.get("receipt_ref") or "")
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return {
        "schema": "lucidota.embedding_provider.embed_file.v1",
        "generated_at_utc": now(),
        "input_path": str(chunks_jsonl),
        "output_path": str(out_jsonl),
        "rows_written": len(rows_out),
        "stats": stats,
        "groq_receipts": groq_receipts,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Embedding provider abstraction for LUCIDOTA corpus ingest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("embed")
    p.add_argument("--text", required=True)
    p.add_argument("--source-path", default="")
    p.add_argument("--chunk-id", default="")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-groq", action="store_true", help="force local degraded provider")
    p = sub.add_parser("embed-file")
    p.add_argument("chunks_jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--no-groq", action="store_true")
    p.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "probe":
        report = probe()
    elif args.cmd == "embed":
        report = embed_text(args.text, source_path=args.source_path, chunk_id=args.chunk_id, prefer_groq=not args.no_groq)
        report = {
            "schema": "lucidota.embedding_provider.embed_text.v1",
            "generated_at_utc": now(),
            "row": report.row,
            "groq_receipt": report.groq_receipt,
        }
    elif args.cmd == "embed-file":
        report = embed_file(Path(args.chunks_jsonl), Path(args.out), prefer_groq=not args.no_groq)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command: {args.cmd}")
    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
