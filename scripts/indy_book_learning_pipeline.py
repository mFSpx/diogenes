#!/usr/bin/env python3
"""INDY_READs book learning pipeline: fetch/read -> <=500-token chunks -> GO packets -> LoRA job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ALGOS.indy_learning_vector import build_learning_vector, build_lora_rows, chunk_text_tokens  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "indy_reads"
RUNTIME = ROOT / "04_RUNTIME" / "lora_cartridges"
API_ENV_NAMES = ["ANNAS_ARCHIVE_API_KEY", "ANNA_ARCHIVE_API_KEY"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.lower()).strip("-")[:80] or "book"


def api_key() -> tuple[str | None, str | None]:
    for name in API_ENV_NAMES:
        v = os.environ.get(name)
        if v:
            return v, name
    return None, None


def read_source_file(path: str) -> tuple[str, dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    data = p.read_bytes()
    return data.decode("utf-8", errors="replace"), {"source_path": rel(p), "source_sha256": sha_bytes(data), "source_bytes": len(data)}


def fetch_url(url: str, *, annas: bool = False, timeout: float = 60.0) -> tuple[str, dict[str, Any]]:
    headers = {"user-agent": "lucidota-indy-reads/1.0"}
    key, key_env = api_key()
    if annas and key:
        headers["authorization"] = "Bearer " + key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        final_url = resp.geturl()
    return data.decode("utf-8", errors="replace"), {"source_url": final_url, "source_sha256": sha_bytes(data), "source_bytes": len(data), "api_key_env_used": key_env if annas else None}


def annas_query_packet(query: str) -> dict[str, Any]:
    key, key_env = api_key()
    qid = "annas_query:" + hashlib.sha256(query.encode()).hexdigest()[:16]
    return {
        "schema": "lucidota.indy_reads.annas_query_packet.v1",
        "generated_at": now(),
        "status": "PASS_QUERY_PACKET",
        "query_id": qid,
        "query": query,
        "api_key_env_names": API_ENV_NAMES,
        "api_key_env_present": bool(key),
        "api_key_env_used": key_env,
        "download_performed": False,
        "purpose": "research_private_study_education",
        "jzloads": [
            {"kind": "OBJECT", "id": qid, "type": "BOOK_SEARCH_QUERY", "provider": "annas_archive", "query": query},
            {"kind": "EVENT", "id": qid + ":requested", "type": "BOOK_SEARCH_PACKET_EMITTED", "source": qid},
        ],
        "canonical_graph_writes_performed": False,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def compile_lora_manifest(*, title: str, source_ref: dict[str, Any], rows: list[dict[str, str]], base: Path) -> dict[str, Any]:
    adapter_id = "indy_reads_book_" + hashlib.sha256(json.dumps(source_ref, sort_keys=True).encode()).hexdigest()[:16]
    root = RUNTIME / adapter_id
    train = root / "train.jsonl"
    valid = root / "validation.jsonl"
    split = max(1, int(len(rows) * 0.9)) if rows else 0
    write_jsonl(train, rows[:split])
    write_jsonl(valid, rows[split:] or rows[:1])
    manifest = {
        "schema": "lucidota.indy_reads.lora_job_manifest.v1",
        "adapter_id": adapter_id,
        "title": title,
        "source_ref": source_ref,
        "training_status": "queued",
        "train_jsonl": rel(train),
        "validation_jsonl": rel(valid),
        "train_rows": len(rows[:split]),
        "validation_rows": len(rows[split:] or rows[:1]),
        "trainer": "scripts/lucidota_indy_lora_train.py",
        "base_model_env": "LUCIDOTA_LORA_BASE_MODEL",
        "model_calls_performed": False,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": rel(manifest_path)}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_file:
        text, meta = read_source_file(args.source_file)
    elif args.source_url:
        text, meta = fetch_url(args.source_url, annas=False, timeout=args.timeout_sec)
    elif args.annas_url:
        text, meta = fetch_url(args.annas_url, annas=True, timeout=args.timeout_sec)
    else:
        raise SystemExit("provide --source-file, --source-url, --annas-url, or --annas-query")
    title = args.title or Path(meta.get("source_path") or meta.get("source_url") or "book").stem or "book"
    source_ref = {
        "custody_id": "indy_book:" + hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:24],
        "title": title,
        "author": args.author or "",
        "purpose": "research_private_study_education",
        **meta,
    }
    chunks = chunk_text_tokens(text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens, source_ref=source_ref)
    vector = build_learning_vector(chunks=chunks, source_ref=source_ref)
    lora_rows = build_lora_rows(chunks=chunks, source_ref=source_ref)
    out_root = OUT / slug(title) / stamp()
    chunks_path = out_root / "chunks.jsonl"
    vector_path = out_root / "learning_vector.json"
    write_jsonl(chunks_path, chunks)
    write_json(vector_path, vector)
    lora = compile_lora_manifest(title=title, source_ref=source_ref, rows=lora_rows, base=out_root)
    receipt = {
        "schema": "lucidota.indy_reads.book_learning_receipt.v1",
        "generated_at": now(),
        "status": "PASS",
        "title": title,
        "author": args.author or "",
        "purpose": "research_private_study_education",
        "source_ref": source_ref,
        "max_tokens_per_chunk": args.max_tokens,
        "chunks_written": len(chunks),
        "max_observed_chunk_tokens": max(c["token_count"] for c in chunks) if chunks else 0,
        "chunks_jsonl": rel(chunks_path),
        "learning_vector": rel(vector_path),
        "lora_manifest": lora["manifest_path"],
        "lora_training_status": lora["training_status"],
        "ontology_hits": vector["ontology_hits"],
        "jzloads": vector["jzloads"],
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
    }
    receipt_path = out_root / "receipt.json"
    write_json(receipt_path, receipt)
    receipt["receipt_path"] = rel(receipt_path)
    write_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="INDY_READs book learning pipeline: bounded chunks, GO packets, LoRA job receipt.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-file")
    source.add_argument("--source-url")
    source.add_argument("--annas-url")
    source.add_argument("--annas-query")
    ap.add_argument("--title", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap-tokens", type=int, default=25)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens > 500:
        raise SystemExit("max_tokens must be <= 500")
    payload = annas_query_packet(args.annas_query) if args.annas_query else run_pipeline(args)
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    if payload.get("receipt_path"):
        print("REPORT_PATH=" + payload["receipt_path"])
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"annas_query_packet_{stamp()}.json"
        write_json(path, payload)
        print("REPORT_PATH=" + rel(path))
    print("INDY_BOOK_LEARNING=" + payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
