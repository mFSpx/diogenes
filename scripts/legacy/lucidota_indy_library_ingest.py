#!/usr/bin/env python3
"""Ingest Indy_READs books into the live Postgres graph.

- Scans BOOKS for epub/pdf/mobi/txt/md.
- Extracts text locally.
- Chunks to <=500 approximate tokens.
- Writes book/chunk graph items plus BOOK_HAS_CHUNK edges.
- Writes deterministic 384-d vector embeddings into pgvector.
- Stages LoRA JSONL examples for <=1.5B models.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Iterable

import psycopg

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == "legacy" else Path(__file__).resolve().parents[1]
BOOKS = ROOT / "BOOKS"
SCHEMA = ROOT / "06_SCHEMA" / "017_indy_reads_library.sql"
GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
RUNTIME_SCHEMA = ROOT / "06_SCHEMA" / "002_model_runtime.sql"
DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
OPERATOR_UUID = os.environ.get("LUCIDOTA_OPERATOR_UUID", "00000000-0000-4000-8000-000000000414")
EMBED_MODEL = "ckdog1_kernel_hash_quantized_embedding_v1_384"
CARTRIDGE_DIR = ROOT / "04_RUNTIME" / "lora_cartridges"
SUPPORTED = {".epub", ".pdf", ".mobi", ".txt", ".md"}
SKIP_MD = {"README_INDY_READS.md", "OFFICIAL_ONTOLOGY_POINTER.md", "ROOT414_GAME_GRADING_SCHEMA.md"}
GRAPH_APPROVAL_MODE = (
    "approved"
    if os.environ.get("LUCIDOTA_GRAPH_APPROVAL_MODE", "").strip().lower() == "approved"
    and os.environ.get("LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL", "").strip().lower() in {"1", "true", "yes", "on"}
    else "staged"
)

sys.path.insert(0, str(ROOT / "01_REPOS" / "doggystyle"))
from kernel.mini_embeddings import INT16_MAX, hash_quantized_embedding  # type: ignore  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def approx_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(data: bytes) -> str:
    s = data.decode("utf-8", errors="ignore")
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n\n", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return clean_text(s)


def extract_epub(path: Path) -> tuple[str, str]:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in sorted(names):
            try:
                t = strip_html(zf.read(name))
            except (KeyError, RuntimeError, UnicodeError, zipfile.BadZipFile):
                continue
            if len(t) > 80:
                parts.append(t)
    return clean_text("\n\n".join(parts)), "zip_epub_html"


def extract_pdf(path: Path) -> tuple[str, str]:
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not found")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "book.txt"
        subprocess.run(["pdftotext", "-layout", str(path), str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return clean_text(out.read_text(encoding="utf-8", errors="ignore")), "pdftotext"


def extract_mobi(path: Path) -> tuple[str, str]:
    # Prefer Calibre if present; otherwise fall back to printable-string extraction.
    if shutil.which("ebook-convert"):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "book.txt"
            subprocess.run(["ebook-convert", str(path), str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return clean_text(out.read_text(encoding="utf-8", errors="ignore")), "ebook-convert"
    raw = path.read_bytes()
    strings = re.findall(rb"[\x20-\x7E]{20,}", raw)
    text = "\n".join(x.decode("utf-8", errors="ignore") for x in strings)
    return clean_text(text), "mobi_printable_strings_fallback"


def extract_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".epub":
        return extract_epub(path)
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".mobi":
        return extract_mobi(path)
    return clean_text(path.read_text(encoding="utf-8", errors="ignore")), "plain_text"


def structural_segments(text: str) -> list[str]:
    """Split on natural reading/math/code boundaries before token packing."""
    raw_blocks = re.split(r"\n\s*\n", text)
    segments: list[str] = []
    pending: list[str] = []
    in_fence = False
    in_display_math = False
    for block in raw_blocks:
        stripped = block.strip()
        if not stripped:
            continue
        fence_flip = stripped.count("```") % 2 == 1
        math_flip = stripped.count("$$") % 2 == 1
        pending.append(stripped)
        if fence_flip:
            in_fence = not in_fence
        if math_flip:
            in_display_math = not in_display_math
        if not in_fence and not in_display_math:
            segments.append("\n\n".join(pending))
            pending = []
    if pending:
        segments.append("\n\n".join(pending))
    return segments


def split_long_segment(segment: str, max_tokens: int) -> Iterable[str]:
    """Fallback splitter that prefers line/sentence boundaries for oversized blocks."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n", segment) if p.strip()]
    buf: list[str] = []
    buf_tokens = 0
    for part in parts or [segment]:
        toks = approx_tokens(part)
        if len(toks) > max_tokens:
            if buf:
                yield " ".join(buf)
                buf = []
                buf_tokens = 0
            for start in range(0, len(toks), max_tokens):
                yield " ".join(toks[start : start + max_tokens])
            continue
        if buf and buf_tokens + len(toks) > max_tokens:
            yield " ".join(buf)
            buf = []
            buf_tokens = 0
        buf.append(part)
        buf_tokens += len(toks)
    if buf:
        yield " ".join(buf)


def chunks(text: str, max_tokens: int = 500, overlap: int = 40) -> Iterable[tuple[int, str, int]]:
    """Yield <=500-token chunks without slicing through normal math/code blocks."""
    idx = 0
    buf: list[str] = []
    buf_tokens = 0
    recent_tail: list[str] = []
    for segment in structural_segments(text):
        segment_parts = list(split_long_segment(segment, max_tokens))
        for part in segment_parts:
            tok_count = len(approx_tokens(part))
            if tok_count == 0:
                continue
            if buf and buf == recent_tail and buf_tokens + tok_count > max_tokens:
                # Overlap is a preference, not a license to violate the 500-token law.
                buf = []
                buf_tokens = 0
            if buf and buf_tokens + tok_count > max_tokens:
                piece = "\n\n".join(buf).strip()
                yield idx, piece, len(approx_tokens(piece))
                idx += 1
                tail_tokens = approx_tokens(piece)[-overlap:] if overlap > 0 else []
                recent_tail = [" ".join(tail_tokens)] if tail_tokens else []
                buf = [x for x in recent_tail if x]
                buf_tokens = len(tail_tokens)
                if buf_tokens + tok_count > max_tokens:
                    buf = []
                    buf_tokens = 0
            if tok_count > max_tokens:
                for sub in split_long_segment(part, max_tokens):
                    sub_tokens = len(approx_tokens(sub))
                    if sub_tokens:
                        yield idx, sub, sub_tokens
                        idx += 1
                buf = []
                buf_tokens = 0
                recent_tail = []
            else:
                buf.append(part)
                buf_tokens += tok_count
    if buf:
        piece = "\n\n".join(buf).strip()
        if piece:
            yield idx, piece, len(approx_tokens(piece))


def embedding384(text: str) -> list[float]:
    # The same deterministic CKDOG1 kernel embedding path used by clawd routing.
    # Store normalized floats in pgvector, while keeping the model id explicit.
    return [round(float(x) / float(INT16_MAX), 8) for x in hash_quantized_embedding(text)]


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def graph_direct_approved() -> bool:
    return GRAPH_APPROVAL_MODE == "approved"


def title_author_from_name(name: str) -> tuple[str, str]:
    base = re.sub(r"\.[^.]+$", "", name)
    parts = [p.strip() for p in base.split(" -- ")]
    title = parts[0].strip() if parts else base
    author = parts[1].strip() if len(parts) > 1 else ""
    return title[:500], author[:500]


def safe_slug(value: str, limit: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return (slug or "book")[:limit]


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(GO_SCHEMA.read_text())
        cur.execute(SCHEMA.read_text())
    conn.commit()


def ensure_runtime_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(RUNTIME_SCHEMA.read_text())
    conn.commit()


def upsert_graph_item(cur, term: str, label: str, location: str, payload: dict) -> str:
    item_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"lucidota:{term}:{location}:{payload.get('sha256','')}"))
    item_payload = {
        **payload,
        "graph_write_mode": GRAPH_APPROVAL_MODE,
        "approval_required": not graph_direct_approved(),
        "approval_source": "lucidota_indy_library_ingest",
    }
    if graph_direct_approved():
        cur.execute(
            """
            INSERT INTO lucidota_go.graph_item (
              uuid, term, label, status, time_on_graph, location_at_on_graph,
              location_real_at_added, time_approved, location_real_at_approved,
              approval_scope, operator_uuid, payload
            ) VALUES (%s,%s,%s,'approved',now(),%s,%s::jsonb,now(),%s::jsonb,'historical',%s,%s::jsonb)
            ON CONFLICT (uuid) DO UPDATE SET label=EXCLUDED.label, payload=EXCLUDED.payload, updated_at=now()
            RETURNING uuid::text
            """,
            (item_uuid, term, label[:500], location, json.dumps({"path": location}), json.dumps({"path": location}), OPERATOR_UUID, json.dumps(item_payload, sort_keys=True)),
        )
    else:
        cur.execute(
            """
            INSERT INTO lucidota_go.graph_item (
              uuid, term, label, status, time_on_graph, location_at_on_graph,
              location_real_at_added, operator_uuid, payload
            ) VALUES (%s,%s,%s,'staged',now(),%s,%s::jsonb,%s,%s::jsonb)
            ON CONFLICT (uuid) DO UPDATE SET label=EXCLUDED.label, payload=EXCLUDED.payload, updated_at=now()
            RETURNING uuid::text
            """,
            (item_uuid, term, label[:500], location, json.dumps({"path": location}), OPERATOR_UUID, json.dumps(item_payload, sort_keys=True)),
        )
    return cur.fetchone()[0]


def link(cur, source: str, target: str, edge_type: str, detail: dict) -> None:
    edge_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"lucidota-edge:{source}:{edge_type}:{target}"))
    cur.execute(
        """
        INSERT INTO lucidota_go.graph_edge(edge_uuid, source_uuid, target_uuid, edge_type, term, status, current_status, current_unknown, operator_uuid, detail)
        VALUES (%s,%s,%s,%s,'RELATIONSHIP',%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT (edge_uuid) DO NOTHING
        """,
        (
            edge_uuid,
            source,
            target,
            edge_type,
            GRAPH_APPROVAL_MODE,
            "yes" if graph_direct_approved() else "unknown",
            not graph_direct_approved(),
            OPERATOR_UUID,
            json.dumps({**detail, "graph_write_mode": GRAPH_APPROVAL_MODE, "approval_required": not graph_direct_approved()}, sort_keys=True),
        ),
    )


def write_lora_cartridge(
    *,
    book_uuid: str,
    title: str,
    author: str,
    file_hash: str,
    text_hash: str,
    rel: str,
    rows: list[dict[str, object]],
) -> dict[str, str | int | bool]:
    slug = safe_slug(title)
    adapter_id = f"indy_reads__{slug}__{file_hash[:12]}"
    cartridge = CARTRIDGE_DIR / adapter_id
    cartridge.mkdir(parents=True, exist_ok=True)
    train_path = cartridge / "train.jsonl"
    validation_path = cartridge / "validation.jsonl"
    train_count = 0
    validation_count = 0
    with train_path.open("w", encoding="utf-8") as train_f, validation_path.open("w", encoding="utf-8") as val_f:
        for row in rows:
            split = str(row.get("split", "train"))
            record = {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "source": row["source"],
                "tokens": row["tokens"],
                "go_terms": row.get("go_terms", ["BOOK", "EVIDENCE", "CLAIM", "PATTERN"]),
            }
            if split == "validation":
                val_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                validation_count += 1
            else:
                train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_count += 1
    manifest = {
        "schema": "lucidota.indy_reads.lora_cartridge.v1",
        "adapter_id": adapter_id,
        "status": "dataset_ready",
        "training_status": "queued",
        "persona": "INDY_READs",
        "target_model_id": "deepseek-1.5b-indy_reads-reads",
        "book_uuid": book_uuid,
        "title": title,
        "author": author,
        "source_path": rel,
        "source_sha256": file_hash,
        "text_sha256": text_hash,
        "embedding_model": EMBED_MODEL,
        "max_source_tokens": 500,
        "train_jsonl": str(train_path.relative_to(ROOT)),
        "validation_jsonl": str(validation_path.relative_to(ROOT)),
        "train_count": train_count,
        "validation_count": validation_count,
        "trainer_script": "scripts/lucidota_indy_lora_train.py",
        "trainer_base_model_env": "LUCIDOTA_LORA_BASE_MODEL",
        "hot_swap_rule": "cartridge is loaded only by deterministic router after graph retrieval",
        "llm_payload_rule": "feed compact GO primitive/context payloads, not raw chat prompts",
    }
    manifest_path = cartridge / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "adapter_id": adapter_id,
        "local_path": str(cartridge.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "train_count": train_count,
        "validation_count": validation_count,
        "dataset_ready": True,
    }


def register_lora_cartridge(cartridge: dict[str, str | int | bool], book_uuid: str, title: str, rel: str) -> None:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_runtime_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_runtime.adapter_cartridge(
                  adapter_id, target_model_id, root_anchor, source_manifest, license_status,
                  local_path, expected_vram_mb, activation_status, notes
                ) VALUES (%s,'deepseek-1.5b-indy_reads-reads',%s,%s::jsonb,'lawful_private',%s,128,'available',%s)
                ON CONFLICT(adapter_id) DO UPDATE SET
                  source_manifest=EXCLUDED.source_manifest,
                  local_path=EXCLUDED.local_path,
                  activation_status=EXCLUDED.activation_status,
                  notes=EXCLUDED.notes,
                  updated_at=now()
                """,
                (
                    cartridge["adapter_id"],
                    f"BOOK:{book_uuid}",
                    json.dumps({"book_uuid": book_uuid, "title": title, "source_path": rel, **cartridge}, sort_keys=True),
                    cartridge["local_path"],
                    "INDY_READs per-book LoRA cartridge dataset. Training may run when a local HF base model is configured.",
                ),
            )
        conn.commit()


def ingest_book(conn: psycopg.Connection, path: Path, max_tokens: int, write_lora: Path | None) -> dict:
    rel = str(path.relative_to(ROOT))
    data = path.read_bytes()
    file_hash = sha256_bytes(data)
    title, author = title_author_from_name(path.name)
    result = {"path": rel, "ok": False, "chunks": 0, "embeddings": 0, "lora_examples": 0, "adapter_id": "", "cartridge": "", "error": ""}
    try:
        text, method = extract_text(path)
        if len(approx_tokens(text)) == 0:
            raise RuntimeError("no extractable text")
        text_hash = sha256_text(text)
    except Exception as e:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_indy.book_source(path,file_name,file_ext,sha256,size_bytes,status,title,author,extraction_error)
                VALUES (%s,%s,%s,%s,%s,'error',%s,%s,%s)
                ON CONFLICT(path) DO UPDATE SET status='error', extraction_error=EXCLUDED.extraction_error, updated_at=now()
                """,
                (rel, path.name, path.suffix.lower(), file_hash, path.stat().st_size, title, author, str(e)[:1000]),
            )
        conn.commit()
        result["error"] = str(e)
        return result

    with conn.cursor() as cur:
        book_graph_uuid = upsert_graph_item(cur, "BOOK", title, rel, {"sha256": file_hash, "author": author, "source": "Indy_READs library"})
        cur.execute(
            """
            INSERT INTO lucidota_indy.book_source(graph_item_uuid,path,file_name,file_ext,sha256,size_bytes,status,title,author,extraction_method,text_sha256,token_count,payload)
            VALUES (%s,%s,%s,%s,%s,%s,'extracted',%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT(path) DO UPDATE SET graph_item_uuid=EXCLUDED.graph_item_uuid, sha256=EXCLUDED.sha256, size_bytes=EXCLUDED.size_bytes,
              status='extracted', title=EXCLUDED.title, author=EXCLUDED.author, extraction_method=EXCLUDED.extraction_method,
              extraction_error='', text_sha256=EXCLUDED.text_sha256, token_count=EXCLUDED.token_count, payload=EXCLUDED.payload, updated_at=now()
            RETURNING book_uuid::text
            """,
            (book_graph_uuid, rel, path.name, path.suffix.lower(), file_hash, path.stat().st_size, title, author, method, text_hash, len(approx_tokens(text)), json.dumps({"max_chunk_tokens": max_tokens})),
        )
        book_uuid = cur.fetchone()[0]
        cur.execute("DELETE FROM lucidota_indy.book_chunk WHERE book_uuid=%s", (book_uuid,))
        lora_rows: list[dict[str, object]] = []
        for idx, piece, tok_count in chunks(text, max_tokens=max_tokens):
            c_hash = sha256_text(piece)
            loc = f"{rel}#chunk-{idx:06d}"
            chunk_graph_uuid = upsert_graph_item(cur, "EVIDENCE", f"{title} chunk {idx}", loc, {"sha256": c_hash, "book_sha256": file_hash, "token_count": tok_count})
            cur.execute(
                """
                INSERT INTO lucidota_indy.book_chunk(book_uuid,graph_item_uuid,chunk_index,token_count,char_count,content_sha256,content,anchor,payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                RETURNING chunk_uuid::text
                """,
                (book_uuid, chunk_graph_uuid, idx, tok_count, len(piece), c_hash, piece, loc, json.dumps({"max_tokens": max_tokens})),
            )
            chunk_uuid = cur.fetchone()[0]
            link(cur, book_graph_uuid, chunk_graph_uuid, "BOOK_HAS_CHUNK", {"chunk_index": idx, "token_count": tok_count})
            emb = vector_literal(embedding384(piece))
            cur.execute(
                f"""
                INSERT INTO lucidota_indy.chunk_embedding(chunk_uuid,embedding_model,embedding_dim,embedding,content_sha256)
                VALUES (%s,%s,384,%s::vector,%s)
                ON CONFLICT(chunk_uuid) DO UPDATE SET embedding_model=EXCLUDED.embedding_model, embedding=EXCLUDED.embedding, content_sha256=EXCLUDED.content_sha256, created_at=now()
                """,
                (chunk_uuid, EMBED_MODEL, emb, c_hash),
            )
            instruction = "Read this source chunk and state the central claim in one concise sentence. Do not invent facts outside the chunk."
            out = "GO payload: BOOK plus EVIDENCE support a local CLAIM/PATTERN. Retrieve this chunk before reasoning; do not invent beyond the visible evidence."
            cur.execute(
                """
                INSERT INTO lucidota_indy.lora_training_example(chunk_uuid,dataset_split,model_ceiling,max_source_tokens,instruction,input,output,content_sha256)
                VALUES (%s, CASE WHEN %s %% 10 = 0 THEN 'validation' ELSE 'train' END, '1.5b', %s, %s, %s, %s, %s)
                ON CONFLICT(chunk_uuid,dataset_split,instruction) DO NOTHING
                """,
                (chunk_uuid, idx, tok_count, instruction, piece, out, c_hash),
            )
            row = {
                "split": "validation" if idx % 10 == 0 else "train",
                "instruction": instruction,
                "input": piece,
                "output": out,
                "source": loc,
                "tokens": tok_count,
                "go_terms": ["BOOK", "EVIDENCE", "CLAIM", "PATTERN"],
            }
            lora_rows.append(row)
            if write_lora:
                write_lora.parent.mkdir(parents=True, exist_ok=True)
                with write_lora.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            result["chunks"] += 1
            result["embeddings"] += 1
            result["lora_examples"] += 1
        cartridge = write_lora_cartridge(
            book_uuid=book_uuid,
            title=title,
            author=author,
            file_hash=file_hash,
            text_hash=text_hash,
            rel=rel,
            rows=lora_rows,
        )
        register_lora_cartridge(cartridge, book_uuid, title, rel)
        result["adapter_id"] = str(cartridge["adapter_id"])
        result["cartridge"] = str(cartridge["local_path"])
        cur.execute(
            """
            UPDATE lucidota_indy.book_source SET status='embedded', chunk_count=%s, embedded_count=%s, payload=payload || %s::jsonb, updated_at=now()
            WHERE book_uuid=%s
            """,
            (result["chunks"], result["embeddings"], json.dumps({"lora_cartridge": cartridge}, sort_keys=True), book_uuid),
        )
    conn.commit()
    result["ok"] = True
    return result


def iter_books(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.name in SKIP_MD:
            continue
        if p.suffix.lower() in SUPPORTED and not p.name.startswith("GO_") and not p.name.startswith("ROOT414_"):
            out.append(p)
    return out


def resolve_book_arg(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return (ROOT / path).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-root", type=Path, default=BOOKS)
    ap.add_argument("--book", type=Path, action="append", help="Ingest only this book path; repeatable.")
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write-lora-jsonl", type=Path, default=ROOT / "04_RUNTIME" / "indy_reads_lora_stage.jsonl")
    ap.add_argument("--append-lora-jsonl", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.max_tokens > 500:
        raise SystemExit("--max-tokens must be <= 500")
    if args.write_lora_jsonl and args.write_lora_jsonl.exists() and not args.append_lora_jsonl:
        args.write_lora_jsonl.unlink()
    if args.book:
        books = [resolve_book_arg(p, args.books_root) for p in args.book]
    else:
        books = iter_books(args.books_root)
    if args.limit:
        books = books[: args.limit]
    with psycopg.connect(DSN) as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO lucidota_indy.ingest_run(source_root,max_tokens,embedding_model,books_seen) VALUES (%s,%s,%s,%s) RETURNING run_uuid::text",
                (str(args.books_root), args.max_tokens, EMBED_MODEL, len(books)),
            )
            run_uuid = cur.fetchone()[0]
        conn.commit()
        results = []
        for p in books:
            results.append(ingest_book(conn, p, args.max_tokens, args.write_lora_jsonl))
        ok = sum(1 for r in results if r["ok"])
        err = len(results) - ok
        chunks_n = sum(r["chunks"] for r in results)
        emb_n = sum(r["embeddings"] for r in results)
        lora_n = sum(r["lora_examples"] for r in results)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lucidota_indy.ingest_run SET status=%s, books_ok=%s, books_error=%s, chunks_written=%s,
                  embeddings_written=%s, lora_examples_written=%s, detail=%s::jsonb, finished_at=now()
                WHERE run_uuid=%s
                """,
                ("succeeded" if err == 0 else "failed", ok, err, chunks_n, emb_n, lora_n, json.dumps({"results": results}, sort_keys=True), run_uuid),
            )
        conn.commit()
    summary = {"ok": err == 0, "run_uuid": run_uuid, "books_seen": len(books), "books_ok": ok, "books_error": err, "chunks": chunks_n, "embeddings": emb_n, "lora_examples": lora_n, "lora_jsonl": str(args.write_lora_jsonl) if args.write_lora_jsonl else None, "results": results}
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else summary)
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
