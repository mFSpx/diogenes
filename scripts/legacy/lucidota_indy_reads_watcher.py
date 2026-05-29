#!/usr/bin/env python3
"""INDY_READs book watcher.

Polls /BOOKS for new or changed books, then runs the simple ingestion lane:
extract -> <=500 token chunks -> CKDOG1 embeddings -> GO graph -> LoRA cartridge dataset.

No external services. This is the "Mamba watches, Indy reads" loop.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == "legacy" else Path(__file__).resolve().parents[1]
BOOKS = ROOT / "BOOKS"
DATA = BOOKS / ".indy_reads"
STATE_PATH = DATA / "watch_state.json"
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
SUPPORTED = {".epub", ".pdf", ".mobi", ".txt", ".md"}
SKIP_PREFIXES = ("GO_", "ROOT414_")
SKIP_NAMES = {"README_INDY_READS.md", "OFFICIAL_ONTOLOGY_POINTER.md", "ROOT414_GAME_GRADING_SCHEMA.md"}


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"schema": "lucidota.indy_reads.watch_state.v1", "files": {}}


def save_state(state: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_PATH)


def iter_books(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES or p.name.startswith(SKIP_PREFIXES):
            continue
        if p.suffix.lower() in SUPPORTED:
            out.append(p)
    return out


def fingerprint(path: Path) -> dict[str, int | str]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "path": str(path.relative_to(ROOT))}


def ensure_state_schema() -> None:
    with psycopg.connect(STATE_DSN) as conn, conn.cursor() as cur:
        cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def workflow_event(run_id: str, phase: str, status: str, detail: dict[str, Any]) -> None:
    try:
        ensure_state_schema()
        with psycopg.connect(STATE_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES ('indy-reads-book-watch', %s, %s, %s, 'mamba-watch-loop', %s::jsonb)
                """,
                (run_id, phase, status, json.dumps(detail, sort_keys=True)),
            )
            conn.commit()
    except psycopg.Error:
        # Watcher must not die merely because the reporting surface is briefly down.
        return


def run_ingest(book: Path, max_tokens: int, append_lora_jsonl: bool) -> dict[str, Any]:
    cmd = [
        str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else "python3",
        str(ROOT / "scripts" / "lucidota_indy_library_ingest.py"),
        "--book",
        str(book),
        "--max-tokens",
        str(max_tokens),
        "--json",
    ]
    if append_lora_jsonl:
        cmd.append("--append-lora-jsonl")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return {"ok": False, "book": str(book.relative_to(ROOT)), "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}
    return json.loads(proc.stdout)


def process_once(root: Path, max_tokens: int, append_lora_jsonl: bool) -> dict[str, Any]:
    state = load_state()
    files = state.setdefault("files", {})
    changed: list[Path] = []
    for book in iter_books(root):
        fp = fingerprint(book)
        key = fp["path"]
        if files.get(key) != fp:
            changed.append(book)
    results: list[dict[str, Any]] = []
    for book in changed:
        run_id = str(book.relative_to(ROOT))
        workflow_event(run_id, "detected", "queued", {"book": run_id, "watcher": "mamba-1.4b-listener"})
        result = run_ingest(book, max_tokens, append_lora_jsonl)
        results.append(result)
        if result.get("ok"):
            files[str(book.relative_to(ROOT))] = fingerprint(book)
            workflow_event(run_id, "embedded", "succeeded", {"book": run_id, "result": result})
        else:
            workflow_event(run_id, "embedded", "failed", {"book": run_id, "result": result})
    state["updated_at"] = time.time()
    save_state(state)
    return {"ok": all(r.get("ok") for r in results), "changed": len(changed), "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books-root", type=Path, default=BOOKS)
    ap.add_argument("--interval", type=float, default=float(os.environ.get("LUCIDOTA_INDY_WATCH_INTERVAL", "5")))
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--append-lora-jsonl", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.max_tokens > 500:
        raise SystemExit("--max-tokens must be <= 500")
    if args.once:
        result = process_once(args.books_root, args.max_tokens, args.append_lora_jsonl)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
        return 0 if result["ok"] else 1

    print(json.dumps({"ok": True, "watching": str(args.books_root), "interval": args.interval, "persona": "INDY_READs", "manager": "mamba-watch-loop"}), flush=True)
    while True:
        process_once(args.books_root, args.max_tokens, args.append_lora_jsonl)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
