#!/usr/bin/env python3
"""File steering prompts into the DB-backed prompt ledger.

The script is bounded: it reads explicit source files, extracts prompt text,
and files each prompt through PostgREST RPCs. It does not build giant prompt
dumps or read raw tables directly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
DEFAULT_SOURCES = [
    ROOT / "GOALS" / "CURRENT_HANDOFF.md",
    ROOT / "GOALS" / "GOAL_LOG.md",
    ROOT / "GOALS" / "GOAL_HANDOFF_PROMPT.md",
    ROOT / "GOALS" / "GOAL_PROMPTS.md",
    ROOT / "GOALS" / "OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md",
    ROOT / "GOALS" / "INDY_CORE_IDENTITY_LAW.md",
    ROOT / "AGENTS.md",
]


def utc_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def source_kind_for_path(path: Path) -> str:
    if "GOALS" in path.parts:
        return "operator"
    if "tests" in path.parts or path.suffix in {".py", ".sh"}:
        return "codex"
    return "system"


def source_model_for_path(path: Path) -> str:
    if path.name.endswith(".md"):
        return "postgrest/manual"
    if path.suffix == ".sh":
        return "shell"
    return "codex"


def extract_prompt_text(path: Path, max_chars: int = 20000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.name == "GOAL_LOG.md":
        blocks: list[str] = []
        marker = "Save This Prompt, Pass on this Handoff:"
        matches = [m.start() for m in re.finditer(re.escape(marker), text)]
        for start in matches[-3:]:
            end = text.find("\n## Step", start)
            if end == -1:
                end = len(text)
            blocks.append(text[start:end].strip())
        if blocks:
            text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        data=data,
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def file_prompt(base_url: str, *, path: Path, decompose: bool = False) -> dict[str, Any]:
    raw_prompt_text = extract_prompt_text(path)
    normalized_prompt_text = normalize_prompt_text(raw_prompt_text)
    payload = {
        "source": source_kind_for_path(path),
        "source_model": source_model_for_path(path),
        "receiving_model": "postgrest",
        "target_model": "indy_reads",
        "raw_prompt_text": raw_prompt_text,
        "normalized_prompt_text": normalized_prompt_text,
        "conversation_session_id": path.stem,
        "linked_goal_id": "active-goal",
        "notes": f"captured from {path.as_posix()}",
        "blockers": "",
        "source_path": path.as_posix(),
        "received_at": utc_from_mtime(path),
        "received_at_confidence": 0.8 if path.name == "GOAL_LOG.md" else 0.95,
        "received_at_basis": "mtime",
        "idempotency_key": sha256_text(
            json.dumps(
                {
                    "path": path.as_posix(),
                    "prompt_hash": sha256_text(normalized_prompt_text),
                    "source": source_kind_for_path(path),
                    "source_model": source_model_for_path(path),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    filed = post_json(base_url, "rpc/file_prompt", payload)
    result = {"path": path.as_posix(), "file_prompt": filed}
    if decompose and isinstance(filed, dict) and filed.get("prompt_id"):
        decomposed = post_json(base_url, "rpc/decompose_prompt_to_work_orders", {"prompt_id": filed["prompt_id"]})
        result["decompose_prompt_to_work_orders"] = decomposed
    return result


def discover_sources(explicit: list[Path]) -> list[Path]:
    paths = [path for path in explicit if path.exists()]
    if paths:
        return paths
    discovered: list[Path] = []
    for path in DEFAULT_SOURCES:
        if path.exists():
            discovered.append(path)
    for dir_path in (ROOT / "GOALS", ROOT / "04_RUNTIME"):
        if dir_path.exists():
            for candidate in sorted(dir_path.glob("*prompt*")):
                if candidate.is_file():
                    discovered.append(candidate)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in discovered:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description="File steering prompts into the prompt ledger.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--source", action="append", dest="sources", default=[], help="Explicit source file to capture.")
    ap.add_argument("--decompose", action="store_true", help="Run prompt decomposition after filing.")
    ap.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = ap.parse_args()

    sources = discover_sources([Path(s) for s in args.sources])
    results = [file_prompt(args.base_url, path=path, decompose=args.decompose) for path in sources]
    payload = {"schema": "lucidota.prompt_ledger_capture.v1", "base_url": args.base_url, "results": results, "source_count": len(results)}
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
