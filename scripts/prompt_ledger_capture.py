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
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = os.environ.get("POSTGREST_BASE_URL", "http://127.0.0.1:3000").rstrip("/")
DEFAULT_SOURCES = [
    ROOT / "GOALS" / "GOAL_HANDOFF_PROMPT.md",
    ROOT / "GOALS" / "GOAL_PROMPTS.md",
    ROOT / "GOALS" / "OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md",
    ROOT / "GOALS" / "INDY_CORE_IDENTITY_LAW.md",
    ROOT / "AGENTS.md",
]
INTERNAL_STATE_FILENAMES = {
    "CURRENT_HANDOFF.md",
    "GOAL_LOG.md",
}
INTERNAL_STATE_MARKERS = (
    "<codex_internal_context",
)


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


def is_internal_state_source(path: Path, text: str | None = None) -> bool:
    if path.name in INTERNAL_STATE_FILENAMES:
        return True
    if text:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in INTERNAL_STATE_MARKERS)
    return False


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


def build_payload(
    *,
    path: Path | None = None,
    text: str | None = None,
    source_path: str | None = None,
    source: str | None = None,
    source_model: str | None = None,
    conversation_session_id: str | None = None,
    message_uuid: str | None = None,
    ontology_tags: list[str] | None = None,
    subsystem_tags: list[str] | None = None,
    work_order_uuid: str | None = None,
    explicit_unlinked_reason: str | None = None,
) -> dict[str, Any]:
    if path is None and text is None:
        raise ValueError("either path or text must be provided")
    if path is not None and text is None:
        raw_prompt_text = extract_prompt_text(path)
        inferred_source_path = path.as_posix()
        inferred_session_id = path.stem
        inferred_source = source_kind_for_path(path)
        inferred_source_model = source_model_for_path(path)
    else:
        raw_prompt_text = str(text or "")
        inferred_source_path = source_path or "stdin://prompt"
        inferred_session_id = conversation_session_id or message_uuid or str(uuid.uuid4())
        inferred_source = source or "system"
        inferred_source_model = source_model or "manual"
    normalized_prompt_text = normalize_prompt_text(raw_prompt_text)
    if is_internal_state_source(Path(inferred_source_path), raw_prompt_text):
        raise ValueError(f"refusing to file internal state source: {inferred_source_path}")
    linked_work_order_uuid = [str(work_order_uuid)] if work_order_uuid else []
    blockers = explicit_unlinked_reason or ("ambient/daemon/probe" if not linked_work_order_uuid else "")
    classification_tags = ontology_tags if ontology_tags else None
    classification_subsystems = subsystem_tags if subsystem_tags else None
    return {
        "source": inferred_source,
        "source_model": inferred_source_model,
        "receiving_model": "postgrest",
        "target_model": "indy_reads",
        "raw_prompt_text": raw_prompt_text,
        "normalized_prompt_text": normalized_prompt_text,
        "conversation_session_id": inferred_session_id,
        "linked_work_order_uuid": linked_work_order_uuid,
        "linked_goal_id": "active-goal",
        "notes": f"captured from {inferred_source_path}",
        "blockers": blockers,
        "source_path": inferred_source_path,
        "received_at": utc_from_mtime(path) if path is not None else datetime.now(timezone.utc).isoformat(),
        "received_at_confidence": 0.8 if path is not None and path.name == "GOAL_LOG.md" else 0.95,
        "received_at_basis": "mtime" if path is not None else "provided",
        "ontology_tags": classification_tags or [],
        "subsystem_tags": classification_subsystems or [],
        "idempotency_key": sha256_text(
            json.dumps(
                {
                    "path": inferred_source_path,
                    "prompt_hash": sha256_text(normalized_prompt_text),
                    "source": inferred_source,
                    "source_model": inferred_source_model,
                    "conversation_session_id": inferred_session_id,
                    "work_order_uuid": str(work_order_uuid or ""),
                    "explicit_unlinked_reason": explicit_unlinked_reason or "",
                    "ontology_tags": classification_tags or [],
                    "subsystem_tags": classification_subsystems or [],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }


def file_prompt(base_url: str, *, path: Path, decompose: bool = False) -> dict[str, Any]:
    payload = build_payload(path=path)
    filed = post_json(base_url, "rpc/file_prompt", payload)
    result = {"path": path.as_posix(), "file_prompt": filed}
    if decompose and isinstance(filed, dict) and filed.get("prompt_id"):
        decomposed = post_json(base_url, "rpc/decompose_prompt_to_work_orders", {"prompt_id": filed["prompt_id"]})
        result["decompose_prompt_to_work_orders"] = decomposed
    return result


def discover_sources(explicit: list[Path]) -> list[Path]:
    paths = [path for path in explicit if path.exists()]
    if paths:
        return [path for path in paths if not is_internal_state_source(path)]
    discovered: list[Path] = []
    for path in DEFAULT_SOURCES:
        if path.exists() and not is_internal_state_source(path):
            discovered.append(path)
    for dir_path in (ROOT / "GOALS", ROOT / "04_RUNTIME"):
        if dir_path.exists():
            for candidate in sorted(dir_path.glob("*prompt*")):
                if candidate.is_file() and not is_internal_state_source(candidate):
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
    raw = ap.add_mutually_exclusive_group()
    raw.add_argument("--text", default="", help="Capture a raw prompt string directly.")
    raw.add_argument("--stdin", action="store_true", help="Capture a raw prompt from stdin.")
    ap.add_argument("--work-order-uuid", default="", help="Optional work order UUID to link when filing prompts.")
    ap.add_argument("--explicit-unlinked-reason", default="", help="Optional reason to file an unlinked prompt without a work order.")
    ap.add_argument("--source-kind", default="system", help="Source kind for raw text capture.")
    ap.add_argument("--source-model", default="manual", help="Source model label for raw text capture.")
    ap.add_argument("--conversation-session-id", default="", help="Session id to attach to raw text capture.")
    ap.add_argument("--message-uuid", default="", help="Stable message UUID to attach to raw text capture.")
    ap.add_argument("--source-path", default="stdin://prompt", help="Source path/URI for raw text capture.")
    ap.add_argument("--ontology-tag", action="append", dest="ontology_tags", default=[], help="Explicit ontology tag for raw text capture.")
    ap.add_argument("--subsystem-tag", action="append", dest="subsystem_tags", default=[], help="Explicit subsystem tag for raw text capture.")
    ap.add_argument("--decompose", action="store_true", help="Run prompt decomposition after filing.")
    ap.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = ap.parse_args()

    sources = discover_sources([Path(s) for s in args.sources])
    results = []
    if args.text or args.stdin:
        raw_text = args.text if args.text else sys.stdin.read()
        try:
            payload = build_payload(
                text=raw_text,
                source=args.source_kind,
                source_model=args.source_model,
                conversation_session_id=args.conversation_session_id,
                message_uuid=args.message_uuid or None,
                ontology_tags=args.ontology_tags or None,
                subsystem_tags=args.subsystem_tags or None,
                source_path=args.source_path,
                work_order_uuid=args.work_order_uuid or None,
                explicit_unlinked_reason=args.explicit_unlinked_reason or None,
            )
        except ValueError as exc:
            result = {"path": args.source_path, "skipped": True, "blockers": [str(exc)]}
        else:
            filed = post_json(args.base_url, "rpc/file_prompt", payload)
            result = {"path": payload["source_path"], "file_prompt": filed}
            if args.decompose and isinstance(filed, dict) and filed.get("prompt_id"):
                decomposed = post_json(args.base_url, "rpc/decompose_prompt_to_work_orders", {"prompt_id": filed["prompt_id"]})
                result["decompose_prompt_to_work_orders"] = decomposed
        results.append(result)
    else:
        for path in sources:
            try:
                payload = build_payload(
                    path=path,
                    work_order_uuid=args.work_order_uuid or None,
                    explicit_unlinked_reason=args.explicit_unlinked_reason or None,
                )
            except ValueError as exc:
                result = {"path": path.as_posix(), "skipped": True, "blockers": [str(exc)]}
            else:
                filed = post_json(args.base_url, "rpc/file_prompt", payload)
                result = {"path": path.as_posix(), "file_prompt": filed}
                if args.decompose and isinstance(filed, dict) and filed.get("prompt_id"):
                    decomposed = post_json(args.base_url, "rpc/decompose_prompt_to_work_orders", {"prompt_id": filed["prompt_id"]})
                    result["decompose_prompt_to_work_orders"] = decomposed
            results.append(result)
    payload = {"schema": "lucidota.prompt_ledger_capture.v1", "base_url": args.base_url, "results": results, "source_count": len(results)}
    if args.json:
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
