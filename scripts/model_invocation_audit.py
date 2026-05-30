#!/usr/bin/env python3
"""Audit every cloud/local model invocation receipt and five-task model-audit coverage."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_invocation_audits"
OBS = ROOT / "04_RUNTIME" / "observation_center" / "model_invocation_audit_latest.json"
BIG_BOARD = ROOT / "05_OUTPUTS" / "big_board.json"
MODEL_DIR = ROOT / "05_OUTPUTS" / "model_invocations"
BOARD_DIR = ROOT / "05_OUTPUTS" / "project2501_board_moves"
LOCAL_AUDIT_DIR = OUT / "local_audit_receipts"

AUDIT_BLOCK_RE = re.compile(r"MODEL_AUDIT_BLOCK:([A-Za-z0-9_.-]+)(?::([0-9a-f]{64}))?")
AUDITOR_CYCLE = ["local", "groq", "cohere"]




def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def block_signature(tasks: list[dict[str, Any]]) -> str:
    """Content signature for a five-task audit block.

    This prevents old model audit receipts for task_block_000N from satisfying
    coverage after the block's task receipt set changes.
    """
    payload = []
    for task in tasks:
        payload.append({
            "task_id": task.get("task_id"),
            "receipt_path": task.get("receipt_path"),
            "verdict": task.get("verdict"),
            "position": task.get("position"),
            "dominant_provider": task.get("dominant_provider"),
        })
    return hashlib.sha256(stable_json(payload).encode("utf-8", errors="replace")).hexdigest()


def audit_marker(block_id: str, signature: str) -> str:
    return f"MODEL_AUDIT_BLOCK:{block_id}:{signature}"

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}:{exc}"}


def text_in_obj(obj: Any, limit: int = 12000) -> str:
    try:
        text = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        text = str(obj)
    return text[:limit]


def extract_audit_marker(data: dict[str, Any]) -> tuple[str | None, str | None]:
    hay = "\n".join([
        text_in_obj(data.get("request", {})),
        text_in_obj(data.get("wire_request", {})),
        text_in_obj(data.get("generation_trace", {})),
        str(data.get("text") or ""),
    ])
    m = AUDIT_BLOCK_RE.search(hay)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def extract_audit_block_id(data: dict[str, Any]) -> str | None:
    return extract_audit_marker(data)[0]


def parse_audit_json(raw_output: str) -> dict[str, Any] | None:
    text = (raw_output or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def valid_audit_output(row: dict[str, Any], *, block_id: str, signature: str, auditor: str) -> bool:
    parsed = row.get("audit_verdict_payload")
    if not isinstance(parsed, dict):
        parsed = parse_audit_json(str(row.get("raw_output") or ""))
    if not isinstance(parsed, dict):
        return False
    if str(parsed.get("block_id")) != block_id:
        return False
    if str(parsed.get("block_signature")) != signature:
        return False
    if str(parsed.get("auditor_provider")) != auditor:
        return False
    return str(parsed.get("verdict")) in {"PASS", "PARTIAL", "FAIL"}


def collect_invocations(paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    paths = list(paths) if paths is not None else sorted(
        list(MODEL_DIR.glob("*.json")) + list(LOCAL_AUDIT_DIR.glob("*.json")),
        key=lambda p: p.stat().st_mtime,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        deterministic_audit = bool(data.get("deterministic_audit_receipt") or str(data.get("schema") or "").startswith("lucidota.model_invocation_audit.local_deterministic"))
        trace = data.get("generation_trace") if isinstance(data.get("generation_trace"), dict) else {}
        provider = str(data.get("provider") or trace.get("target") or "unknown")
        model = str(data.get("model") or data.get("model_name") or trace.get("model_name") or "unknown")
        raw_output = str(trace.get("raw_output") if trace.get("raw_output") is not None else data.get("text") or "")
        audit_block_id, audit_block_signature = extract_audit_marker(data)
        audit_block_id = str(data.get("audit_block_id") or audit_block_id or "") or None
        audit_block_signature = str(data.get("audit_block_signature") or audit_block_signature or "") or None
        # If raw_output is empty, check if the data dict carries audit_verdict_payload directly
        direct_payload = data.get("audit_verdict_payload")
        if not raw_output and isinstance(direct_payload, dict):
            audit_verdict_payload = direct_payload
        else:
            audit_verdict_payload = parse_audit_json(raw_output) or (direct_payload if isinstance(direct_payload, dict) else None)
        rows.append({
            "schema": "lucidota.model_invocation_audit.row.v1",
            "receipt_path": rel(path),
            "generated_at": data.get("generated_at") or data.get("generated_at_utc") or "",
            "provider": provider,
            "model": model,
            "deterministic_audit_receipt": deterministic_audit,
            "execute_performed": bool(data.get("execute_performed")),
            "real_inference_performed": bool(data.get("real_inference_performed", data.get("execute_performed", False))),
            "status": str(data.get("status") or data.get("verdict") or "UNKNOWN"),
            "latency_ms": data.get("latency_ms", trace.get("latency_ms")),
            "raw_output": raw_output,
            "raw_output_chars": int(trace.get("raw_output_chars") or len(raw_output)),
            "audit_block_id": audit_block_id,
            "audit_block_signature": audit_block_signature,
            "audit_verdict_payload": audit_verdict_payload,
            "audit_verdict_parse_status": "valid_json" if audit_verdict_payload else "invalid_or_missing",
            "audit_verdict": audit_verdict_payload.get("verdict") if audit_verdict_payload else None,
            "blockers": data.get("blockers") or [],
            "canonical_graph_writes_performed": bool(data.get("canonical_graph_writes_performed", False)),
        })
    return rows


def collect_tasks(paths: Iterable[Path] | None = None) -> list[dict[str, Any]]:
    paths = list(paths) if paths is not None else sorted(BOARD_DIR.glob("project2501_board_move_*.json"), key=lambda p: p.stat().st_mtime)
    tasks: list[dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        data = load_json(path)
        move = data.get("board_move") if isinstance(data.get("board_move"), dict) else {}
        env = data.get("event_envelope") if isinstance(data.get("event_envelope"), dict) else {}
        task_id = str(data.get("event_envelope", {}).get("event_id") or move.get("event_id") or path.stem)
        tasks.append({
            "schema": "lucidota.model_invocation_audit.task.v1",
            "task_id": task_id,
            "ordinal": idx,
            "receipt_path": rel(path),
            "generated_at": data.get("generated_at") or env.get("ts") or "",
            "actor": move.get("actor") or env.get("actor") or "unknown",
            "dominant_provider": "codex" if (move.get("actor") or env.get("actor")) in {"operator", "codex", "system", None, ""} else str(move.get("actor") or env.get("actor")),
            "position": move.get("position") or "unknown",
            "verdict": move.get("verdict") or data.get("status") or "UNKNOWN",
        })
    return tasks


def build_five_task_blocks(tasks: list[dict[str, Any]], auditor_cycle: list[str] | None = None, invocations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    cycle = auditor_cycle or AUDITOR_CYCLE
    inv_by_block: dict[str, list[dict[str, Any]]] = {}
    for inv in invocations or []:
        bid = inv.get("audit_block_id")
        if bid:
            inv_by_block.setdefault(str(bid), []).append(inv)
    blocks: list[dict[str, Any]] = []
    for block_idx, start in enumerate(range(0, len(tasks), 5), start=1):
        chunk = tasks[start:start + 5]
        dominant = Counter(str(t.get("dominant_provider") or "unknown") for t in chunk).most_common(1)[0][0] if chunk else "unknown"
        auditor = cycle[(block_idx - 1) % len(cycle)]
        if auditor == dominant:
            auditor = cycle[block_idx % len(cycle)]
        block_id = f"task_block_{block_idx:04d}"
        signature = block_signature(chunk)
        all_receipts = inv_by_block.get(block_id, [])
        signed_current = [r for r in all_receipts if r.get("audit_block_signature") == signature]
        candidate_provider = [r for r in signed_current if str(r.get("provider")) == auditor]
        fresh = [r for r in candidate_provider if valid_audit_output(r, block_id=block_id, signature=signature, auditor=auditor)]
        invalid_audit = [r for r in candidate_provider if not valid_audit_output(r, block_id=block_id, signature=signature, auditor=auditor)]
        wrong_provider = [r for r in signed_current if str(r.get("provider")) != auditor]
        stale = [r for r in all_receipts if r.get("audit_block_signature") != signature]
        if len(chunk) < 5:
            status = "PENDING_UNTIL_5_TASKS"
        else:
            if fresh:
                status = "MODEL_AUDIT_RECEIPT_PRESENT"
            elif invalid_audit:
                status = "MISSING_VALID_AUDIT_OUTPUT"
            elif wrong_provider:
                status = "MISSING_REQUIRED_AUDITOR"
            else:
                status = "MISSING_FRESH_MODEL_AUDIT" if stale else "MISSING_DEDICATED_MODEL_AUDIT"
        blocks.append({
            "schema": "lucidota.model_invocation_audit.five_task_block.v1",
            "block_id": block_id,
            "block_signature": signature,
            "audit_marker": audit_marker(block_id, signature),
            "task_count": len(chunk),
            "task_receipts": [t["receipt_path"] for t in chunk],
            "dominant_provider": dominant,
            "auditor_provider": auditor,
            "audit_receipts": [r["receipt_path"] for r in fresh],
            "invalid_audit_receipts": [r["receipt_path"] for r in invalid_audit],
            "wrong_provider_audit_receipts": [r["receipt_path"] for r in wrong_provider],
            "stale_or_unsigned_audit_receipts": [r["receipt_path"] for r in stale],
            "audit_status": status,
            "requires_different_model": True,
        })
    return blocks


def markdown(rows: list[dict[str, Any]], blocks: list[dict[str, Any]], payload: dict[str, Any]) -> str:
    lines = [
        "# Model Invocation Audit",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Invocation count: `{len(rows)}`",
        f"Five-task audit blocks: `{len(blocks)}`",
        "",
        "## Five-task audit coverage",
        "",
    ]
    for b in blocks:
        lines.append(f"- `{b['block_id']}` sig=`{b.get('block_signature','')[:12]}` auditor=`{b['auditor_provider']}` tasks=`{b['task_count']}` status=`{b['audit_status']}` receipts={b['audit_receipts']} stale={b.get('stale_or_unsigned_audit_receipts', [])}")
    lines += ["", "## Every model invocation receipt", ""]
    for row in rows:
        output = (row.get("raw_output") or "").replace("\n", "\\n")
        if len(output) > 600:
            output = output[:600] + "…"
        lines.append(
            f"- `{row['generated_at']}` provider=`{row['provider']}` model=`{row['model']}` "
            f"execute=`{row['execute_performed']}` status=`{row['status']}` receipt=`{row['receipt_path']}` output=`{output}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_model_audit(invocations: list[dict[str, Any]], tasks: list[dict[str, Any]], out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = build_five_task_blocks(tasks, invocations=invocations)
    model_invocations = [r for r in invocations if not r.get("deterministic_audit_receipt")]
    by_provider = Counter(str(r.get("provider")) for r in model_invocations)
    by_execute = Counter("execute" if r.get("execute_performed") else "dry_run" for r in model_invocations)
    missing_blocks = sum(1 for b in blocks if str(b["audit_status"]).startswith("MISSING_"))
    verdict = "PASS" if missing_blocks == 0 else "FAIL"
    base = stamp()
    json_path = out_dir / f"model_invocation_audit_{base}.json"
    md_path = out_dir / f"model_invocation_audit_{base}.md"
    payload = {
        "schema": "lucidota.model_invocation_audit.v1",
        "generated_at": now(),
        "invocation_count": len(model_invocations),
        "audit_evidence_count": len(invocations),
        "by_provider": dict(sorted(by_provider.items())),
        "by_execution_mode": dict(sorted(by_execute.items())),
        "five_task_audit_blocks": blocks,
        "missing_dedicated_model_audit_blocks": missing_blocks,
        "verdict": verdict,
        "status": verdict,
        "signature_enforced": True,
        "invocations": invocations,
        "canonical_graph_writes_performed": False,
    }
    payload["json_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(markdown(invocations, blocks, payload), encoding="utf-8")
    payload["json_path"] = rel(json_path)
    payload["markdown_path"] = rel(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    OBS.parent.mkdir(parents=True, exist_ok=True)
    OBS.write_text(json.dumps({k: v for k, v in payload.items() if k != "invocations"}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    update_big_board(payload)
    print("MODEL_INVOCATION_AUDIT_JSON=" + rel(json_path))
    print("MODEL_INVOCATION_AUDIT_MARKDOWN=" + rel(md_path))
    return payload


def update_big_board(payload: dict[str, Any]) -> None:
    try:
        board = json.loads(BIG_BOARD.read_text(encoding="utf-8")) if BIG_BOARD.exists() else {}
    except Exception:
        board = {}
    board.setdefault("observation_center", {})["model_invocation_audit"] = {
        "source_receipt": payload.get("json_path"),
        "markdown": payload.get("markdown_path"),
        "verdict": payload.get("verdict"),
        "status": payload.get("status"),
        "invocation_count": payload.get("invocation_count"),
        "missing_dedicated_model_audit_blocks": payload.get("missing_dedicated_model_audit_blocks"),
        "canonical_graph_writes_performed": False,
    }
    board.setdefault("counters", {})["model_invocation_audit_invocations"] = payload.get("invocation_count", 0)
    BIG_BOARD.parent.mkdir(parents=True, exist_ok=True)
    BIG_BOARD.write_text(json.dumps(board, indent=2, sort_keys=True, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Print/audit all cloud and local model invocation receipts.")
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    inv = collect_invocations()
    tasks = collect_tasks()
    payload = write_model_audit(inv, tasks)
    if args.json:
        print(json.dumps({k: v for k, v in payload.items() if k != "invocations"}, sort_keys=True, default=str))
    verdict = str(payload.get("verdict") or "FAIL")
    print("MODEL_INVOCATION_AUDIT=" + verdict)
    return 0 if verdict == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
