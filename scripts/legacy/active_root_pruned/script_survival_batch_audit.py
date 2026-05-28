#!/usr/bin/env python3
"""Batch baseline survival-audit every TICKLETRUNK script entry.

This is a coverage repair tool: it appends deterministic audit rows for every
TICKLETRUNK SCRIPTS path not already present in SCRIPT_AUDIT_MANIFEST.jsonl.
It never deletes or moves scripts. Legacy-path scripts are corpse-logged and
queued for Krampuschewing; active or sovereign-toolbox scripts get evidence
rows with callers, dataflow hints, LOC pressure, and a conservative verdict.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from script_survival_audit import (
    DEFAULT_AUDIT_MANIFEST,
    DEFAULT_CORPSE_MANIFEST,
    DEFAULT_KRAMPUS_QUEUE,
    DEFAULT_UNKNOWN_HOLD,
    KRAMPUS_ACTIONS,
    append_jsonl,
    rel,
    script_facts,
)

ROOT = Path(__file__).resolve().parents[1]
TICKLETRUNK = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
OUT = ROOT / "05_OUTPUTS" / "script_audit"
SCAN_ROOTS = {"scripts", "tests", "06_SCHEMA", "00_PROJECT_BRAIN", ".github", "services", "src", "ALGOS"}
SCAN_SUFFIXES = {".py", ".sh", ".sql", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".service", ".timer"}
SCAN_SKIP = {"00_PROJECT_BRAIN/TICKLETRUNK.json", "00_PROJECT_BRAIN/TICKLETRUNK.md"}
MAX_CORPUS_BYTES = 512 * 1024
MAX_SCRIPT_BYTES_FOR_FULL_TEXT = 8 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def read_jsonl_paths(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("path"):
                out.add(rel(row["path"]))
    return out


def tickletrunk_script_rows() -> list[dict[str, Any]]:
    data = json.loads(TICKLETRUNK.read_text(encoding="utf-8"))
    rows = data.get("toolboxes", {}).get("SCRIPTS", [])
    return sorted(rows, key=lambda row: row.get("relative_path") or row.get("path") or "")


def read_script_text(path: Path) -> tuple[str, bool]:
    if not path.exists() or not path.is_file():
        return "", False
    size = path.stat().st_size
    if size > MAX_SCRIPT_BYTES_FOR_FULL_TEXT:
        return path.read_bytes()[:MAX_SCRIPT_BYTES_FOR_FULL_TEXT].decode("utf-8", errors="replace"), False
    return path.read_text(encoding="utf-8", errors="replace"), True


def python_metrics(path: Path, text: str) -> dict[str, Any]:
    if path.suffix != ".py":
        return {"branch_count": 0, "syntax_error": None}
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {"branch_count": 0, "syntax_error": str(exc)}
    branches = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.Match, ast.IfExp)):
            branches += 1
    return {"branch_count": branches, "syntax_error": None}


def build_scan_corpus() -> list[tuple[str, str]]:
    corpus: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        base = Path(dirpath)
        try:
            rel_base = base.relative_to(ROOT)
        except ValueError:
            continue
        first = rel_base.parts[0] if rel_base.parts else ""
        if first and first not in SCAN_ROOTS:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"})
        for name in sorted(filenames):
            path = base / name
            r = rel(path)
            if r in SCAN_SKIP or path.suffix not in SCAN_SUFFIXES:
                continue
            try:
                data = path.read_bytes()[:MAX_CORPUS_BYTES].decode("utf-8", errors="replace")
            except Exception:
                continue
            corpus.append((r, data))
    return corpus


def find_callers(script_path: str, basename: str, known_uses: list[str], corpus: list[tuple[str, str]]) -> list[str]:
    callers = set(known_uses or [])
    for fpath, text in corpus:
        if fpath == script_path:
            continue
        if script_path in text or basename in text:
            callers.add(fpath)
    callers.discard("00_PROJECT_BRAIN/TICKLETRUNK.md")
    callers.discard("00_PROJECT_BRAIN/TICKLETRUNK.json")
    return sorted(callers)[:200]


def classify_role(script_path: str, row: dict[str, Any]) -> str:
    name = Path(script_path).name.lower()
    text = f"{script_path} {row.get('what_it_does') or ''} {row.get('one_line_description') or ''}".lower()
    if "/legacy/" in script_path:
        return "legacy_script_corpse"
    if script_path.startswith("01_REPOS/"):
        return "sovereign_toolbox_prior"
    if Path(script_path).suffix == "" and (ROOT / script_path).is_dir():
        return "script_directory_container"
    if any(k in text for k in ["absurd", "queue", "worker"]):
        return "absurd_queue_worker_or_helper"
    if "chrono" in text:
        return "chrono_ledger_helper"
    if "graph" in text:
        return "graph_promotion_or_materialization_helper"
    if "korpus" in text or "krampus" in text:
        return "korpus_krampus_ingestion_helper"
    if "document" in text or "ocr" in text or "parse" in text:
        return "document_ingestion_helper"
    if "ahoy" in text:
        return "ahoy_strategy_lab_helper"
    if "model" in text or "gpu" in text or "ncnn" in text:
        return "model_runtime_helper"
    if any(k in name for k in ["audit", "check", "gate", "smoke", "test", "validator"]):
        return "test_checker_auditor"
    return "script_helper"


def dataflow_hints(text: str, facts: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    tables = sorted(set(re.findall(r"lucidota_[a-z_]+\.[A-Za-z0-9_]+", text)))[:40]
    output_paths = sorted(set(re.findall(r"05_OUTPUTS/[A-Za-z0-9_./{}-]+", text)))[:40]
    env_vars = sorted(
        set(re.findall(r"os\.environ\.get\(['\"]([A-Z0-9_]+)['\"]", text))
        | set(re.findall(r"os\.environ\[['\"]([A-Z0-9_]+)['\"]\]", text))
    )[:40]
    inputs = [f"cli:{flag}" for flag in facts.get("argparse_flags", [])[:40]]
    inputs += [f"env:{var}" for var in env_vars]
    outputs = output_paths + [f"db:{table}" for table in tables]
    side_effects: list[str] = []
    if re.search(r"\bINSERT\s+INTO\b", text, re.I):
        side_effects.append("db_insert")
    if re.search(r"\bUPDATE\b", text, re.I):
        side_effects.append("db_update")
    if re.search(r"\bDELETE\s+FROM\b", text, re.I):
        side_effects.append("db_delete")
    if any(tok in text for tok in [".write_text(", ".write_bytes(", "open(", "Path.open("]):
        side_effects.append("filesystem_write_possible")
    if "subprocess." in text:
        side_effects.append("subprocess_possible")
    return sorted(set(inputs)), sorted(set(outputs)), sorted(set(side_effects))


def score_script(script_path: str, text: str, facts: dict[str, Any], metrics: dict[str, Any], callers: list[str]) -> tuple[int, int, int, int]:
    lines = text.splitlines()
    loc = len(lines)
    nonblank = sum(1 for line in lines if line.strip())
    if "/legacy/" in script_path:
        return 9, 2, loc, nonblank
    if script_path.startswith("01_REPOS/"):
        return 3 if nonblank < 300 else 5, 6, loc, nonblank
    slop = 0
    if facts.get("syntax_error") or metrics.get("syntax_error"):
        slop += 4
    if nonblank > 300:
        slop += 2
    if nonblank > 700:
        slop += 2
    if len(facts.get("functions", [])) > 35:
        slop += 1
    if int(metrics.get("branch_count") or 0) > 45:
        slop += 1
    if any(token in text for token in ("read_text(", "read_bytes(", ".read(")) and "max_" not in text[:3000].lower():
        slop += 1
    if "TODO" in text or "FIXME" in text:
        slop += 1
    if facts.get("language") == "py" and not facts.get("has_module_docstring"):
        slop += 1
    slop = min(10, slop)
    survival = 5
    if callers:
        survival += 1
    if script_path.startswith("scripts/absurd") or script_path.startswith("scripts/chrono") or script_path.startswith("scripts/graph") or script_path.startswith("scripts/korpus") or script_path.startswith("scripts/spine"):
        survival += 2
    if facts.get("executable"):
        survival += 1
    if facts.get("syntax_error") or metrics.get("syntax_error"):
        survival -= 2
    return slop, max(0, min(10, survival)), loc, nonblank


def decide_verdict(script_path: str, role: str, slop: int, survival: int, facts: dict[str, Any], metrics: dict[str, Any]) -> str:
    if role == "legacy_script_corpse":
        return "LEGACY_CORPSE"
    if role in {"sovereign_toolbox_prior", "script_directory_container"}:
        return "ACTIVE_KEEP"
    if facts.get("syntax_error") or metrics.get("syntax_error"):
        return "UNKNOWN_HOLD"
    if slop >= 8 and survival < 8:
        return "UNKNOWN_HOLD"
    if slop >= 6 and survival >= 8:
        return "ACTIVE_REPAIR"
    return "ACTIVE_KEEP"


def comparison_strings(slop: int, survival: int, nonblank: int) -> tuple[str, str]:
    ncnn = max(0, 10 - min(5, nonblank // 160) - (2 if slop >= 7 else 0))
    flow = max(0, min(10, survival + (1 if slop <= 3 else 0) - (2 if slop >= 8 else 0)))
    return f"{ncnn}/10 static LOC/directness pressure; NCNN/FLOW-style follow-up required for high slop", f"{flow}/10 static runtime/dataflow fit; receipt-backed live proof still separate"


def manifest_row(
    script_path: str,
    verdict: str,
    role: str,
    callers: list[str],
    inputs: list[str],
    outputs: list[str],
    side_effects: list[str],
    slop: int,
    survival: int,
    loc: int,
    nonblank: int,
    facts: dict[str, Any],
    row: dict[str, Any],
    executed: bool,
) -> dict[str, Any]:
    ncnn, flow = comparison_strings(slop, survival, nonblank)
    if verdict == "LEGACY_CORPSE":
        reason = "Path is under scripts/legacy; retired from active trust, preserved as historical evidence, and queued for Krampuschewing without deletion."
        next_action = "Index corpse and only copy/adapt through current ABSURD/graph/Chrono gates if needed."
    elif verdict == "UNKNOWN_HOLD":
        reason = "Static batch audit could not safely grant active trust; script remains preserved and requires targeted caller/runtime proof before promotion."
        next_action = "Run focused deep audit before active production use."
    elif verdict == "ACTIVE_REPAIR":
        reason = "Script appears operationally important but static slop pressure is high; active trust requires focused repair before claiming clean status."
        next_action = "Schedule targeted shrink/hardening pass."
    elif role == "sovereign_toolbox_prior":
        reason = "Sovereign toolbox/reference script; preserved as reusable prior, not production-gated or mutated."
        next_action = "Copy/adapt into production only through current LUCIDOTA gates."
    else:
        reason = "Static audit found existing file, readable contract hints, and no immediate corpse trigger; keep with future focused hardening as needed."
        next_action = "Re-audit deeply when this script becomes the next active target."
    facts = dict(facts)
    facts.update({
        "loc": loc,
        "nonblank_loc": nonblank,
        "tickletrunk_id": row.get("id"),
        "tickletrunk_status": row.get("status"),
        "proof_hoard_role": row.get("proof_hoard_role"),
        "what_it_does": row.get("what_it_does"),
        "full_file_read": facts.get("exists") and facts.get("size_bytes", 0) <= MAX_SCRIPT_BYTES_FOR_FULL_TEXT,
    })
    return {
        "timestamp_utc": now(),
        "path": script_path,
        "verdict": verdict,
        "role": role,
        "callers": callers,
        "inputs": inputs,
        "outputs": outputs,
        "side_effects": side_effects,
        "slop_score": slop,
        "survival_score": survival,
        "golden_comparison": {
            "ncnn_alignment": ncnn,
            "flow_alignment": flow,
        },
        "decision_reason": reason,
        "changes_made": ["scripts/SCRIPT_AUDIT_MANIFEST.jsonl"] if executed else [],
        "validation": ["static_full_file_read_or_bounded_large_file_read", "tickletrunk_cross_reference", "callgraph_text_scan"],
        "next_action": next_action,
        "facts": facts,
    }


def corpse_row(script_path: str, callers: list[str], slop: int, survival: int, loc: int, facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "path": script_path,
        "verdict": "LEGACY_CORPSE",
        "death_reason": "scripts/legacy path is retired from active trust; preserved, not deleted",
        "evidence_read": [script_path, "00_PROJECT_BRAIN/TICKLETRUNK.json", "scripts/SCRIPT_AUDIT_MANIFEST.jsonl"],
        "callers_found": callers,
        "superseded_by": None,
        "risk_if_kept_active": "legacy implementation may bypass current ABSURD, Chrono, graph-promotion, or command-envelope contracts",
        "historical_value": "implementation history and reusable prior patterns for Krampuschewing preservation",
        "slop_score": slop,
        "survival_score": survival,
        "current_loc": loc,
        "estimated_minimum_sane_loc": max(20, int(loc * 0.65)),
        "krampuschewing_ingest": True,
        "hash_blake3_or_sha256": facts.get("hash_blake3_or_sha256"),
        "auditor": "ouroboros_batch_script_survival_audit",
        "notes": "corpse-logged by deterministic batch audit; no deletion performed",
    }


def krampus_row(script_path: str) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "source_path": script_path,
        "source_manifest": "/scripts/CORPSE_MANIFEST.jsonl",
        "ingest_class": "SCRIPT_CORPSE",
        "desired_actions": KRAMPUS_ACTIONS,
        "never_delete": True,
    }


def unknown_row(script_path: str, verdict: str) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "path": script_path,
        "source_manifest": "/scripts/SCRIPT_AUDIT_MANIFEST.jsonl",
        "missing_evidence": ["focused_runtime_call_proof", "human_deep_semantic_review"],
        "next_inspection_action": "Run targeted deep audit before promoting active trust.",
        "verdict": verdict,
        "auditor": "ouroboros_batch_script_survival_audit",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Append baseline survival audit rows for all unaudited TICKLETRUNK scripts.")
    ap.add_argument("--execute", action="store_true", help="Write manifest/corpse/ingest rows. Without this, dry-run only.")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows processed; 0 means all missing.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = tickletrunk_script_rows()
    audited = read_jsonl_paths(DEFAULT_AUDIT_MANIFEST)
    missing_rows = [row for row in rows if rel(row.get("relative_path") or row.get("path")) not in audited]
    if args.limit:
        missing_rows = missing_rows[: args.limit]

    corpus = build_scan_corpus()
    verdict_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    processed: list[dict[str, Any]] = []
    for row in missing_rows:
        script_path = rel(row.get("relative_path") or row.get("path"))
        target = ROOT / script_path
        facts = script_facts(target)
        text, full_read = read_script_text(target)
        facts["full_file_read"] = full_read
        metrics = python_metrics(target, text)
        if metrics.get("syntax_error"):
            facts["syntax_error"] = metrics["syntax_error"]
        callers = find_callers(script_path, target.name, row.get("known_uses") or [], corpus)
        role = classify_role(script_path, row)
        inputs, outputs, side_effects = dataflow_hints(text, facts)
        slop, survival, loc, nonblank = score_script(script_path, text, facts, metrics, callers)
        verdict = decide_verdict(script_path, role, slop, survival, facts, metrics)
        audit_row = manifest_row(script_path, verdict, role, callers, inputs, outputs, side_effects, slop, survival, loc, nonblank, facts, row, args.execute)
        if args.execute:
            append_jsonl(DEFAULT_AUDIT_MANIFEST, audit_row)
            if verdict == "LEGACY_CORPSE":
                append_jsonl(DEFAULT_CORPSE_MANIFEST, corpse_row(script_path, callers, slop, survival, loc, facts))
                append_jsonl(DEFAULT_KRAMPUS_QUEUE, krampus_row(script_path))
            elif verdict == "UNKNOWN_HOLD":
                append_jsonl(DEFAULT_UNKNOWN_HOLD, unknown_row(script_path, verdict))
        verdict_counts[verdict] += 1
        role_counts[role] += 1
        processed.append({"path": script_path, "verdict": verdict, "role": role, "slop_score": slop, "survival_score": survival, "loc": loc, "callers": callers[:20]})

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.script_survival_batch_audit.v1",
        "generated_at": now(),
        "execute": args.execute,
        "tickletrunk_scripts": len(rows),
        "already_audited_before": len(audited),
        "processed": len(processed),
        "remaining_after_estimate": max(0, len(rows) - len(audited) - len(processed)),
        "verdict_counts": dict(verdict_counts),
        "role_counts": dict(role_counts),
        "audit_manifest": rel(DEFAULT_AUDIT_MANIFEST),
        "corpse_manifest": rel(DEFAULT_CORPSE_MANIFEST),
        "krampus_queue": rel(DEFAULT_KRAMPUS_QUEUE),
        "unknown_hold_manifest": rel(DEFAULT_UNKNOWN_HOLD),
        "processed_sample": processed[:50],
        "processed_all_paths_count": len(processed),
        "status": "PASS",
    }
    report = OUT / f"script_survival_batch_audit_{stamp()}.json"
    payload["report_path"] = rel(report)
    report.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    (OUT / "script_survival_batch_audit_latest.json").write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("REPORT_PATH=" + rel(report))
    print("SCRIPT_SURVIVAL_BATCH_AUDIT=PASS")
    print("PROCESSED=" + str(len(processed)))
    print("VERDICT_COUNTS=" + json.dumps(dict(verdict_counts), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
