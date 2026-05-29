#!/usr/bin/env python3
"""Project 2501 audit-lane script classifier: script -> manifest/corpse evidence -> receipt."""
from __future__ import annotations

import argparse
import ast
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from project2501_board_move import build_board_move, persist_bundle, redacted, stable_json
from script_survival_audit import read_text, script_facts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "project2501_script_audit"
CORE_SCHEMA = ROOT / "06_SCHEMA" / "112_project2501_core_board.sql"
SCHEMA = ROOT / "06_SCHEMA" / "115_project2501_script_audit_worker.sql"
SCAN_ROOTS = ("scripts", "tests", "00_PROJECT_BRAIN", "06_SCHEMA", "GOALS")
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".sql", ".sh", ".toml", ".yaml", ".yml", ".txt"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_script_audit_{stamp()}.json"
    payload["report_path"] = rel(path)
    if isinstance(payload.get("watch_metric"), dict):
        payload["watch_metric"]["source_receipt"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def apply_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(CORE_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def source_text(path: Path, limit: int = 512 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return read_text(path, limit)


def text_counts(text: str) -> dict[str, int]:
    lines = text.splitlines()
    return {"loc": len(lines), "nonblank_loc": sum(1 for line in lines if line.strip())}


def facts_for_script(path: Path) -> dict[str, Any]:
    facts = script_facts(path)
    facts.update(text_counts(source_text(path)))
    return facts


def module_purpose(path: Path) -> str:
    text = source_text(path, 64 * 1024)
    if path.suffix == ".py" and text:
        try:
            doc = ast.get_docstring(ast.parse(text))
            if doc:
                return doc.strip().splitlines()[0][:240]
        except SyntaxError:
            return "python syntax error; quarantine for inspection"
    if "scripts/legacy/" in rel(path):
        return "legacy script preserved as historical evidence; not active authority"
    return "script purpose inferred by Project 2501 script audit"


def iter_scan_files(max_files: int = 2500):
    count = 0
    for root_name in SCAN_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if count >= max_files:
                return
            if path.is_file() and path.suffix in TEXT_SUFFIXES and path.stat().st_size <= 768 * 1024:
                count += 1
                yield path


def find_callers(target: Path, *, max_callers: int = 32) -> list[str]:
    target_rel = rel(target)
    needles = {target_rel, target.name}
    callers: list[str] = []
    for path in iter_scan_files():
        path_rel = rel(path)
        if path_rel == target_rel:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if any(needle in text for needle in needles):
            callers.append(path_rel)
            if len(callers) >= max_callers:
                break
    return sorted(dict.fromkeys(callers))


def receipt_contract(facts: dict[str, Any], text: str, callers: list[str]) -> dict[str, Any]:
    writes_receipt = "REPORT_PATH=" in text or "write_report" in text or "05_OUTPUTS" in text
    return {
        "writes_receipt": bool(writes_receipt),
        "has_cli_flags": bool(facts.get("argparse_flags")),
        "has_tests": any(c.startswith("tests/") for c in callers),
        "db_writes_possible": "psycopg" in facts.get("imports", []) or "INSERT INTO" in text.upper(),
        "verification_path": [c for c in callers if c.startswith("tests/")][:8],
    }


def score_and_verdict(path: Path, facts: dict[str, Any], callers: list[str], contract: dict[str, Any]) -> tuple[int, int, str, list[str]]:
    reasons: list[str] = []
    score = 50
    path_rel = rel(path)
    if not facts.get("exists"):
        score -= 55; reasons.append("missing_file")
    if "scripts/legacy/" in path_rel:
        score -= 45; reasons.append("legacy_path_retired_from_active_trust")
    if callers:
        score += 15; reasons.append("callers_found")
    else:
        score -= 15; reasons.append("no_callers_found")
    if contract["has_tests"]:
        score += 15; reasons.append("test_caller_found")
    if contract["writes_receipt"]:
        score += 15; reasons.append("receipt_contract_detected")
    if facts.get("has_module_docstring"):
        score += 5; reasons.append("module_docstring_present")
    if facts.get("syntax_error"):
        score -= 25; reasons.append("syntax_error")
    loc = int(facts.get("nonblank_loc") or 0)
    if loc > 500:
        score -= 10; reasons.append("large_active_surface")
    score = max(0, min(100, score))
    slop = max(0, min(100, 100 - score + (15 if loc > 500 else 0)))
    if not facts.get("exists") or facts.get("syntax_error"):
        verdict = "QUARANTINE"
    elif "scripts/legacy/" in path_rel or score < 30:
        score = min(score, 35)
        verdict = "CORPSE_MANIFEST"
    elif score >= 70 and contract["writes_receipt"]:
        verdict = "KEEP"
    elif score >= 45:
        verdict = "WRAP"
    else:
        verdict = "REWRITE"
    return score, slop, verdict, reasons


def classify_script(path: Path | str, *, max_callers: int = 32) -> dict[str, Any]:
    target = resolve(path)
    facts = facts_for_script(target)
    text = source_text(target)
    callers = find_callers(target, max_callers=max_callers)
    contract = receipt_contract(facts, text, callers)
    survival, slop, verdict, reasons = score_and_verdict(target, facts, callers, contract)
    path_rel = rel(target)
    purpose = module_purpose(target)
    caller = ";".join(callers)
    manifest = {
        "script_path": path_rel,
        "caller": caller,
        "purpose": purpose,
        "receipt_contract": contract,
        "survival_score": survival,
        "verdict": verdict,
        "detail": {"facts": facts, "callers": callers, "slop_score": slop, "reasons": reasons, "classifier": "project2501_script_audit_worker.v1"},
    }
    corpse = None
    sha = facts.get("hash_blake3_or_sha256")
    if verdict in {"CORPSE_MANIFEST", "KRAMPUSCHEW"}:
        corpse = {
            "original_path": path_rel,
            "corpse_ref": f"script://{path_rel}@sha256:{sha or 'missing'}",
            "reason": "; ".join(reasons) or "classified for corpse preservation",
            "source_sha256": sha,
            "krampuschewing_status": "queued",
            "detail": {
                "never_delete": True,
                "historical_value": "preserve as evidence and reusable prior",
                "risk_if_kept_active": "active trust may mislead routing or duplicate current workers",
            },
        }
    metric = {
        "metric_key": "project2501_script_audit_once",
        "metric_value": {"script_path": path_rel, "verdict": verdict, "survival_score": survival, "slop_score": slop},
        "source_receipt": "",
        "source_db_ref": f"lucidota_control.script_manifest:{path_rel}",
        "operator_requested": True,
        "operator_feature_authority_required": True,
    }
    return {
        "schema": "lucidota.project2501.script_audit.result.v1",
        "status": "PASS",
        "script_manifest": manifest,
        "corpse_manifest": corpse,
        "watch_metric": metric,
        "canonical_graph_writes_performed": False,
    }


def select_batch_candidates(*, limit: int, min_nonblank_loc: int, max_files: int = 2500) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in iter_scan_files(max_files=max_files):
        path_rel = rel(path)
        if not path_rel.startswith("scripts/") or "/legacy/" in path_rel or not path_rel.endswith(".py"):
            continue
        facts = facts_for_script(path)
        nonblank = int(facts.get("nonblank_loc") or 0)
        if nonblank >= min_nonblank_loc:
            candidates.append({"script_path": path_rel, "nonblank_loc": nonblank, "size_bytes": facts.get("size_bytes") or 0})
    candidates.sort(key=lambda row: (-int(row["nonblank_loc"]), row["script_path"]))
    return candidates[: max(0, int(limit))]


def summarize_classifications(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in classifications:
        verdict = item["script_manifest"]["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
    return {
        "total": len(classifications),
        "verdict_counts": counts,
        "scripts": [item["script_manifest"]["script_path"] for item in classifications],
    }


def build_batch_result(*, paths: list[Path | str], execute: bool, max_callers: int = 32) -> dict[str, Any]:
    started = time.perf_counter()
    classifications = [classify_script(path, max_callers=max_callers) for path in paths]
    summary = summarize_classifications(classifications)
    metric = {
        "metric_key": "project2501_script_audit_batch",
        "metric_value": summary,
        "source_receipt": "",
        "source_db_ref": "lucidota_control.script_audit_run:batch",
        "operator_requested": True,
        "operator_feature_authority_required": True,
    }
    return {
        "schema": "lucidota.project2501.script_audit.batch.v1",
        "status": "PASS",
        "generated_at": now(),
        "execute_performed": bool(execute),
        "classifications": classifications,
        "summary": summary,
        "watch_metric": metric,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "canonical_graph_writes_performed": False,
        "db_rows": [],
    }


def persist_batch_result(result: dict[str, Any], database_url: str, receipt_path: str) -> list[dict[str, Any]]:
    apply_schema(database_url)
    rows: list[dict[str, Any]] = []
    for item in result["classifications"]:
        item["watch_metric"]["source_receipt"] = receipt_path
        rows.append(persist_classification(item, database_url, receipt_path))
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            metric = result["watch_metric"]
            cur.execute(
                """
                INSERT INTO lucidota_control.watch_metric(metric_key, metric_value, source_receipt, source_db_ref, operator_requested, operator_feature_authority_required)
                VALUES (%s,%s::jsonb,%s,%s,%s,%s)
                RETURNING metric_uuid::text
                """,
                (
                    metric["metric_key"],
                    json.dumps(metric["metric_value"]),
                    receipt_path,
                    metric["source_db_ref"],
                    metric["operator_requested"],
                    metric["operator_feature_authority_required"],
                ),
            )
            rows.append({"batch_metric_uuid": cur.fetchone()[0]})
        conn.commit()
    return rows


def persist_classification(result: dict[str, Any], database_url: str, receipt_path: str) -> dict[str, str | None]:
    apply_schema(database_url)
    manifest = result["script_manifest"]
    corpse = result["corpse_manifest"]
    move = build_board_move(
        actor="auditor",
        source="script_audit_worker",
        position="script_survival_audit",
        execute=True,
        text=f"Audit script {manifest['script_path']} verdict {manifest['verdict']} survival {manifest['survival_score']} emit manifest/corpse receipt only.",
    )
    move.update({"generated_at": now(), "execute_performed": True, "database_url": redacted(database_url), "status": "PASS", "db_rows": {}})
    move["work_receipt"]["receipt_path"] = receipt_path
    move["board_move"]["receipt"] = receipt_path
    move_rows = persist_bundle(move, database_url)
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.script_manifest(script_path, caller, purpose, receipt_contract, survival_score, verdict, detail)
                VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT(script_path) DO UPDATE SET
                  caller=EXCLUDED.caller,
                  purpose=EXCLUDED.purpose,
                  receipt_contract=EXCLUDED.receipt_contract,
                  survival_score=EXCLUDED.survival_score,
                  verdict=EXCLUDED.verdict,
                  detail=EXCLUDED.detail,
                  updated_at=now()
                RETURNING script_manifest_uuid::text
                """,
                (
                    manifest["script_path"],
                    manifest["caller"],
                    manifest["purpose"],
                    json.dumps(manifest["receipt_contract"]),
                    manifest["survival_score"],
                    manifest["verdict"],
                    json.dumps(manifest["detail"]),
                ),
            )
            script_manifest_uuid = cur.fetchone()[0]
            corpse_uuid = None
            if corpse:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.corpse_manifest(original_path, corpse_ref, reason, source_sha256, krampuschewing_status, detail)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(corpse_ref) DO UPDATE SET
                      reason=EXCLUDED.reason,
                      krampuschewing_status=EXCLUDED.krampuschewing_status,
                      detail=EXCLUDED.detail
                    RETURNING corpse_uuid::text
                    """,
                    (
                        corpse["original_path"],
                        corpse["corpse_ref"],
                        corpse["reason"],
                        corpse["source_sha256"],
                        corpse["krampuschewing_status"],
                        json.dumps(corpse["detail"]),
                    ),
                )
                corpse_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.script_audit_run(event_id, work_receipt_uuid, script_manifest_uuid, corpse_uuid, script_path, verdict, survival_score, slop_score, source_receipt, detail)
                VALUES (%s,%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s::jsonb)
                RETURNING script_audit_run_uuid::text
                """,
                (
                    move["event_envelope"]["event_id"],
                    move_rows["work_receipt_uuid"],
                    script_manifest_uuid,
                    corpse_uuid,
                    manifest["script_path"],
                    manifest["verdict"],
                    manifest["survival_score"],
                    int(manifest["detail"]["slop_score"]),
                    receipt_path,
                    json.dumps({"reasons": manifest["detail"]["reasons"], "callers": manifest["detail"]["callers"]}),
                ),
            )
            audit_run_uuid = cur.fetchone()[0]
            metric = result["watch_metric"]
            cur.execute(
                """
                INSERT INTO lucidota_control.watch_metric(metric_key, metric_value, source_receipt, source_db_ref, operator_requested, operator_feature_authority_required)
                VALUES (%s,%s::jsonb,%s,%s,%s,%s)
                RETURNING metric_uuid::text
                """,
                (
                    metric["metric_key"],
                    json.dumps(metric["metric_value"]),
                    receipt_path,
                    metric["source_db_ref"],
                    metric["operator_requested"],
                    metric["operator_feature_authority_required"],
                ),
            )
            metric_uuid = cur.fetchone()[0]
        conn.commit()
    return {
        **move_rows,
        "script_manifest_uuid": script_manifest_uuid,
        "corpse_uuid": corpse_uuid,
        "script_audit_run_uuid": audit_run_uuid,
        "metric_uuid": metric_uuid,
    }


def cmd_classify(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result = classify_script(args.path, max_callers=args.max_callers)
    result.update({"generated_at": now(), "execute_performed": bool(args.execute), "database_url": redacted(db_url(args)), "db_rows": {}})
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    path = write_report(result)
    if args.execute:
        try:
            result["watch_metric"]["source_receipt"] = rel(path)
            result["db_rows"] = persist_classification(result, db_url(args), rel(path))
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        except Exception as exc:
            result["status"] = "FAIL"
            result["error"] = f"{type(exc).__name__}:{exc}"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return result


def cmd_batch(args: argparse.Namespace) -> dict[str, Any]:
    if args.paths:
        paths = [Path(p.strip()) for value in args.paths for p in value.split(";") if p.strip()]
    else:
        paths = [row["script_path"] for row in select_batch_candidates(limit=args.limit, min_nonblank_loc=args.min_nonblank_loc, max_files=args.max_files)]
    result = build_batch_result(paths=paths, execute=bool(args.execute), max_callers=args.max_callers)
    result.update({"database_url": redacted(db_url(args)), "candidate_count": len(paths), "candidate_policy": {"limit": args.limit, "min_nonblank_loc": args.min_nonblank_loc, "max_files": args.max_files}})
    path = write_report(result)
    if args.execute:
        try:
            result["watch_metric"]["source_receipt"] = rel(path)
            result["db_rows"] = persist_batch_result(result, db_url(args), rel(path))
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        except Exception as exc:
            result["status"] = "FAIL"
            result["error"] = f"{type(exc).__name__}:{exc}"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return result


def cmd_init_schema(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "lucidota.project2501.script_audit.init_schema.v1",
        "status": "PASS",
        "generated_at": now(),
        "execute_performed": bool(args.execute),
        "database_url": redacted(db_url(args)),
        "schema_paths": [rel(CORE_SCHEMA), rel(SCHEMA)],
        "canonical_graph_writes_performed": False,
    }
    if args.execute:
        apply_schema(db_url(args))
    write_report(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Project 2501 audit-lane script survival classifier.")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-schema")
    init.add_argument("--execute", action="store_true")
    init.add_argument("--json", action="store_true")
    classify = sub.add_parser("classify")
    classify.add_argument("--path", required=True)
    classify.add_argument("--max-callers", type=int, default=32)
    classify.add_argument("--execute", action="store_true")
    classify.add_argument("--json", action="store_true")
    batch = sub.add_parser("batch")
    batch.add_argument("--limit", type=int, default=5)
    batch.add_argument("--min-nonblank-loc", type=int, default=400)
    batch.add_argument("--max-files", type=int, default=2500)
    batch.add_argument("--max-callers", type=int, default=32)
    batch.add_argument("--paths", action="append", help="Optional semicolon-separated explicit script paths.")
    batch.add_argument("--execute", action="store_true")
    batch.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "init-schema":
        payload = cmd_init_schema(args)
    elif args.cmd == "batch":
        payload = cmd_batch(args)
    else:
        payload = cmd_classify(args)
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True, default=str))
    print("PROJECT2501_SCRIPT_AUDIT=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
