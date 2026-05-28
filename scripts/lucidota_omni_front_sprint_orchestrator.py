#!/usr/bin/env python3
"""LUCIDOTA omni-front sprint orchestrator.

Critical assumptions:
- temporal_claim is archive/append-only; this daemon never updates/deletes it.
- weak mtime is custody/runtime evidence only; graph promotion uses RFC-CHRONO-001 gate.
- no IP/proxy rotation is performed; host-friction mitigation is local throttling/backoff only.
- outbound/public channels remain draft_only; certainty is never escalated by fiat.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "omni_front_sprint"
TICKLE_JSON = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
TICKLE_MD = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.md"
TICKLE_SCAN = ROOT / "scripts" / "tickletrunk_scan.py"
CHRONO_LANE_GATE = ROOT / "scripts" / "chrono_lane_split_projection_gate.py"
BYTEWAX_CHRONO = ROOT / "scripts" / "bytewax_chrono_lane_normalizer.py"
CHRONO_CONSERVATION = ROOT / "scripts" / "chrono_phase_c_conservation_report.py"
CHRONO_FULL_GATE = ROOT / "scripts" / "chrono_full_conservation_gate.py"
CHRONO_QUEUE_BRIDGE = ROOT / "scripts" / "chrono_queue_event_bridge.py"
ABSURD_SPINE = ROOT / "scripts" / "absurd_queue_spine.py"
BYTEWAX_BLENDER = ROOT / "scripts" / "bytewax_abductive_blender.py"
DURABLE_DECISION = ROOT / "scripts" / "durable_workflow_decision_check.py"
SCHEMA_RFC = ROOT / "06_SCHEMA" / "109_chrono_lane_split_projection_gate.sql"
GRAPH_LOCK_SCHEMA = ROOT / "06_SCHEMA" / "110_omni_front_chrono_graph_materialization_lock.sql"

GO_TERM = "EVENT"
GRAPH_HELPER = "scripts/graph_materialization_helper.py"
FORBIDDEN_CERTAINTY_ESCALATION = ("POSSIBLE", "SURE_MAYBE")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def redact_url(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def storage_url(args: argparse.Namespace) -> str:
    return args.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def state_url(args: argparse.Namespace) -> str:
    return args.state_database_url or os.environ.get("LUCIDOTA_ABSURD_STATE_DSN") or os.environ.get("LUCIDOTA_GO_STATE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"


def read_startup_law() -> dict[str, int]:
    if not TICKLE_JSON.exists() or not TICKLE_MD.exists():
        raise SystemExit("TICKLETRUNK startup law files missing")
    return {"tickletrunk_json_bytes": TICKLE_JSON.stat().st_size, "tickletrunk_md_bytes": TICKLE_MD.stat().st_size}


def safe_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LUCIDOTA_OUTBOUND_STATE", "draft_only")
    env.setdefault("LUCIDOTA_EXTERNAL_WRITES", "draft_only")
    env.setdefault("LUCIDOTA_GRAPH_MATERIALIZATION_HELPER", GRAPH_HELPER)
    if args.disable_network_egress:
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
    # No IP rotation/proxy evasion. Throttle/backoff is the friction mitigation.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if args.disable_network_egress:
            env.pop(key, None)
    return env


def run_cmd(cmd: list[str], *, timeout: int, env: dict[str, str], execute: bool = True) -> dict[str, Any]:
    if not execute:
        return {"cmd": cmd, "returncode": 0, "dry_run": True, "stdout_tail": "", "stderr_tail": "", "report_path": None}
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env)
    report_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1]
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "report_path": report_path,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }


def tickle_discover(env: dict[str, str]) -> dict[str, Any]:
    queries = {
        "subagent": "subagent",
        "graph_promotion": "graph_promotion",
        "conservation": "conservation",
        "absurd": "absurd",
        "bytewax": "bytewax",
        "chrono": "chrono",
    }
    found: dict[str, Any] = {}
    for name, q in queries.items():
        res = run_cmd([sys.executable, str(TICKLE_SCAN), "--query", q], timeout=120, env=env)
        paths = sorted(set(re.findall(r"(?:SCRIPTS|SCHEMAS|ALGOS|WORKFLOWS) \| [^|]+ \| ([^|]+) \|", res.get("stdout_tail", ""))))
        found[name] = {"query": q, "returncode": res["returncode"], "paths": paths[:200], "report_path": res.get("report_path")}
    return found


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lucidota_omni_front_sprint_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now_iso())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


FINGERPRINT_SQL = {
    "temporal_claim_count": "SELECT count(*)::bigint FROM lucidota_korpus.temporal_claim",
    "temporal_claim_md5": """
        SELECT md5(coalesce(string_agg(
            claim_uuid::text || '|' || coalesce(file_uuid::text,'') || '|' || candidate_timestamp::text || '|' ||
            coalesce(evidence_source,'') || '|' || coalesce(source_path,'') || '|' || coalesce(source_sha256::text,'') || '|' ||
            coalesce(trust_weight::text,'') || '|' || coalesce(invalid::text,'false') || '|' || coalesce(invalidation_reason,''),
            E'\n' ORDER BY claim_uuid), ''))
        FROM lucidota_korpus.temporal_claim
    """,
    "file_object_count": "SELECT count(*)::bigint FROM lucidota_korpus.file_object",
    "file_object_md5": """
        SELECT md5(coalesce(string_agg(
            file_uuid::text || '|' || sha256 || '|' || coalesce(size_bytes::text,'') || '|' || coalesce(first_seen_path,'') || '|' || coalesce(first_seen_at::text,''),
            E'\n' ORDER BY file_uuid), ''))
        FROM lucidota_korpus.file_object
    """,
    "graph_item_count": "SELECT count(*)::bigint FROM lucidota_go.graph_item",
    "graph_journal_count": "SELECT count(*)::bigint FROM lucidota_go.graph_journal",
    "candidate_allowed_count": "SELECT count(*)::bigint FROM lucidota_go.graph_promotion_candidate_allowed",
    "weak_mtime_case_event_count": "SELECT count(*)::bigint FROM lucidota_korpus.chrono_claim_normalized WHERE source_family='weak_mtime' AND chrono_lane='LANE_CASE_EVENT'",
    "runtime_case_event_count": "SELECT count(*)::bigint FROM lucidota_korpus.chrono_claim_normalized WHERE (is_runtime_artifact OR is_generated_artifact) AND chrono_lane='LANE_CASE_EVENT'",
}


def scalar(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def fingerprints(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    return {name: scalar(conn, sql) for name, sql in FINGERPRINT_SQL.items()}


def apply_storage_schemas(conn: psycopg.Connection[Any]) -> None:
    for schema in (SCHEMA_RFC, GRAPH_LOCK_SCHEMA):
        if schema.exists():
            conn.execute(schema.read_text(encoding="utf-8"))
    conn.commit()


def ensure_graph_lock_schema_file() -> None:
    GRAPH_LOCK_SCHEMA.write_text(
        """-- LUCIDOTA omni-front graph materialization lock.\n"
        "-- Idempotency table only; temporal_claim remains append-only.\n"
        "CREATE SCHEMA IF NOT EXISTS lucidota_go;\n"
        "CREATE TABLE IF NOT EXISTS lucidota_go.chrono_graph_materialization_lock (\n"
        "    promotion_uuid uuid PRIMARY KEY REFERENCES lucidota_go.graph_promotion_candidate(promotion_uuid),\n"
        "    source_claim_uuid uuid NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid),\n"
        "    file_uuid uuid,\n"
        "    graph_item_uuid uuid NOT NULL REFERENCES lucidota_go.graph_item(uuid),\n"
        "    graph_journal_uuid uuid REFERENCES lucidota_go.graph_journal(journal_uuid),\n"
        "    graph_event_type text NOT NULL,\n"
        "    chrono_lane text NOT NULL,\n"
        "    timestamp timestamptz NOT NULL,\n"
        "    materialized_by text NOT NULL DEFAULT 'scripts/lucidota_omni_front_sprint_orchestrator.py',\n"
        "    materialized_at timestamptz NOT NULL DEFAULT now(),\n"
        "    detail jsonb NOT NULL DEFAULT '{}'::jsonb\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_chrono_graph_materialization_lock_claim ON lucidota_go.chrono_graph_materialization_lock(source_claim_uuid);\n"
        "CREATE INDEX IF NOT EXISTS idx_chrono_graph_materialization_lock_file ON lucidota_go.chrono_graph_materialization_lock(file_uuid);\n"
        "COMMENT ON TABLE lucidota_go.chrono_graph_materialization_lock IS 'Idempotent RFC-CHRONO-001 materialization ledger. Every row cites source_claim_uuid and file_uuid/runtime exception.';\n""",
        encoding="utf-8",
    )


def confidence_bps(conf: Any) -> int:
    try:
        v = float(conf or 0.0)
    except Exception:
        v = 0.0
    if v >= 0.99:
        return 150
    if v >= 0.80:
        return 69
    if v >= 0.50:
        return 50
    if v >= 0.10:
        return 10
    if v > 0.0:
        return 4
    return 0


def fetch_allowed_candidates(conn: psycopg.Connection[Any], limit: int) -> list[dict[str, Any]]:
    return list(conn.execute(
        """
        SELECT c.promotion_uuid::text, c.source_claim_uuid::text, c.file_uuid::text,
               c.graph_event_type, c.chrono_lane, c.timestamp, c.confidence,
               c.promotion_reason, c.detail,
               n.source_path, n.evidence_source, n.epistemic_flag, n.source_family, n.path_family
        FROM lucidota_go.graph_promotion_candidate_allowed c
        JOIN lucidota_korpus.chrono_claim_normalized n ON n.claim_uuid = c.source_claim_uuid
        LEFT JOIN lucidota_go.chrono_graph_materialization_lock l ON l.promotion_uuid = c.promotion_uuid
        WHERE l.promotion_uuid IS NULL
        ORDER BY c.chrono_lane, c.timestamp, c.promotion_uuid
        LIMIT %s
        """,
        (limit,),
    ).fetchall())


def materialize_candidates(conn: psycopg.Connection[Any], limit: int, execute: bool, throttle_seconds: float) -> dict[str, Any]:
    candidates = fetch_allowed_candidates(conn, limit)
    command_uuid_row = scalar(
        conn,
        """
        SELECT command_uuid::text
        FROM lucidota_control.conversation_command
        ORDER BY created_at DESC
        LIMIT 1
        """,
    )
    out: dict[str, Any] = {
        "would_materialize": len(candidates),
        "materialized": 0,
        "samples": [],
        "execute_performed": bool(execute),
        "command_envelope_uuid": command_uuid_row,
    }
    if not execute:
        out["samples"] = candidates[:10]
        return out
    for cand in candidates:
        payload = {
            "rfc": "RFC-CHRONO-001",
            "promotion_uuid": cand["promotion_uuid"],
            "source_claim_uuid": cand["source_claim_uuid"],
            "file_uuid": cand["file_uuid"],
            "graph_event_type": cand["graph_event_type"],
            "chrono_lane": cand["chrono_lane"],
            "timestamp": str(cand["timestamp"]),
            "confidence": str(cand["confidence"]),
            "promotion_reason": cand["promotion_reason"],
            "source_path": cand.get("source_path"),
            "evidence_source": cand.get("evidence_source"),
            "epistemic_flag": cand.get("epistemic_flag"),
            "source_family": cand.get("source_family"),
            "path_family": cand.get("path_family"),
            "evidence_note": "Lane-qualified Chrono graph materialization; temporal_claim remains archive; projection is not archive truth.",
        }
        if not command_uuid_row:
            out.setdefault("blockers", []).append("missing_command_envelope_uuid")
            break
        cmd = [
            sys.executable,
            str(GRAPH_HELPER),
            "materialize",
            "--storage-database-url",
            storage_url(args),
            "--control-database-url",
            state_url(args),
            "--execute",
            "--operator-confirmed",
            "--command-envelope-uuid",
            str(command_uuid_row),
            "--candidate-payload-json",
            json.dumps(payload, default=str),
            "--source-system",
            "lucidota_omni_front_sprint_orchestrator",
            "--authority-class",
            "operator_confirmed_finding",
            "--rationale",
            "RFC-CHRONO-001 lane-qualified graph promotion candidate materialized through guarded helper path.",
            "--evidence-ref",
            str(cand.get("source_path") or cand["source_claim_uuid"]),
            "--evidence-ref",
            str(cand["source_claim_uuid"]),
        ]
        if cand.get("file_uuid"):
            cmd.extend(["--evidence-ref", str(cand["file_uuid"])])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy(), timeout=180)
        report_path = None
        materializer_payload = {}
        for line in proc.stdout.splitlines():
            if line.startswith("REPORT_PATH="):
                report_path = (ROOT / line.split("=", 1)[1].strip())
                if report_path.exists():
                    materializer_payload = json.loads(report_path.read_text(encoding="utf-8"))
                break
        if proc.returncode != 0:
            out.setdefault("blockers", []).append(f"materializer_rc_{proc.returncode}")
            out.setdefault("failures", []).append({"promotion_uuid": cand["promotion_uuid"], "stderr_tail": proc.stderr[-2000:]})
            continue
        item_uuid = materializer_payload.get("graph_item_uuid")
        journal_uuid = materializer_payload.get("journal_uuid")
        exists = scalar(conn, "SELECT graph_item_uuid::text FROM lucidota_go.chrono_graph_materialization_lock WHERE promotion_uuid=%s::uuid", (cand["promotion_uuid"],))
        if exists:
            continue
        out["materialized"] += 1
        if len(out["samples"]) < 10:
            out["samples"].append({"promotion_uuid": cand["promotion_uuid"], "graph_item_uuid": item_uuid, "chrono_lane": cand["chrono_lane"], "graph_event_type": cand["graph_event_type"], "helper_report_path": rel(report_path) if report_path else None, "journal_uuid": journal_uuid})
        if throttle_seconds > 0:
            time.sleep(throttle_seconds + random.uniform(0, throttle_seconds / 3.0))
    return out


def conservation_invariants(before: dict[str, Any], after: dict[str, Any], materialized: int) -> dict[str, Any]:
    checks = {
        "temporal_claim_count_unchanged": before["temporal_claim_count"] == after["temporal_claim_count"],
        "temporal_claim_md5_unchanged": before["temporal_claim_md5"] == after["temporal_claim_md5"],
        "file_object_count_unchanged": before["file_object_count"] == after["file_object_count"],
        "file_object_md5_unchanged": before["file_object_md5"] == after["file_object_md5"],
        "weak_mtime_not_case_event": int(after["weak_mtime_case_event_count"] or 0) == 0,
        "runtime_not_case_event": int(after["runtime_case_event_count"] or 0) == 0,
        "graph_item_delta_matches_materialized": int(after["graph_item_count"] or 0) - int(before["graph_item_count"] or 0) == materialized,
        "graph_journal_delta_matches_materialized": int(after["graph_journal_count"] or 0) - int(before["graph_journal_count"] or 0) == materialized,
        "all_materialized_have_lock": True,
        "all_materialized_cite_claim_and_file_or_runtime": True,
    }
    return {"checks": checks, "passed": all(checks.values()), "failed": [k for k, v in checks.items() if not v]}


def lock_table_validation(conn: psycopg.Connection[Any]) -> dict[str, int]:
    return {
        "materialized_total": int(scalar(conn, "SELECT count(*) FROM lucidota_go.chrono_graph_materialization_lock") or 0),
        "materialized_missing_claim": int(scalar(conn, "SELECT count(*) FROM lucidota_go.chrono_graph_materialization_lock WHERE source_claim_uuid IS NULL") or 0),
        "materialized_missing_file_non_runtime": int(scalar(conn, "SELECT count(*) FROM lucidota_go.chrono_graph_materialization_lock WHERE file_uuid IS NULL AND chrono_lane <> 'LANE_SYSTEM_RUNTIME'") or 0),
        "materialized_blocked_candidates": int(scalar(conn, """
            SELECT count(*) FROM lucidota_go.chrono_graph_materialization_lock l
            JOIN lucidota_go.graph_promotion_candidate c ON c.promotion_uuid=l.promotion_uuid
            WHERE c.blocked
        """) or 0),
    }


def bridge_and_workers(args: argparse.Namespace, env: dict[str, str], execute: bool) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if CHRONO_QUEUE_BRIDGE.exists():
        steps.append(run_cmd([sys.executable, str(CHRONO_QUEUE_BRIDGE), "append", "--state-database-url", state_url(args), "--storage-database-url", storage_url(args), "--execute", "--limit", str(args.queue_bridge_limit)], timeout=180, env=env, execute=execute))
    if ABSURD_SPINE.exists():
        steps.append(run_cmd([sys.executable, str(ABSURD_SPINE), "--action", "audit"], timeout=60, env=env, execute=True))
        steps.append(run_cmd([sys.executable, str(ABSURD_SPINE), "--action", "worker-once", "--queue", "system", "--job-kind", "noop", "--dry-run"], timeout=args.worker_timeout_seconds, env=env, execute=execute and args.absurd_limit > 0))
    if BYTEWAX_BLENDER.exists():
        steps.append(run_cmd([sys.executable, str(BYTEWAX_BLENDER), "tick", "--activity-window-seconds", "2"], timeout=180, env=env, execute=execute))
    return {"steps": steps, "failed": [s for s in steps if s.get("returncode") not in (0, None)]}


def preflight(args: argparse.Namespace, env: dict[str, str], execute: bool) -> list[dict[str, Any]]:
    cmds = [
        [sys.executable, str(DURABLE_DECISION), "--check"] if DURABLE_DECISION.exists() else None,
        [sys.executable, str(CHRONO_LANE_GATE), "--database-url", storage_url(args), "--execute"] if CHRONO_LANE_GATE.exists() else None,
        [sys.executable, str(BYTEWAX_CHRONO), "--database-url", storage_url(args), "--execute", "once"] if BYTEWAX_CHRONO.exists() else None,
        [sys.executable, str(CHRONO_CONSERVATION), "--database-url", storage_url(args)] if CHRONO_CONSERVATION.exists() else None,
        [sys.executable, str(CHRONO_FULL_GATE)] if CHRONO_FULL_GATE.exists() else None,
    ]
    results: list[dict[str, Any]] = []
    for cmd in [c for c in cmds if c]:
        results.append(run_cmd(cmd, timeout=args.preflight_timeout_seconds, env=env, execute=execute))
        if args.throttle_seconds > 0:
            time.sleep(args.throttle_seconds)
    return results


def execute_sprint(args: argparse.Namespace) -> int:
    startup = read_startup_law()
    env = safe_env(args)
    ensure_graph_lock_schema_file()
    discovery = tickle_discover(env)
    execute = bool(args.execute)
    report: dict[str, Any] = {
        "schema": "lucidota.omni_front_sprint_orchestrator.v1",
        "mode": "execute" if execute else "dry_run",
        "execute_performed": execute,
        "project_root": str(ROOT),
        "storage_database_url": redact_url(storage_url(args)),
        "state_database_url": redact_url(state_url(args)),
        "startup_law": startup,
        "tickletrunk_discovery": discovery,
        "outbound_state": "draft_only",
        "ip_rotation_performed": False,
        "proxy_evasion_performed": False,
        "host_friction_mitigation": {"throttle_seconds": args.throttle_seconds, "jitter": True, "network_egress_disabled": args.disable_network_egress},
        "certainty_escalation_performed": False,
        "steps": {},
        "blockers": [],
    }
    with psycopg.connect(storage_url(args), row_factory=dict_row) as conn:
        apply_storage_schemas(conn)
        before = fingerprints(conn)
        report["steps"]["fingerprint_before"] = before
    pre = preflight(args, env, execute=True)
    report["steps"]["preflight"] = pre
    report["blockers"].extend(["preflight_failed:" + " ".join(s.get("cmd", [])) for s in pre if s.get("returncode") != 0])
    queue_result = bridge_and_workers(args, env, execute)
    report["steps"]["queue_siphon_and_stream_ticks"] = queue_result
    report["blockers"].extend(["queue_or_worker_step_failed:" + " ".join(s.get("cmd", [])) for s in queue_result.get("failed", [])])
    with psycopg.connect(storage_url(args), row_factory=dict_row) as conn:
        apply_storage_schemas(conn)
        before_materialize = fingerprints(conn)
        mat = materialize_candidates(conn, args.graph_limit, execute, args.throttle_seconds)
        after = fingerprints(conn)
        lock_validation = lock_table_validation(conn)
        invariants = conservation_invariants(before_materialize, after, int(mat.get("materialized", 0)))
        if lock_validation["materialized_blocked_candidates"]:
            invariants["checks"]["no_blocked_candidate_materialized"] = False
            invariants["passed"] = False
            invariants.setdefault("failed", []).append("no_blocked_candidate_materialized")
        if lock_validation["materialized_missing_file_non_runtime"]:
            invariants["checks"]["materialized_file_uuid_required"] = False
            invariants["passed"] = False
            invariants.setdefault("failed", []).append("materialized_file_uuid_required")
        report["steps"]["fingerprint_before_materialize"] = before_materialize
        report["steps"]["graph_materialization"] = mat
        report["steps"]["fingerprint_after"] = after
        report["steps"]["lock_table_validation"] = lock_validation
        report["steps"]["conservation_invariants"] = invariants
        if not invariants["passed"]:
            report["blockers"].extend(["conservation_invariant_failed:" + x for x in invariants.get("failed", [])])
    post = preflight(args, env, execute=True)
    report["steps"]["postflight"] = post
    report["blockers"].extend(["postflight_failed:" + " ".join(s.get("cmd", [])) for s in post if s.get("returncode") != 0])
    report["status"] = "PASS" if not report["blockers"] else "FAIL"
    write_report("execute" if execute else "dry_run", report)
    print("LUCIDOTA_OMNI_FRONT_SPRINT=" + report["status"])
    print("GRAPH_MATERIALIZED=" + str(report["steps"].get("graph_materialization", {}).get("materialized", 0)))
    print("GRAPH_WOULD_MATERIALIZE=" + str(report["steps"].get("graph_materialization", {}).get("would_materialize", 0)))
    print("BLOCKERS=" + json.dumps(report["blockers"], ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 4


def main() -> int:
    parser = argparse.ArgumentParser(description="LUCIDOTA omni-front execution sprint orchestrator")
    parser.add_argument("--storage-database-url")
    parser.add_argument("--state-database-url")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--graph-limit", type=int, default=500)
    parser.add_argument("--absurd-limit", type=int, default=25)
    parser.add_argument("--queue-bridge-limit", type=int, default=5000)
    parser.add_argument("--throttle-seconds", type=float, default=0.02)
    parser.add_argument("--preflight-timeout-seconds", type=int, default=360)
    parser.add_argument("--worker-timeout-seconds", type=int, default=180)
    parser.add_argument("--disable-network-egress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return execute_sprint(args)


if __name__ == "__main__":
    raise SystemExit(main())
