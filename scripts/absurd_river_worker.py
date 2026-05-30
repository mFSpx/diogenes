#!/usr/bin/env python3
"""ABSURD queue-spine wrapper for River/Bytewax and Phase 0.5 GLiNER extraction.

Primary queue for this pass:
  absurd.phase05_streaming_brain / gliner_zero_shot_extract

Safety laws:
- Writes ABSURD job/event/workflow receipts and lucidota_learning GLiNER staging rows only.
- Does not mutate Chrono temporal claims.
- Does not mutate KORPUS custody rows.
- Does not write canonical graph tables; spans remain graph-promotion candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import shutil
import socket
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from absurd_worker_contracts import gate_worker_payload_hygiene, record_worker_contract_rejection, validate_worker_contract

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "absurd"
STATE_SCHEMA = ROOT / "06_SCHEMA" / "035_absurd_queue_spine.sql"
REAL_WORK_LOOP_SCHEMA = ROOT / "06_SCHEMA" / "039_absurd_real_work_loop.sql"
LEARNING_SCHEMA = ROOT / "06_SCHEMA" / "004_learning_reflex.sql"
BYTEWAX_SCHEMA = ROOT / "06_SCHEMA" / "007_bytewax_stream.sql"
RIVER_SCHEMA = ROOT / "06_SCHEMA" / "038_absurd_river_wrapper.sql"
CLAIM_PACKET_SCHEMA = ROOT / "06_SCHEMA" / "073_absurd_river_claim_packet_job.sql"
LABEL_FIXTURE = ROOT / "05_OUTPUTS" / "contracts" / "operator_ontology_labels.json"
RIVER_SCRIPT = ROOT / "scripts" / "lucidota_river_reflex.py"
BYTEWAX_SCRIPT = ROOT / "scripts" / "lucidota_bytewax_mini.py"
STREAM_WORKER = ROOT / "scripts" / "lucidota_stream_river_worker.sh"
RUNTIME_LOG = ROOT / "04_RUNTIME" / "lucidota_stream_river_worker.log"
DEFAULT_GLINER_MODEL = ROOT / "03_VAULT" / "models" / "gliner" / "urchade_gliner_small-v2.1"
DEFAULT_QUEUE = "absurd.phase05_streaming_brain"
LEGACY_QUEUE = "river"
DEFAULT_JOB_KIND = "gliner_zero_shot_extract"
CLAIM_PACKET_JOB_KIND = "gliner_claim_packet_extract"
LEGACY_JOB_KIND = "river_bytewax_health_check"
DEFAULT_WORKFLOW = "absurd-phase05-streaming-brain-gliner-extract"
CLAIM_PACKET_WORKFLOW = "absurd-phase05-streaming-brain-claim-packet-extract"
LEGACY_WORKFLOW = "absurd-river-bytewax-health-check"
WORKER_KEY = "river_worker"
LEGACY_WORKER_KEY = "river_legacy_worker"
CANONICAL_GRAPH_TABLES = ["lucidota_go.graph_item", "lucidota_go.graph_edge", "lucidota_go.graph_journal"]
TEMPORAL_TABLES = ["lucidota_korpus.temporal_claim"]
LEARNING_TABLES = [
    "lucidota_control.workflow_event",
    "lucidota_learning.river_event_cursor",
    "lucidota_learning.river_score",
    "lucidota_learning.river_run",
    "lucidota_learning.bytewax_stream_run",
    "lucidota_learning.bytewax_hint",
    "lucidota_learning.gliner_extraction_run",
    "lucidota_learning.gliner_entity_span",
]
INSTALL_COMMAND = "pip install gliner"
MAX_COMPONENT_LIMIT = 100
MAX_CLAIM_PACKET_LIMIT = 1000
MAX_BYTEWAX_LIMIT = 1000
MAX_RIVER_LIMIT = 100000
MAX_MIN_CHARS = 20000


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(dumps(obj).encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def first_value(row: Any) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def redact(url: str | None) -> str:
    if not url:
        return "unset"
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def state_url(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> str:
    if args.state_database_url:
        return args.state_database_url
    if payload and payload.get("state_database_url"):
        return str(payload["state_database_url"])
    return os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def storage_url(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> str:
    if args.storage_database_url:
        return args.storage_database_url
    if payload and payload.get("storage_database_url"):
        return str(payload["storage_database_url"])
    return os.environ.get("KORPUS_DATABASE_URL") or "postgresql:///lucidota_storage"


def python_bin(args: argparse.Namespace) -> str:
    if args.python:
        return args.python
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return shutil.which("python3") or sys.executable


def workflow_for(job_kind: str) -> str:
    if job_kind == CLAIM_PACKET_JOB_KIND:
        return CLAIM_PACKET_WORKFLOW
    return LEGACY_WORKFLOW if job_kind == LEGACY_JOB_KIND else DEFAULT_WORKFLOW


def worker_key_for_queue(queue_name: str) -> str:
    return LEGACY_WORKER_KEY if queue_name == LEGACY_QUEUE else WORKER_KEY


def count_table(conn, table: str) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if first_value(cur.fetchone()) is None:
                return None
            cur.execute(f"SELECT count(*) FROM {table}")
            return int(first_value(cur.fetchone()))
    except Exception:
        return None


def count_tables(conn, tables: list[str]) -> dict[str, int | None]:
    return {table: count_table(conn, table) for table in tables}


def write_report(action: str, report: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"absurd_river_wrapper_{action}_{stamp()}.json"
    report["report_path"] = str(out.relative_to(ROOT))
    out.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return out


def bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except Exception as exc:
        raise ValueError(f"{name}_must_be_int") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name}_out_of_range:{minimum}..{maximum}")
    return number


def run_cmd(cmd: list[str], env: dict[str, str], timeout: int = 180) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    parsed_json: Any = None
    if proc.stdout.strip():
        try:
            parsed_json = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            parsed_json = None
    return {"cmd": [str(x) for x in cmd], "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:], "parsed_json": parsed_json}


def dependency_probe(py: str) -> dict[str, Any]:
    probe = """import json\nmods=['psycopg','river','bytewax','gliner']\nout={}\nfor m in mods:\n    try:\n        __import__(m); out[m]='ok'\n    except Exception as exc:\n        out[m]=type(exc).__name__+': '+str(exc)\nprint(json.dumps(out, sort_keys=True))\n"""
    proc = subprocess.run([py, "-c", probe], cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True, timeout=60)
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except Exception:
        parsed = {}
    return {"python": py, "returncode": proc.returncode, "modules": parsed, "stderr_tail": proc.stderr[-1000:]}


def ensure_state_schema(conn) -> None:
    with conn.cursor() as cur:
        for schema in (STATE_SCHEMA, REAL_WORK_LOOP_SCHEMA, LEARNING_SCHEMA, BYTEWAX_SCHEMA, RIVER_SCHEMA, CLAIM_PACKET_SCHEMA):
            cur.execute(schema.read_text(encoding="utf-8"))


def load_labels(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> list[str]:
    raw = args.labels or (payload or {}).get("labels")
    if raw:
        if isinstance(raw, list):
            return [str(x) for x in raw if str(x).strip()]
        return [part.strip() for part in str(raw).split(",") if part.strip()]
    if LABEL_FIXTURE.exists():
        data = json.loads(LABEL_FIXTURE.read_text(encoding="utf-8"))
        return [str(x) for x in data.get("required_exact_labels", [])]
    return ["Operator", "KORPUS", "Chrono-Ledger", "Command Envelope Protocol"]


def require_gliner_ready(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import gliner  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"gliner_dependency_missing:{type(exc).__name__}:{exc}") from exc
    if not args.gliner_model:
        raise RuntimeError("gliner_model_required")
    model_path = Path(str(args.gliner_model))
    if not model_path.exists() and not args.allow_remote_model:
        raise RuntimeError(f"gliner_model_path_missing:{args.gliner_model}")
    return {"backend": "gliner", "model": args.gliner_model, "threshold": args.gliner_threshold}


def run_gliner_strict(text: str, labels: list[str], args: argparse.Namespace) -> tuple[list[Span], dict[str, Any]]:
    detail = require_gliner_ready(args)
    try:
        from transformers.utils import logging as hf_logging  # type: ignore
        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore", message="The sentencepiece tokenizer.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Sentence of length .* has been truncated.*", category=UserWarning)
        from gliner import GLiNER  # type: ignore
        model = GLiNER.from_pretrained("03_VAULT/models/gliner/urchade_gliner_small-v2.1")
        ents = model.predict_entities(text, labels, threshold=args.gliner_threshold)
    except Exception as exc:
        raise RuntimeError(f"gliner_runtime_failed:{type(exc).__name__}:{exc}") from exc
    spans: list[Span] = []
    for ent in ents:
        start = int(ent.get("start", ent.get("start_pos", 0)))
        end = int(ent.get("end", ent.get("end_pos", start)))
        label = str(ent.get("label", ent.get("class", "")))
        matched = str(ent.get("text", text[start:end]))
        score = float(ent.get("score", ent.get("confidence", 0.0)))
        if 0 <= start <= end <= len(text) and label:
            spans.append(Span(start, end, matched, label, score, "gliner"))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label)), detail

def load_korpus_components(args: argparse.Namespace, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    limit = bounded_int((payload or {}).get("component_limit") or args.component_limit, name="component_limit", minimum=1, maximum=MAX_COMPONENT_LIMIT)
    min_chars = bounded_int((payload or {}).get("min_chars") or args.min_chars, name="min_chars", minimum=1, maximum=MAX_MIN_CHARS)
    db = storage_url(args, payload)
    with psycopg.connect(db, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.component_uuid::text, c.file_uuid::text, c.component_index, c.sha256 AS component_sha256,
                       c.content, f.sha256 AS file_sha256, COALESCE(f.locked_relative_path, f.first_seen_path, '') AS source_path,
                       f.status AS file_status, c.created_at::text AS component_created_at
                FROM lucidota_korpus.component c
                JOIN lucidota_korpus.file_object f ON f.file_uuid = c.file_uuid
                WHERE c.content IS NOT NULL
                  AND length(c.content) >= %s
                  AND COALESCE(f.status, '') IN ('indexed', 'digested', 'ready', 'active')
                ORDER BY c.created_at DESC, c.component_uuid DESC
                LIMIT %s
            """, (min_chars, limit))
            return [dict(r) for r in cur.fetchall()]


def river_bytewax_health(args: argparse.Namespace, payload: dict[str, Any] | None = None, run_tick: bool = False) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    db = state_url(args, payload)
    py = python_bin(args)
    dep = dependency_probe(py)
    for mod in ("psycopg", "river", "bytewax"):
        if dep.get("modules", {}).get(mod) != "ok":
            blockers.append(f"python_dependency_missing:{mod}")
    for path, key in [(RIVER_SCRIPT, "river_reflex_script_missing"), (BYTEWAX_SCRIPT, "bytewax_mini_script_missing"), (STREAM_WORKER, "stream_worker_script_missing")]:
        if not path.exists():
            blockers.append(key)
    before = after = graph_before = graph_after = temporal_before = temporal_after = {}
    command_results: dict[str, Any] = {}
    bytewax_limit = river_limit = None
    try:
        bytewax_limit = bounded_int((payload or {}).get("bytewax_limit") or args.bytewax_limit, name="bytewax_limit", minimum=1, maximum=MAX_BYTEWAX_LIMIT)
        river_limit = bounded_int((payload or {}).get("river_limit") or args.river_limit, name="river_limit", minimum=1, maximum=MAX_RIVER_LIMIT)
    except ValueError as exc:
        blockers.append(str(exc).split(":", 1)[0])
    try:
        with psycopg.connect(db) as conn:
            ensure_state_schema(conn)
            before = count_tables(conn, LEARNING_TABLES)
            graph_before = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_before = count_tables(conn, TEMPORAL_TABLES)
            conn.commit()
    except Exception as exc:
        blockers.append(f"state_db_unreachable:{exc}")
    if run_tick and not blockers:
        env = os.environ.copy(); env["ABSURD_SYSTEM_DATABASE_URL"] = db; env["LUCIDOTA_HOME"] = str(ROOT)
        command_results["bytewax"] = run_cmd([py, str(BYTEWAX_SCRIPT), "--json", "--live-cursor", "--limit", str(bytewax_limit)], env)
        if command_results["bytewax"]["returncode"] != 0: blockers.append("bytewax_tick_failed")
        command_results["river"] = run_cmd([py, str(RIVER_SCRIPT), "--json", "--limit", str(river_limit)], env)
        if command_results["river"]["returncode"] != 0: blockers.append("river_tick_failed")
    try:
        with psycopg.connect(db) as conn:
            after = count_tables(conn, LEARNING_TABLES)
            graph_after = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_after = count_tables(conn, TEMPORAL_TABLES)
    except Exception as exc:
        blockers.append(f"state_db_postcheck_failed:{exc}")
    graph_writes = graph_before != graph_after
    temporal_writes = temporal_before != temporal_after
    if graph_writes: blockers.append("canonical_graph_counts_changed")
    if temporal_writes: blockers.append("temporal_claim_counts_changed")
    return {"mode": "river_bytewax_health", "dependency_probe": dep, "bytewax_limit": bytewax_limit, "river_limit": river_limit, "learning_counts_before": before, "learning_counts_after": after, "command_results": command_results, "canonical_graph_counts_before": graph_before, "canonical_graph_counts_after": graph_after, "temporal_counts_before": temporal_before, "temporal_counts_after": temporal_after, "learning_writes_performed": bool(run_tick), "canonical_graph_writes_performed": graph_writes, "temporal_claims_mutated_by_wrapper": temporal_writes}, blockers


def gliner_streaming_brain(args: argparse.Namespace, payload: dict[str, Any] | None, job_uuid: str | None, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    labels = load_labels(args, payload)
    state_db = state_url(args, payload)
    storage_db = storage_url(args, payload)
    components: list[dict[str, Any]] = []
    graph_before = graph_after = temporal_before = temporal_after = {}
    learning_before = learning_after = {}
    spans_payload: list[dict[str, Any]] = []
    backend_detail: dict[str, Any] = {}
    try:
        backend_detail = require_gliner_ready(args)
    except RuntimeError as exc:
        blockers.append(str(exc).split(":", 1)[0])
        backend_detail = {"backend": "gliner_unavailable", "error": str(exc), "install_command": INSTALL_COMMAND}
    try:
        with psycopg.connect(state_db) as conn:
            ensure_state_schema(conn); conn.commit()
            learning_before = count_tables(conn, LEARNING_TABLES)
    except Exception as exc:
        blockers.append(f"state_db_unreachable:{exc}")
    try:
        with psycopg.connect(storage_db) as conn:
            graph_before = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_before = count_tables(conn, TEMPORAL_TABLES)
        components = load_korpus_components(args, payload)
    except Exception as exc:
        blockers.append(f"storage_db_unreachable:{exc}")
    if components and not blockers:
        for comp in components:
            text = comp.get("content") or ""
            spans, detail = run_gliner_strict(text, labels, args)
            backend_detail = detail if not backend_detail else backend_detail
            for span in spans:
                item = asdict(span)
                item.update({
                    "file_uuid": comp["file_uuid"],
                    "component_uuid": comp["component_uuid"],
                    "component_index": comp.get("component_index"),
                    "source_sha256": comp.get("component_sha256") or comp.get("file_sha256"),
                    "source_path": comp.get("source_path") or "",
                    "payload_sha256": sha256_obj({"component_uuid": comp["component_uuid"], "span": asdict(span)}),
                })
                spans_payload.append(item)
    spans_inserted = 0
    run_uuid = None
    if execute and not blockers:
        status = "succeeded"
        with psycopg.connect(state_db) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lucidota_learning.gliner_extraction_run
                      (job_uuid, queue_name, workflow_name, backend, model_ref, label_source, labels, components_seen, spans_found, status, detail)
                    VALUES (%s::uuid,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb)
                    RETURNING run_uuid::text
                """, (job_uuid, args.queue, workflow_for(args.job_kind), backend_detail.get("backend", "none"), args.gliner_model or "", "operator_ontology_fixture", json.dumps(labels), len(components), len(spans_payload), status, json.dumps({"backend_detail": backend_detail, "storage_database_url": redact(storage_db)}, default=str)))
                run_uuid = first_value(cur.fetchone())
                for sp in spans_payload:
                    cur.execute("""
                        INSERT INTO lucidota_learning.gliner_entity_span
                          (run_uuid, job_uuid, file_uuid, component_uuid, source_sha256, source_path, component_index,
                           label, matched_text, start_char, end_char, score, backend, payload_sha256, detail)
                        VALUES (%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT (component_uuid, label, start_char, end_char, matched_text, backend) DO NOTHING
                    """, (run_uuid, job_uuid, sp["file_uuid"], sp["component_uuid"], sp["source_sha256"], sp["source_path"], sp.get("component_index"), sp["label"], sp["text"], sp["start"], sp["end"], sp["score"], sp["backend"], sp["payload_sha256"], json.dumps({"graph_promotion_candidate": True}, default=str)))
                    spans_inserted += cur.rowcount
                cur.execute("UPDATE lucidota_learning.gliner_extraction_run SET spans_inserted=%s WHERE run_uuid=%s::uuid", (spans_inserted, run_uuid))
            conn.commit()
    try:
        with psycopg.connect(storage_db) as conn:
            graph_after = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_after = count_tables(conn, TEMPORAL_TABLES)
        with psycopg.connect(state_db) as conn:
            learning_after = count_tables(conn, LEARNING_TABLES)
    except Exception as exc:
        blockers.append(f"postcheck_failed:{exc}")
    graph_writes = graph_before != graph_after
    temporal_writes = temporal_before != temporal_after
    if graph_writes: blockers.append("canonical_graph_counts_changed")
    if temporal_writes: blockers.append("temporal_claim_counts_changed")
    return {
        "mode": "phase05_streaming_brain_gliner",
        "state_database_url": redact(state_db),
        "storage_database_url": redact(storage_db),
        "labels": labels,
        "component_limit": bounded_int((payload or {}).get("component_limit") or args.component_limit, name="component_limit", minimum=1, maximum=MAX_COMPONENT_LIMIT),
        "components_seen": len(components),
        "spans_found": len(spans_payload),
        "spans_inserted": spans_inserted,
        "run_uuid": run_uuid,
        "backend_detail": backend_detail,
        "sample_spans": spans_payload[:10],
        "learning_counts_before": learning_before,
        "learning_counts_after": learning_after,
        "canonical_graph_counts_before": graph_before,
        "canonical_graph_counts_after": graph_after,
        "temporal_counts_before": temporal_before,
        "temporal_counts_after": temporal_after,
        "learning_writes_performed": bool(execute),
        "canonical_graph_writes_performed": graph_writes,
        "temporal_claims_mutated_by_wrapper": temporal_writes,
        "korpus_custody_mutated_by_wrapper": False,
    }, blockers


def claim_packet_extract(args: argparse.Namespace, payload: dict[str, Any] | None, job_uuid: str | None, execute: bool) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    storage_db = storage_url(args, payload)
    state_db = state_url(args, payload)
    try:
        limit = bounded_int((payload or {}).get("claim_packet_limit") or args.claim_packet_limit, name="claim_packet_limit", minimum=1, maximum=MAX_CLAIM_PACKET_LIMIT)
    except ValueError as exc:
        return {
            "mode": "phase05_streaming_brain_claim_packet_extract",
            "state_database_url": redact(state_db),
            "storage_database_url": redact(storage_db),
            "job_uuid": job_uuid,
            "claim_packet_limit": None,
            "claim_limit_error": str(exc),
            "learning_writes_performed": False,
            "claim_packet_writes_performed": False,
            "canonical_graph_writes_performed": False,
            "temporal_claims_mutated_by_wrapper": False,
            "korpus_custody_mutated_by_wrapper": False,
            "truth_status": "not_truth_claim_candidate",
        }, [str(exc).split(":", 1)[0]]
    py = python_bin(args)
    graph_before = graph_after = temporal_before = temporal_after = {}
    claim_packets_before = claim_packets_after = None
    commands: dict[str, Any] = {}
    try:
        with psycopg.connect(storage_db) as conn:
            graph_before = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_before = count_tables(conn, TEMPORAL_TABLES)
            claim_packets_before = count_table(conn, "lucidota_korpus.document_claim_packet")
    except Exception as exc:
        blockers.append(f"storage_db_precheck_failed:{exc}")
    if execute and not blockers:
        env = os.environ.copy()
        env["KORPUS_DATABASE_URL"] = storage_db
        commands["init_schema"] = run_cmd([py, "scripts/document_claim_packet_worker.py", "--database-url", storage_db, "init-schema", "--execute"], env)
        if commands["init_schema"]["returncode"] != 0:
            blockers.append("document_claim_packet_init_failed")
        if not blockers:
            commands["extract"] = run_cmd([py, "scripts/document_claim_packet_worker.py", "--database-url", storage_db, "extract", "--execute", "--limit", str(limit)], env, timeout=300)
            if commands["extract"]["returncode"] != 0:
                blockers.append("document_claim_packet_extract_failed")
    try:
        with psycopg.connect(storage_db) as conn:
            graph_after = count_tables(conn, CANONICAL_GRAPH_TABLES)
            temporal_after = count_tables(conn, TEMPORAL_TABLES)
            claim_packets_after = count_table(conn, "lucidota_korpus.document_claim_packet")
    except Exception as exc:
        blockers.append(f"storage_db_postcheck_failed:{exc}")
    graph_writes = graph_before != graph_after
    temporal_writes = temporal_before != temporal_after
    if graph_writes:
        blockers.append("canonical_graph_counts_changed")
    if temporal_writes:
        blockers.append("temporal_claim_counts_changed")
    inserted_delta = None
    if claim_packets_before is not None and claim_packets_after is not None:
        inserted_delta = int(claim_packets_after) - int(claim_packets_before)
    return {
        "mode": "phase05_streaming_brain_claim_packet_extract",
        "state_database_url": redact(state_db),
        "storage_database_url": redact(storage_db),
        "job_uuid": job_uuid,
        "claim_packet_limit": limit,
        "claim_packets_before": claim_packets_before,
        "claim_packets_after": claim_packets_after,
        "claim_packets_inserted_delta": inserted_delta,
        "commands": commands,
        "canonical_graph_counts_before": graph_before,
        "canonical_graph_counts_after": graph_after,
        "temporal_counts_before": temporal_before,
        "temporal_counts_after": temporal_after,
        "learning_writes_performed": False,
        "claim_packet_writes_performed": bool(execute),
        "canonical_graph_writes_performed": graph_writes,
        "temporal_claims_mutated_by_wrapper": temporal_writes,
        "korpus_custody_mutated_by_wrapper": False,
        "truth_status": "not_truth_claim_candidate",
    }, blockers


def init_schema(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = state_url(args)
    schemas = [STATE_SCHEMA, REAL_WORK_LOOP_SCHEMA, LEARNING_SCHEMA, BYTEWAX_SCHEMA, RIVER_SCHEMA, CLAIM_PACKET_SCHEMA]
    result: dict[str, Any] = {"state_database_url": redact(url), "execute_performed": False, "would_apply": [str(schema.relative_to(ROOT)) for schema in schemas]}
    if not execute:
        return result, []
    with psycopg.connect(url) as conn:
        ensure_state_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM lucidota_control.absurd_queue WHERE queue_name=%s", (args.queue,))
            result["queue_registered"] = int(first_value(cur.fetchone())) == 1
            cur.execute("SELECT count(*) FROM lucidota_control.workflow_registry WHERE workflow_name=%s", (workflow_for(args.job_kind),))
            result["workflow_registered"] = int(first_value(cur.fetchone())) == 1
            cur.execute("SELECT to_regclass('lucidota_learning.gliner_entity_span')")
            result["gliner_span_table"] = first_value(cur.fetchone()) is not None
        conn.commit()
    result["execute_performed"] = True
    return result, []


def enqueue(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = state_url(args)
    try:
        component_limit = bounded_int(args.component_limit, name="component_limit", minimum=1, maximum=MAX_COMPONENT_LIMIT)
        min_chars = bounded_int(args.min_chars, name="min_chars", minimum=1, maximum=MAX_MIN_CHARS)
        claim_packet_limit = bounded_int(args.claim_packet_limit, name="claim_packet_limit", minimum=1, maximum=MAX_CLAIM_PACKET_LIMIT)
        bytewax_limit = bounded_int(args.bytewax_limit, name="bytewax_limit", minimum=1, maximum=MAX_BYTEWAX_LIMIT)
        river_limit = bounded_int(args.river_limit, name="river_limit", minimum=1, maximum=MAX_RIVER_LIMIT)
    except ValueError as exc:
        return {
            "state_database_url": redact(url),
            "queue": args.queue,
            "workflow": workflow_for(args.job_kind),
            "job_kind": args.job_kind,
            "error": str(exc),
            "execute_performed": False,
            "job_uuid": None,
            "inserted_new": False,
        }, [str(exc).split(":", 1)[0]]
    payload = {"check_kind": args.job_kind, "state_database_url": url, "storage_database_url": storage_url(args), "component_limit": component_limit, "min_chars": min_chars, "claim_packet_limit": claim_packet_limit, "labels": load_labels(args), "no_graph_write": True, "no_temporal_claim_mutation": True}
    if args.job_kind == LEGACY_JOB_KIND:
        payload.update({"bytewax_limit": bytewax_limit, "river_limit": river_limit})
    idempotency_key = args.idempotency_key or sha256_obj({"queue": args.queue, "job_kind": args.job_kind, "payload": {k: v for k, v in payload.items() if not k.endswith("database_url")}})
    result: dict[str, Any] = {"state_database_url": redact(url), "queue": args.queue, "workflow": workflow_for(args.job_kind), "job_kind": args.job_kind, "idempotency_key": idempotency_key, "payload_sha256": sha256_obj(payload), "execute_performed": False, "job_uuid": None, "inserted_new": False}
    if not execute:
        return result, []
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.absurd_queue_job
                  (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                RETURNING job_uuid, (xmax = 0) AS inserted_new
            """, (args.queue, workflow_for(args.job_kind), args.job_kind, idempotency_key, json.dumps(payload), args.priority, args.max_attempts, json.dumps({"source": "absurd_river_worker"})))
            job_uuid, inserted_new = cur.fetchone()
            if inserted_new:
                cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s,%s,'enqueued','absurd_river_worker',%s::jsonb)", (job_uuid, args.queue, json.dumps({"job_kind": args.job_kind, "idempotency_key": idempotency_key})))
            result.update({"execute_performed": True, "job_uuid": str(job_uuid), "inserted_new": bool(inserted_new)})
        conn.commit()
    return result, []


def worker_once(args: argparse.Namespace, execute: bool) -> tuple[dict[str, Any], list[str]]:
    url = state_url(args)
    blockers: list[str] = []
    worker_id = args.worker_id or f"absurd_river:{socket.gethostname()}:{os.getpid()}"
    result: dict[str, Any] = {"state_database_url": redact(url), "queue": args.queue, "worker_id": worker_id, "execute_performed": False, "job_processed": False}
    row: dict[str, Any] | None = None
    with psycopg.connect(url) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            lock_clause = "FOR UPDATE SKIP LOCKED" if execute else ""
            cur.execute(f"""
                SELECT job_uuid::text, workflow_name, job_kind, idempotency_key, status::text, payload
                FROM lucidota_control.absurd_queue_job
                WHERE queue_name=%s AND status='queued' AND run_after <= now()
                ORDER BY priority ASC, created_at ASC
                {lock_clause}
                LIMIT 1
            """, (args.queue,))
            fetched = cur.fetchone()
            if not execute:
                result["would_process"] = dict(fetched) if fetched else None
                if fetched:
                    result["worker_contract"] = validate_worker_contract(
                        cur,
                        queue_name=args.queue,
                        job_kind=str(fetched["job_kind"]),
                        worker_key=worker_key_for_queue(args.queue),
                    ).as_result()
                return result, blockers
            if not fetched:
                result["no_job_available"] = True
                return result, blockers
            row = dict(fetched)
            job_uuid = row["job_uuid"]
            args.job_kind = str(row["job_kind"])
            contract = validate_worker_contract(
                cur,
                queue_name=args.queue,
                job_kind=str(row["job_kind"]),
                worker_key=worker_key_for_queue(args.queue),
            )
            result["worker_contract"] = contract.as_result()
            if not contract.ok:
                gate_result, error_kind = record_worker_contract_rejection(
                    cur,
                    job_uuid=job_uuid,
                    queue_name=args.queue,
                    payload=row["payload"],
                    contract=contract,
                    event_source="absurd_river_worker",
                )
                conn.commit()
                result.update({
                    "execute_performed": True,
                    "job_processed": True,
                    "job_uuid": job_uuid,
                    "status": "failed",
                    "health": gate_result,
                    "learning_writes_performed": False,
                    "canonical_graph_writes_performed": False,
                    "temporal_claims_mutated_by_wrapper": False,
                })
                return result, [error_kind]
            cur.execute("UPDATE lucidota_control.absurd_queue_job SET status='running', leased_by=%s, lease_expires_at=now()+interval '5 minutes', attempt_count=attempt_count+1, updated_at=now() WHERE job_uuid=%s::uuid", (worker_id, job_uuid))
            cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,'started','absurd_river_worker',%s::jsonb)", (job_uuid, args.queue, json.dumps({"worker_id": worker_id})))
        conn.commit()
    assert row is not None
    job_uuid = row["job_uuid"]
    payload = row["payload"]
    if args.job_kind == LEGACY_JOB_KIND:
        health, hblockers = river_bytewax_health(args, payload, run_tick=True)
    elif args.job_kind == CLAIM_PACKET_JOB_KIND:
        health, hblockers = claim_packet_extract(args, payload, job_uuid, execute=True)
    else:
        health, hblockers = gliner_streaming_brain(args, payload, job_uuid, execute=True)
    ok = not hblockers
    if ok:
        health_payload = {"health": health, "outcome": "succeeded"}
        payload_ok, hygiene = gate_worker_payload_hygiene(
            health_payload,
            queue_name=args.queue,
            worker_key=worker_key_for_queue(args.queue),
            job_kind=args.job_kind,
            required_keys=(),
            min_score=0,
        )
        if not payload_ok:
            ok = False
            hblockers.append(hygiene.get("error", "decision_hygiene_failed"))
            health = {"health": health_payload, "hygiene": hygiene, "outcome": "failed"}
    status = "succeeded" if ok else "failed"
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE lucidota_control.absurd_queue_job SET status=%s, result=%s::jsonb, completed_at=CASE WHEN %s THEN now() ELSE completed_at END, updated_at=now(), last_error=%s WHERE job_uuid=%s::uuid", (status, json.dumps(health, default=str), ok, ";".join(hblockers), job_uuid))
            cur.execute("INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,%s,'absurd_river_worker',%s::jsonb)", (job_uuid, args.queue, status, json.dumps({"health_blockers": hblockers, "spans_inserted": health.get("spans_inserted"), "backend": (health.get("backend_detail") or {}).get("backend")})))
            cur.execute("INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail) VALUES (%s,%s,'absurd_river_wrapper',%s,'absurd_river_worker',%s::jsonb) RETURNING event_id::text", (workflow_for(args.job_kind), job_uuid, status, json.dumps({"queue": args.queue, "job_uuid": job_uuid, "health": health}, default=str)))
            event_id = first_value(cur.fetchone())
            if not ok:
                cur.execute("""
                    INSERT INTO lucidota_control.absurd_queue_dead_letter
                      (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
                    SELECT job_uuid, queue_name, workflow_name, job_kind, idempotency_key, 'river_wrapper_failed', %s, attempt_count, %s, %s::jsonb
                    FROM lucidota_control.absurd_queue_job WHERE job_uuid=%s::uuid
                    ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET error_message=EXCLUDED.error_message,last_seen_at=now(),context=EXCLUDED.context
                """, (";".join(hblockers), sha256_obj(payload), json.dumps(health, default=str), job_uuid))
        conn.commit()
    result.update({"execute_performed": True, "job_processed": True, "job_uuid": job_uuid, "workflow_event_id": event_id, "status": status, "health": health, "learning_writes_performed": health.get("learning_writes_performed", False), "canonical_graph_writes_performed": health.get("canonical_graph_writes_performed", False), "temporal_claims_mutated_by_wrapper": health.get("temporal_claims_mutated_by_wrapper", False)})
    blockers.extend(hblockers)
    return result, blockers

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["audit", "init-schema", "enqueue-health-check", "worker-once"], required=True)
    mode = ap.add_mutually_exclusive_group(); mode.add_argument("--dry-run", action="store_true"); mode.add_argument("--execute", action="store_true")
    ap.add_argument("--state-database-url", default=os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
    ap.add_argument("--storage-database-url", default=os.environ.get("KORPUS_DATABASE_URL", "postgresql:///lucidota_storage"))
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--job-kind", default=DEFAULT_JOB_KIND, choices=[DEFAULT_JOB_KIND, CLAIM_PACKET_JOB_KIND, LEGACY_JOB_KIND])
    ap.add_argument("--idempotency-key")
    ap.add_argument("--priority", type=int, default=60)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--worker-id")
    ap.add_argument("--python", default=os.environ.get("LUCIDOTA_PYTHON"))
    ap.add_argument("--bytewax-limit", type=int, default=25)
    ap.add_argument("--river-limit", type=int, default=5000)
    ap.add_argument("--component-limit", type=int, default=8)
    ap.add_argument("--claim-packet-limit", type=int, default=25)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--labels")
    ap.add_argument("--gliner-model", default=os.environ.get("GLINER_MODEL_PATH") or (str(DEFAULT_GLINER_MODEL) if DEFAULT_GLINER_MODEL.exists() else None))
    ap.add_argument("--gliner-threshold", type=float, default=0.35)
    ap.add_argument("--allow-remote-model", action="store_true")
    args = ap.parse_args()
    execute = bool(args.execute)
    try:
        if args.action == "init-schema":
            action_result, blockers = init_schema(args, execute)
        elif args.action == "enqueue-health-check":
            action_result, blockers = enqueue(args, execute)
        elif args.action == "worker-once":
            action_result, blockers = worker_once(args, execute)
        else:
            if args.job_kind == LEGACY_JOB_KIND:
                health, hblockers = river_bytewax_health(args, run_tick=False)
            else:
                health, hblockers = gliner_streaming_brain(args, None, None, execute=False)
            action_result = {"health": health, "execute_performed": False}
            blockers = hblockers
    except Exception as exc:
        action_result = {}; blockers = [f"exception:{type(exc).__name__}:{exc}"]
    report = {
        "schema": "lucidota.absurd_river_wrapper.report.v2",
        "generated_at": now_iso(),
        "action": args.action,
        "mode": "execute" if execute else "dry_run",
        "execute_requested": execute,
        "queue": args.queue,
        "job_kind": args.job_kind,
        "state_database_url": redact(state_url(args)),
        "storage_database_url": redact(storage_url(args)),
        "action_result": action_result,
        "db_writes_performed": bool(action_result.get("execute_performed")) if isinstance(action_result, dict) else False,
        "learning_writes_performed": bool(action_result.get("learning_writes_performed") or (action_result.get("health") or {}).get("learning_writes_performed")) if isinstance(action_result, dict) else False,
        "canonical_graph_writes_performed": bool(action_result.get("canonical_graph_writes_performed")) if isinstance(action_result, dict) else False,
        "temporal_claims_mutated_by_wrapper": bool(action_result.get("temporal_claims_mutated_by_wrapper")) if isinstance(action_result, dict) else False,
        "blockers": blockers,
    }
    write_report(args.action, report)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
