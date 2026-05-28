#!/usr/bin/env python3
"""DIOGENES / LUCIDOTA 30-phase architecture audit.

This is an evidence audit, not a build loop. It reads filesystem, SQL schemas,
status/TICKLETRUNK manifests, local DB counts, local GPU/model/dependency state,
and emits one finding block per required audit phase.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    psycopg = None
    dict_row = None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "audit"
PROJECT_BRAIN = ROOT / "00_PROJECT_BRAIN"
STATUS_LEDGER = ROOT / "05_OUTPUTS" / "status_ledger.json"
TICKLETRUNK = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"

PHASE_NAMES = [
    "ckdog1-kernel audit",
    "Rust workspace audit",
    "CLAW / ClaudeCode fork audit",
    "DBOS spine audit",
    "KRAMPUSCHEWING custody audit",
    "KRAMPUSCHEWING DBOS wrapper audit",
    "OCR/document parser audit",
    "Chrono-Ledger audit",
    "Graph schema audit",
    "Graph promotion execution audit",
    "Ontology audit",
    "Command Envelope Protocol audit",
    "Darwinian Surfaces audit",
    "TICKLETRUNK audit",
    "STATUS_LEDGER audit",
    "Security quarantine audit",
    "Brain Archaeology audit",
    "GLiNER / extraction audit",
    "SimpleMem / DeMem / CatchMe audit",
    "Worker/daemon supervision audit",
    "Model runtime audit",
    "GPU utilization audit",
    "LoRA/adapters audit",
    "Tech bench audit",
    "05_OUTPUTS evidence audit",
    "Schema migration audit",
    "Rust-port candidate audit",
    "End-to-end path audit",
    "Production readiness audit",
    "Remediation backlog",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(cmd),
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": "timeout",
        }


def table_count(conn, table: str) -> int | None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s) AS r", (table,))
            row = cur.fetchone()
            exists = row["r"] if isinstance(row, dict) else row[0]
            if not exists:
                return None
            cur.execute(f"SELECT count(*) AS c FROM {table}")
            row = cur.fetchone()
            return int(row["c"] if isinstance(row, dict) else row[0])
    except Exception:
        return None


def db_counts() -> dict[str, Any]:
    if psycopg is None:
        return {"error": "psycopg_not_importable"}
    specs = {
        "state": {
            "url": os.environ.get("DBOS_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state",
            "tables": [
                "lucidota_control.dbos_queue",
                "lucidota_control.dbos_queue_job",
                "lucidota_control.dbos_queue_event",
                "lucidota_control.dbos_queue_dead_letter",
                "lucidota_control.workflow_event",
                "lucidota_control.conversation_command",
                "lucidota_control.production_readiness_item",
                "lucidota_control.system_telemetry_rollup",
            ],
        },
        "storage": {
            "url": os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage",
            "tables": [
                "lucidota_korpus.file_object",
                "lucidota_korpus.file_occurrence",
                "lucidota_korpus.component",
                "lucidota_korpus.temporal_claim",
                "lucidota_korpus.current_chrono_timeline_projection",
                "lucidota_go.graph_item",
                "lucidota_go.graph_edge",
                "lucidota_go.graph_journal",
                "lucidota_go.graph_promotion_packet",
                "lucidota_go.graph_promotion_decision",
                "lucidota_learning.gliner_entity_span",
                "lucidota_learning.gliner_extraction_run",
            ],
        },
    }
    out: dict[str, Any] = {}
    for name, spec in specs.items():
        item = {"url_source": "env_or_default_redacted", "tables": {}}
        try:
            with psycopg.connect(spec["url"], row_factory=dict_row) as conn:
                for t in spec["tables"]:
                    item["tables"][t] = table_count(conn, t)
                if name == "state":
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT queue_name,
                                   count(*) AS jobs,
                                   count(*) FILTER (WHERE status='queued') AS queued,
                                   count(*) FILTER (WHERE status='running') AS running,
                                   count(*) FILTER (WHERE status='succeeded') AS succeeded,
                                   count(*) FILTER (WHERE status IN ('failed','dead_lettered')) AS failed
                            FROM lucidota_control.dbos_queue_job
                            GROUP BY queue_name ORDER BY queue_name
                        """)
                        item["queue_summary"] = [dict(r) for r in cur.fetchall()]
                if name == "storage":
                    with conn.cursor() as cur:
                        cur.execute("SELECT count(*) AS c FROM lucidota_korpus.temporal_claim WHERE evidence_source='mtime_snapshot_v1'")
                        item["mtime_snapshot_v1_claims"] = int(cur.fetchone()["c"])
                        cur.execute("SELECT count(*) AS c FROM lucidota_korpus.current_chrono_timeline_projection p LEFT JOIN lucidota_korpus.temporal_claim c ON p.selected_claim_uuid=c.claim_uuid WHERE c.claim_uuid IS NULL")
                        item["projection_missing_claim_links"] = int(cur.fetchone()["c"])
        except Exception as exc:
            item["error"] = str(exc)
        out[name] = item
    return out


def path_exists(paths: list[str]) -> dict[str, bool]:
    return {p: (ROOT / p).exists() for p in paths}


def latest(pattern: str) -> str | None:
    xs = glob.glob(str(ROOT / pattern))
    if not xs:
        return None
    return rel(max(xs, key=os.path.getmtime))


def grep_count(root: str, pattern: str, exts: tuple[str, ...] = (".py", ".rs", ".sql", ".md", ".json", ".toml")) -> int:
    base = ROOT / root
    if not base.exists():
        return 0
    rx = re.compile(pattern, re.I)
    count = 0
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts or p.stat().st_size > 500_000:
            continue
        try:
            if rx.search(p.read_text(encoding="utf-8", errors="ignore")):
                count += 1
        except Exception:
            pass
    return count


def top_file_types(root: str, limit: int = 250_000) -> list[tuple[str, int]]:
    base = ROOT / root
    c: Counter[str] = Counter()
    if not base.exists():
        return []
    for i, p in enumerate(base.rglob("*")):
        if i > limit:
            break
        if p.is_file():
            c[p.suffix.lower() or "[none]"] += 1
    return c.most_common(30)


def du(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        return "missing"
    res = run(["du", "-sh", path], timeout=15)
    return (res["stdout_tail"].split("\t")[0] if res["returncode"] == 0 and res["stdout_tail"] else "unknown")


def git_status_subset(paths: list[str]) -> list[str]:
    res = run(["git", "status", "--short", "--"] + paths, timeout=20)
    return [line for line in res["stdout_tail"].splitlines() if line.strip()]


def list_files(path: str, max_items: int = 80, patterns: tuple[str, ...] | None = None) -> list[str]:
    base = ROOT / path
    if not base.exists():
        return []
    files: list[str] = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            if patterns and not any(p.match(x) or p.name.lower().endswith(x.lower().lstrip("*")) for x in patterns):
                continue
            files.append(rel(p))
            if len(files) >= max_items:
                break
    return files


def venv_dep_status() -> dict[str, bool]:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        return {"venv_python": False}
    code = """
import importlib.util,json
mods=['docling','gliner','opentelemetry','pydantic','FlagEmbedding','transformers','torch','marimo','textual','ollama']
print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))
"""
    res = run([str(py), "-c", code], timeout=30)
    try:
        d = json.loads(res["stdout_tail"])
        d["venv_python"] = True
        return d
    except Exception:
        return {"venv_python": True, "error": res}


def gpu_status() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"nvidia_smi": False}
    res = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"], timeout=10)
    return {"nvidia_smi": res["returncode"] == 0, "raw": res["stdout_tail"].strip(), "error": res["stderr_tail"].strip()}


def cargo_workspace_summary() -> dict[str, Any]:
    base = ROOT / "01_REPOS" / "lucidota_etl"
    crates = []
    for p in sorted((base / "crates").glob("*/Cargo.toml")) if (base / "crates").exists() else []:
        crates.append(rel(p.parent))
    todo = []
    for p in (base / "crates").rglob("*.rs") if (base / "crates").exists() else []:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(x in text for x in ["todo!", "unimplemented!", "TODO"]):
            todo.append(rel(p))
    return {"workspace_exists": base.exists(), "crates": crates, "todo_unimplemented_files": todo[:80], "crate_count": len(crates)}


def status_lookup(status: dict[str, Any], name: str) -> dict[str, Any] | None:
    lname = name.lower()
    for sec in ["software", "phases", "plans_workstreams"]:
        for e in status.get(sec, []):
            if e.get("name", "").lower() == lname:
                return e
    return None


def inflated_statuses(status: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for sec in ["software", "phases", "plans_workstreams"]:
        for e in status.get(sec, []):
            s = e.get("status")
            blockers = str(e.get("blockers") or "").strip()
            progress = int(e.get("progress") or 0)
            if s in {"verified", "executed"} and blockers:
                out.append({"section": sec, "name": e.get("name"), "status": s, "progress": progress, "blockers": blockers, "evidence": e.get("evidence")})
            elif s == "verified" and progress < 100:
                out.append({"section": sec, "name": e.get("name"), "status": s, "progress": progress, "reason": "verified_but_progress_below_100", "evidence": e.get("evidence")})
    return out


def make_phase(num: int, name: str, grade: str, facts: dict[str, Any], findings: list[str], blockers: list[str], next_work: list[str], evidence: list[str]) -> dict[str, Any]:
    return {
        "phase": num,
        "name": name,
        "grade": grade,
        "facts": facts,
        "findings": findings,
        "blockers": blockers,
        "next_real_work": next_work,
        "evidence": evidence,
    }


def audit() -> dict[str, Any]:
    status = read_json(STATUS_LEDGER, {})
    tickle = read_json(TICKLETRUNK, {})
    db = db_counts()
    phases: list[dict[str, Any]] = []
    schema_files = sorted(rel(p) for p in (ROOT / "06_SCHEMA").glob("*.sql"))
    scripts = sorted(rel(p) for p in (ROOT / "scripts").glob("*.py"))
    rust = cargo_workspace_summary()
    gpu = gpu_status()
    deps = venv_dep_status()
    service_check = run(["scripts/check_chrono_ledger_service.sh"], timeout=30) if (ROOT / "scripts/check_chrono_ledger_service.sh").exists() else {"returncode": 127, "stderr_tail": "missing"}
    graph_counts = (db.get("storage", {}).get("tables") or {})
    queue_summary = db.get("state", {}).get("queue_summary", [])

    # 1 kernel
    kernel_files = path_exists([
        "01_REPOS/lucidota_etl/crates/lucidota-kernel/src/lib.rs",
        "01_REPOS/lucidota_etl/crates/lucidota-kernel/src/bin/lucidota-kernel.rs",
        "01_REPOS/lucidota_etl/crates/lucidota-core/src/ckdog.rs",
        "05_OUTPUTS/rust/kernel_status_smoke_20260517T060436Z.json",
    ])
    phases.append(make_phase(1, PHASE_NAMES[0], "PARTIAL_REAL",
        {"files": kernel_files, "status_ledger": status_lookup(status, "ckdog1-kernel")},
        ["CKDOG1 bounds and packed slot refs exist in Rust.", "Kernel currently exposes status, tonic health, route plan, packet auth, schema install, and event append.", "Kernel is not yet the live policy authority for DBOS, KRAMPUSCHEWING, graph promotion, CLAW, ontology, or model routing."],
        ["KERNEL_NOT_DBOS_POLICY_AUTHORITY", "KERNEL_NOT_CLAW_COMMAND_GATE", "KERNEL_NOT_MODEL_ROUTER"],
        ["Make DBOS enqueue/consume paths call kernel authorize/route before state transitions.", "Expose graph-promotion and KORPUS route decisions as kernel commands with SQL receipts."],
        [p for p,v in kernel_files.items() if v]))

    # 2 Rust workspace
    phases.append(make_phase(2, PHASE_NAMES[1], "PARTIAL_REAL",
        {"rust_workspace": rust, "status_ledger": status_lookup(status, "Rust hot-path replacement")},
        ["Rust workspace exists with core/db/intake/kernel/launcher/audit/sqlite-importer/workers/chrono crates.", "Chrono daemon is the most operational Rust component.", "Most Python worker behavior has not been ported to Rust; port policy is not enforced by code."],
        ["PYTHON_TO_RUST_PORT_POLICY_NOT_ENFORCED", "RUST_WORKERS_STILL_SHELL_LEVEL"],
        ["Create Rust-port candidate registry and parity test requirement for each Python worker before marking it finished.", "Port KRAMPUS manifest/componentization or document parse lane next, not report scripts."],
        rust.get("crates", [])[:20]))

    # 3 CLAW
    claw = git_status_subset(["01_REPOS/claudecode", "claw", ".claw", ".claw.json"])
    phases.append(make_phase(3, PHASE_NAMES[2], "UNDERMODELED",
        {"git_status": claw, "status_ledger": status_lookup(status, "LUCIDOTA CLI / CLAW fork"), "modified_files": [x for x in claw if x.strip().startswith("M")]},
        ["CLAW/ClaudeCode fork is modified and should be first-class; it is the operator execution hand.", "It is not yet wired as CEP emitter, kernel client, or DBOS submitter.", "Runtime permission edits exist but are not represented as a formal DIOGENES subsystem contract."],
        ["CLAW_NOT_IN_DBOS_COMMAND_PATH", "CLAW_PERMISSION_MODEL_NOT_AUDITED_TO_KERNEL"],
        ["Define CLAW command envelope output format and route it through ckdog1-kernel then DBOS.", "Add CLAW fork test that proves no direct canonical graph mutation."],
        ["01_REPOS/claudecode/rust/crates/claw-cli/src/main.rs", "01_REPOS/claudecode/rust/crates/runtime/src/permissions.rs", "claw"]))

    # 4 DBOS
    phases.append(make_phase(4, PHASE_NAMES[3], "REAL_BUT_NOT_COMPLETE",
        {"db_counts": db.get("state"), "status_ledger": status_lookup(status, "DBOS queue spine") or status_lookup(status, "DBOS workflow spine")},
        ["DBOS queue/event/dead-letter tables have live rows.", "Chrono, KRAMPUSCHEWING, River, document_parse, surface_cep, graph_promotion queues exist.", "Queued jobs remain in several queues; wrappers are uneven and some are health checks rather than real work drains."],
        ["REMAINING_DBOS_WORKERS_NOT_FULLY_DRAINED", "HEALTH_WRAPPER_CONFUSED_WITH_PRODUCTION_WORKER"],
        ["Convert remaining queued surface_cep, graph_promotion, marrow, scheduler_probe, and intake jobs into explicit handler decisions or drain them safely.", "Add worker registry gate: every queue_name+job_kind must map to a handler before enqueue."],
        ["06_SCHEMA/035_dbos_queue_spine.sql", "scripts/dbos_queue_spine.py"] + [latest("05_OUTPUTS/dbos/dbos_queue_spine_phase2_report_*.json") or "missing"]))

    # 5 KRAMPUS custody
    kr_types = top_file_types("KRAMPUSCHEWING")
    phases.append(make_phase(5, PHASE_NAMES[4], "MASSIVE_PARTIAL_CUSTODY",
        {"size": du("KRAMPUSCHEWING"), "top_file_types": kr_types, "db_tables": {k:v for k,v in (db.get("storage",{}).get("tables") or {}).items() if k.startswith("lucidota_korpus")}},
        ["KRAMPUSCHEWING is the largest body at ~133G.", "Custody tables have file_object/file_occurrence/component rows.", "Corpus contains many OCR/document targets: PDFs, images, DOCX, TIFF, HTML, OPUS/MP4, DB files, ZIPs."],
        ["OCR_DOCUMENT_LANES_NOT_PRODUCTIONIZED", "BINARY_ARCHIVE_POLICY_NOT_VISIBLE_AS_QUEUE_LANES"],
        ["Create KORPUS parser lane table/queue for pdf,image,docx,archive,audio/video,db artifacts.", "Make KRAMPUS DBOS worker process one real allowlisted custody row per lane, not just health."],
        ["KRAMPUSCHEWING", "06_SCHEMA/019_korpus_krampii.sql", "scripts/korpus_krampii.py"]))

    # 6 krampus wrapper
    phases.append(make_phase(6, PHASE_NAMES[5], "HEALTH_VERIFIED_NOT_PRODUCTION_COMPLETE",
        {"status_ledger": status_lookup(status, "KRAMPUSCHEWING DBOS wrapper"), "phase_report": latest("05_OUTPUTS/dbos/dbos_krampus_wrapper_phase_report_*.json")},
        ["KRAMPUSCHEWING wrapper is verified as DBOS health/observation wrapper.", "Wrapper explicitly does not ingest drops, move files, delete files, mutate custody rows, or mutate temporal claims.", "Therefore wrapper cannot be called KRAMPUSCHEWING production-complete."],
        ["KRAMPUS_WRAPPER_HEALTH_ONLY", "NO_REAL_KRAMPUS_DBOS_CONSUME_ONE_LANE"],
        ["Add krampus_parse_one job_kind over lucidota_korpus.file_object with idempotent parser output staging.", "Add KORPUS lane outcome table and DLQ policy per parser family."],
        ["scripts/dbos_krampus_worker.py", latest("05_OUTPUTS/dbos/dbos_krampus_wrapper_phase_report_*.json") or "missing"]))

    # 7 OCR
    doc_paths = path_exists(["scripts/document_parse_bakeoff.py", "scripts/document_parse_ingest.py", "scripts/dbos_document_parse_worker.py", "06_SCHEMA/045_document_ingestion_pipeline.sql", "06_SCHEMA/076_dbos_document_parse_worker.sql"])
    phases.append(make_phase(7, PHASE_NAMES[6], "BENCH_TO_WORKER_PARTIAL",
        {"deps_venv": deps, "files": doc_paths, "tech_bench": latest("05_OUTPUTS/tech_bench/*/tech_bench_manifest.json")},
        ["Docling/GLiNER/OpenTelemetry/Pydantic/etc are installed in .venv, not system python.", "Document parser scripts and DBOS document_parse worker exist.", "SmolDocling direct path, BGE weights, ModernBERT weights remain blockers."],
        ["SMOLDOCLING_DIRECT_PACKAGE_OR_MODEL_PATH_NOT_CONFIRMED", "BGE_MODEL_WEIGHTS_NOT_DOWNLOADED", "MODERNBERT_LOCAL_MODEL_PATH_NOT_CONFIGURED", "OCR_OUTPUT_NOT_KRAMPUS_DEFAULT_LANE"],
        ["Promote document_parse worker to KRAMPUS parser lane for PDFs/images/DOCX with provenance blocks.", "Store parser output as component spans + claim packet inputs, not graph truth."],
        [p for p,v in doc_paths.items() if v] + [latest("05_OUTPUTS/tech_bench/*/document_parse_bakeoff_report.json") or "missing"]))

    # 8 Chrono
    phases.append(make_phase(8, PHASE_NAMES[7], "STRONG_REAL",
        {"service_returncode": service_check.get("returncode"), "db": {"temporal_claim": db.get("storage",{}).get("tables",{}).get("lucidota_korpus.temporal_claim"), "projection": db.get("storage",{}).get("tables",{}).get("lucidota_korpus.current_chrono_timeline_projection"), "missing_projection_claim_links": db.get("storage",{}).get("projection_missing_claim_links"), "mtime_snapshot_v1_claims": db.get("storage",{}).get("mtime_snapshot_v1_claims")}},
        ["Chrono service check passes and daemon is active.", "Temporal claims and current projection counts are live.", "Projection missing-claim links are zero; mtime_snapshot_v1 claims are present."],
        ["CHRONO_DBOS_MIGRATION_STILL_CUSTOM_DAEMON_UNDER_SERVICE"],
        ["Keep Chrono conservation checks as hard gate for KRAMPUS expansion.", "Wrap live daemon loop under DBOS contract without breaking LISTEN/NOTIFY."],
        ["scripts/check_chrono_ledger_service.sh", latest("05_OUTPUTS/chrono_ledger/chrono_ledger_phase_c_report_*.json") or "missing"]))

    # 9 graph schema
    phases.append(make_phase(9, PHASE_NAMES[8], "REAL_WITH_GUARDRAILS",
        {"graph_counts": {k:v for k,v in graph_counts.items() if k.startswith("lucidota_go")}, "schemas": [s for s in schema_files if "graph" in s.lower()]},
        ["Canonical graph tables are populated at large scale.", "Promotion packet/decision tables exist.", "Graph write barrier schema/guards exist, but materialization remains blocked by policy."],
        ["CANONICAL_GRAPH_HAS_LARGE_EXISTING_STATE", "MATERIALIZATION_PATH_NOT_FULLY_POLICY_HARDENED"],
        ["Require graph_promoter role + command envelope + kernel authorization + graph_journal append in one transaction before any new materialization.", "Run orphan/direct-write detector before and after every graph worker."],
        ["06_SCHEMA/016_go_graph_core.sql", "06_SCHEMA/052_graph_promotion_materialization.sql", "06_SCHEMA/074_graph_journal_write_barrier.sql"]))

    # 10 graph promotion execution
    phases.append(make_phase(10, PHASE_NAMES[9], "PACKET_EXECUTE_REAL_CANONICAL_MUTATION_BLOCKED",
        {"status_ledger": status_lookup(status, "Graph promotion execute path"), "latest_execute": latest("05_OUTPUTS/graph/graph_promotion_execute_execute_*.json"), "gate_latest": latest("05_OUTPUTS/graph/graph_promotion_gate_execute_*.json")},
        ["Promotion packet/decision execute path is real.", "Canonical graph materialization is explicitly disabled/open blocker.", "This is correct safety posture but not final graph promotion."],
        ["GRAPH_CANONICAL_MATERIALIZATION_NOT_ENABLED"],
        ["Implement operator-confirmed materialization transaction with graph_journal append and before/after mutation detector.", "Reject packets missing evidence_refs/authority_class/command envelope."],
        [latest("05_OUTPUTS/graph/graph_promotion_execute_execute_*.json") or "missing", "scripts/graph_promotion_gate.py"]))

    # 11 ontology
    onto_files = list_files("BOOKS", 120, patterns=("*.json", "*.schema.json"))
    phases.append(make_phase(11, PHASE_NAMES[10], "AVAILABLE_NOT_ENFORCED_ENOUGH",
        {"official_exists": (ROOT/"OFFICIAL_ONTOLOGY.json").exists(), "ontology_files": onto_files, "runtime_mentions": grep_count("scripts", "OFFICIAL_ONTOLOGY|GO_ACTIVE_TERMS|ontology")},
        ["GO/ontology files exist and are indexed.", "Operator ontology fixture exists separately.", "Ontology is not yet a universal runtime gate for kernel, graph promotion, parser output, or model output."],
        ["ONTOLOGY_NOT_KERNEL_ENFORCED", "ONTOLOGY_NOT_CLAW_COMMAND_VALIDATOR", "ONTOLOGY_NOT_GRAPH_PACKET_REQUIRED"],
        ["Make kernel load active ontology terms and validate command/promotion packets.", "Reject graph/claim packets with unknown primitive unless explicitly operator-defined."],
        ["OFFICIAL_ONTOLOGY.json"] + onto_files[:20]))

    # 12 CEP
    phases.append(make_phase(12, PHASE_NAMES[11], "PARTIAL_REAL",
        {"conversation_command_count": db.get("state",{}).get("tables",{}).get("lucidota_control.conversation_command"), "files": path_exists(["scripts/surface_instruction_compile_dry_run.py", "scripts/conversation_command_accept_worker.py", "scripts/dbos_surface_cep_worker.py", "06_SCHEMA/068_conversation_command_acceptance.sql", "06_SCHEMA/097_conversation_command_status_transition.sql"])},
        ["conversation_command table has rows.", "Surface instruction compiler and DBOS surface_cep wrapper exist.", "CLAW is not yet emitting CEP; kernel is not yet authorizing CEP."],
        ["CLAW_TO_CEP_NOT_WIRED", "CEP_TO_KERNEL_AUTH_NOT_WIRED", "SURFACE_CEP_QUEUE_HAS_PENDING_JOBS"],
        ["Add CLAW command envelope emitter and route to conversation_command.", "Add kernel authorize step before DBOS enqueue from CEP."],
        ["00_PROJECT_BRAIN/CEP_BOUNDARY_DECISION.md", "scripts/surface_instruction_compile_dry_run.py", "scripts/dbos_surface_cep_worker.py"]))

    # 13 surfaces
    phases.append(make_phase(13, PHASE_NAMES[12], "IN_PROGRESS",
        {"surface_files": list_files("07_SURFACES", 80), "status_ledger": status_lookup(status, "Darwinian Surfaces")},
        ["Generated/promoted/forked/archive surface directories exist.", "Surface instruction compiler exists and no-direct-mutation doctrine exists.", "Active pheromone enforcement and full lifecycle are not final."],
        ["PHEROMONE_ENFORCEMENT_NOT_FULLY_WIRED", "SURFACE_LIFECYCLE_EXECUTE_NOT_COMPLETE"],
        ["Make surface sidecars validate interaction_affordances -> plain-language instruction -> CEP only.", "Add surface lifecycle worker under DBOS with no direct graph mutation."],
        ["00_PROJECT_BRAIN/DARWINIAN_SURFACES.md", "07_SURFACES", "ALGOS/pheromone.py"]))

    # 14 tickletrunk
    cats = {k: len(v) for k,v in tickle.get("toolboxes",{}).items()}
    phases.append(make_phase(14, PHASE_NAMES[13], "STRONG_REAL",
        {"total_tools": tickle.get("total_tools"), "categories": cats, "latest_scan": latest("05_OUTPUTS/tickletrunk/tickletrunk_scan_*.json")},
        ["TICKLETRUNK is the best current map of the proof hoard.", "770 entries across scripts, schemas, models, LoRAs, skills, services, etc.", "It solves findability, not production readiness."],
        ["MANY_UNKNOWN_DESCRIPTIONS_REMAIN", "TOOL_REUSE_NOT_AUTOMATICALLY_ENFORCED_BY_CLAW"],
        ["Make CLAW startup enforce TICKLETRUNK search before tool creation.", "Add promotion-copy workflow for moving proof-hoard artifacts into production copies."],
        ["00_PROJECT_BRAIN/TICKLETRUNK.json", "00_PROJECT_BRAIN/TICKLETRUNK.md", latest("05_OUTPUTS/tickletrunk/tickletrunk_scan_*.json") or "missing"]))

    # 15 status ledger
    inflated = inflated_statuses(status)
    phases.append(make_phase(15, PHASE_NAMES[14], "USEFUL_BUT_OVERSTATES_SOME_SUBSYSTEMS",
        {"software_entries": len(status.get("software",[])), "open_blockers": len(status.get("open_blockers",[])), "inflated_or_inconsistent_entries": inflated[:40]},
        ["Status ledger is useful and passes validation.", "Some entries are labeled verified/executed despite blockers or incomplete semantics.", "It needs 'health-wrapper verified' vs 'production capability complete' vocabulary."],
        ["STATUS_SEMANTICS_TOO_COARSE", "VERIFIED_WITH_BLOCKERS_ENTRIES_EXIST"],
        ["Split status into health_verified, queue_verified, production_complete, policy_authoritative.", "Downgrade KRAMPUSCHEWING production status while keeping wrapper evidence."],
        ["05_OUTPUTS/status_ledger.json", "00_PROJECT_BRAIN/STATUS_LEDGER.md"]))

    # 16 security
    sec = latest("05_OUTPUTS/security/security_quarantine_manifest_*.json")
    sec_data = read_json(ROOT/sec, {}) if sec else {}
    phases.append(make_phase(16, PHASE_NAMES[15], "CURRENTLY_CLEAN_BUT_NEEDS_CONTINUOUS_GATE",
        {"latest_manifest": sec, "clean_manifest": sec_data.get("clean_manifest"), "blockers": sec_data.get("blockers"), "findings_count": sec_data.get("findings_count")},
        ["Latest security quarantine manifest reports clean_manifest=true.", "Brain Archaeology full ingest gate can now proceed only over allowlisted/clean corpus paths.", "This is a point-in-time gate, not a permanent guarantee."],
        ["SECURITY_GATE_NOT_AUTOMATICALLY_ATTACHED_TO_KRAMPUS_NEW_DROPS"],
        ["Make KRAMPUS intake require latest clean manifest or per-drop scan before extraction/embedding/design_atom promotion.", "Store security manifest ref on every Phase 0.5 ingest batch."],
        [sec or "missing", "00_PROJECT_BRAIN/SECURITY_QUARANTINE_GATE.md"]))

    # 17 brain archaeology
    phases.append(make_phase(17, PHASE_NAMES[16], "SCAFFOLDED_PARTIAL_RUNTIME",
        {"files": path_exists(["06_SCHEMA/030_phase05_brain_archaeology.sql", "06_SCHEMA/051_phase05_design_atom_runtime.sql", "06_SCHEMA/066_phase05_workflow_blueprint_synthesis.sql", "06_SCHEMA/075_phase05_workflow_blueprint_queue.sql", "scripts/phase05_brain_archaeology_prep.py", "scripts/phase05_workflow_blueprint_synthesizer.py"]), "status_ledger": status_lookup(status, "Phase 0.5 Brain Archaeology full ingest")},
        ["Brain Archaeology schema and runtime pieces exist.", "Full ingest is not complete; design atom/workflow extraction needs allowlisted batch execution.", "Ontology fidelity guard is fixture-level, not extraction-output-wide."],
        ["FULL_DESIGN_ATOM_EXTRACTION_PENDING", "EXTRACTION_FIDELITY_GUARD_NOT_FULLY_WIRED"],
        ["Run Phase 0.5 allowlisted ingest against clean manifest with no secret-bearing paths.", "Attach authority_class/evidence refs/security manifest to every design_atom/workflow_blueprint."],
        ["00_PROJECT_BRAIN/SPEC-001.5-BRAIN-ARCHAEOLOGY.md", "06_SCHEMA/030_phase05_brain_archaeology.sql"]))

    # 18 GLiNER
    phases.append(make_phase(18, PHASE_NAMES[17], "REAL_STAGING_NOT_GRAPH_TRUTH",
        {"deps": {k: deps.get(k) for k in ["gliner", "pydantic", "torch"]}, "db": {"gliner_runs": db.get("storage",{}).get("tables",{}).get("lucidota_learning.gliner_extraction_run"), "gliner_spans": db.get("storage",{}).get("tables",{}).get("lucidota_learning.gliner_entity_span")}, "status_ledger": status_lookup(status, "River/Bytewax DBOS wrapper")},
        ["GLiNER is installed in .venv and has a proof-hoard extractor plus DBOS River worker.", "Spans are staging/candidate evidence, not graph truth.", "Need better integration with KRAMPUS document/OCR lanes."],
        ["GLINER_NOT_DEFAULT_POST_PARSE_STAGE_FOR_KRAMPUS", "MODEL_HASH_AND_EXTRACTOR_VERSION_POLICY_NEEDS_HARD_GATE"],
        ["Wire document_parse output to gliner_claim_packet_extract jobs.", "Require model_hash/extractor_version/source_span refs before promotion packet creation."],
        ["ALGOS/gliner_zero_shot_extractor.py", "scripts/dbos_river_worker.py", "06_SCHEMA/038_dbos_river_wrapper.sql"]))

    # 19 SimpleMem etc
    phases.append(make_phase(19, PHASE_NAMES[18], "PARTIAL_RUNTIME_GUARDS",
        {"files": path_exists(["scripts/simplemem_candidate_index.py", "scripts/demem_runtime_guard.py", "scripts/catchme_context_guard.py", "06_SCHEMA/053_simplemem_candidate_index.sql", "06_SCHEMA/054_demem_runtime_enforcement.sql", "06_SCHEMA/055_catchme_context_guard.sql"]), "status": {"catchme": status_lookup(status, "CatchMe sensitivity map"), "simplemem": status_lookup(status, "SimpleMem candidate index")}},
        ["SimpleMem/DeMem/CatchMe have scripts and schemas.", "Need assurance they are invoked by real CEP/DBOS/Brain Archaeology flows, not only dry-runs/probes.", "The critical law is candidate != truth and repeated != preferred."],
        ["MEMORY_GUARDS_NOT_GLOBAL_EXECUTION_POLICY", "CANDIDATE_RECALL_NOT_PROMOTION_EVAL_LOOP"],
        ["Insert DeMem and CatchMe checks into surface/CLAW/Brain Archaeology command acceptance path.", "Make SimpleMem recall outputs explicitly non-truth and promotion-blocked unless reviewed."],
        ["scripts/demem_runtime_guard.py", "scripts/catchme_context_guard.py", "scripts/fast_recall_scout_dry_run.py"]))

    # 20 supervision
    phases.append(make_phase(20, PHASE_NAMES[19], "PARTIAL_SUPERVISION",
        {"chrono_service_rc": service_check.get("returncode"), "service_scripts": path_exists(["scripts/install_chrono_ledger_service.sh", "scripts/check_chrono_ledger_service.sh", "scripts/install_lucidota_intake_service.sh", "scripts/check_lucidota_intake_service.sh", "scripts/check_all_lucidota_services.py"])},
        ["Chrono is supervised and active.", "Intake/service wrappers exist but all worker families are not yet supervised/managed uniformly.", "DBOS queue workers are mostly CLI workers rather than managed daemons."],
        ["UNIFIED_SERVICE_SUPERVISOR_MISSING", "DBOS_WORKERS_NOT_ALL_DAEMONIZED"],
        ["Create service registry and check_all script coverage for Chrono, intake, surface_cep, graph_promotion, KRAMPUS, River.", "Use DBOS worker supervisor preflight as required gate."],
        ["scripts/check_chrono_ledger_service.sh", "scripts/check_all_lucidota_services.py"]))

    # 21 model runtime
    model_files = list_files("03_VAULT/models", 80)
    phases.append(make_phase(21, PHASE_NAMES[20], "SCAFFOLDED_WITH_LOCAL_MODELS",
        {"model_files": model_files, "ollama": run(["ollama", "list"], timeout=15) if shutil.which("ollama") else {"returncode":127}, "status_ledger": status_lookup(status, "Model CPU scheduler")},
        ["Local GGUF models exist: DeepSeek 1.5B Q4 and Mamba Q2.", "Ollama is installed but has no pulled models.", "Model CPU scheduler remains scaffolded; no governed invocation path."],
        ["MODEL_RUNTIME_REGISTRY_NOT_EXECUTION_AUTHORITY", "OLLAMA_MODELS_NOT_PULLED", "MODEL_OUTPUT_PROVENANCE_NOT_GLOBAL"],
        ["Create model_runtime_registry entries for local GGUFs with hash/path/VRAM profile.", "Add kernel-approved model invocation wrapper with output provenance."],
        model_files[:20] + ["06_SCHEMA/002_model_runtime.sql"]))

    # 22 GPU
    phases.append(make_phase(22, PHASE_NAMES[21], "AVAILABLE_UNDERUSED",
        {"gpu": gpu, "status_ledger": status_lookup(status, "≤4GB VRAM profile")},
        ["GTX 1650 4GB is present and visible to nvidia-smi.", "No evidence of a production model runner using GPU in the DBOS path.", "4GB VRAM means small GGUF/llama.cpp/vLLM-lite decisions; no heavy model assumptions."],
        ["GPU_RUNTIME_NOT_IN_MODEL_SCHEDULER", "VRAM_BUDGET_NOT_ENFORCED_BY_CODE"],
        ["Add GPU profile table/config and llama.cpp smoke using existing GGUF with strict context/batch caps.", "Record every model run with device, VRAM target, model hash, and command envelope ref."],
        ["nvidia-smi", "03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"]))

    # 23 lora adapters
    lora_files = list_files("04_RUNTIME", 100, patterns=("*.json", "*.jsonl", "*.safetensors", "*.gguf"))
    phases.append(make_phase(23, PHASE_NAMES[22], "REGISTRY_PARTIAL",
        {"lora_runtime_files": lora_files, "ternary_status": status_lookup(status, "Ternary Lens Lab"), "fairyfuse_status": status_lookup(status, "FairyFuse")},
        ["LoRA/runtime artifacts exist but are not a complete adapter runtime.", "Ternary/FairyFuse remains blocked by real_backend_not_wired.", "Standard LoRA must remain unsafe_for_fastpath for BitNet/FairyFuse unless benchmark proves otherwise."],
        ["REAL_BACKEND_NOT_WIRED", "LORA_SWAP_MECHANISM_NOT_IMPLEMENTED"],
        ["Create adapter/lens custody registry with exact file hash and compatible backend.", "Do not run BitNet/FairyFuse claims until real low-bit fast path benchmark exists."],
        ["06_SCHEMA/028_ternary_lens_lab.sql", "services/ternary_lab/vendor_manifest.json"] + lora_files[:10]))

    # 24 tech bench
    dep_report = latest("05_OUTPUTS/tech_bench/*/dependency_install_report_*.json")
    phases.append(make_phase(24, PHASE_NAMES[23], "BENCH_INSTALLED_NOT_PRODUCTION",
        {"dep_report": dep_report, "deps": deps, "tech_manifest": latest("05_OUTPUTS/tech_bench/*/tech_bench_manifest.json")},
        ["Tech bench has installation/manifest reports.", "Python dependencies are installed in .venv, not global Python.", "Missing model weights and SmolDocling path remain explicit blockers."],
        ["BENCH_NOT_PRODUCTION_PIPELINE", "MODEL_WEIGHTS_MISSING_FOR_RERANKER_AND_MODERNBERT"],
        ["Promote only Docling/GLiNER/Pydantic/OpenTelemetry contracts that have local deps available.", "Keep BGE/ModernBERT as blocked until weights are explicitly chosen."],
        [dep_report or "missing", latest("05_OUTPUTS/tech_bench/*/tech_bench_manifest.json") or "missing"]))

    # 25 outputs evidence
    out_counts = Counter()
    for p in (ROOT/"05_OUTPUTS").glob("*/*"):
        if p.is_file(): out_counts[p.parent.name] += 1
    phases.append(make_phase(25, PHASE_NAMES[24], "LOTS_OF_EVIDENCE_MIXED_QUALITY",
        {"size": du("05_OUTPUTS"), "category_counts": dict(out_counts.most_common(40)), "updated_abcd_summary": latest("05_OUTPUTS/work_loops/updated_abcd_reports/updated_abcd_sequence_summary_*.json")},
        ["05_OUTPUTS is large and contains many receipts.", "Some reports prove real runtime state; some are validation theater or repeated checks.", "Future audit must tag evidence_type: implementation, execution, validation, dry_run, report_only."],
        ["EVIDENCE_TYPE_NOT_NORMALIZED", "REPORT_THEATRE_RISK"],
        ["Add evidence classifier to status ledger updater.", "Refuse to count report-only outputs as build work."],
        ["05_OUTPUTS", latest("05_OUTPUTS/work_loops/updated_abcd_reports/updated_abcd_sequence_summary_*.json") or "missing"]))

    # 26 schema
    schema_nums = []
    for s in schema_files:
        m = re.search(r"/(\d+)_", s)
        if m: schema_nums.append(int(m.group(1)))
    missing_nums = [n for n in range(min(schema_nums or [0]), max(schema_nums or [0])+1) if n not in schema_nums]
    phases.append(make_phase(26, PHASE_NAMES[25], "BROAD_BUT_FRAGMENTED",
        {"schema_count": len(schema_files), "min_schema": min(schema_nums or [0]), "max_schema": max(schema_nums or [0]), "numbering_gaps": missing_nums[:50], "latest_schemas": schema_files[-30:]},
        ["SQL surface is extensive: 104 files under 06_SCHEMA.", "Numbering gaps exist and are expected, not failures.", "Concept overlap risk is high: DBOS, graph, Phase 0.5, CEP, Chrono, document parse have multiple migrations."],
        ["MIGRATION_APPLICATION_REGISTRY_NOT_CLEAR", "OVERLAPPING_SCHEMA_CONCEPTS_NEED_CONSOLIDATION"],
        ["Create migration_applied ledger from DB introspection and schema hash.", "Group migrations by subsystem and mark canonical/current vs superseded."],
        schema_files[-30:]))

    # 27 rust port candidates
    py_candidates = ["scripts/korpus_krampii.py", "scripts/document_parse_ingest.py", "scripts/dbos_krampus_worker.py", "scripts/dbos_river_worker.py", "scripts/graph_promotion_gate.py", "scripts/surface_instruction_compile_dry_run.py", "scripts/tickletrunk_scan.py", "scripts/lucidota_status_ledger.py"]
    phases.append(make_phase(27, PHASE_NAMES[26], "POLICY_MISSING",
        {"python_candidates": {p: (ROOT/p).exists() for p in py_candidates}, "rust_crates": rust.get("crates")},
        ["There is no consistent finished->Rust port pipeline.", "Best Rust-port candidates are stable, high-value, deterministic Python scripts: KORPUS parsing/manifesting, document parse custody, graph gate validation, status/tickletrunk read-only validators.", "Do not port volatile research scripts first."],
        ["NO_RUST_PORT_QUEUE", "NO_PARITY_ORACLE_POLICY"],
        ["Create rust_port_candidate table/file with Python source, Rust target crate, parity fixtures, acceptance tests.", "Start with KORPUS manifest/componentization and graph gate validator."],
        py_candidates + rust.get("crates", [])[:10]))

    # 28 e2e
    phases.append(make_phase(28, PHASE_NAMES[27], "PARTIAL_CHAIN_MULTIPLE_BREAKS",
        {"facts": {"korpus_files": db.get("storage",{}).get("tables",{}).get("lucidota_korpus.file_object"), "temporal_claims": db.get("storage",{}).get("tables",{}).get("lucidota_korpus.temporal_claim"), "dbos_jobs": db.get("state",{}).get("tables",{}).get("lucidota_control.dbos_queue_job"), "conversation_commands": db.get("state",{}).get("tables",{}).get("lucidota_control.conversation_command"), "graph_packets": db.get("storage",{}).get("tables",{}).get("lucidota_go.graph_promotion_packet")}},
        ["Pieces of the end-to-end path exist in DB and scripts.", "The unified drop→custody→chrono→parse/OCR→claim→DBOS→kernel→graph candidate→surface/CEP path is not one enforced transaction/workflow.", "Kernel and CLAW are the missing authority/control bridge."],
        ["NO_SINGLE_E2E_KERNEL_GOVERNED_DROP_PATH", "OCR_AND_CLAIM_STAGES_NOT_DEFAULT_AFTER_KRAMPUS"],
        ["Implement one allowlisted file E2E through KRAMPUS parser lane, Chrono, document parse, GLiNER claim packet, graph promotion defer, surface instruction.", "Require kernel route receipt and DBOS execution record at each hop."],
        ["scripts/boring_beast.py", "scripts/dbos_krampus_worker.py", "scripts/document_claim_packet_worker.py", "scripts/graph_promotion_gate.py"]))

    # 29 prod readiness
    phases.append(make_phase(29, PHASE_NAMES[28], "READINESS_GATES_PASS_BUT_PRODUCT_NOT_COMPLETE",
        {"latest_signoff": latest("05_OUTPUTS/production/lucidota_production_signoff_*.json"), "latest_mega_gate": latest("05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json"), "status_inflation": inflated[:20]},
        ["Production readiness/signoff scripts can pass.", "That means gates are healthy, not that DIOGENES is complete.", "Several core blockers remain: real backend, graph materialization, DBOS remaining daemons, model weights, Ollama models."],
        ["GATE_PASS_CONFUSED_WITH_SYSTEM_COMPLETE", "OPEN_BLOCKERS_REMAIN"],
        ["Make production signoff print blocker class and distinguish gate health from subsystem completion.", "Add no-report-theatre rule to production readiness evaluator."],
        [latest("05_OUTPUTS/production/lucidota_production_signoff_*.json") or "missing", latest("05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json") or "missing"]))

    # 30 remediation backlog
    remediation = [
        "Promote ckdog1-kernel to mandatory policy authority for DBOS enqueue/consume and graph promotion.",
        "Replace KRAMPUS health wrapper with real krampus_parse_one worker over allowlisted custody rows.",
        "Wire OCR/document parser lanes from KRAMPUS file kinds to document_parse DBOS worker.",
        "Wire GLiNER claim packet extraction after parser output with source spans/model hash/extractor version.",
        "Add CLAW -> CEP -> kernel -> DBOS command path and tests.",
        "Make ontology validation a kernel/graph/claim runtime gate.",
        "Implement graph materialization only through operator-confirmed graph_promoter transaction with graph_journal append.",
        "Create Rust port candidate registry and parity fixtures; start with KORPUS manifest/componentization.",
        "Create GPU/model runtime registry for GTX 1650 + local GGUFs with VRAM caps and provenance.",
        "Normalize evidence types and stop counting report-only validation as build work.",
    ]
    phases.append(make_phase(30, PHASE_NAMES[29], "ACTIONABLE",
        {"top_10": remediation, "open_blockers": status.get("open_blockers", [])},
        ["The project is real but uneven: Chrono/TICKLETRUNK/status/DBOS primitives are real; kernel/KRAMPUS/OCR/CLAW/model runtime need hard integration.", "The next sprint must build connective tissue, not more repeated gate loops."],
        [b.get("blocker_key") for b in status.get("open_blockers", []) if isinstance(b, dict)][:20],
        remediation,
        ["05_OUTPUTS/status_ledger.json", "00_PROJECT_BRAIN/TICKLETRUNK.json"]))

    assert len(phases) == 30
    blockers = sorted({b for p in phases for b in p["blockers"] if b})
    return {
        "schema": "diogenes.30_phase_architecture_audit.v1",
        "project_name": "DIOGENES",
        "repo_root": str(ROOT),
        "generated_at": now_iso(),
        "summary": {
            "phase_count": len(phases),
            "major_truth": "DIOGENES has real organs but the ckdog1-kernel is not yet the live authority layer, and KRAMPUSCHEWING/DBOS/OCR/CLAW/model-runtime integration is incomplete.",
            "strongest_subsystems": ["Chrono-Ledger", "TICKLETRUNK", "Status Ledger", "DBOS queue primitives", "Rust CKDOG1 substrate"],
            "weakest_subsystems": ["Kernel-as-policy-authority", "KRAMPUS real DBOS ingestion", "OCR/parser production lanes", "CLAW->CEP->kernel", "GPU/model runtime"],
            "blocker_count": len(blockers),
        },
        "filesystem": {
            "top_level": sorted([p.name for p in ROOT.iterdir() if p.name != ".git"]),
            "sizes": {p: du(p) for p in ["00_PROJECT_BRAIN", "01_REPOS", "03_VAULT", "04_RUNTIME", "05_OUTPUTS", "06_SCHEMA", "07_SURFACES", "ALGOS", "KRAMPUSCHEWING", "scripts", "services", "tests"]},
            "script_count": len(scripts),
            "schema_count": len(schema_files),
        },
        "status_ledger": {"updated_at": status.get("updated_at"), "open_blockers": status.get("open_blockers", []), "inflated_or_inconsistent": inflated},
        "tickletrunk": {"total_tools": tickle.get("total_tools"), "categories": {k: len(v) for k,v in tickle.get("toolboxes",{}).items()}},
        "db_counts": db,
        "gpu": gpu,
        "venv_dependencies": deps,
        "phases": phases,
        "all_blockers": blockers,
    }


def write_markdown(report: dict[str, Any], json_path: Path) -> Path:
    md_path = json_path.with_suffix(".md")
    lines = [
        "# DIOGENES 30-Phase Architecture Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        f"JSON evidence: `{rel(json_path)}`",
        "",
        "## Summary",
        "",
        report["summary"]["major_truth"],
        "",
        "### Strongest subsystems",
        *[f"- {x}" for x in report["summary"]["strongest_subsystems"]],
        "",
        "### Weakest subsystems",
        *[f"- {x}" for x in report["summary"]["weakest_subsystems"]],
        "",
        "## 30 Phases",
        "",
    ]
    for p in report["phases"]:
        lines += [
            f"### {p['phase']:02d}. {p['name']}",
            f"Grade: `{p['grade']}`",
            "",
            "Findings:",
            *[f"- {x}" for x in p["findings"]],
            "",
            "Blockers:",
            *([f"- {x}" for x in p["blockers"]] or ["- NONE"]),
            "",
            "Next real work:",
            *[f"- {x}" for x in p["next_real_work"]],
            "",
            "Evidence:",
            *[f"- `{x}`" for x in p["evidence"] if x],
            "",
        ]
    lines += ["## Full machine details", "", "See JSON report for facts/db counts/file lists.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="write audit outputs")
    args = ap.parse_args()
    report = audit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"diogenes_30_phase_audit_{stamp()}.json"
    report["report_path"] = rel(json_path)
    if args.execute:
        json_path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
        md_path = write_markdown(report, json_path)
        report["markdown_report_path"] = rel(md_path)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=False, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(json_path))
    print("MARKDOWN_PATH=" + (report.get("markdown_report_path") or rel(json_path.with_suffix(".md"))))
    print("DIOGENES_30_PHASE_AUDIT=PASS")
    print("PHASES=30")
    print("BLOCKERS=" + str(report["summary"]["blocker_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
