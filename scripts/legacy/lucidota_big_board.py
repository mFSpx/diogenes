#!/usr/bin/env python3
"""LUCIDOTA terminal Big Board v0.

Read-only dashboard: docs are truth for bars; Postgres supplies live counters.
No Drive access, no writes, no secrets.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "00_PROJECT_BRAIN" / "BUILD_PLAN_AUDIT.md"
STATE_DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
GRAPH_DB = os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", "postgresql://mfspx@/lucidota_graph")
STORAGE_DB = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
KRAMPUS_DIR = ROOT / "KRAMPUSCHEWING"
LIVE_HTML = ROOT / "05_OUTPUTS" / "ingestion_live_dashboard.html"


def bars() -> dict:
    if PLAN.exists():
        text = PLAN.read_text(encoding="utf-8")
        overall = re.search(r"## Overall Build Bar: (.+)", text)
        phases: list[str] = []
        in_bars = False
        for line in text.splitlines():
            if line == "## Phase Bars":
                in_bars = True
                continue
            if in_bars and line.startswith("## "):
                break
            if in_bars and line.startswith("- **"):
                phases.append(re.sub(r"^- \*\*(.*?)\*\* `(.*?)`(.*)$", r"\1 \2\3", line))
        return {"overall": overall.group(1) if overall else "unknown", "phases": phases, "source": str(PLAN.relative_to(ROOT))}
    return {
        "overall": "graph_first_no_markdown_bar",
        "phases": [
            f"graph_items {scalar(os.environ.get('LUCIDOTA_GO_STORAGE_DSN', 'postgresql:///lucidota_storage'), 'SELECT count(*) FROM lucidota_go.graph_item')}",
            f"indy_books {scalar(os.environ.get('LUCIDOTA_GO_STORAGE_DSN', 'postgresql:///lucidota_storage'), 'SELECT count(*) FROM lucidota_indy.book_source')}",
            f"indy_chunks {scalar(os.environ.get('LUCIDOTA_GO_STORAGE_DSN', 'postgresql:///lucidota_storage'), 'SELECT count(*) FROM lucidota_indy.book_chunk')}",
        ],
        "source": "postgres",
    }


def scalar(db_url: str, sql: str):
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            return conn.execute(sql).fetchone()[0]
    except Exception as exc:
        return f"error:{str(exc)[:80]}"


def rows(db_url: str, sql: str, params: tuple = ()) -> list[tuple]:
    try:
        with psycopg.connect(db_url, connect_timeout=3) as conn:
            return list(conn.execute(sql, params).fetchall())
    except psycopg.Error:
        return []


def drive_manifest_count() -> int:
    try:
        from lucidota_drive_manifest import scan
        return len(scan())
    except (ImportError, OSError):
        return -1


def counters() -> dict:
    return {
        "workflow_events": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.workflow_event"),
        "registered_workflows": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.workflow_registry"),
        "scheduled_workflows": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.workflow_schedule WHERE enabled=true"),
        "signoff_pending": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.governance_gate WHERE approval_status='pending'"),
        "signoff_approved": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.governance_gate WHERE approval_status='approved'"),
        "wake_pending": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.event_outbox WHERE status='pending'"),
        "wake_delivered": scalar(STATE_DB, "SELECT count(*) FROM lucidota_control.event_outbox WHERE status='delivered'"),
        "cas_artifacts": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_vault.cas_manifest"),
        "body_capture_captures": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_body_capture.capture"),
        "body_capture_bundles": scalar(GRAPH_DB, "SELECT count(*) FROM lucidota_body_capture.evidence_bundle"),
        "river_scores": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.river_score"),
        "bytewax_hints": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.bytewax_hint"),
        "treelite_runs": scalar(STATE_DB, "SELECT count(*) FROM lucidota_learning.treelite_router_run"),
        "indy_queue": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.side_queue WHERE status='queued'"),
        "auth_records": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.auth_inventory"),
        "operator_corrections": scalar(STATE_DB, "SELECT count(*) FROM lucidota_indy.task_memory WHERE kind='correction'"),
        "drive_manifest_targets": drive_manifest_count(),
        "model_governor_decisions": scalar(STATE_DB, "SELECT count(*) FROM lucidota_runtime.load_governor_decision"),
    }


def ingestion_report() -> dict:
    target = str(KRAMPUS_DIR)
    raw = rows(
        STORAGE_DB,
        """
        SELECT count(*), count(DISTINCT sha256_hash), count(*)-count(DISTINCT sha256_hash),
               coalesce(sum(file_size_bytes),0), max(ingested_at)
        FROM lucidota_archaeology.raw_file_inventory
        WHERE absolute_path LIKE %s
        """,
        (target + "/%",),
    )
    latest_batches = rows(
        STORAGE_DB,
        """
        SELECT batch_uuid::text, status, batch_label, file_count, new_file_count,
               duplicate_file_count, component_count, entity_count, concept_count,
               error_count, skipped_count, started_at::text, finished_at::text
        FROM lucidota_korpus.ingest_batch
        ORDER BY started_at DESC LIMIT 5
        """,
    )
    wf = rows(
        STATE_DB,
        """
        SELECT workflow_id, phase, status, count(*), max(created_at)::text
        FROM lucidota_control.workflow_event
        WHERE workflow_id IN ('korpus-krampii-mass-ingest','korpus-krampii-embed-pending','hard-truth-math','commdump-universal-ingest','body_capture-capture')
           OR workflow_id LIKE 'korpus%%'
        GROUP BY 1,2,3 ORDER BY max(created_at) DESC LIMIT 12
        """,
    )
    procs = subprocess.run(
        ["pgrep", "-af", "krampuschewing|korpus_krampii|krampus_rechronologize|lucidota_hard_truth_math|002_chrono_harvester"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip().splitlines()
    log_tail = []
    log_path = ROOT / "04_RUNTIME" / "krampuschewing_watcher.log"
    if log_path.exists():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-28:]
    size = 0
    file_count = 0
    if KRAMPUS_DIR.exists():
        for root, _dirs, files in os.walk(KRAMPUS_DIR):
            file_count += len(files)
            for name in files:
                try:
                    size += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    def safe_count(table: str) -> Any:
        return scalar(STORAGE_DB, f"SELECT count(*) FROM {table}")
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "watch_dir": target,
        "watch_dir_exists": KRAMPUS_DIR.exists(),
        "watch_dir_files": file_count,
        "watch_dir_size_bytes": size,
        "raw_inventory": raw[0] if raw else (0, 0, 0, 0, None),
        "korpus": {
            "files": safe_count("lucidota_korpus.file_object"),
            "occurrences": safe_count("lucidota_korpus.file_occurrence"),
            "components": safe_count("lucidota_korpus.component"),
            "entities": safe_count("lucidota_korpus.entity"),
            "concepts": safe_count("lucidota_korpus.concept"),
            "near_duplicates": safe_count("lucidota_korpus.near_duplicate"),
            "vibe_telemetry": safe_count("lucidota_korpus.vibe_telemetry"),
            "river_decisions": safe_count("lucidota_korpus.river_decision"),
            "deferred": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_korpus.file_object WHERE status='deferred'"),
        },
        "commdump": {
            "exports": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_commdump.export_object"),
            "threads": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_commdump.thread"),
            "messages": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_commdump.message"),
        },
        "graph": {
            "items": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_go.graph_item"),
            "edges": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_go.graph_edge"),
        },
        "cas": {"objects": scalar(STORAGE_DB, "SELECT count(*) FROM lucidota_vault.cas_manifest")},
        "latest_batches": latest_batches,
        "workflow_events": wf,
        "processes": procs,
        "log_tail": log_tail,
        "invariants": [
            "No LLM call in KORPUS/KRAMPUS fast path.",
            "SHA256 dedupe is the file identity spine.",
            "UUIDs are assigned in Postgres tables.",
            "Components/entities/concepts/vibe telemetry are deterministic extractors.",
            "River ML learns one component at a time in chronological mode.",
            "Unsupported binaries are stored/deferred instead of crashing ingestion.",
            "Workflow events are recorded in lucidota_control.workflow_event.",
        ],
    }


def fmt_bytes(n: Any) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return str(n)


def render_ingest_html(report: dict) -> str:
    raw_total, raw_distinct, raw_dupes, raw_bytes, raw_latest = report["raw_inventory"]
    def esc(x: Any) -> str:
        return html.escape(str(x if x is not None else ""))
    def metric(label: str, value: Any, sub: str = "") -> str:
        return f"<div class='metric'><b>{esc(value)}</b><span>{esc(label)}</span><small>{esc(sub)}</small></div>"
    batch_rows = "\n".join(
        f"<tr><td>{esc(r[1])}</td><td>{esc(r[2])}</td><td>{esc(r[3])}</td><td>{esc(r[6])}</td><td>{esc(r[7])}</td><td>{esc(r[8])}</td><td>{esc(r[9])}</td><td>{esc(r[11])}</td></tr>"
        for r in report["latest_batches"]
    ) or "<tr><td colspan='8'>no batches yet</td></tr>"
    wf_rows = "\n".join(
        f"<tr><td>{esc(r[0])}</td><td>{esc(r[1])}</td><td>{esc(r[2])}</td><td>{esc(r[3])}</td><td>{esc(r[4])}</td></tr>"
        for r in report["workflow_events"]
    ) or "<tr><td colspan='5'>no workflow events yet</td></tr>"
    log_lines = "\n".join(f"<div>{esc(line)}</div>" for line in report["log_tail"]) or "<div>no watcher log yet</div>"
    proc_lines = "\n".join(f"<div>{esc(p)}</div>" for p in report["processes"]) or "<div>no active ingestion processes detected</div>"
    inv = "".join(f"<li>{esc(x)}</li>" for x in report["invariants"])
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="1">
<title>LUCIDOTA Live Ingestion Dashboard</title>
<style>
:root{{--bg:#07080d;--panel:#101623;--panel2:#151d2d;--text:#edf6ff;--muted:#8ea4bd;--good:#55ffb0;--warn:#ffd166;--bad:#ff5c8a;--line:#26364f;--cyan:#66e3ff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top left,#182641,#07080d 38%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Arial,sans-serif}}
header{{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:end;background:rgba(7,8,13,.75);position:sticky;top:0}}
h1{{margin:0;font-size:25px;letter-spacing:.08em}} .clock{{font-size:22px;color:var(--good);font-variant-numeric:tabular-nums}} .sub{{color:var(--muted);font-size:12px}}
main{{padding:22px;display:grid;gap:18px;grid-template-columns:1.1fr 1fr}} section{{background:linear-gradient(180deg,var(--panel),#0b101a);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 16px 40px #0008}}
.wide{{grid-column:1/-1}} h2{{margin:0 0 12px;font-size:15px;text-transform:uppercase;letter-spacing:.14em;color:var(--cyan)}} .grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}
.metric{{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:14px;min-height:86px}} .metric b{{display:block;font-size:25px;color:var(--good);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .metric span{{display:block;color:var(--text);font-size:12px;text-transform:uppercase;letter-spacing:.08em}} .metric small{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.08em}}
.log{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;line-height:1.35;max-height:360px;overflow:hidden;color:#cfe3ff;background:#060912;border-radius:12px;padding:12px;border:1px solid var(--line)}}
.pill{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;margin:3px;background:#0d1421;color:var(--muted);font-size:12px}} ul{{margin:0;padding-left:20px;color:#dcecff}} li{{margin:6px 0}} .ok{{color:var(--good)}} .warn{{color:var(--warn)}} @media(max-width:1100px){{main{{grid-template-columns:1fr}}.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head>
<body><header><div><h1>LUCIDOTA / KRAMPUS LIVE INGESTION</h1><div class="sub">non-interactive · deterministic · auto-refresh 1s · {esc(report['watch_dir'])}</div></div><div><div class="clock">{esc(report['generated_at'])}</div><div class="sub">open this file; watcher keeps chewing</div></div></header>
<main>
<section class="wide"><h2>Intake State</h2><div class="grid">
{metric('KRAMPUSCHEWING files on disk', report['watch_dir_files'], fmt_bytes(report['watch_dir_size_bytes']))}
{metric('raw chrono inventory rows', raw_total, f'{raw_distinct} unique hashes · {raw_dupes} duplicate paths')}
{metric('raw inventory bytes', fmt_bytes(raw_bytes), f'latest {raw_latest}')}
{metric('active processes', len(report['processes']), 'watcher / korpus / chrono / hardmath')}
</div></section>
<section><h2>KORPUS / River / CAS</h2><div class="grid">
{metric('file objects', report['korpus']['files'])}{metric('occurrences', report['korpus']['occurrences'])}{metric('components', report['korpus']['components'])}{metric('entities', report['korpus']['entities'])}
{metric('concepts', report['korpus']['concepts'])}{metric('near duplicates', report['korpus']['near_duplicates'])}{metric('vibe telemetry', report['korpus']['vibe_telemetry'])}{metric('River decisions', report['korpus']['river_decisions'])}
{metric('deferred files', report['korpus']['deferred'])}{metric('CAS objects', report['cas']['objects'])}{metric('graph items', report['graph']['items'])}{metric('graph edges', report['graph']['edges'])}
</div></section>
<section><h2>Communications / DBOS Control</h2><div class="grid">
{metric('comm exports', report['commdump']['exports'])}{metric('threads', report['commdump']['threads'])}{metric('messages', report['commdump']['messages'])}{metric('workflow event rows', counters().get('workflow_events'))}
</div><h2 style="margin-top:18px">Execution Invariants</h2><ul>{inv}</ul></section>
<section class="wide"><h2>Latest KORPUS Batches</h2><table><thead><tr><th>Status</th><th>Label</th><th>Files</th><th>Components</th><th>Entities</th><th>Concepts</th><th>Errors</th><th>Started</th></tr></thead><tbody>{batch_rows}</tbody></table></section>
<section class="wide"><h2>DBOS / Workflow Events</h2><table><thead><tr><th>Workflow</th><th>Phase</th><th>Status</th><th>Count</th><th>Latest</th></tr></thead><tbody>{wf_rows}</tbody></table></section>
<section><h2>Active Processes</h2><div class="log">{proc_lines}</div></section>
<section><h2>Watcher Log Tail</h2><div class="log">{log_lines}</div></section>
</main></body></html>"""


def write_ingest_html(path: Path = LIVE_HTML) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ingest_html(ingestion_report()), encoding="utf-8")
    return path


def scraper_fleet() -> dict:
    """Summarize local scraper/extractor readiness without touching Drive/web."""
    script_names = [
        "lucidota_survey.py",
        "lucidota_hop_pivot.py",
        "lucidota_body_capture.py",
        "lucidota_browser_body_capture.py",
        "lucidota_extractor_registry.py",
    ]
    local_scripts = [name for name in script_names if (ROOT / "scripts" / name).exists()]

    manifest_targets = []
    try:
        from lucidota_drive_manifest import scan

        manifest_targets = [
            hit
            for hit in scan()
            if "scraper" in (ROOT / hit["path"]).read_text(errors="ignore").lower()
        ][:8]
    except (ImportError, OSError, KeyError):
        manifest_targets = []

    adapters: list[dict] = []
    try:
        with psycopg.connect(GRAPH_DB, connect_timeout=3) as conn:
            rows = conn.execute(
                """
                SELECT adapter_id, adapter_kind, stability, default_priority, browser_required
                FROM lucidota_extract.adapter
                ORDER BY default_priority ASC, adapter_id
                """
            ).fetchall()
        adapters = [
            dict(
                zip(
                    ["adapter_id", "adapter_kind", "stability", "default_priority", "browser_required"],
                    row,
                )
            )
            for row in rows
        ]
    except psycopg.Error:
        adapters = []

    return {
        "status": "ok" if local_scripts and adapters else "partial",
        "policy": "adapters_first_browser_last",
        "local_scripts": local_scripts,
        "authorized_adapters": adapters,
        "browser_required_adapters": sum(1 for a in adapters if a.get("browser_required")),
        "manifest_scraper_targets": [
            {"path": hit["path"], "score": hit["score"]} for hit in manifest_targets
        ],
    }


def gpu() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"status": "missing"}
    try:
        out = subprocess.check_output([exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
        return {"status": "ok", "summary": out}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:120]}


def render(report: dict) -> str:
    lines = ["LUCIDOTA BIG BOARD v0", "=" * 22, f"Overall {report['bars']['overall']}", "", "Build phases:"]
    lines.extend(f"  {p}" for p in report["bars"]["phases"][:16])
    lines += ["", "Live counters:"]
    lines.extend(f"  {k}: {v}" for k, v in report["counters"].items())
    lines += ["", "GPU:", f"  {report['gpu']}"]
    sf = report.get("scraper_fleet", {})
    lines += [
        "",
        "Scraper fleet:",
        f"  status: {sf.get('status', 'unknown')}  policy: {sf.get('policy', 'unknown')}",
        f"  scripts: {', '.join(sf.get('local_scripts', []))}",
        f"  authorized_adapters: {len(sf.get('authorized_adapters', []))}  browser_required: {sf.get('browser_required_adapters', '?')}",
        f"  manifest_targets: {len(sf.get('manifest_scraper_targets', []))}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-big-board")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ingest-html", action="store_true", help="write one-page live ingestion dashboard HTML")
    ap.add_argument("--output", type=Path, default=LIVE_HTML)
    ap.add_argument("--watch", action="store_true", help="rewrite dashboard every interval seconds")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    if args.ingest_html:
        while True:
            out = write_ingest_html(args.output)
            print(out)
            if not args.watch:
                return 0
            time.sleep(max(0.2, args.interval))
    report = {"ok": True, "bars": bars(), "counters": counters(), "gpu": gpu(), "scraper_fleet": scraper_fleet()}
    print(json.dumps(report, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
