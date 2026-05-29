#!/usr/bin/env python3
"""Canonical LUCIDOTA status ledger maintainer.

Hard law:
If it exists, it is listed.
If it ran, it has evidence.
If it is blocked, the blocker is named.
If it changed, the ledger changed.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "00_PROJECT_BRAIN" / "STATUS_LEDGER.md"
JSON_PATH = ROOT / "05_OUTPUTS" / "status_ledger.json"
HTML_PATH = ROOT / "07_SURFACES" / "generated" / "status_ledger.html"

STATUSES = {
    "not_started",
    "scaffolded",
    "in_progress",
    "blocked",
    "dry_run_passed",
    "executed",
    "executed_local",
    "verified",
    "deprecated",
    "archived",
}
SECTIONS = ("software", "hardware_runtime_targets", "phases", "plans_workstreams")
SECTION_TITLES = {
    "software": "Software",
    "hardware_runtime_targets": "Hardware / Runtime Targets",
    "phases": "Phases",
    "plans_workstreams": "Plans / Workstreams",
}
LAW = "If it exists, it is listed.\nIf it ran, it has evidence.\nIf it is blocked, the blocker is named.\nIf it changed, the ledger changed."
STANDING_INSTRUCTION = """Before finishing any task, update the canonical LUCIDOTA status ledger.

Maintain:
- 00_PROJECT_BRAIN/STATUS_LEDGER.md
- 05_OUTPUTS/status_ledger.json

Hard law:
If it exists, it is listed.
If it ran, it has evidence.
If it is blocked, the blocker is named.
If it changed, the ledger changed.

Track, at minimum:
- SOFTWARE
- HARDWARE / RUNTIME TARGETS
- PHASES
- PLANS / WORKSTREAMS
- executed yes/no
- status
- progress percentage
- loading bar
- evidence path or command
- next action
- blocker

Do not claim completion without evidence.
Do not use vague statuses.
Do not end the task until the ledger passes `scripts/lucidota_status_ledger.py --check`."""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def bar(progress: int) -> str:
    progress = max(0, min(100, int(progress)))
    filled = progress // 10
    return f"[{'█' * filled}{'░' * (10 - filled)}] {progress}%"


@dataclass
class Entry:
    name: str
    path_or_profile: str
    status: str
    executed: str
    progress: int
    evidence: str
    next_action: str
    blockers: str = ""
    owner_or_subsystem: str = ""
    purpose: str = ""
    last_updated: str = ""

    def normalized(self) -> dict[str, Any]:
        d = asdict(self)
        d["progress"] = int(d["progress"])
        d["loading_bar"] = bar(d["progress"])
        if not d["last_updated"]:
            d["last_updated"] = now()
        return d


def seed_data() -> dict[str, Any]:
    ts = now()
    def e(name: str, path: str, status: str, executed: str, progress: int, evidence: str, next_action: str, blockers: str = "", owner: str = "", purpose: str = "") -> dict[str, Any]:
        return Entry(name, path, status, executed, progress, evidence, next_action, blockers, owner, purpose, ts).normalized()

    data = {
        "schema": "lucidota.status_ledger.v1",
        "law": LAW,
        "standing_instruction": STANDING_INSTRUCTION,
        "updated_at": ts,
        "canonical_status_vocabulary": sorted(STATUSES),
        "software": [
            e("KORPUS", "06_SCHEMA/019_korpus_krampii.sql", "executed", "yes", 70, "lucidota_storage:lucidota_korpus.file_object count=18625", "Wire continuous workers to derived queues."),
            e("KRAMPUSCHEWING", "KRAMPUSCHEWING/", "in_progress", "yes", 65, "03_VAULT/korpus_krampii/DIGESTED + 05_OUTPUTS/lucidota_krampus_printout_20260515T123339Z", "Route all new drops through Rust intake."),
            e("Chrono-Ledger", "06_SCHEMA/025_chrono_ledger_core.sql;06_SCHEMA/026_chrono_absurd_triggers.sql;01_REPOS/lucidota_etl/crates/lucidota-chrono-ledger", "verified", "yes", 82, "05_OUTPUTS/chrono_ledger/chrono_ledger_phase_b_report_20260517T002500Z.json", "Install supervised daemon and replay cursor."),
            e("ABSURD workflow spine", "06_SCHEMA/006_workflow_registry.sql;01_REPOS/lucidota_etl/crates/lucidota-launcher", "scaffolded", "yes", 35, "cargo test --workspace rc=0; 06_SCHEMA/006_workflow_registry.sql", "Bind queues to workflow registry and worker jobs."),
            e("Graph promotion engine", "06_SCHEMA/016_go_graph_core.sql", "scaffolded", "no", 25, "06_SCHEMA/016_go_graph_core.sql", "Add promotion CLI and evidence gate."),
            e("Model CPU scheduler", "06_SCHEMA/002_model_runtime.sql", "scaffolded", "no", 20, "06_SCHEMA/002_model_runtime.sql", "Connect model inventory to job fair allocator."),
            e("DIOGENES", "/home/mfspx/DIOGENES -> /home/mfspx/LUCIDOTA", "scaffolded", "no", 18, "DIOGENES symlink inspected; README references pending", "Create edge-runtime service map."),
            e("ckdog1-kernel", "01_REPOS/lucidota_etl/crates/lucidota-kernel", "verified", "yes", 45, "cargo test --workspace rc=0", "Replace health-only surface with route/control API."),
            e("FairyFuse", "services/fairyfuse/", "not_started", "no", 0, "not_created", "Create inference scaffold.", "service directory missing"),
            e("Darwinian Surfaces", "07_SURFACES/", "scaffolded", "no", 10, "07_SURFACES/generated/status_ledger.html", "Keep surfaces read-only; commands emit envelopes only."),
            e("Command Envelope Protocol", "pending schema/script", "not_started", "no", 0, "not_created", "Define command envelope JSON schema.", "schema missing"),
            e("Pheromone decay engine", "pending ALGOS/schema", "not_started", "no", 0, "not_created", "Define decay tables and algorithm scaffold.", "schema missing"),
            e("Fidelity Guard", "planned Phase 0.5 Brain Archaeology", "scaffolded", "no", 15, "Phase 0.5 prompt/spec in conversation; no repo file yet", "Materialize schema and checker."),
            e("Phase 0.5 Brain Archaeology", "planned lucidota_archaeology schema", "scaffolded", "no", 20, "Phase 0.5 architecture prompt; STATUS_LEDGER entry", "Create schema/workflow scaffolds."),
            e("Job Fair Allocator", "ALGOS/job_fair_allocator.py", "not_started", "no", 0, "not_created", "Create allocator scaffold.", "file missing"),
            e("Biometric Stream", "ALGOS/biometric_stream.py", "not_started", "no", 0, "not_created", "Create stream scaffold.", "file missing"),
            e("LUCIDOTA CLI / CLAW fork", "01_REPOS/claudecode;claw", "in_progress", "yes", 55, "git status shows modified CLAW fork files", "Record CLAW runtime contract and tests."),
            e("Status Ledger", "00_PROJECT_BRAIN/STATUS_LEDGER.md;05_OUTPUTS/status_ledger.json;scripts/lucidota_status_ledger.py", "verified", "yes", 100, "scripts/lucidota_status_ledger.py --check rc=0", "Maintain before finishing every task."),
        ],
        "hardware_runtime_targets": [
            e("post-2020 laptop profile", "Intel i5-10300H/8GB/GTX1650-class", "in_progress", "yes", 50, "developer hardware constraint; local builds executed", "Keep daemon/resource profiles conservative."),
            e("≤4GB VRAM profile", "GTX1650 4GB target", "scaffolded", "no", 20, "README/runtime planning; GGUF model present", "Create explicit FairyFuse/llama config."),
            e("CPU-only fallback", "local CPU runtime", "scaffolded", "yes", 35, "cargo builds/tests executed locally", "Add model CPU fallback policy."),
            e("Samsung S22 Ultra+ class mobile profile", "mobile edge target", "not_started", "no", 0, "not_created", "Define storage/model constraints.", "device runtime profile missing"),
            e("local Postgres over Unix socket", "postgresql:///lucidota_storage", "verified", "yes", 85, "psql lucidota_storage; Chrono migrations executed", "Add service health check."),
            e("llama.cpp / local GGUF runtime", "03_VAULT/models/*.gguf", "scaffolded", "no", 25, "summary shows DeepSeek GGUF model in 03_VAULT/models", "Wire governed model CPU scheduler."),
        ],
        "phases": [
            e("Phase 0: snapshot/review", "05_OUTPUTS/lucidota_krampus_printout_20260515T123339Z", "verified", "yes", 90, "PRINT_ME.json; summary.json; components.json", "Use as baseline evidence."),
            e("Phase 0.5: Brain Archaeology prep", "planned lucidota_archaeology", "scaffolded", "no", 20, "Phase 0.5 architecture prompt; ledger seeded", "Create SPEC-001.5 schema and workflows."),
            e("Phase 1: canonical graph promotion", "06_SCHEMA/016_go_graph_core.sql", "scaffolded", "no", 25, "06_SCHEMA/016_go_graph_core.sql", "Implement evidence-gated promotion."),
            e("Phase 2: ABSURD queue spine", "06_SCHEMA/006_workflow_registry.sql", "scaffolded", "yes", 35, "workflow registry schema; launcher crate tests", "Connect workflow events to ABSURD workers."),
            e("Phase 3: governed model CPUs", "06_SCHEMA/002_model_runtime.sql", "scaffolded", "no", 20, "06_SCHEMA/002_model_runtime.sql", "Implement scheduler and model policy."),
            e("Phase 4: DIOGENES edge runtime", "/home/mfspx/DIOGENES", "scaffolded", "no", 15, "DIOGENES symlink; ckdog kernel scaffold", "Create FairyFuse and edge service topology."),
            e("Phase 5: Darwinian Surfaces", "07_SURFACES/generated/status_ledger.html", "scaffolded", "no", 10, "status ledger HTML projection", "Define read-only projections and envelope actions."),
        ],
        "plans_workstreams": [
            e("Chrono-Ledger Phase A", "06_SCHEMA/025_chrono_ledger_core.sql", "verified", "yes", 100, "temporal_claims=24575; ranking_violations=0", "Archived into Phase B metabolism.", owner="Chrono-Ledger"),
            e("Chrono-Ledger Phase B", "06_SCHEMA/026_chrono_absurd_triggers.sql; chrono daemon", "verified", "yes", 85, "phase_b_report; daemon smoke inserted comm claims", "Install supervised daemon and replay cursor.", owner="Chrono-Ledger"),
            e("Chrono-Ledger Phase C", "systemd/replay/dead-letter planned", "not_started", "no", 0, "not_created", "Create user-systemd unit, replay cursor, dead-letter table.", owner="Chrono-Ledger"),
            e("Rust hot-path replacement", "01_REPOS/lucidota_etl", "verified", "yes", 55, "cargo test --workspace rc=0", "Replace remaining Python state ownership.", owner="Rust control plane"),
            e("KORPUS continuous ingestion", "KRAMPUSCHEWING -> lucidota_korpus", "in_progress", "yes", 60, "file_object_count=18625; DIGESTED exists", "Bind Rust intake daemon to live drop path.", owner="KORPUS"),
            e("Status ledger canon", "STATUS_LEDGER.md/status_ledger.json", "verified", "yes", 100, "scripts/lucidota_status_ledger.py --check rc=0", "Update on every implementation pass.", owner="Project governance"),
        ],
    }
    return data


def load_data() -> dict[str, Any]:
    if not JSON_PATH.exists():
        return seed_data()
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def save_data(data: dict[str, Any], render_html: bool = False) -> None:
    data["updated_at"] = now()
    for section in SECTIONS:
        for entry in data.get(section, []):
            entry["progress"] = int(entry.get("progress", 0))
            entry["loading_bar"] = bar(entry["progress"])
            entry.setdefault("last_updated", data["updated_at"])
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(render_markdown(data), encoding="utf-8")
    if render_html:
        HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        HTML_PATH.write_text(render_html_view(data), encoding="utf-8")


def esc_cell(value: Any) -> str:
    text = str(value or "")
    return text.replace("\n", "<br>").replace("|", "\\|")


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# LUCIDOTA STATUS LEDGER",
        "",
        LAW,
        "",
        f"Last updated: `{data.get('updated_at', now())}`",
        "",
        "## Standing Codex Instruction",
        "",
        STANDING_INSTRUCTION,
        "",
    ]
    if data.get("hard_laws"):
        lines += ["## Hard Laws", ""]
        for law in data.get("hard_laws", []):
            lines += [f"### {law.get('title', law.get('law_key', 'Hard Law'))}", "", str(law.get("body", "")).strip(), ""]

    if data.get("open_blockers"):
        lines += ["## Open Blockers", "", "| Blocker | Status | Severity | Applies To | Meaning | Next Action |", "|---|---:|---:|---|---|---|"]
        for blocker in data.get("open_blockers", []):
            lines.append("| " + " | ".join(esc_cell(x) for x in [
                blocker.get("blocker_key", ""),
                blocker.get("status", ""),
                blocker.get("severity", ""),
                ", ".join(blocker.get("applies_to", [])),
                blocker.get("meaning", ""),
                blocker.get("next_action", ""),
            ]) + " |")
        lines.append("")
    render_section(lines, data, "software", ["Module", "Path", "Status", "Executed?", "Progress", "Evidence", "Next Action", "Blockers"])
    render_section(lines, data, "hardware_runtime_targets", ["Target", "Profile", "Status", "Executed?", "Progress", "Evidence", "Next Action", "Blockers"])
    render_section(lines, data, "phases", ["Phase", "Purpose", "Status", "Executed?", "Progress", "Evidence", "Next Action", "Blockers"])
    render_section(lines, data, "plans_workstreams", ["Plan", "Owner/Subsystem", "Status", "Executed?", "Progress", "Evidence", "Next Action", "Blockers"])
    return "\n".join(lines).rstrip() + "\n"


def render_section(lines: list[str], data: dict[str, Any], section: str, headers: list[str]) -> None:
    lines += [f"## {SECTION_TITLES[section]}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for entry in data.get(section, []):
        if section == "software":
            row = [entry["name"], entry["path_or_profile"], entry["status"], entry["executed"], entry["loading_bar"], entry["evidence"], entry["next_action"], entry.get("blockers", "")]
        elif section == "hardware_runtime_targets":
            row = [entry["name"], entry["path_or_profile"], entry["status"], entry["executed"], entry["loading_bar"], entry["evidence"], entry["next_action"], entry.get("blockers", "")]
        elif section == "phases":
            row = [entry["name"], entry.get("purpose") or entry["path_or_profile"], entry["status"], entry["executed"], entry["loading_bar"], entry["evidence"], entry["next_action"], entry.get("blockers", "")]
        else:
            row = [entry["name"], entry.get("owner_or_subsystem") or entry["path_or_profile"], entry["status"], entry["executed"], entry["loading_bar"], entry["evidence"], entry["next_action"], entry.get("blockers", "")]
        lines.append("| " + " | ".join(esc_cell(x) for x in row) + " |")
    lines.append("")


def render_html_view(data: dict[str, Any]) -> str:
    sections = []
    for section in SECTIONS:
        rows = []
        for entry in data.get(section, []):
            envelope = {
                "protocol": "lucidota.command_envelope.v1",
                "intent": "status_ledger_update_request",
                "target": entry["name"],
                "section": section,
                "direct_db_write": False,
            }
            rows.append(
                "<tr>"
                f"<td>{html.escape(entry['name'])}</td>"
                f"<td>{html.escape(entry.get('path_or_profile',''))}</td>"
                f"<td>{html.escape(entry['status'])}</td>"
                f"<td>{html.escape(entry['executed'])}</td>"
                f"<td><code>{html.escape(entry['loading_bar'])}</code></td>"
                f"<td>{html.escape(entry.get('evidence',''))}</td>"
                f"<td>{html.escape(entry.get('next_action',''))}</td>"
                f"<td>{html.escape(entry.get('blockers',''))}</td>"
                f"<td><code>{html.escape(json.dumps(envelope, separators=(',',':')))}</code></td>"
                "</tr>"
            )
        sections.append(f"<h2>{html.escape(SECTION_TITLES[section])}</h2><table><thead><tr><th>Name</th><th>Path/Profile</th><th>Status</th><th>Executed</th><th>Progress</th><th>Evidence</th><th>Next</th><th>Blockers</th><th>Command Envelope Seed</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")
    return """<!doctype html><html><head><meta charset="utf-8"><title>LUCIDOTA Status Ledger</title><style>body{font-family:system-ui,sans-serif;margin:2rem;background:#111;color:#eee}table{border-collapse:collapse;width:100%;margin-bottom:2rem}td,th{border:1px solid #444;padding:.35rem;vertical-align:top}th{background:#222}code{color:#9ef}pre{white-space:pre-wrap}</style></head><body>""" + f"<h1>LUCIDOTA STATUS LEDGER</h1><pre>{html.escape(LAW)}</pre><p>Last updated: <code>{html.escape(data.get('updated_at',''))}</code></p>" + "".join(sections) + "</body></html>\n"


def find_entry(data: dict[str, Any], name: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    lower = name.lower()
    for section in SECTIONS:
        for entry in data.get(section, []):
            if entry.get("name", "").lower() == lower:
                return section, entry
    return None, None


def set_entry(args: argparse.Namespace) -> None:
    data = load_data()
    section, entry = find_entry(data, args.set_name)
    if entry is None:
        section = "software"
        entry = Entry(args.set_name, args.evidence or "", args.status, args.executed, args.progress, args.evidence or "", args.next or "", args.blocker or "", last_updated=now()).normalized()
        data.setdefault(section, []).append(entry)
    entry["status"] = args.status
    entry["progress"] = args.progress
    entry["executed"] = args.executed
    if args.evidence is not None:
        entry["evidence"] = args.evidence
    if args.next is not None:
        entry["next_action"] = args.next
    if args.blocker is not None:
        entry["blockers"] = args.blocker
    entry["last_updated"] = now()
    save_data(data, render_html=HTML_PATH.exists())


def set_next_from_target_report(path: str) -> None:
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    data_report = json.loads(report_path.read_text(encoding="utf-8"))
    if data_report.get("status") != "PASS":
        raise SystemExit("target report status is not PASS")
    target_key = data_report.get("target_key") or (data_report.get("target") or {}).get("target_key") or data_report.get("recommended")
    if not target_key:
        raise SystemExit("target report missing target_key/recommended")
    data = load_data()
    section, entry = find_entry(data, "Post-Gate Target Selection")
    if entry is None:
        entry = Entry("Post-Gate Target Selection", "scripts/post_gate_target_selector.py", "verified", "yes", 100, rel_path(report_path), f"Execute selected target: {target_key}", last_updated=now()).normalized()
        data.setdefault("software", []).append(entry)
    entry["status"] = "verified"
    entry["executed"] = "yes"
    entry["progress"] = 100
    entry["evidence"] = rel_path(report_path)
    entry["next_action"] = f"Execute selected target: {target_key}"
    entry["blockers"] = ""
    entry["last_updated"] = now()
    save_data(data, render_html=HTML_PATH.exists())

def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)

def check() -> list[str]:
    errors: list[str] = []
    if not MD_PATH.exists():
        errors.append(f"missing {MD_PATH.relative_to(ROOT)}")
    if not JSON_PATH.exists():
        errors.append(f"missing {JSON_PATH.relative_to(ROOT)}")
        return errors
    try:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid json: {exc}"]
    for section in SECTIONS:
        if section not in data:
            errors.append(f"missing section {section}")
            continue
        for idx, entry in enumerate(data[section]):
            name = entry.get("name", f"{section}[{idx}]")
            status = entry.get("status")
            if status not in STATUSES:
                errors.append(f"{name}: invalid status {status}")
            try:
                progress = int(entry.get("progress"))
                if not 0 <= progress <= 100:
                    errors.append(f"{name}: progress outside 0-100")
            except Exception:
                errors.append(f"{name}: progress not integer")
            executed = entry.get("executed")
            if executed not in {"yes", "no"}:
                errors.append(f"{name}: executed must be yes/no")
            if executed == "yes" and not str(entry.get("evidence", "")).strip():
                errors.append(f"{name}: executed=yes missing evidence")
            if status == "blocked" and not str(entry.get("blockers", "")).strip():
                errors.append(f"{name}: blocked without blocker")
            expected_bar = bar(int(entry.get("progress", 0))) if str(entry.get("progress", "")).isdigit() else None
            if expected_bar and entry.get("loading_bar") != expected_bar:
                errors.append(f"{name}: loading_bar mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Maintain canonical LUCIDOTA status ledger")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--render-html", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--set", dest="set_name")
    parser.add_argument("--status", choices=sorted(STATUSES))
    parser.add_argument("--progress", type=int)
    parser.add_argument("--executed", choices=["yes", "no"])
    parser.add_argument("--evidence")
    parser.add_argument("--next")
    parser.add_argument("--blocker", default=None)
    parser.add_argument("--set-next-from-target-report")
    args = parser.parse_args()

    if args.init:
        save_data(seed_data(), render_html=args.render_html)
    if args.set_next_from_target_report:
        set_next_from_target_report(args.set_next_from_target_report)
    if args.set_name:
        missing = [flag for flag in ("status", "progress", "executed") if getattr(args, flag) is None]
        if missing:
            print(f"missing required --set fields: {', '.join(missing)}", file=sys.stderr)
            return 2
        set_entry(args)
    if args.render_html and not args.init:
        data = load_data()
        HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
        HTML_PATH.write_text(render_html_view(data), encoding="utf-8")
    if args.check:
        errors = check()
        if errors:
            for error in errors:
                print(f"CHECK_FAIL {error}", file=sys.stderr)
            return 1
        print("CHECK_OK status ledger valid")
    if not any([args.init, args.set_name, args.render_html, args.check, args.set_next_from_target_report]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
