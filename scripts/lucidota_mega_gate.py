#!/usr/bin/env python3
"""Full LUCIDOTA mega-gate.

Repairs v2:
1. compile critical gate scripts before running the gate;
2. regenerate TICKLETRUNK before checking it;
3. parse child JSON reports instead of trusting rc/stdout alone;
4. enforce cross-system invariants from parsed reports/service output;
5. require evidence report paths for report-producing steps;
6. emit compact component metrics so PASS is auditable at a glance.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "mega_gate"
CRITICAL_PY = [
    "scripts/cep_full_e2e.py",
    "scripts/conversation_command_accept_worker.py",
    "scripts/spine_kernel_authorization.py",
    "scripts/spine_queue_soak_test.py",
    "scripts/spine_surface_cep_worker.py",
    "scripts/chrono_full_conservation_gate.py",
    "scripts/graph_canonical_mutation_detector.py",
    "scripts/graph_promotion_gate.py",
    "scripts/system_graph_safety_audit.py",
    "scripts/system_archaeology_evidence_audit.py",
    "scripts/surface_instruction_compile_dry_run.py",
    "scripts/surface_lineage_validator.py",
    "scripts/tickletrunk_scan.py",
    "scripts/lucidota_status_ledger.py",
    "scripts/system_runtime_facts_refresh.py",
    "scripts/lucidota_mega_gate.py",
    "scripts/dev_order_matrix_wrapper.py",
    "scripts/matrix_trace_checker.py",
    "scripts/dev_order_gate.py",
    "scripts/run_dev_order_methodology_checks.py",
]
REPAIRS = [
    "compile_critical_gate_scripts_first",
    "run_tickletrunk_execute_before_check",
    "parse_child_json_reports_and_validate_status",
    "enforce_cross_system_invariants",
    "require_report_paths_for_report_steps",
    "emit_component_metrics_summary",
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def parse_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = ROOT / path
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def run(cmd: list[str], *, name: str, require_report: bool = True, expected_status: str | None = "PASS") -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    report_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            report_path = line.split("=", 1)[1]
    report = parse_report(report_path)
    step = {
        "name": name,
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "report_path": report_path,
        "report": report,
        "stdout_head": proc.stdout[:1800],
        "stdout_tail": proc.stdout[-1800:],
        "stderr_head": proc.stderr[:1800],
        "stderr_tail": proc.stderr[-1800:],
        "passed": proc.returncode == 0,
        "blockers": [],
    }
    if require_report and not report_path:
        step["blockers"].append("missing_report_path")
    if require_report and report_path and report is None:
        step["blockers"].append("report_path_unreadable")
    if isinstance(report, dict) and report.get("_parse_error"):
        step["blockers"].append(f"report_parse_error:{report['_parse_error']}")
    if expected_status and isinstance(report, dict):
        status = report.get("status")
        if status is not None and status != expected_status:
            step["blockers"].append(f"report_status_not_{expected_status}:{status}")
        if report.get("pass") is False:
            step["blockers"].append("report_pass_false")
        if report.get("blockers"):
            step["blockers"].append("child_report_blockers_present")
    if proc.returncode != 0:
        step["blockers"].append(f"nonzero_rc:{proc.returncode}")
    step["passed"] = not step["blockers"]
    return step


def compile_step() -> dict[str, Any]:
    missing = [p for p in CRITICAL_PY if not (ROOT / p).exists()]
    cmd = [sys.executable, "-m", "py_compile", *CRITICAL_PY]
    if missing:
        return {"name": "py_compile", "command": " ".join(cmd), "rc": 2, "report_path": None, "report": None, "stdout_head": "", "stdout_tail": "", "stderr_head": "missing:" + ",".join(missing), "stderr_tail": "missing:" + ",".join(missing), "passed": False, "blockers": ["critical_python_missing"]}
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    blockers = [] if proc.returncode == 0 else [f"py_compile_rc:{proc.returncode}"]
    return {"name": "py_compile", "command": " ".join(cmd), "rc": proc.returncode, "report_path": None, "report": None, "stdout_head": proc.stdout[:1800],
        "stdout_tail": proc.stdout[-1800:], "stderr_head": proc.stderr[:1800],
        "stderr_tail": proc.stderr[-1800:], "passed": not blockers, "blockers": blockers}


def extract_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"steps_total": len(steps), "steps_passed": sum(1 for s in steps if s.get("passed"))}
    for step in steps:
        report = step.get("report") if isinstance(step.get("report"), dict) else {}
        name = step.get("name")
        if name == "golden_path":
            metrics["golden_path"] = {
                "status": report.get("status"),
                "receipt_refs": len(report.get("receipt_refs") or []),
                "db_writes_performed": report.get("db_writes_performed"),
                "graph_writes_performed": report.get("graph_writes_performed"),
                "canonical_graph_writes_performed": report.get("canonical_graph_writes_performed"),
            }
        elif name == "absurd_soak":
            metrics["absurd_soak"] = {k: report.get(k) for k in ("inserted_new", "duplicates_reused", "claimed", "forced_failures", "succeeded", "dead_lettered")}
        elif name == "chrono_conservation":
            metrics["chrono_conservation_status"] = report.get("status")
        elif name == "graph_safety":
            metrics["graph_safety_status"] = report.get("status")
        elif name == "archaeology_evidence":
            metrics["archaeology_evidence_status"] = report.get("status")
        elif name == "surface_lineage":
            metrics["surface_lineage"] = {k: report.get(k) for k in ("lineage_checked", "failures", "status")}
        elif name == "tickletrunk_execute":
            metrics["tickletrunk_total_tools"] = report.get("total_entries") or report.get("total_tools")
        elif name == "status_ledger_check":
            metrics["status_ledger_check"] = "PASS" if step.get("rc") == 0 else "FAIL"
        elif name == "chrono_service":
            out = (step.get("stdout_head") or "") + "\n" + (step.get("stdout_tail") or "") + "\n" + (step.get("stderr_head") or "") + "\n" + (step.get("stderr_tail") or "")
            m = re.search(r"ranking_violations\s+-+\s+(\d+)", out, re.S)
            metrics["chrono_service_active"] = (step.get("rc") == 0) and ("ranking_violations" in out or "Active: active (running)" in out)
            metrics["chrono_ranking_violations"] = int(m.group(1)) if m else None
    return metrics


def invariant_blockers(steps: list[dict[str, Any]], metrics: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    golden = metrics.get("golden_path") or {}
    if golden.get("status") not in {"PASS_DRY_RUN", "PASS"}:
        blockers.append("golden_path_not_passed")
    if int(golden.get("receipt_refs") or 0) < 3:
        blockers.append("golden_path_receipts_insufficient")
    if golden.get("db_writes_performed") is not False:
        blockers.append("golden_path_db_write_detected")
    if golden.get("graph_writes_performed") is not False:
        blockers.append("golden_path_graph_write_detected")
    if golden.get("canonical_graph_writes_performed") is not False:
        blockers.append("golden_path_canonical_graph_write_detected")
    if metrics.get("chrono_ranking_violations") not in (0, None):
        blockers.append("chrono_ranking_violations_nonzero")
    if metrics.get("chrono_service_active") is not True:
        blockers.append("chrono_service_not_confirmed_active")
    soak = metrics.get("absurd_soak") or {}
    if soak and (int(soak.get("inserted_new") or 0) < 1 or int(soak.get("duplicates_reused") or 0) < 1 or int(soak.get("succeeded") or 0) < 1):
        blockers.append("absurd_soak_metrics_insufficient")
    for step in steps:
        report = step.get("report") if isinstance(step.get("report"), dict) else {}
        if report.get("canonical_graph_writes_performed") is True and step.get("name") not in {"graph_promotion_materialization_allowed"}:
            blockers.append(f"unexpected_canonical_graph_write:{step.get('name')}")
        if report.get("graph_writes_performed") is True and step.get("name") not in {"graph_promotion_materialization_allowed"}:
            blockers.append(f"unexpected_graph_write:{step.get('name')}")
    return blockers


def validate_existing_report(report_path: str) -> int:
    path = ROOT / report_path if not Path(report_path).is_absolute() else Path(report_path)
    blockers: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        data = {}
        blockers.append(f"report_unreadable:{exc}")
    if data.get("schema") != "lucidota.mega_gate.v2":
        blockers.append("schema_not_v2")
    missing_repairs = sorted(set(REPAIRS) - set(data.get("repairs_applied") or []))
    if missing_repairs:
        blockers.append("missing_repairs:" + ",".join(missing_repairs))
    if data.get("status") != "PASS":
        blockers.append("status_not_PASS")
    if data.get("blockers"):
        blockers.append("report_blockers_present")
    metrics = data.get("metrics") or {}
    if metrics.get("steps_total") != metrics.get("steps_passed"):
        blockers.append("steps_not_all_passed")
    golden = metrics.get("golden_path") or {}
    if golden.get("status") not in {"PASS_DRY_RUN", "PASS"}:
        blockers.append("golden_path_not_passed")
    if golden.get("canonical_graph_writes_performed") is not False:
        blockers.append("golden_path_canonical_graph_write_detected")
    if golden.get("graph_writes_performed") is not False:
        blockers.append("golden_path_graph_write_detected")
    if golden.get("db_writes_performed") is not False:
        blockers.append("golden_path_db_write_detected")
    if metrics.get("chrono_ranking_violations") != 0:
        blockers.append("chrono_ranking_violations_nonzero")
    if not metrics.get("chrono_service_active"):
        blockers.append("chrono_service_not_active")
    payload = {"action":"mega_gate_validate_report","validated_report":rel(path),"blockers":blockers,"status":"PASS" if not blockers else "FAIL"}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"lucidota_mega_gate_validate_report_{stamp()}.json"
    payload["generated_at"] = utc(); payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("REPORT_PATH=" + rel(out)); print("MEGA_GATE_REPORT_VALID=" + payload["status"])
    return 0 if not blockers else 4

def main() -> int:
    parser = argparse.ArgumentParser(description="Full LUCIDOTA mega-gate")
    parser.add_argument("--absurd-soak-jobs", type=int, default=12)
    parser.add_argument("--validate-report")
    parser.add_argument("--include-target-selection", action="store_true")
    args = parser.parse_args()
    if args.validate_report:
        return validate_existing_report(args.validate_report)
    steps = [compile_step()]
    if args.include_target_selection:
        target_selector = ROOT / "scripts/post_gate_target_selector.py"
        if target_selector.exists():
            steps.append(run([sys.executable, "scripts/post_gate_target_selector.py", "list"], name="target_selection", require_report=True))
        else:
            steps.append({
                "name": "target_selection",
                "command": "python3 scripts/post_gate_target_selector.py list",
                "status": "SKIPPED",
                "ok": True,
                "report_path": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "blockers": [],
            })
    steps.extend([
        run([sys.executable, "scripts/cep_full_e2e.py", "--dry-run", "--operator-instruction", "Mega-gate golden path dry-run guard"], name="golden_path", require_report=True, expected_status=None),
        run([sys.executable, "scripts/spine_queue_soak_test.py", "--execute", "--jobs", str(args.absurd_soak_jobs)], name="absurd_soak"),
        run([sys.executable, "scripts/chrono_full_conservation_gate.py"], name="chrono_conservation"),
        run([sys.executable, "scripts/system_graph_safety_audit.py"], name="graph_safety"),
        run([sys.executable, "-m", "pytest", "tests/test_graph_materialization_command_policy.py", "tests/test_graph_promotion_gate_safety.py::test_boring_beast_direct_graph_materialize_default_refuses", "-q"], name="graph_materialization_policy_tests", require_report=False, expected_status=None),
        run([sys.executable, "scripts/system_archaeology_evidence_audit.py"], name="archaeology_evidence"),
        run([sys.executable, "scripts/surface_lineage_validator.py"], name="surface_lineage"),
        run([sys.executable, "scripts/tickletrunk_scan.py", "--execute"], name="tickletrunk_execute"),
        run([sys.executable, "scripts/tickletrunk_scan.py", "--check"], name="tickletrunk_check"),
        run([sys.executable, "scripts/run_dev_order_methodology_checks.py"], name="dev_order_methodology", require_report=True, expected_status=None),
        run([sys.executable, "scripts/lucidota_status_ledger.py", "--check"], name="status_ledger_check", require_report=False, expected_status=None),
        run(["scripts/check_chrono_ledger_service.sh"], name="chrono_service", require_report=False, expected_status=None),
    ])
    metrics = extract_metrics(steps)
    blockers = [f"{s['name']}:{b}" for s in steps for b in s.get("blockers", [])]
    blockers.extend(invariant_blockers(steps, metrics))
    payload = {
        "action": "lucidota_mega_gate",
        "schema": "lucidota.mega_gate.v2",
        "repairs_applied": REPAIRS,
        "components": ["ABSURD", "Chrono", "Graph", "Archaeology", "Surfaces", "TICKLETRUNK", "Status Ledger", "Dev Order Methodology", "Target Selection" if args.include_target_selection else "Target Selection Optional"],
        "metrics": metrics,
        "steps": steps,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lucidota_mega_gate_{stamp()}.json"
    payload["generated_at"] = utc()
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    print("LUCIDOTA_MEGA_GATE=" + payload["status"])
    print("MEGA_GATE_REPAIRS=" + ",".join(REPAIRS))
    return 0 if not blockers else 4


if __name__ == "__main__":
    raise SystemExit(main())
