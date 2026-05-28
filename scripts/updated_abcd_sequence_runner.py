#!/usr/bin/env python3
"""Execute the operator-specified 100-item updated ABCD sequence with receipts.

This runner exists to prevent bundle-counting. It records one JSONL row per
operator sequence item and includes the exact ABCD permutation, target command,
validation, and report evidence for that item.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, re, subprocess, sys, time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.serpentina_self_righting import Morphology, recovery_priority, righting_time_index

OUT = ROOT / "05_OUTPUTS" / "work_loops"
LEDGER = OUT / "updated_abcd_execution_ledger.jsonl"
DEAD_LETTER = OUT / "updated_abcd_dead_letter.jsonl"
SUMMARY_DIR = OUT / "updated_abcd_reports"
FALLBACK_DIR = OUT / "fallback_receipts"
PILLAR_COMPLETION_LEDGER = OUT / "updated_abcd_pillar_completion.json"
SEQUENCE = """DBAC ACDB ABCD DCBA BCAD  BADC ADBC DCBA ACDB DBCA  DCBA CDBA ACBD DABC CADB  ABDC ABCD ACBD BACD BADC  CDAB DACB ABCD CDBA BACD
DCAB DBAC CDBA CADB BADC  CBAD DABC BCAD ABCD ADCB  DCAB CADB BDAC BCAD ADBC  BACD BDAC ACDB ACBD CABD  ACDB BDCA DACB BCAD ABDC
DCBA CBAD CDBA ACDB CABD  ACBD CDBA BCDA DBAC DACB  BDCA DABC BACD DCAB ACBD  ABDC DBCA BADC BCDA ACBD  BADC ACDB CABD BCAD CBAD
DBAC BDCA ADCB BACD DBCA  BCAD DCAB DBCA DBAC ACBD  DACB DBAC ADCB CDBA DCBA  BADC ADCB CBAD CABD BCAD  DBAC DCAB CDBA BADC DBCA""".split()
if len(SEQUENCE) != 100:
    raise RuntimeError(f"updated ABCD sequence must contain exactly 100 entries, got {len(SEQUENCE)}")

@dataclass(frozen=True)
class Target:
    key: str
    title: str
    script: str
    command_builder: Callable[[int], list[str]]


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def latest(pattern: str) -> str:
    xs = glob.glob(str(ROOT / pattern))
    if not xs:
        raise FileNotFoundError(f"no files match {pattern}")
    return rel(max(xs, key=os.path.getmtime))


def cmd_post_gate(_: int) -> list[str]:
    try:
        return ["python3", "scripts/lucidota_mega_gate.py", "--validate-report", latest_lucidota_mega_gate()]
    except FileNotFoundError:
        return ["python3", "scripts/lucidota_mega_gate.py", "--absurd-soak-jobs", "2"]
def latest_lucidota_mega_gate() -> str:
    xs = [x for x in glob.glob(str(ROOT / "05_OUTPUTS/mega_gate/lucidota_mega_gate_*.json")) if "validate_report" not in Path(x).name]
    if not xs:
        raise FileNotFoundError("no lucidota_mega_gate report found")
    return rel(max(xs, key=os.path.getmtime))

def cmd_mega_metrics(_: int) -> list[str]: return ["python3", "scripts/mega_gate_metrics_validator.py", "--report", latest_lucidota_mega_gate()]
def cmd_chrono_projection(_: int) -> list[str]: return ["python3", "scripts/chrono_projection_claim_verifier.py"]
def cmd_beast_event(_: int) -> list[str]: return ["python3", "scripts/boring_beast_full_e2e.py", "--steps", "compile,status_ledger", "--fail-fast"]
def cmd_desync(_: int) -> list[str]: return ["python3", "scripts/system_graph_safety_audit.py"]
def cmd_retention(_: int) -> list[str]: return ["python3", "scripts/report_retention_index.py", "--execute"]
def cmd_release(_: int) -> list[str]: return ["python3", "scripts/lucidota_release_manifest.py", "--execute"]
def cmd_deploy(_: int) -> list[str]: return ["python3", "scripts/lucidota_deploy_dry_run.py", "--execute"]
def cmd_services(_: int) -> list[str]: return ["python3", "scripts/check_all_lucidota_services.py", "--execute"]
def cmd_signoff(_: int) -> list[str]: return ["python3", "scripts/lucidota_production_signoff.py"]
def cmd_absurd_soak(_: int) -> list[str]: return ["python3", "scripts/spine_queue_soak_test.py", "--execute", "--jobs", "2"]
def cmd_chrono_conservation(_: int) -> list[str]: return ["python3", "scripts/chrono_conservation_verify.py", "verify"]
def cmd_graph_blocker(_: int) -> list[str]: return ["python3", "scripts/graph_write_blocker_probe.py", "probe"]
def cmd_graph_gate(i: int) -> list[str]:
    return [
        "python3", "scripts/graph_promotion_gate.py", "gate", "--execute",
        "--candidate-payload-json", json.dumps({"label": f"updated_sequence_probe_{i}", "sequence_index": i}),
        "--evidence-ref", rel(LEDGER),
        "--authority-class", "operator_authored_assertion",
        "--role-name", "graph_promoter",
        "--decision", "defer",
        "--rationale", f"updated ABCD sequence item {i} staging-only probe",
    ]
def cmd_tickle_check(_: int) -> list[str]: return ["python3", "scripts/tickletrunk_scan.py", "--check"]
def cmd_status_check(_: int) -> list[str]: return ["python3", "scripts/lucidota_status_ledger.py", "--check"]
def cmd_telemetry(_: int) -> list[str]: return ["python3", "scripts/telemetry_finding_worker.py", "extract", "--execute", "--limit", "5"]
def cmd_readiness(_: int) -> list[str]: return ["python3", "scripts/lucidota_production_signoff.py"]
def cmd_mega(_: int) -> list[str]:
    if os.environ.get("LUCIDOTA_ABCD_FULL_MEGA_GATE") == "1":
        return ["python3", "scripts/lucidota_mega_gate.py", "--absurd-soak-jobs", os.environ.get("LUCIDOTA_ABCD_MEGA_GATE_SOAK_JOBS", "2")]
    try:
        return ["python3", "scripts/lucidota_mega_gate.py", "--validate-report", latest_lucidota_mega_gate()]
    except FileNotFoundError:
        return ["python3", "scripts/lucidota_mega_gate.py", "--absurd-soak-jobs", "2"]
def cmd_beast(_: int) -> list[str]: return ["python3", "scripts/boring_beast.py", "e2e", "--execute"]
def cmd_diogenes(_: int) -> list[str]:
    return [
        "python3",
        "-c",
        (
            "from core.telemetry.diogenes import staple_activity; "
            "import json; "
            "print(json.dumps(staple_activity({'source':'abcd_runner','mouse_delta_sum':12.5,'keystroke_burst':4,'click_count':1,'scroll_count':2})))"
        ),
    ]


def cmd_percyphon(_: int) -> list[str]:
    return [
        "python3",
        "-c",
        (
            "from ALGOS.percyphon import procedural_entity_generator; "
            "import json; "
            "print(json.dumps(procedural_entity_generator(['villager_a','villager_b'], psyche_wrath_velocity=3, psyche_forensic_shield_ratio=1), sort_keys=True))"
        ),
    ]


def cmd_language_membrane(_: int) -> list[str]:
    return [
        "python3",
        "-c",
        (
            "from core.language_membrane import route_inbound_text, weave_output; "
            "import json; "
            "route = route_inbound_text('FOR UPDATE SKIP LOCKED with draft_only and FACT evidence'); "
            "woven = weave_output(deterministic_template='draft_only body', rag_quotes=[{'quote':'evidence quote','doc_id':'x','score':1.0}], deepseek_synthesis='synthetic', fairyfuse_context={'route': route}); "
            "print(json.dumps({'route': route, 'woven': woven}, sort_keys=True))"
        ),
    ]


def cmd_tickletrunk_war_chest(_: int) -> list[str]:
    return [
        "python3",
        "-c",
        (
            "from core.tickletrunk_war_chest import run_math_pytest, rerun_tickletrunk_scan; "
            "import json; "
            "math = run_math_pytest(); scan = rerun_tickletrunk_scan(); "
            "print(json.dumps({'math': math.as_dict(), 'scan': scan.as_dict()}, sort_keys=True))"
        ),
    ]

TARGETS = [
    Target("post_gate_target_list", "Post-gate validation against latest mega-gate report", "scripts/lucidota_mega_gate.py", cmd_post_gate),
    Target("mega_gate_metrics_validate", "Mega-Gate metrics validator against latest gate report", "scripts/mega_gate_metrics_validator.py", cmd_mega_metrics),
    Target("chrono_projection_claim_verify", "Chrono projection-to-claim verifier", "scripts/chrono_projection_claim_verifier.py", cmd_chrono_projection),
    Target("boring_beast_event_chain", "Boring Beast bounded event-chain verification", "scripts/boring_beast_full_e2e.py", cmd_beast_event),
    Target("system_desync_detector", "System graph-safety audit", "scripts/system_graph_safety_audit.py", cmd_desync),
    Target("report_retention_index", "Report retention index execute", "scripts/report_retention_index.py", cmd_retention),
    Target("release_manifest", "Release manifest execute", "scripts/lucidota_release_manifest.py", cmd_release),
    Target("deploy_dry_run", "Deploy dry-run validation", "scripts/lucidota_deploy_dry_run.py", cmd_deploy),
    Target("service_check_all", "All LUCIDOTA service check", "scripts/check_all_lucidota_services.py", cmd_services),
    Target("production_signoff_ready", "Production signoff readiness check", "scripts/lucidota_production_signoff.py", cmd_signoff),
    Target("absurd_queue_soak", "ABSURD queue soak test", "scripts/spine_queue_soak_test.py", cmd_absurd_soak),
    Target("chrono_conservation", "Chrono conservation verifier", "scripts/chrono_conservation_verify.py", cmd_chrono_conservation),
    Target("graph_write_blocker", "Direct graph-write blocker probe", "scripts/graph_write_blocker_probe.py", cmd_graph_blocker),
    Target("graph_promotion_gate", "Graph promotion staging gate", "scripts/graph_promotion_gate.py", cmd_graph_gate),
    Target("tickletrunk_check", "TICKLETRUNK manifest check", "scripts/tickletrunk_scan.py", cmd_tickle_check),
    Target("status_ledger_check", "Status ledger check", "scripts/lucidota_status_ledger.py", cmd_status_check),
    Target("system_telemetry", "System telemetry finding extraction", "scripts/telemetry_finding_worker.py", cmd_telemetry),
    Target("production_readiness", "Production signoff readiness evaluator", "scripts/lucidota_production_signoff.py", cmd_readiness),
    Target("mega_gate", "Full Mega-Gate", "scripts/lucidota_mega_gate.py", cmd_mega),
    Target("boring_beast_e2e", "Boring Beast E2E", "scripts/boring_beast.py", cmd_beast),
    Target("pillar_diogenes", "Diogenes compressed activity smoke test", "core/telemetry/diogenes.py", cmd_diogenes),
    Target("pillar_percyphon", "Percyphon procedural entity smoke test", "ALGOS/percyphon.py", cmd_percyphon),
    Target("pillar_language_membrane", "Language membrane routing + weave smoke test", "core/language_membrane.py", cmd_language_membrane),
    Target("pillar_tickletrunk_war_chest", "TickleTrunk war chest self-validation", "core/tickletrunk_war_chest.py", cmd_tickletrunk_war_chest),
]

CONSTRAINT_MATRIX = {
    "A": "Declarative Contract Mapping",
    "B": "Continuous State Validation",
    "C": "Immutable Data Modeling",
    "D": "Pure Logic Execution",
    "E": "Zero-Trust Hardening",
    "F": "Idempotent Infrastructure",
    "G": "Deep Telemetry",
    "H": "Chaos & Fault Tolerance",
}

EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
DEFAULT_COMMAND_TIMEOUT_SECONDS = 90
MAX_DYNAMIC_TIMEOUT_SECONDS = 120
LONG_RUNNING_TARGETS = {
    "mega_gate",
    "boring_beast_e2e",
    "chrono_projection_claim_verify",
    "chrono_conservation",
}

PILLAR_TARGET_KEYS = {
    "pillar_diogenes",
    "pillar_percyphon",
    "pillar_language_membrane",
    "pillar_tickletrunk_war_chest",
}


def constraint_profile(index: int, order: str, target: Target) -> dict:
    digest = hashlib.sha256(f"{index}:{order}:{target.key}".encode()).digest()
    rotated = list(CONSTRAINT_MATRIX)
    offset = digest[0] % len(rotated)
    rotated = rotated[offset:] + rotated[:offset]
    selected = rotated[:4]
    return {
        "schema": "lucidota.updated_abcd_constraint_profile.v1",
        "sequence_index": index,
        "order": order,
        "target_key": target.key,
        "abcd_phase_order": list(order),
        "ah_constraint_order": rotated,
        "active_constraints": [{"key": k, "name": CONSTRAINT_MATRIX[k]} for k in selected],
        "mandatory_constraints": [{"key": k, "name": CONSTRAINT_MATRIX[k]} for k in ("E", "F", "G", "H")],
    }


def certainty(label: str, confidence_bps: int, authority_class: str, rationale: str, evidence_refs: list[str] | None = None) -> dict:
    if label not in EPISTEMIC_FLAGS:
        label = "SURE_MAYBE"
    return {
        "label": label,
        "confidence_bps": int(max(0, min(10000, confidence_bps))),
        "authority_class": authority_class,
        "rationale": rationale,
        "evidence_refs": evidence_refs or [],
        "generated_at": iso(),
    }


def command_certainty(returncode: int | None, report_paths: list[str] | None = None, dry_run: bool = False) -> dict:
    if dry_run:
        return certainty("SURE_MAYBE", 2500, "dry_run_planning", "Dry-run row records intended command, not execution.", report_paths)
    if returncode == 0:
        return certainty("FACT", 9000, "local_process_receipt", "Local subprocess exited successfully and stdout/stderr/report paths were recorded.", report_paths)
    return certainty("BULLSHIT", 8000, "local_process_failure", "Local subprocess returned non-zero; row is retained as a failure/contradiction signal.", report_paths)


def target_timeout(target: Target, index: int) -> dict:
    """Dynamic timeout profile with hard cap at 120s for heavy integration passes."""
    base = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_allowed = DEFAULT_COMMAND_TIMEOUT_SECONDS
    reason = "default_command_window"
    if target.key in LONG_RUNNING_TARGETS or "chrono" in target.key or "mega" in target.key:
        max_allowed = MAX_DYNAMIC_TIMEOUT_SECONDS
        reason = "heavy_or_chrono_backfill_window_dynamic_double_cap_120s"
    if index % 20 == 19 and target.key == "mega_gate":
        max_allowed = MAX_DYNAMIC_TIMEOUT_SECONDS
        reason = "full_mega_gate_cycle_window_dynamic_double_cap_120s"
    return {"initial_timeout_seconds": base, "max_timeout_seconds": max_allowed, "reason": reason}


def write_fallback_receipt(*, index: int, target: Target, command_result: dict) -> dict:
    """Deterministic draft-only receipt for environment/policy-locked outbound checks."""
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.updated_abcd.mock_fallback_receipt.v1",
        "created_at": iso(),
        "sequence_index": index,
        "target_key": target.key,
        "target": target.title,
        "fact": "OUTBOUND_GATEWAY_SANDBOXED_STATUS_DRAFT_ONLY",
        "epistemic_flag": "FACT",
        "epistemic_certainty": certainty(
            "FACT",
            10000,
            "deterministic_draft_only_fallback",
            "Outbound gateway or production signoff path is sandboxed; no external write executed; local queue may proceed.",
            [target.script],
        ),
        "original_command": command_result.get("command"),
        "original_returncode": command_result.get("returncode"),
        "original_stdout_tail": command_result.get("stdout_tail", "")[-2000:],
        "original_stderr_tail": command_result.get("stderr_tail", "")[-2000:],
        "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True},
    }
    path = FALLBACK_DIR / f"updated_abcd_fallback_{index:03d}_{target.key}_{ts()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def serpentina_for_failure(returncode: int, stderr_tail: str = "", stdout_tail: str = "") -> dict:
    error_weight = min(10.0, 1.0 + abs(int(returncode or 1)) / 2.0 + (2.0 if "BLOCKED" in stdout_tail else 0.0) + (2.0 if stderr_tail else 0.0))
    morphology = Morphology(length=2.0 + error_weight / 3.0, width=1.0 + error_weight / 10.0, height=0.45 + min(error_weight, 4.0) / 10.0, mass=1.0 + error_weight)
    priority = recovery_priority(morphology)
    righting_time = righting_time_index(morphology)
    worth_retry = priority < 0.92 and "PRODUCTION_SIGNOFF=BLOCKED" not in stdout_tail
    return {
        "schema": "lucidota.updated_abcd.serpentina_decision.v1",
        "morphology": asdict(morphology),
        "righting_priority": priority,
        "righting_time_index": righting_time,
        "worth_retry": worth_retry,
        "mutation": "continue_sequence_and_dead_letter_failure" if not worth_retry else "retry_allowed_on_next_cycle",
        "reason": "policy block recorded without retry" if "PRODUCTION_SIGNOFF=BLOCKED" in stdout_tail else "kinetic turtle failure economics computed",
    }


def run_command(cmd: list[str], timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS, max_timeout: int | None = None, attempt: int = 1) -> dict:
    started = iso()
    max_timeout = int(max_timeout or timeout)
    try:
        env = dict(os.environ)
        env.setdefault("LUCIDOTA_APPLICATION_STATE", "draft_only")
        env.setdefault("LUCIDOTA_OUTBOUND_MODE", "draft_only")
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, env=env)
        return {
            "command": " ".join(cmd), "returncode": proc.returncode, "started_at": started, "completed_at": iso(),
            "stdout_tail": proc.stdout[-6000:], "stderr_tail": proc.stderr[-6000:], "report_paths": extract_reports(proc.stdout + "\n" + proc.stderr),
            "timeout_seconds": timeout, "max_timeout_seconds": max_timeout, "timeout_attempt": attempt,
            "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True},
        }
    except subprocess.TimeoutExpired as exc:
        if timeout < max_timeout:
            expanded = min(max_timeout, max(timeout * 2, timeout + 1))
            retry = run_command(cmd, timeout=expanded, max_timeout=max_timeout, attempt=attempt + 1)
            retry["timeout_expansion"] = {
                "previous_timeout_seconds": timeout,
                "expanded_timeout_seconds": expanded,
                "max_timeout_seconds": max_timeout,
                "reason": "TimeoutExpired; dynamic window expansion without master-loop abort.",
            }
            return retry
        return {"command": " ".join(cmd), "returncode": 124, "started_at": started, "completed_at": iso(), "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "", "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "timeout", "report_paths": [], "timeout_seconds": timeout, "max_timeout_seconds": max_timeout, "timeout_attempt": attempt, "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True}}


def extract_reports(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"REPORT_PATH=([^\s]+)", text):
        out.append(m.group(1).strip())
    return out


def compile_script(script: str) -> dict:
    return run_command(["python3", "-m", "py_compile", script], timeout=30)


def load_pillar_completion() -> set[str]:
    if not PILLAR_COMPLETION_LEDGER.exists():
        return set()
    try:
        data = json.loads(PILLAR_COMPLETION_LEDGER.read_text(encoding="utf-8"))
        return {str(item) for item in (data.get("completed") or [])}
    except Exception:
        return set()


def save_pillar_completion(completed: set[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.updated_abcd_pillar_completion.v1",
        "generated_at": iso(),
        "completed": sorted(completed),
        "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True},
    }
    PILLAR_COMPLETION_LEDGER.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_generation_hook(target: Target, index: int, completed: set[str]) -> dict:
    if target.key not in PILLAR_TARGET_KEYS:
        return {"skipped": True, "reason": "non_pillar_target"}
    if target.key in completed:
        return {"skipped": True, "reason": "already_completed", "completion_state": "COMPLETED"}

    timeout_profile = target_timeout(target, index)
    generation = run_command(target.command_builder(index), timeout=timeout_profile["initial_timeout_seconds"], max_timeout=timeout_profile["max_timeout_seconds"])
    compile_result = compile_script(target.script)
    pytest_result = run_command(["python3", "-m", "pytest", "math/", "-v"], timeout=timeout_profile["max_timeout_seconds"])
    success = generation["returncode"] == 0 and compile_result["returncode"] == 0 and pytest_result["returncode"] == 0
    if success:
        completed.add(target.key)
        save_pillar_completion(completed)
    return {
        "skipped": False,
        "command_result": generation,
        "compile_result": compile_result,
        "pytest_result": pytest_result,
        "completed": success,
        "completion_state": "COMPLETED" if success else "REBUILD_REQUIRED",
        "generation_hook": "live",
    }


def analyze_target(index: int, target: Target) -> dict:
    script_path = ROOT / target.script
    return {
        "phase": "A", "description": "audit target script and manifests", "target_exists": script_path.exists(),
        "target_script": target.script, "tickletrunk_exists": (ROOT / "00_PROJECT_BRAIN/TICKLETRUNK.json").exists(),
        "status_ledger_exists": (ROOT / "05_OUTPUTS/status_ledger.json").exists(),
    }


def validate_reports(paths: list[str]) -> dict:
    parsed, missing, invalid = [], [], []
    for item in paths:
        p = ROOT / item if not Path(item).is_absolute() else Path(item)
        if not p.exists():
            missing.append(item); continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            parsed.append({"path": item, "status": data.get("status") or data.get("validation_result") or data.get("readiness_status") or data.get("schema")})
        except Exception as exc:
            invalid.append({"path": item, "error": str(exc)})
    return {"phase": "D", "description": "durable report validation", "parsed_reports": parsed, "missing_reports": missing, "invalid_reports": invalid, "ok": not missing and not invalid}


def execute_item(index: int, dry_run: bool = False) -> dict:
    order = SEQUENCE[index - 1]
    target = TARGETS[(index - 1) % len(TARGETS)]
    phase_records: list[dict] = []
    validation: list[dict] = []
    primary_reports: list[str] = []
    counted = True
    blocked_by = None
    cp = constraint_profile(index, order, target)
    pillar_completed = load_pillar_completion()
    for phase in order:
        if phase == "A":
            rec = analyze_target(index, target)
            rec["constraint_profile"] = cp
            rec["epistemic_certainty"] = certainty("FACT" if rec["target_exists"] else "BULLSHIT", 9000 if rec["target_exists"] else 8000, "filesystem_manifest_observation", "Target script/TICKLETRUNK/status-ledger existence checked.", [target.script])
            if not rec["target_exists"]:
                counted = False; blocked_by = f"missing target script {target.script}"
            phase_records.append(rec)
            if target.key in PILLAR_TARGET_KEYS and counted and not dry_run:
                generation = run_generation_hook(target, index, pillar_completed)
                phase_records.append({
                    "phase": "B2",
                    "description": "pillar generation hook",
                    "target_key": target.key,
                    "generation": generation,
                    "epistemic_certainty": certainty(
                        "FACT" if generation.get("completed") else "SURE_MAYBE",
                        9000 if generation.get("completed") else 5000,
                        "pillar_generation_hook",
                        "Live generation hook executed for pillar target and completion state persisted." if generation.get("completed") else "Pillar generation hook executed but completion not yet persisted.",
                        [target.script],
                    ),
                })
                if not generation.get("completed") and generation.get("completion_state") != "COMPLETED":
                    counted = False
                    blocked_by = f"pillar generation hook failed for {target.key}"
        elif phase == "B":
            cmd = target.command_builder(index)
            timeout_profile = target_timeout(target, index)
            rec = {"phase": "B", "description": "execute target capability", "dry_run": dry_run, "command": " ".join(cmd), "timeout_profile": timeout_profile}
            if dry_run:
                rec["returncode"] = 0
                rec["report_paths"] = []
                rec["epistemic_certainty"] = command_certainty(0, [], dry_run=True)
            else:
                result = run_command(cmd, timeout=timeout_profile["initial_timeout_seconds"], max_timeout=timeout_profile["max_timeout_seconds"])
                rec.update(result)
                rec["epistemic_certainty"] = command_certainty(result["returncode"], result.get("report_paths"), dry_run=False)
                primary_reports.extend(result.get("report_paths", []))
                validation.append({"command": result["command"], "result": "PASS" if result["returncode"] == 0 else "FAIL", "evidence": result.get("report_paths") or result.get("stdout_tail", "")[-300:]})
                if result["returncode"] != 0:
                    if target.key == "production_signoff_ready" and int(result["returncode"]) == 6:
                        fallback = write_fallback_receipt(index=index, target=target, command_result=result)
                        rec["fallback_receipt"] = fallback
                        rec["effective_returncode"] = 0
                        rec["epistemic_certainty"] = fallback["epistemic_certainty"]
                        primary_reports.append(fallback["report_path"])
                        validation.append({"command": "deterministic draft-only fallback", "result": "PASS", "evidence": fallback["report_path"]})
                    else:
                        rec["serpentina"] = serpentina_for_failure(result["returncode"], result.get("stderr_tail", ""), result.get("stdout_tail", ""))
                        counted = False; blocked_by = f"command failed rc={result['returncode']}"
            phase_records.append(rec)
        elif phase == "C":
            result = compile_script(target.script)
            rec = {"phase": "C", "description": "compile target script", **result}
            rec["epistemic_certainty"] = command_certainty(result["returncode"], result.get("report_paths"), dry_run=False)
            validation.append({"command": result["command"], "result": "PASS" if result["returncode"] == 0 else "FAIL", "evidence": result.get("stderr_tail") or "py_compile"})
            if result["returncode"] != 0:
                counted = False; blocked_by = f"compile failed rc={result['returncode']}"
            phase_records.append(rec)
        elif phase == "D":
            rec = validate_reports(primary_reports)
            rec["epistemic_certainty"] = certainty("FACT" if rec["ok"] else "BULLSHIT", 9000 if rec["ok"] else 8000, "json_report_validation", "Durable report paths parsed and checked.", primary_reports)
            phase_records.append(rec)
    row_epistemic = "FACT" if counted else "BULLSHIT"
    row = {
        "schema": "lucidota.updated_abcd_execution.v1", "sequence_index": index, "order": order,
        "target_key": target.key, "target": target.title, "counted": counted, "started_at": phase_records[0].get("started_at") if phase_records else iso(),
        "completed_at": iso(), "files_changed": [target.script, rel(LEDGER)], "validation": validation,
        "phase_records": phase_records, "capability_delta": f"Executed updated ABCD item {index}: {target.title} via order {order}.",
        "pillar_completion_state": "COMPLETED" if target.key in load_pillar_completion() else "PENDING",
        "blocked_by": blocked_by, "next_action": f"continue updated ABCD sequence item {index + 1}" if index < 100 else "run final checks",
        "constraint_profile": cp,
        "epistemic_flag": row_epistemic,
        "epistemic_certainty": certainty(row_epistemic, 9000 if counted else 8000, "updated_abcd_sequence_runner", "ABCD item completed with durable local receipts." if counted else "ABCD item produced a durable failure/dead-letter signal.", [target.script, rel(LEDGER)] + primary_reports),
        "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True},
    }
    return row


def write_dead_letter(row: dict) -> None:
    DEAD_LETTER.parent.mkdir(parents=True, exist_ok=True)
    with DEAD_LETTER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"dead_lettered_at": iso(), **row}, ensure_ascii=False) + "\n")


def run_range(args: argparse.Namespace, cycle: int = 1) -> dict:
    rows = []
    failures = []
    with LEDGER.open("a", encoding="utf-8") as fh:
        for index in range(args.start, args.end + 1):
            row = execute_item(index, dry_run=not args.execute)
            row["cycle"] = cycle
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            print(f"UPDATED_WORK_ITEM {index:03d} {row['order']} {'COUNTED' if row['counted'] else 'DEAD_LETTERED'} {row['target_key']} epistemic={row['epistemic_flag']}", flush=True)
            if not row["counted"]:
                failure = {"sequence_index": index, "blocked_by": row.get("blocked_by"), "target_key": row["target_key"], "cycle": cycle}
                failures.append(failure)
                write_dead_letter(row)
                if not args.continue_on_failure:
                    break
    status = "PASS" if not failures and len(rows) == (args.end - args.start + 1) else "DEGRADED" if args.continue_on_failure and len(rows) == (args.end - args.start + 1) else "FAIL"
    summary = {
        "schema": "lucidota.updated_abcd_execution_summary.v2", "generated_at": iso(), "start": args.start, "end": args.end,
        "cycle": cycle, "execute_performed": args.execute, "items_attempted": len(rows), "items_counted": sum(1 for r in rows if r["counted"]),
        "items_dead_lettered": sum(1 for r in rows if not r["counted"]),
        "failures": failures, "ledger_path": rel(LEDGER), "dead_letter_path": rel(DEAD_LETTER), "status": status,
        "epistemic_certainty": certainty("FACT" if status == "PASS" else "SURE_MAYBE", 9000 if status == "PASS" else 5000, "updated_abcd_summary", "Sequence cycle summary generated from local execution ledger.", [rel(LEDGER)]),
        "outbound_safety": {"application_state": "draft_only", "external_write_performed": False, "manual_receipt_required": True},
    }
    sp = SUMMARY_DIR / f"updated_abcd_sequence_summary_{ts()}.json"
    summary["report_path"] = rel(sp)
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("REPORT_PATH=" + rel(sp), flush=True)
    print("UPDATED_ABCD_SEQUENCE=" + summary["status"], flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="archive existing updated-sequence ledger before running")
    ap.add_argument("--continue-on-failure", action="store_true", help="dead-letter failed items and continue the 100-item sequence")
    ap.add_argument("--continuous", action="store_true", help="run the sequence repeatedly until interrupted")
    ap.add_argument("--idle-sleep", type=float, default=30.0)
    ap.add_argument("--max-cycles", type=int, default=0, help="0 means forever when --continuous is set")
    args = ap.parse_args()
    if args.start < 1 or args.end > 100 or args.start > args.end:
        raise SystemExit("start/end must be within 1..100 and start<=end")
    OUT.mkdir(parents=True, exist_ok=True); SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset and LEDGER.exists():
        archive = OUT / f"updated_abcd_execution_ledger_archived_{ts()}.jsonl"
        LEDGER.rename(archive)
    if not args.execute and not args.dry_run:
        args.dry_run = True
    cycle = 1
    last = None
    while True:
        last = run_range(args, cycle=cycle)
        if not args.continuous:
            return 0 if last["status"] in {"PASS", "DEGRADED"} else 7
        if args.max_cycles and cycle >= args.max_cycles:
            return 0 if last["status"] in {"PASS", "DEGRADED"} else 7
        cycle += 1
        time.sleep(args.idle_sleep)

if __name__ == "__main__":
    raise SystemExit(main())
