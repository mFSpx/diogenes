#!/usr/bin/env python3
"""LUCIDOTA fabric press diagnostic + orchestration sweep.

This is a thin coordinator over existing production scripts. It does not invent
duplicate simulator or runtime infrastructure; it only wires what already exists
and emits a structured receipt for the operator.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.telemetry.diogenes import sample_hardware_telemetry
OUT = ROOT / "05_OUTPUTS" / "fabric_press"
VENV_PY = ROOT / ".venv" / "bin" / "python"
BYTEWAX = ROOT / "scripts" / "bytewax_abductive_blender.py"
AHOY = ROOT / "scripts" / "legacy" / "ahoy" / "ahoy_million_replay.py"
AHOY_WORKFLOW = ROOT / "scripts" / "legacy" / "ahoy" / "ahoy_absurd_workflow.py"
GRAPH_HELPER = ROOT / "scripts" / "graph_materialization_helper.py"
GATE = ROOT / "scripts" / "graph_promotion_gate.py"
SPINE = ROOT / "scripts" / "spine_authority_checker.py"
SCANNER = ROOT / "scripts" / "canonical_graph_write_scanner.py"
DEEPSEEK_START = ROOT / "scripts" / "lucidota_start_deepseek_llama.sh"
MAMBA_START = ROOT / "scripts" / "lucidota_start_mamba_llama.sh"
DEEPSEEK_MODEL = ROOT / "03_VAULT" / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
MAMBA_MODEL = ROOT / "03_VAULT" / "models" / "mamba-1.4b-hf-Q2_K.gguf"


def ahoy_paused() -> bool:
    return os.environ.get("LUCIDOTA_AHOY_PAUSED", "1").strip().lower() not in {"0", "false", "no", "off"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False, default=str), encoding="utf-8")


def runtime_python() -> str:
    return str(VENV_PY if VENV_PY.exists() else Path(sys.executable))


def run_cmd(argv: list[str], *, timeout: float | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    cp = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": argv,
        "returncode": cp.returncode,
        "stdout": cp.stdout[-8000:],
        "stderr": cp.stderr[-8000:],
        "duration_seconds": round(time.time() - started, 3),
    }


def package_probe() -> dict[str, Any]:
    py = runtime_python()
    expr = (
        "import importlib.util, json; "
        "mods=['bytewax','onnxruntime','river','xgboost','duckdb','psycopg']; "
        "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}, sort_keys=True))"
    )
    probe = run_cmd([py, "-c", expr], timeout=30)
    avail: dict[str, bool] = {}
    try:
        avail = json.loads((probe["stdout"] or "{}").splitlines()[-1])
    except Exception:
        avail = {}
    return {
        "python": py,
        "availability": avail,
        "probe": probe,
    }


def runtime_plan() -> dict[str, Any]:
    from pypeline.math.model_vram_scheduler import (
        gpu_memory,
        load_lora_manifests,
        plan_dual_engine_residency,
        plan_lora_preemption,
    )
    from pypeline.math.small_model_zoo import quantize_to_onnx, train_setfit_classifier

    payload = {
        "indys": {
            "Indy_FAST": "DeepSeek-R1-Distill-Qwen 1.5B Q4 resident in VRAM",
            "Indy_DEEP": "long-context DeepSeek warm RAM fallback",
            "Mamba": "dynamic graph cache lane",
            "Needle": "high-throughput verifier lane",
            "ONNX": "tiny triage classifiers on CPU/VRAM boundary",
        }
    }
    state = {"context": "fabric_press"}
    plan = plan_dual_engine_residency(payload, state)
    preemption = plan_lora_preemption(payload, state)
    manifests = load_lora_manifests()
    onnx_probe = None
    try:
        dummy = train_setfit_classifier("triage_probe", [{"text": "probe", "label": "triage"}], domain="runtime")
        onnx_probe = asdict(quantize_to_onnx(dummy))
    except Exception as exc:
        onnx_probe = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "hardware_state": sample_hardware_telemetry(),
        "gpu_memory": gpu_memory(),
        "dual_engine_plan": plan,
        "lora_preemption_plan": preemption,
        "lora_manifest_count": len(manifests),
        "lora_manifests": manifests[:10],
        "onnx_probe": onnx_probe,
    }


def ensure_bytewax_loop(socket_path: str | None = None) -> dict[str, Any]:
    py = runtime_python()
    cmd = [py, str(BYTEWAX), "tick", "--limit", "40", "--prefer-replication", "--no-activitywatch"]
    if socket_path:
        cmd.extend(["--unix-socket", socket_path])
    return run_cmd(cmd, timeout=120)


def latest_graph_receipts(limit: int = 4) -> list[Path]:
    return sorted((ROOT / "05_OUTPUTS" / "graph").glob("graph_materialization_helper_materialize_execute_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def verify_graph_receipts() -> dict[str, Any]:
    py = runtime_python()
    receipts = []
    results = []
    for path in latest_graph_receipts():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts.append(rel(path))
        materialization_uuid = data.get("materialization_uuid")
        command_uuid = data.get("command_envelope_uuid")
        evidence_refs = data.get("evidence_refs") or []
        if not materialization_uuid:
            continue
        cmd = [py, str(GRAPH_HELPER), "verify", "--materialization-uuid", str(materialization_uuid)]
        if command_uuid:
            cmd.extend(["--command-envelope-uuid", str(command_uuid)])
        for ref in evidence_refs:
            cmd.extend(["--evidence-ref", str(ref)])
        results.append(run_cmd(cmd, timeout=120))
    gate_candidate = json.dumps({"check": "smoke", "time": now(), "receipt_refs": receipts[:2]})
    gate = run_cmd(
        [
            py,
            str(GATE),
            "gate",
            "--dry-run",
            "--source-system",
            "fabric_press",
            "--candidate-kind",
            "workflow",
            "--candidate-payload-json",
            gate_candidate,
            "--evidence-ref",
            receipts[0] if receipts else "fabric_press_empty_receipt",
            "--authority-class",
            "operator_confirmed_finding",
            "--role-name",
            "graph_promoter",
            "--decision",
            "defer",
            "--rationale",
            "receipt verification smoke",
        ],
        timeout=120,
    )
    spine = run_cmd([py, str(SPINE)], timeout=120)
    scanner = run_cmd([py, str(SCANNER)], timeout=120)
    return {
        "latest_receipts": receipts,
        "verify_runs": results,
        "gatekeeper": {"graph_promotion_gate": gate, "spine_authority_checker": spine, "canonical_graph_write_scanner": scanner},
    }


def ahoy_factory() -> dict[str, Any]:
    if ahoy_paused():
        return {"returncode": 0, "status": "PAUSED", "reason": "LUCIDOTA_AHOY_PAUSED=1", "cmd": []}
    py = runtime_python()
    cmd = [
        py,
        str(AHOY_WORKFLOW),
        "launch",
        "--start",
        "--iterations",
        "10",
        "--games-per-iteration",
        "100000",
        "--workers",
        str(max(1, os.cpu_count() or 1)),
        "--quick-grid",
    ]
    return run_cmd(cmd, timeout=None)


def ahoy_stop() -> dict[str, Any]:
    if ahoy_paused():
        return {"returncode": 0, "status": "PAUSED", "reason": "LUCIDOTA_AHOY_PAUSED=1", "cmd": []}
    py = runtime_python()
    cmd = [py, str(AHOY_WORKFLOW), "stop", "--reason", "fabric_press stop request"]
    return run_cmd(cmd, timeout=60)


def inventory() -> dict[str, Any]:
    return {
        "bytewax_script_exists": BYTEWAX.exists(),
        "ahoy_runner_exists": AHOY.exists(),
        "ahoy_workflow_controller_exists": AHOY_WORKFLOW.exists(),
        "ahoy_paused": ahoy_paused(),
        "graph_helper_exists": GRAPH_HELPER.exists(),
        "gate_exists": GATE.exists(),
        "spine_exists": SPINE.exists(),
        "scanner_exists": SCANNER.exists(),
        "llama_scripts": {
            "deepseek_start": rel(DEEPSEEK_START),
            "mamba_start": rel(MAMBA_START),
            "deepseek_model": {"path": rel(DEEPSEEK_MODEL), "exists": DEEPSEEK_MODEL.exists()},
            "mamba_model": {"path": rel(MAMBA_MODEL), "exists": MAMBA_MODEL.exists()},
        },
        "bytewax_binary": shutil.which("bytewax"),
    }


def algorithm_audit() -> dict[str, Any]:
    terms = {
        "Tarjan": ["Tarjan", "tarjan", "low-link", "bridge detection"],
        "Mahalanobis": ["Mahalanobis", "mahalanobis"],
        "CUSUM": ["CUSUM", "cusum"],
        "RETE": ["RETE", "rete"],
    }
    roots = [ROOT / "ALGOS", ROOT / "scripts", ROOT / "pypeline", ROOT / "math", ROOT / "02_RECORDS_OFFICE"]
    found: dict[str, list[str]] = {k: [] for k in terms}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json", ".sql", ".txt", ".sh"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for label, needles in terms.items():
                if any(n in text for n in needles):
                    rel_path = rel(path)
                    if rel_path not in found[label]:
                        found[label].append(rel_path)
    return {
        "found": found,
        "status": {k: bool(v) for k, v in found.items()},
        "notes": {
            "Tarjan": "not found in current search scope" if not found["Tarjan"] else "found",
            "CUSUM": "not found in current search scope" if not found["CUSUM"] else "found",
            "Mahalanobis": "present" if found["Mahalanobis"] else "missing",
            "RETE": "present" if found["RETE"] else "missing",
        },
    }


def peripheral_audit() -> dict[str, Any]:
    import importlib.util

    home = Path.home()
    comms = {
        "aerc_conf": str(home / ".config/aerc/aerc.conf"),
        "aerc_conf_exists": (home / ".config/aerc/aerc.conf").exists(),
        "aerc_accounts_exists": (home / ".config/aerc/accounts.conf").exists(),
        "mbsync_rc_exists": (home / ".mbsyncrc").exists(),
        "mbsync_dir_exists": (home / ".config/mbsync").exists(),
        "jmap_dir_exists": (home / ".config/jmap").exists(),
        "burn_registry_exists": any(p.exists() for p in [home / ".config/burn", home / ".config/burn_registry", home / ".local/share/burn"]),
    }
    activitywatch_paths = [
        home / ".local/share/activitywatch/aw-server/peewee-sqlite.v2.db",
        home / ".local/share/activitywatch/aw-server-rust/peewee-sqlite.v2.db",
        ROOT / "04_RUNTIME" / "activitywatch" / "events.sqlite",
    ]
    activitywatch = {
        "db_candidates": [str(p) for p in activitywatch_paths],
        "db_present": [str(p) for p in activitywatch_paths if p.exists()],
        "collector_modules": {
            "activitywatch": bool(importlib.util.find_spec("activitywatch")),
            "aw_client": bool(importlib.util.find_spec("aw_client")),
        },
    }
    crawler = {
        "crawl4ai": bool(importlib.util.find_spec("crawl4ai")),
        "scrapegraphai": bool(importlib.util.find_spec("scrapegraphai")),
        "missing": [name for name, present in {
            "crawl4ai": bool(importlib.util.find_spec("crawl4ai")),
            "scrapegraphai": bool(importlib.util.find_spec("scrapegraphai")),
        }.items() if not present],
    }
    return {"comms_surface": comms, "activitywatch": activitywatch, "crawler_dependencies": crawler}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LUCIDOTA fabric press diagnostic/orchestrator")
    ap.add_argument("cmd", nargs="?", default="diagnose", choices=["diagnose", "wire-bytewax", "run-ahoy", "stop-ahoy", "verify-gates", "inventory", "full"])
    ap.add_argument("--unix-socket", default=os.environ.get("LUCIDOTA_DIIOGENES_SOCKET"))
    args = ap.parse_args(argv)

    report: dict[str, Any] = {
        "schema": "lucidota.fabric_press.report.v1",
        "generated_at": now(),
        "runtime_python": runtime_python(),
        "inventory": inventory(),
        "package_probe": package_probe(),
    }
    if args.cmd in {"diagnose", "full"}:
        report["runtime_plan"] = runtime_plan()
        report["algorithm_audit"] = algorithm_audit()
        report["peripheral_audit"] = peripheral_audit()
    if args.cmd in {"wire-bytewax", "full"}:
        report["bytewax_tick"] = ensure_bytewax_loop(args.unix_socket)
    if args.cmd in {"verify-gates", "full"}:
        report["gatekeeper"] = verify_graph_receipts()
    if args.cmd in {"run-ahoy", "full"}:
        report["ahoy_factory"] = ahoy_factory()
    if args.cmd in {"stop-ahoy", "full"}:
        report["ahoy_stop"] = ahoy_stop()

    out = OUT / f"fabric_press_{args.cmd}_{stamp()}.json"
    report["report_path"] = rel(out)
    dump_json(out, report)
    print(json.dumps({
        "cmd": args.cmd,
        "report_path": rel(out),
        "runtime_python": report["runtime_python"],
        "bytewax": report.get("bytewax_tick", {}).get("returncode"),
        "ahoy": report.get("ahoy_factory", {}).get("returncode"),
        "ahoy_stop": report.get("ahoy_stop", {}).get("returncode"),
        "gatekeeper": report.get("gatekeeper", {}).get("gatekeeper", {}).get("graph_promotion_gate", {}).get("returncode"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
