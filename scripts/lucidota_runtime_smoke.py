#!/usr/bin/env python3
"""LUCIDOTA runtime smoke.

This intentionally checks plumbing, not performance. Benchmarks come after the
stack imports cleanly and the control-plane schema exists.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys


IMPORTS = [
    "dbos",
    "absurd_queue_spine",
    "river",
    "bytewax",
    "treelite",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "sentencepiece",
    "datasets",
    "bitsandbytes",
    "needle",
]


def module_status(name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(name)
        return {
            "status": "ok",
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except Exception as exc:  # noqa: BLE001 - smoke report should capture exact failure class
        return {
            "status": "missing",
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def command_json(command: list[str]) -> dict[str, object]:
    if shutil.which(command[0]) is None:
        return {"status": "missing", "command": command[0]}
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def verify_control_schema() -> dict[str, object]:
    sql = """
    SELECT table_schema || '.' || table_name
    FROM information_schema.tables
    WHERE table_schema IN ('lucidota_control', 'lucidota_runtime', 'lucidota_learning', 'lucidota_vault', 'lucidota_pivot', 'lucidota_bus')
    ORDER BY table_schema, table_name;
    """
    result = subprocess.run(
        ["psql", "postgresql://mfspx@/lucidota_state", "-At", "-c", sql],
        check=False,
        text=True,
        capture_output=True,
    )
    tables = [line for line in result.stdout.splitlines() if line]
    expected = {
        "lucidota_control.governance_gate",
        "lucidota_control.model_runtime_inventory",
        "lucidota_control.source_policy",
        "lucidota_control.workflow_event",
        "lucidota_control.event_outbox",
        "lucidota_runtime.adapter_cartridge",
        "lucidota_runtime.model_candidate",
        "lucidota_runtime.resident_loadout",
        "lucidota_runtime.resident_loadout_slot",
        "lucidota_learning.river_event_cursor",
        "lucidota_learning.river_run",
        "lucidota_learning.river_score",
        "lucidota_learning.treelite_router_run",
        "lucidota_bus.wake_run",
        "lucidota_pivot.promotion",
        "lucidota_pivot.hop_node",
        "lucidota_pivot.hop_job",
        "lucidota_vault.cas_integrity_check",
        "lucidota_vault.cas_gc_run",
        "lucidota_vault.cas_gc_candidate",
        "lucidota_vault.cas_manifest",
    }
    return {
        "status": "ok" if expected.issubset(set(tables)) else "failed",
        "tables": tables,
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    report = {
        "imports": {name: module_status(name) for name in IMPORTS},
        "gpu": command_json(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cuda": {
            "nvcc": command_json(["nvcc", "--version"]),
            "mps_control": command_json(["nvidia-cuda-mps-control", "-h"]),
        },
        "control_schema": verify_control_schema(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    import_failures = [
        name
        for name, status in report["imports"].items()
        if status.get("status") != "ok"
    ]
    if import_failures:
        print(f"import failures: {', '.join(import_failures)}", file=sys.stderr)
        return 1
    if report["control_schema"]["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
