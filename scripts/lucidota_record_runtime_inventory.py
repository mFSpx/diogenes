#!/usr/bin/env python3
"""Capture current runtime inventory into lucidota_control.model_runtime_inventory."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess

import psycopg


MODULES = [
    "dbos",
    "river",
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


def run(command: list[str]) -> dict[str, object]:
    if shutil.which(command[0]) is None:
        return {"status": "missing", "command": command[0]}
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    return {
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def python_imports() -> dict[str, object]:
    status = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            status[name] = {
                "status": "ok",
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as exc:  # noqa: BLE001 - inventory wants exact failure
            status[name] = {
                "status": "missing",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
    return status


def main() -> int:
    db_url = os.environ.get(
        "LUCIDOTA_CONTROL_DATABASE_URL",
        "postgresql://mfspx@/lucidota_state",
    )
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap,pstate,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    cuda_tools = {
        "nvcc": run(["nvcc", "--version"]),
        "mps_control": run(["nvidia-cuda-mps-control", "-h"]),
        "mps_server": run(["nvidia-cuda-mps-server", "-h"]),
    }
    imports = python_imports()

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.model_runtime_inventory
                    (gpu, cuda_tools, python_imports, notes)
                VALUES (%s, %s, %s, %s)
                RETURNING inventory_id, captured_at
                """,
                (
                    json.dumps(gpu),
                    json.dumps(cuda_tools),
                    json.dumps(imports),
                    "captured by scripts/lucidota_record_runtime_inventory.py",
                ),
            )
            inventory_id, captured_at = cur.fetchone()
        conn.commit()

    print(f"recorded runtime inventory {inventory_id} at {captured_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
