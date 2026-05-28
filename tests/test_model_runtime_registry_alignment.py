#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from scripts.lucidota_strict_model_stack_admission import build_strict_stack_plan


def test_runtime_registry_sql_uses_current_strict_mamba_artifact() -> None:
    sql = Path("scripts/06_SCHEMA/002_model_runtime.sql").read_text(encoding="utf-8")
    strict = build_strict_stack_plan(env={})
    mamba = next(service for service in strict["services"] if service["name"] == "mamba7b_ram")

    assert "falcon3-mamba-7b-listener" in sql
    assert mamba["model_path"] in sql
    assert "'listener',\n        'falcon3-mamba-7b-listener'" in sql
    assert "'listener',\n        'mamba-1.4b-listener'" not in sql


def test_spark_alias_routes_to_always_on_mamba_cpu_lane() -> None:
    from scripts.local_model_chat_cli import LLAMA_LANES, NEEDLE_LANES

    assert "spark" not in LLAMA_LANES
    assert NEEDLE_LANES["spark"] == NEEDLE_LANES["needle_0"]


def test_gpu_runtime_registry_records_current_strict_stack_models() -> None:
    registry = __import__("json").loads(Path("00_PROJECT_BRAIN/gpu_model_runtime_registry.json").read_text(encoding="utf-8"))
    inventory_paths = {item["path"] for item in registry["gguf_inventory"]}
    strict_paths = {
        service["model_path"]
        for service in build_strict_stack_plan(env={})["services"]
        if service.get("model_path", "").startswith("03_VAULT/models/")
    }

    assert {path for path in strict_paths if path.endswith(".gguf")} <= inventory_paths
    assert strict_paths <= set(registry["strict_runtime_alignment"]["active_model_paths"])
    assert registry["strict_runtime_alignment"]["source"] == "scripts/lucidota_strict_model_stack_admission.py"
    assert registry["strict_runtime_alignment"]["spark_alias"] == "needle_0"
