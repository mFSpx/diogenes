#!/usr/bin/env python3
"""Active LUCIDOTA/Claw local graph runtime shim.

Purpose: keep Claw wired to the vetted legacy GO/ABSURD runtime without copying its
417-line body back into active script space. The shim is the production entrypoint;
the source implementation remains preserved under scripts/legacy/ until collapsed
into the Rust/core runtime.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "scripts" / "legacy" / "lucidota_clawd_runtime.py"


def _load_impl():
    spec = importlib.util.spec_from_file_location("_lucidota_legacy_clawd_runtime", IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load clawd runtime implementation: {IMPL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
_impl.ROOT = ROOT
_impl.GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
_impl.CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
_impl.WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
_impl.TERMS_PATH = ROOT / "BOOKS" / "GO_ACTIVE_TERMS.json"


def _deterministic_model_lanes_7b(routes, graph_item_uuid):
    terms = [str(r["term"]) for r in routes]
    lanes = [
        {
            "lane": "mamba_watch",
            "model_role": "listener_manager",
            "target_model_id": "mamba-7b-ternary-ram-watch",
            "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:7]},
        }
    ]
    if any(t in terms for t in ("ACTION", "ALGORITHM", "HYPOTHESIS", "CLAIM", "PATTERN")):
        lanes.append(
            {
                "lane": "deepseek_think",
                "model_role": "deep_reasoner",
                "target_model_id": "deepseek-r1-qwen-distill-1.5b-gpu",
                "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:7], "instruction": "reason_over_go_payload_only"},
            }
        )
    lanes.append(
        {
            "lane": "needle_report",
            "model_role": "fast_token_reporter",
            "target_model_id": "google-distill-gemini-needle-swarm-6x",
            "payload": {"graph_item_uuid": graph_item_uuid, "go_terms": terms[:5], "instruction": "stream_status_tokens_only"},
        }
    )
    return lanes


_impl.deterministic_model_lanes = _deterministic_model_lanes_7b
for _name, _value in vars(_impl).items():
    if not (_name.startswith("__") and _name != "__doc__"):
        globals()[_name] = _value
globals()["deterministic_model_lanes"] = _deterministic_model_lanes_7b


if __name__ == "__main__":
    raise SystemExit(_impl.main())
