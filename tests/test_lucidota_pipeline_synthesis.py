#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lucidota_pipeline_synthesis import build_synthesis, render_markdown


def test_pipeline_synthesis_maps_existing_contracts_and_jobs(tmp_path):
    manifest = {
        "generated_at": "2026-05-21T00:00:00Z",
        "total_tools": 4,
        "toolboxes": {
            "SCRIPTS": [
                {
                    "relative_path": "scripts/absurd_queue_spine.py",
                    "name": "absurd_queue_spine.py",
                    "category": "SCRIPTS",
                    "kind": "script",
                    "one_line_description": "ABSURD durable queue spine",
                    "known_use_count": 3,
                    "tags": ["workflow_hint"],
                },
                {
                    "relative_path": "scripts/lucidota_pipeline.py",
                    "name": "lucidota_pipeline.py",
                    "category": "SCRIPTS",
                    "kind": "script",
                    "one_line_description": "case pipeline",
                    "known_use_count": 2,
                    "tags": [],
                },
            ],
            "SCHEMAS": [
                {
                    "relative_path": "06_SCHEMA/035_absurd_queue_spine.sql",
                    "name": "035_absurd_queue_spine.sql",
                    "category": "SCHEMAS",
                    "kind": "schema",
                    "one_line_description": "ABSURD queue schema",
                    "known_use_count": 1,
                    "tags": [],
                }
            ],
            "OTHER": [
                {
                    "relative_path": "tests/test_pipeline_worker_execution.py",
                    "name": "test_pipeline_worker_execution.py",
                    "category": "OTHER",
                    "kind": "other",
                    "one_line_description": "pipeline worker test",
                    "known_use_count": 0,
                    "tags": [],
                }
            ],
        },
    }
    manifest_path = tmp_path / "TICKLETRUNK.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    payload = build_synthesis(
        manifest_path=manifest_path,
        case_id="case-x",
        source_folder=str(tmp_path / "drop"),
        base_dir=str(tmp_path / "cases"),
    )

    assert payload["canonical_graph_writes_performed"] is False
    assert [row["stage"] for row in payload["stage_map"]] == [
        "intake",
        "parse",
        "timeline",
        "staging",
        "graph_candidate",
        "case_packet",
    ]
    assert payload["planned_pipeline_job_count"] == 6
    assert payload["families"]["absurd_queue"]["schemas_count"] == 1
    assert payload["families"]["case_pipeline"]["scripts_count"] >= 1
    assert "run_case_pipeline" in [step["name"] for step in payload["command_plan"]]


def test_pipeline_synthesis_markdown_contains_command_plan(tmp_path):
    manifest_path = tmp_path / "TICKLETRUNK.json"
    manifest_path.write_text(json.dumps({"toolboxes": {}, "total_tools": 0}), encoding="utf-8")
    payload = build_synthesis(manifest_path=manifest_path)
    md = render_markdown(payload)
    assert "# LUCIDOTA Pipeline Synthesis Map" in md
    assert "`pipeline.intake`" in md
    assert "Command plan" in md
