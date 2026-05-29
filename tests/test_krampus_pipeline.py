from __future__ import annotations

import subprocess
import json
from pathlib import Path

import scripts.krampus_pipeline as kp


def test_build_lane_plan_splits_inventory_by_lane_and_skips_large_archives(tmp_path) -> None:
    corpus_map = {
        "recommendations": {"huge_file_threshold_bytes": 104857600},
        "categories": {
            "easy_text": {"count": 2, "total_bytes": 3000},
            "heavy_pdf": {"count": 1, "total_bytes": 2000},
            "heavy_image": {"count": 1, "total_bytes": 1000},
            "heavy_archive": {"count": 1, "total_bytes": 500000000},
        },
    }
    inventory = tmp_path / "inventory.jsonl"
    rows = [
        {"path": "a.md", "suffix": ".md", "size_bytes": 1000, "hash_status": "HASHED"},
        {"path": "b.json", "suffix": ".json", "size_bytes": 2000, "hash_status": "HASHED"},
        {"path": "c.pdf", "suffix": ".pdf", "size_bytes": 3000, "hash_status": "HASHED"},
        {"path": "d.png", "suffix": ".png", "size_bytes": 4000, "hash_status": "HASHED"},
        {"path": "e.zip", "suffix": ".zip", "size_bytes": 150_000_000, "hash_status": "SKIPPED"},
    ]
    inventory.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    plan = kp.build_pipeline_plan(inventory, corpus_map=corpus_map)

    assert plan["lane_counts"] == {"lane1_text": 2, "lane2_groq": 2, "lane3_registration": 4, "lane4_skips": 1}
    assert plan["skip_manifest_rows"][0]["reason"] == "heavy_archive_or_over_100mb"
    assert plan["skip_manifest_rows"][0]["path"] == "e.zip"


def test_build_lane_commands_wrap_all_lanes_in_resource_governor_spawn(tmp_path) -> None:
    plan = {
        "lane_counts": {"lane1_text": 1, "lane2_groq": 1, "lane3_registration": 2, "lane4_skips": 1},
        "inventory_path": "05_OUTPUTS/krampus_inventory/krampus_queue_eligible.jsonl",
        "corpus_map_path": "05_OUTPUTS/goals/corpus_map_20260528T091724121300Z.json",
        "skip_manifest_path": "05_OUTPUTS/runtime/krampus_skip_manifest.jsonl",
    }
    commands = kp.build_lane_commands(plan, root=tmp_path)

    assert len(commands) == 4
    joined = [" ".join(cmd) for cmd in commands]
    assert all("scripts/resource_governor.py" in cmd for cmd in joined)
    assert any("--lane lane1" in cmd for cmd in joined)
    assert any("--lane lane2" in cmd for cmd in joined)
    assert any("--lane lane3" in cmd for cmd in joined)
    assert any("--lane lane4" in cmd for cmd in joined)


def test_default_script_execution_refuses_unauthorized_pipeline(tmp_path) -> None:
    proc = subprocess.run(
        [str(Path(".venv/bin/python")), "scripts/krampus_pipeline.py", "--json"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "KRAMPUS_PIPELINE=BLOCKED_UNAUTHORIZED_ARCHITECTURE" in proc.stdout
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("RECEIPT_PATH="))
    receipt_path = Path(receipt_line.split("=", 1)[1])
    payload = json.loads(receipt_path.read_text())
    assert payload["schema"] == "lucidota.krampus_pipeline.blocked_unauthorized.v1"
    assert payload["reason"] == "operator_did_not_authorize_four_lane_plan"
