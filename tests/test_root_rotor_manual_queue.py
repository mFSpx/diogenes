from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_root_rotor_manual_queue_builds_one_job_per_manifest_file_and_routes_sql_to_groq(tmp_path: Path) -> None:
    import scripts.root_rotor_manual_queue as queue

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "06_SCHEMA/test.sql", "sha256": "a" * 64, "size_bytes": 64, "bytes_read": 32},
                    {"path": "scripts/sample.py", "sha256": "b" * 64, "size_bytes": 128, "bytes_read": 128},
                    {"path": "scripts/sample.rs", "sha256": "c" * 64, "size_bytes": 256, "bytes_read": 200},
                    {"path": "docs/readme.md", "sha256": "d" * 64, "size_bytes": 12, "bytes_read": 12},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out = tmp_path / "manual_queue.jsonl"
    node_dir = tmp_path / "05_OUTPUTS" / "root_rotor_nodes"
    result = queue.build_and_write(manifest_path, output_jsonl=out, node_dir=node_dir)
    rows = _read_jsonl(out)

    assert result["jobs_planned"] == 4
    assert result["jobs_written"] == 4
    assert len(rows) == 4  # no batching; exactly one per manifest entry

    by_path = {row["path"]: row for row in rows}
    assert by_path["06_SCHEMA/test.sql"]["model"] == "groq"
    assert by_path["06_SCHEMA/test.sql"]["model_preference"][0] == "groq"

    py = by_path["scripts/sample.py"]
    rs = by_path["scripts/sample.rs"]
    for row in (py, rs):
        assert row["model"] in {"codestral", "ministral", "groq", "auto"}
        assert row["model"] == "codestral"
        assert row["model_preference"][0] == "vibes:codestral"
        assert row["model_preference"][1] == "groq"

    for row in rows:
        assert "source_path" not in row
        assert row["target_output_contract"] == queue.TARGET_OUTPUT_CONTRACT
        assert row["target_manual_volume_guess"]["volume_id"] >= 1
        assert row["schema"] == queue.SCHEMA


def test_root_rotor_manual_queue_prompt_contains_required_contract_terms_and_node_path(tmp_path: Path) -> None:
    import scripts.root_rotor_manual_queue as queue

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "scripts/analysis.py",
                        "sha256": "e" * 64,
                        "size_bytes": 10,
                        "bytes_read": 10,
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "manual_queue.jsonl"
    node_dir = tmp_path / "05_OUTPUTS" / "root_rotor_nodes"
    queue.build_and_write(manifest_path, output_jsonl=out, node_dir=node_dir)
    row = _read_jsonl(out)[0]

    prompt = row["prompt"]
    assert "ASD-STE100" in prompt
    assert "Law-of-Root" in prompt
    assert "source_path: scripts/analysis.py" in prompt
    assert f"source_sha256: {'e' * 64}" in prompt
    assert queue.NODE_PAYLOAD_SCHEMA in prompt
    assert row["target_output_contract"] in prompt

    expected_dir = (tmp_path / "05_OUTPUTS" / "root_rotor_nodes").resolve()
    target = (tmp_path / row["target_file"]).resolve()
    assert target.parent == expected_dir
    assert target.suffix == ".json"
    assert target.name.endswith(".json")


def test_root_rotor_manual_queue_prompt_includes_capped_source_text(tmp_path: Path) -> None:
    import scripts.root_rotor_manual_queue as queue

    source = tmp_path / "scripts" / "analysis.py"
    source.parent.mkdir()
    source.write_text("print('real source')\n" + "x" * 200, encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"files": [{"path": "scripts/analysis.py", "sha256": "f" * 64, "size_bytes": 221, "bytes_read": 40, "truncated": True}]}),
        encoding="utf-8",
    )
    out = tmp_path / "queue.jsonl"
    queue.build_and_write(manifest_path, output_jsonl=out, node_dir=tmp_path / "nodes", root=tmp_path, max_source_bytes=40)
    row = _read_jsonl(out)[0]

    assert "SOURCE_TEXT_BEGIN" in row["prompt"]
    assert "print('real source')" in row["prompt"]
    assert "SOURCE_TEXT_END" in row["prompt"]
    assert "source_text_truncated: true" in row["prompt"]


def test_root_rotor_manual_queue_prompt_demands_deep_artifact_fields(tmp_path: Path) -> None:
    import scripts.root_rotor_manual_queue as queue

    source = tmp_path / "scripts" / "tool.py"
    source.parent.mkdir()
    source.write_text("import json\nprint(json.dumps({}))", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"files": [{"path": "scripts/tool.py", "sha256": "a" * 64, "size_bytes": 30, "bytes_read": 30, "truncated": False}]}),
        encoding="utf-8",
    )
    out = tmp_path / "queue.jsonl"
    queue.build_and_write(manifest_path, output_jsonl=out, node_dir=tmp_path / "nodes", root=tmp_path)
    prompt = _read_jsonl(out)[0]["prompt"]

    assert "what_it_is_and_does" in prompt
    assert "exact_interactions" in prompt
    assert "operating_limits_failure_modes" in prompt
    assert "integration_points" in prompt
    assert "manual_id" in prompt
    assert "node_title" in prompt
    assert "payload_asd_ste100" in prompt
    assert "node_kind" in prompt
    assert "ontology_tags" in prompt
    assert "WORKFLOW" in prompt
    assert "OBJECT" in prompt
    assert "RECEIPT" in prompt
    assert "Do not return pending_dedicated_model_analysis" in prompt
