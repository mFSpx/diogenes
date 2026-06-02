from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def test_root_rotor_queue_batch_skips_existing_outputs_and_limits(tmp_path: Path) -> None:
    import scripts.root_rotor_queue_batch as batch

    node_dir = tmp_path / "nodes"
    node_dir.mkdir()
    existing = node_dir / "done.json"
    existing.write_text('{"schema":"lucidota.root_rotor.bible_node_payload.v1","what_it_is_and_does":"done"}', encoding="utf-8")
    queue = tmp_path / "queue.jsonl"
    rows = [
        {"label": "done", "target_file": str(existing), "model": "codestral"},
        {"label": "next1", "target_file": str(node_dir / "next1.json"), "model": "codestral"},
        {"label": "next2", "target_file": str(node_dir / "next2.json"), "model": "groq"},
    ]
    write_jsonl(queue, rows)
    out = tmp_path / "batch.jsonl"

    result = batch.write_next_batch(queue, out, limit=1)
    selected = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert result["jobs_selected"] == 1
    assert result["jobs_skipped_existing_output"] == 1
    assert selected[0]["label"] == "next1"


def test_root_rotor_queue_batch_can_select_specific_model(tmp_path: Path) -> None:
    import scripts.root_rotor_queue_batch as batch

    queue = tmp_path / "queue.jsonl"
    rows = [
        {"label": "py", "target_file": str(tmp_path / "py.json"), "model": "codestral"},
        {"label": "sql", "target_file": str(tmp_path / "sql.json"), "model": "groq"},
    ]
    write_jsonl(queue, rows)
    out = tmp_path / "batch.jsonl"

    result = batch.write_next_batch(queue, out, limit=10, model="groq")
    selected = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert result["jobs_selected"] == 1
    assert selected[0]["label"] == "sql"
