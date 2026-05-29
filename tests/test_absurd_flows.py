from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import absurd_flows as af


def test_file_kind_and_batch_helpers() -> None:
    assert af.file_kind(Path("x.pdf"), "application/pdf") == "document"
    assert af.file_kind(Path("x.png"), "") == "image"
    assert af.batch([Path("a"), Path("b"), Path("c")], 2) == [[Path("a"), Path("b")], [Path("c")]]


def test_batch_by_bytes_aggregates_until_payload_cap() -> None:
    items = [
        Path("a.txt"),
        Path("b.txt"),
        Path("c.txt"),
    ]
    sizes = {
        "a.txt": 10,
        "b.txt": 20,
        "c.txt": 30,
    }

    groups = af.batch_by_bytes(items, max_bytes=30, size_lookup=lambda p: sizes[p.name])

    assert groups == [[Path("a.txt"), Path("b.txt")], [Path("c.txt")]]


def test_process_files_can_cap_payload_bytes(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_bytes(b"aaaa")
    (root / "two.txt").write_bytes(b"bbbb")
    (root / "three.txt").write_bytes(b"cc")
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")

    payload = af.process_files(root, max_files=None, max_bytes=8, chunk_size=64, execute=False)

    assert payload["ok"] is True
    assert payload["file_count"] == 2
    assert payload["processed_count"] == 0
    assert payload["batch_size_final"] == 64
    assert len(payload["batch_history"]) == 1
    assert payload["batch_history"][0]["batch_size"] == 2
    assert payload["batch_history"][0]["batch_bytes"] == 6


def test_process_files_can_use_inventory_jsonl_without_rescanning(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "keep.txt").write_bytes(b"keep")
    (root / "skip.bin").write_bytes(b"x" * 11)
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "\n".join(
            [
                json.dumps({"path": "keep.txt", "size_bytes": 4, "status": "OK"}),
                json.dumps({"path": "skip.bin", "size_bytes": 11, "status": "OK"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")

    payload = af.process_files(root, input_jsonl=inventory, max_bytes=10, chunk_size=64, execute=False)

    assert payload["file_count"] == 1
    assert payload["records"][0]["source_path"].endswith("keep.txt")
    assert payload["records"][0]["db_action"] == "dry-run"


def test_process_files_inventory_jsonl_respects_cursor(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for name in ["a.txt", "b.txt", "c.txt"]:
        (root / name).write_bytes(name.encode("utf-8"))
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "\n".join(
            [
                json.dumps({"path": "a.txt", "size_bytes": 5, "status": "OK"}),
                json.dumps({"path": "b.txt", "size_bytes": 5, "status": "OK"}),
                json.dumps({"path": "c.txt", "size_bytes": 5, "status": "OK"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")

    payload = af.process_files(root, input_jsonl=inventory, start_after="b.txt", max_bytes=10, chunk_size=64, execute=False)

    assert payload["file_count"] == 1
    assert payload["records"][0]["source_path"].endswith("c.txt")


def test_process_files_dry_run_uses_receipts_without_db(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("alpha", encoding="utf-8")
    (root / "two.txt").write_text("beta", encoding="utf-8")
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")
    payload = af.process_files(root, max_files=None, chunk_size=2, execute=False)
    assert payload["ok"] is True
    assert payload["file_count"] == 2
    assert payload["processed_count"] == 0
    assert len(payload["records"]) == 2


def test_discover_files_can_resume_after_cursor(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "c.txt").write_text("c", encoding="utf-8")
    monkeypatch.setattr(af, "ROOT", root)
    files = af.discover_files(root, start_after="b.txt")
    assert [p.name for p in files] == ["c.txt"]


def test_record_learning_run_emits_river_run_sql() -> None:
    captured: dict[str, object] = {}

    class FakeConn:
        def execute(self, sql, params):
            captured.setdefault("calls", []).append((sql, params))

    af.record_learning_run(
        FakeConn(),
        {
            "schema": "lucidota.absurd_flows.v1",
            "root": "09_STORAGE/krampuschewing_unpacked",
            "file_count": 3,
            "processed_count": 2,
            "deduped_count": 1,
            "db_skipped_count": 0,
            "batch_size_final": 4,
            "batch_history": [{"batch_size": 4, "seconds": 0.5}],
            "records": [{"source_path": "a", "db_action": "inserted", "file_kind": "text"}],
        },
    )
    assert "INSERT INTO lucidota_learning.river_run" in captured["calls"][0][0]
    assert captured["calls"][0][1][0] == "succeeded"
    assert captured["calls"][0][1][1] == 3
    assert captured["calls"][0][1][2] == 2
    assert "INSERT INTO lucidota_learning.river_score" in captured["calls"][1][0]
    assert captured["calls"][1][1][0] == "absurd_flows"
    assert captured["calls"][1][1][1] == "krampuschew"
    assert captured["calls"][1][1][2] == "batch_size_final"


def test_process_files_stop_file_triggers_graceful_stop(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("alpha", encoding="utf-8")
    (root / "two.txt").write_text("beta", encoding="utf-8")
    stop_file = tmp_path / "pause.stop"
    stop_file.write_text("stop", encoding="utf-8")
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")
    payload = af.process_files(root, max_files=None, chunk_size=2, execute=False, stop_file=stop_file)
    assert payload["stopped"] is True
    assert payload["stop_reason"] == f"stop_file_present:{stop_file}"
    assert payload["processed_count"] == 0
    assert len(payload["records"]) == 0


def test_phase1_edge_dedupe_script_dedupes_hashes(tmp_path) -> None:
    target = tmp_path / "krampuschewing_unpacked"
    target.mkdir()
    (target / "a.txt").write_text("same", encoding="utf-8")
    (target / "b.txt").write_text("same", encoding="utf-8")
    (target / "c.txt").write_text("different", encoding="utf-8")
    script = Path("scripts/phase1_edge_dedupe.sh").resolve()
    proc = subprocess.run([str(script), str(target)], cwd=Path.cwd(), text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    receipt_path = Path(proc.stdout.strip())
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    mapping = Path(receipt["mapping_path"])
    assert mapping.exists()
    lines = [line for line in mapping.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
