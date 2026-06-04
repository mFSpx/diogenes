from __future__ import annotations

import json
from pathlib import Path


def test_queue_dispatch_skips_existing_receipts_and_executes_only_missing_jobs(tmp_path, monkeypatch):
    import scripts.runpod_queue_dispatch as mod

    monkeypatch.chdir(tmp_path)
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    queue_path = receipts_dir / "TALKIE_TRAINING_QUEUE.jsonl"
    skipped_receipt = receipts_dir / "talkie_adapter_train.json"
    pending_receipt = receipts_dir / "talkie_book_reader.json"
    skipped_receipt.write_text(json.dumps({"status": "PASS"}) + "\n", encoding="utf-8")
    queue_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "label": "talkie-adapter",
                        "command": ["python3", "-c", "print('skip')"],
                        "receipt_path": "talkie_adapter_train.json",
                    }
                ),
                json.dumps(
                    {
                        "label": "talkie-book-reader",
                        "command": ["python3", "-c", "print('run')"],
                        "receipt_path": "talkie_book_reader.json",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(cmd, text=None, capture_output=None, check=None, cwd=None):
        calls.append(cmd)
        pending_receipt.write_text(json.dumps({"status": "PASS"}) + "\n", encoding="utf-8")

        class Result:
            returncode = 0
            stdout = "ok\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    receipt_path = tmp_path / "dispatch_receipt.json"

    assert mod.main(["--queue-path", str(queue_path), "--receipt", str(receipt_path), "--execute", "--json"]) == 0

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    jobs = {job["label"]: job for job in payload["jobs"]}

    assert payload["schema"] == "lucidota.runpod.queue_dispatch.v1"
    assert payload["queue_rows"] == 2
    assert payload["skipped_count"] == 1
    assert payload["executed_count"] == 1
    assert payload["blocked_count"] == 0
    assert jobs["talkie-adapter"]["state"] == "skipped_existing_receipt"
    assert jobs["talkie-adapter"]["receipt_exists"] is True
    assert jobs["talkie-book-reader"]["state"] == "executed"
    assert jobs["talkie-book-reader"]["receipt_exists"] is False
    assert len(calls) == 1


def test_queue_dispatch_resolves_receipts_relative_to_queue_parent(tmp_path, monkeypatch):
    import scripts.runpod_queue_dispatch as mod

    monkeypatch.chdir(tmp_path)
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    queue_path = receipts_dir / "TALKIE_TRAINING_QUEUE.jsonl"
    existing = receipts_dir / "bare-name-receipt.json"
    existing.write_text(json.dumps({"status": "PASS"}) + "\n", encoding="utf-8")
    queue_path.write_text(
        json.dumps({"label": "bare-name", "receipt_path": "bare-name-receipt.json"}) + "\n",
        encoding="utf-8",
    )

    report = mod.build_report(queue_path, execute=False)
    job = report["jobs"][0]

    assert job["receipt_path"].endswith("receipts/bare-name-receipt.json")
    assert job["state"] == "skipped_existing_receipt"


def test_queue_dispatch_defaults_to_runpod_accel_queue_location(tmp_path, monkeypatch):
    import scripts.runpod_queue_dispatch as mod

    monkeypatch.chdir(tmp_path)
    queue_dir = tmp_path / "04_RUNTIME" / "RUNPOD_ACCEL"
    queue_dir.mkdir(parents=True)
    queue_path = queue_dir / "TALKIE_TRAINING_QUEUE.jsonl"
    receipt = queue_dir / "talkie_moe_router_build.json"
    receipt.write_text(json.dumps({"status": "PASS"}) + "\n", encoding="utf-8")
    queue_path.write_text(
        json.dumps(
            {
                "label": "talkie-moe-router-build",
                "command": ["python3", "-c", "print('unused')"],
                "receipt_path": "talkie_moe_router_build.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt_path = tmp_path / "dispatch_receipt.json"

    assert mod.main(["--receipt", str(receipt_path), "--json"]) == 0

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    job = payload["jobs"][0]

    assert payload["queue_path"].endswith("04_RUNTIME/RUNPOD_ACCEL/TALKIE_TRAINING_QUEUE.jsonl")
    assert payload["queue_exists"] is True
    assert job["state"] == "skipped_existing_receipt"
    assert job["receipt_path"].endswith("04_RUNTIME/RUNPOD_ACCEL/talkie_moe_router_build.json")
