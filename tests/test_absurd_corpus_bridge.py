#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kernel_control_packet import absurd_enqueue_packet


def run(cmd, *, expect: int = 0):
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == expect, (cmd, proc.returncode, proc.stdout, proc.stderr)
    return proc


def receipt_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def write_packet(packet: dict, name: str) -> Path:
    out = ROOT / "05_OUTPUTS" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_execute_blocks_without_explicit_control_packet_before_job_creation():
    idem = f"pytest-missing-packet-{uuid.uuid4()}"
    proc = run([
        sys.executable,
        "KRAMPUSCHEWING/System_Archive_Docs/legacy_scripts/absurd_corpus_bridge.py",
        "--source",
        "README.md",
        "--lane",
        "manifest_inventory",
        "--idempotency-key",
        idem,
        "--max-files",
        "1",
        "--no-dry-run",
    ], expect=2)
    receipt = receipt_from(proc.stdout)
    assert receipt["status"] == "BLOCKED"
    assert receipt["error"] == "MISSING_CONTROL_PACKET"
    assert receipt["job_uuid"] is None
    assert receipt["inserted_new"] is False


def test_execute_blocks_tampered_control_packet_before_job_creation():
    idem = f"pytest-tampered-packet-{uuid.uuid4()}"
    packet = absurd_enqueue_packet(
        queue_name="korpus",
        lane="manifest_inventory",
        source_path="README.md",
        idempotency_key=idem,
        authorized_by="operator_cli",
    )
    packet["payload"]["lane"] = "other_lane"
    packet_path = write_packet(packet, f"tampered_{idem}.json")
    proc = run([
        sys.executable,
        "KRAMPUSCHEWING/System_Archive_Docs/legacy_scripts/absurd_corpus_bridge.py",
        "--source",
        "README.md",
        "--lane",
        "manifest_inventory",
        "--idempotency-key",
        idem,
        "--control-packet",
        str(packet_path),
        "--max-files",
        "1",
        "--no-dry-run",
    ], expect=2)
    receipt = receipt_from(proc.stdout)
    assert receipt["status"] == "BLOCKED"
    assert receipt["error"] == "invalid_kernel_authorization"
    assert receipt["job_uuid"] is None
    assert receipt["inserted_new"] is False


def test_execute_accepts_valid_control_packet_and_creates_bounded_job():
    idem = f"pytest-valid-packet-{uuid.uuid4()}"
    packet = absurd_enqueue_packet(
        queue_name="korpus",
        lane="manifest_inventory",
        source_path="README.md",
        idempotency_key=idem,
        authorized_by="operator_cli",
    )
    packet_path = write_packet(packet, f"valid_{idem}.json")
    proc = run([
        sys.executable,
        "KRAMPUSCHEWING/System_Archive_Docs/legacy_scripts/absurd_corpus_bridge.py",
        "--source",
        "README.md",
        "--lane",
        "manifest_inventory",
        "--idempotency-key",
        idem,
        "--control-packet",
        str(packet_path),
        "--max-files",
        "1",
        "--no-dry-run",
    ])
    receipt = receipt_from(proc.stdout)
    assert receipt["status"] == "PASSED"
    assert receipt["job_uuid"]
    assert receipt["kernel_authorization"]["valid"] is True
    assert receipt["max_files"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
