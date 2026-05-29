#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kernel_control_packet import absurd_enqueue_packet, verify_control_packet


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, (cmd, p.returncode, p.stdout, p.stderr)
    return p.stdout


def receipt_path(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            return ROOT / line.split("=", 1)[1]
    raise AssertionError(stdout)


def test_control_packet_rejects_tamper():
    packet = absurd_enqueue_packet(
        queue_name="korpus",
        lane="manifest_inventory",
        source_path="README.md",
        idempotency_key="unit-test-key",
        authorized_by="operator_cli",
    )
    ok, error = verify_control_packet(packet)
    assert ok, error
    packet["payload"]["queue_name"] = "graph_promotion"
    ok, error = verify_control_packet(packet)
    assert not ok
    assert "hash mismatch" in error


def test_absurd_corpus_bridge_embeds_kernel_authorization_in_dry_run():
    out = run([
        sys.executable,
        "scripts/absurd_corpus_job_bridge.py",
        "--source-path",
        "README.md",
        "--lane",
        "manifest_inventory",
        "--dry-run",
        "--authorized-by",
        "operator_cli",
    ])
    receipt = json.load(open(receipt_path(out)))
    assert receipt["schema"] == "diogenes.absurd_corpus_job_bridge.v2"
    assert receipt["kernel_authorization"]["authorized_by"] == "operator_cli"
    assert receipt["kernel_authorization"]["packet_hash"]
    assert receipt["kernel_authorization"]["lane"] == "absurd:korpus:manifest_inventory"


def test_absurd_corpus_bridge_rejects_unbounded_max_files():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/absurd_corpus_job_bridge.py",
            "--source-path",
            "README.md",
            "--lane",
            "manifest_inventory",
            "--dry-run",
            "--max-files",
            "100001",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    receipt = json.load(open(receipt_path(proc.stdout)))
    assert receipt["status"] == "BLOCKED"
    assert receipt["error"] == "MAX_FILES_OUT_OF_RANGE:1..100000"


from spine_kernel_authorization import validate_job_kernel_authorization


def test_absurd_worker_rejects_required_job_without_kernel_authorization():
    verdict = validate_job_kernel_authorization(
        queue_name="korpus",
        job_kind="korpus_lane_job",
        payload={"lane": "manifest_inventory", "bridge_version": "v2"},
    )
    assert verdict.required is True
    assert verdict.ok is False
    assert verdict.error_kind == "missing_kernel_authorization"


def test_absurd_worker_rejects_tampered_kernel_authorization():
    packet = absurd_enqueue_packet(
        queue_name="korpus",
        lane="manifest_inventory",
        source_path="README.md",
        idempotency_key="unit-test-key",
        authorized_by="operator_cli",
    )
    packet["payload"]["lane"] = "other_lane"
    verdict = validate_job_kernel_authorization(
        queue_name="korpus",
        job_kind="korpus_lane_job",
        payload={"lane": "manifest_inventory", "bridge_version": "v2", "kernel_authorization": packet},
    )
    assert verdict.required is True
    assert verdict.ok is False
    assert verdict.error_kind == "invalid_kernel_authorization"


def test_absurd_worker_accepts_valid_kernel_authorization():
    packet = absurd_enqueue_packet(
        queue_name="korpus",
        lane="manifest_inventory",
        source_path="README.md",
        idempotency_key="unit-test-key",
        authorized_by="operator_cli",
    )
    verdict = validate_job_kernel_authorization(
        queue_name="korpus",
        job_kind="korpus_lane_job",
        payload={"lane": "manifest_inventory", "bridge_version": "v2", "kernel_authorization": packet},
    )
    assert verdict.required is True
    assert verdict.ok is True
    assert verdict.packet_hash == packet["packet_hash"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
