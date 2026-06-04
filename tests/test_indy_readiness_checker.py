from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/indy_readiness_checker.py"


def _run_checker(*, service_manifest: Path, startup_manifest: Path, receipt: Path) -> tuple[int, Path]:
    proc = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--service-manifest",
            str(service_manifest),
            "--startup-manifest",
            str(startup_manifest),
            "--receipt",
            str(receipt),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode in (0, 4), proc.stderr
    return proc.returncode, receipt


def test_readiness_checker_reports_pass_on_configured_indy_setup(tmp_path: Path) -> None:
    service_file = tmp_path / "services" / "indy_reads_watcher.service"
    service_file.parent.mkdir(parents=True, exist_ok=True)
    service_file.write_text(
        """[Unit]
Description=Test INDY_READs watcher

[Service]
Type=simple
ExecStart=/home/mfspx/LUCIDOTA/scripts/lucidota_start_indy_reads_watcher.sh
""",
        encoding="utf-8",
    )

    startup = tmp_path / "indy_reads_startup_comms_manifest.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "lucidota.indy_reads.startup_comms.v1",
                "startup": {"service_file": str(service_file)},
                "email": {
                    "mode": "queue_only_until_operator_send_approval",
                    "recipients": ["ops@example.com"],
                },
                "signal": {"mode": "optional_connector_requires_signal_cli_and_operator_approval"},
                "direct_chat": {"preferred": "local_unix_socket_or_loopback_sse"},
                "luci_app": {
                    "indy_response_helper": True,
                    "response_policy": "test",
                    "speed_slo_ms": {"status_p95": 10},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    service = tmp_path / "indy_service_manifest.json"
    service.write_text(
        json.dumps(
            {
                "boot": {
                    "startup_manifest": str(startup),
                },
                "inventory": {
                    "actual_book_file_count": 6,
                    "staged_book_count": 7,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    receipt = tmp_path / "indy_readiness_status.json"
    rc, receipt_path = _run_checker(service_manifest=service, startup_manifest=startup, receipt=receipt)
    assert rc == 0

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "lucidota.indy_reads.readiness_status.v1"
    assert payload["status"] == "PASS"
    assert payload["checks"]["auto_start"]["status"] == "PASS"
    assert payload["checks"]["book_count"]["status"] == "PASS"
    assert payload["checks"]["comms"]["status"] == "PASS"
    assert payload["checks"]["response_helper"]["indy_response_helper"] is True
    assert payload["model_calls_performed"] is False
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["db_writes_performed"] is False


def test_readiness_checker_blocks_when_auto_start_and_comms_are_unconfigured(tmp_path: Path) -> None:
    startup = tmp_path / "indy_reads_startup_comms_manifest.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "lucidota.indy_reads.startup_comms.v1",
                "email": {
                    "mode": "",
                    "recipients": [],
                },
                "signal": {},
                "direct_chat": {},
                "luci_app": {"indy_response_helper": False},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    service = tmp_path / "indy_service_manifest.json"
    service.write_text(
        json.dumps(
            {
                "boot": {
                    "startup_manifest": str(startup),
                },
                "inventory": {},
                "book_training_plans": {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    receipt = tmp_path / "indy_readiness_status_blocked.json"
    rc, receipt_path = _run_checker(service_manifest=service, startup_manifest=startup, receipt=receipt)
    assert rc == 4

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    blockers = set(payload["blockers"])
    assert "auto-start service_file is missing" in blockers
    assert "email config missing recipients or mode" in blockers
    assert "signal config missing mode" in blockers
    assert "direct chat config missing preferred transport" in blockers
    assert "luci_app.indy_response_helper is false or missing" in blockers
    assert "boot.startup_manifest is missing" not in blockers
    assert payload["checks"]["auto_start"]["status"] == "FAIL"
    assert payload["checks"]["book_count"]["status"] == "FAIL"
