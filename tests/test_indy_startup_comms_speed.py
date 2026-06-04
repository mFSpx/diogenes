import json
import subprocess
import sys
from pathlib import Path

MANIFEST = Path("04_RUNTIME/indy_reads_startup_comms_manifest.json")
SERVICE = Path("services/ironclaw-indy-reads.service")
START_SCRIPT = Path("scripts/lucidota_start_indy_reads_watcher.sh")
COMMS = Path("scripts/indy_reads_comms.py")
SPEED = Path("scripts/luci_speed_probe.py")


def test_indy_manifest_boot_email_signal_direct_chat_policy():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.indy_reads.startup_comms.v1"
    assert data["startup"]["boot_target"] == "laptop_user_login"
    assert data["startup"]["service_file"] == str(SERVICE)
    assert data["startup"]["existing_log_file"] == "journalctl --user -u ironclaw-indy-reads.service"
    assert data["startup"]["background_fallback_log_file"] == "04_RUNTIME/indy_daemon.log"
    assert "services/ironclaw-indy-reads.service" in data["startup"]["install_hint"]
    assert "lucidota-indy-reads-watcher.service" not in data["startup"]["install_hint"]
    assert data["email"]["mode"] == "queue_only_until_operator_send_approval"
    assert set(data["email"]["recipients"]) == {"MaroonedPilot@gmail.com", "mfspx@proton.me"}
    assert data["signal"]["mode"] == "optional_connector_requires_signal_cli_and_operator_approval"
    assert data["direct_chat"]["preferred"] == "local_unix_socket_or_loopback_sse"
    assert data["luci_app"]["indy_response_helper"] is True
    assert data["luci_app"]["speed_slo_ms"]["status_p95"] <= 1000


def test_indy_systemd_service_starts_db_daemon_front_door():
    text = SERVICE.read_text(encoding="utf-8")
    assert "Description=IronClaw Indy_READs 24/7 DB daemon" in text
    assert "ExecStart=/home/mfspx/LUCIDOTA/scripts/lucidota_start_indy_reads_watcher.sh --foreground" in text
    assert "lucidota_indy_reads_watcher.py --interval" not in text
    assert "WantedBy=default.target" in text
    assert "Restart=on-failure" in text
    assert "MemoryMax=768M" in text
    assert "TasksMax=64" in text


def test_indy_start_script_has_pidfile_independent_single_instance_guard():
    text = START_SCRIPT.read_text(encoding="utf-8")
    assert "is_daemon_pid()" in text
    assert 'ps -p "$pid" -o args=' in text
    assert "find_existing_daemon_pids()" in text
    assert "pgrep -f" in text
    assert "indy_daemon.py" in text
    assert 'is_running_pid "$pid" && is_daemon_pid "$pid"' in text
    assert "existing_pid=\"$(find_existing_daemon_pids" in text
    assert "echo \"$existing_pid\" > \"$PID_FILE\"" in text
    assert 'LUCIDOTA_INDY_USE_ULIMIT_V:-0' in text
    assert "--foreground" in text


def test_ironclaw_host_gate_checks_canonical_indy_watcher_service():
    text = Path("scripts/ironclaw_host_gate.sh").read_text(encoding="utf-8")
    assert 'INDY_DAEMON_SERVICE="${INDY_DAEMON_SERVICE:-ironclaw-indy-reads.service}"' in text
    assert 'INDY_DAEMON_UNIT_TEXT="$(systemctl --user cat "$INDY_DAEMON_SERVICE"' in text
    assert "canonical Indy daemon service not installed/readable" in text
    assert "legacy Indy watcher service is active" in text
    assert "lucidota-indy-reads-watcher.service" in text


def test_legacy_indy_watcher_service_is_retired_safely():
    legacy = Path("services/lucidota-indy-reads-watcher.service").read_text(encoding="utf-8")
    assert "LEGACY RETIRED UNIT" in legacy
    assert "ConditionPathExists=/run/lucidota-enable-legacy-indy-reads-watcher" in legacy
    assert "ExecStart=/bin/false" in legacy
    assert "kill $(cat" not in legacy


def test_comms_cli_queues_email_without_sending_and_reports_signal_missing_policy(tmp_path):
    outbox = tmp_path / "outbox.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(COMMS),
            "queue-email",
            "--manifest",
            str(MANIFEST),
            "--outbox",
            str(outbox),
            "--subject",
            "test note",
            "--body",
            "hello from Indy",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "QUEUED_NOT_SENT"
    assert result["recipients"] == ["MaroonedPilot@gmail.com", "mfspx@proton.me"]
    queued = json.loads(outbox.read_text(encoding="utf-8").strip())
    assert queued["send_requires_operator_approval"] is True

    status = subprocess.run(
        [sys.executable, str(COMMS), "status", "--manifest", str(MANIFEST)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert status.returncode == 0
    data = json.loads(status.stdout)
    assert data["email_mode"] == "queue_only_until_operator_send_approval"
    assert data["direct_chat_preferred"] == "local_unix_socket_or_loopback_sse"


def test_speed_probe_runs_luci_status_with_budgeted_receipt():
    proc = subprocess.run(
        [
            sys.executable,
            str(SPEED),
            "--command",
            "./luci sheet list --json",
            "--runs",
            "2",
            "--p95-budget-ms",
            "2500",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["schema"] == "lucidota.luci_speed_probe.v1"
    assert data["runs"] == 2
    assert data["p95_ms"] <= 2500
    assert data["status"] == "PASS"
