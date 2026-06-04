import json
import subprocess
import sys
from pathlib import Path
import importlib.util

SCRIPT = Path("scripts/runpod_talkie_control.py")
PACK = Path("05_OUTPUTS/runpod/talkie_book_lora/talkie_book_lora_runpod_pack.tar.gz")


def load_control_module():
    spec = importlib.util.spec_from_file_location("runpod_talkie_control", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runpod_talkie_control_plan_contains_upload_start_poll_commands():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.runpod.talkie_control.plan.v1"
    assert payload["host"] == "213.192.6.98"
    assert payload["port"] == 40100
    assert payload["pack"] == str(PACK)
    assert payload["pack_sha256"]
    assert "talkie-lm/talkie-1930-13b-it" in payload["remote_commands"]["start_download"]
    assert "rl-refined.pt" in payload["remote_commands"]["start_download"]
    assert "scp -P 40100" in payload["local_commands"]["upload_pack"]
    assert "talkie_source_custody.json" in payload["remote_commands"]["poll_receipt"]
    assert payload["dolphin_touched"] is False


def test_runpod_talkie_control_probe_no_execute_writes_receipt_without_scp():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "probe", "--no-network", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.runpod.talkie_control.probe_receipt.v1"
    assert payload["status"] == "NOT_PROBED_NO_NETWORK"
    assert payload["db_writes_performed"] is False
    assert payload["graph_writes_performed"] is False
    assert Path(payload["receipt_path"]).exists()


def test_runpod_control_hard_gate_stops_ssh_when_key_refusal_already_diagnosed(tmp_path):
    receipt = tmp_path / "latest.json"
    receipt.write_text(json.dumps({
        "schema": "lucidota.runpod.talkie_control.probe_receipt.v1",
        "status": "WAITING_FOR_PUBLIC_KEY_AUTH",
        "ssh_stderr_tail": "Offering public key: /home/mfspx/.ssh/id_ed25519 ED25519 SHA256:x explicit\nPermission denied (publickey,password).",
        "public_key": "ssh-ed25519 TEST luci-runpod-20260602"
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--receipt", str(receipt), "probe", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "WAITING_FOR_PUBLIC_KEY_AUTH"
    assert payload["ssh_attempted"] is False
    assert payload["scp_attempted"] is False
    assert payload["reason"] == "remote refused correct offered key; PUBLIC_KEY/authorized_keys must change before retry"
    assert payload["next_required_action"] == "Set PUBLIC_KEY env on runpod/pytorch pod or paste key into /root/.ssh/authorized_keys via web terminal, then restart/redeploy if env changed."


def test_runpod_control_hard_gate_blocks_run_path_too(tmp_path):
    receipt = tmp_path / "latest.json"
    receipt.write_text(json.dumps({
        "schema": "lucidota.runpod.talkie_control.probe_receipt.v1",
        "status": "PUBLIC_KEY_AUTH_REQUIRED",
        "public_key": "ssh-ed25519 TEST luci-runpod-20260602"
    }), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--receipt", str(receipt), "run", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "WAITING_FOR_PUBLIC_KEY_AUTH"
    assert payload["ssh_attempted"] is False
    assert payload["scp_attempted"] is False
    assert payload["next_allowed_retry"] == "force_after_auth_change_or_auth_material_changed_receipt"


def test_auth_material_changed_after_refusal_requires_newer_timestamp(tmp_path):
    mod = load_control_module()
    refusal = tmp_path / "refusal.json"
    changed = tmp_path / "changed.json"
    refusal.write_text(json.dumps({
        "status": "WAITING_FOR_PUBLIC_KEY_AUTH",
        "generated_at": "2026-06-02T20:00:00Z",
    }), encoding="utf-8")
    changed.write_text(json.dumps({
        "status": "PUBLIC_KEY_SET",
        "changed_at": "2026-06-02T20:01:00+00:00",
    }), encoding="utf-8")
    assert mod.auth_material_changed_after_refusal(changed, refusal) is True
    changed.write_text(json.dumps({
        "status": "PUBLIC_KEY_SET",
        "changed_at": "2026-06-02T19:59:00Z",
    }), encoding="utf-8")
    assert mod.auth_material_changed_after_refusal(changed, refusal) is False
