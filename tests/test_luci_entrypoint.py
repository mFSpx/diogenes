from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.luci_operator import is_learning_prompt, is_source_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CLAW_BIN = REPO_ROOT / "claw"
LUCI_BIN = REPO_ROOT / "luci"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_luci_wrapper_exists_and_delegates_help():
    assert LUCI_BIN.is_file(), "expected ./luci wrapper script"
    assert LUCI_BIN.stat().st_mode & 0o111, "luci wrapper must be executable"

    proc = _run([str(LUCI_BIN), "--help"])
    assert proc.returncode == 0
    assert "LUCI" in proc.stdout
    assert "luci operate --text" in proc.stdout
    assert "luci flow batch" in proc.stdout


def test_luci_operate_front_door_routes_and_writes_receipt():
    proc = _run([str(LUCI_BIN), "operate", "--text", "health check ontology workflow", "--json"])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout
    payload = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
    assert payload["verdict"] in {"PASS", "DEGRADED"}
    assert payload["receipt_path"].startswith("05_OUTPUTS/luci/")
    assert payload["postgres_workflow_event"]["event_id"]
    assert payload["visible_response"]["summary"].startswith("Indy_READs:")
    assert payload["routing_fabric"]["route_targets"]
    assert any(target["target_kind"] == "workflow" for target in payload["routing_fabric"]["route_targets"])
    assert payload["routing_fabric"]["provider_lanes"]["groq"]["lane_kind"] == "external_provider"
    assert payload["routing_fabric"]["local_model_admission"]["mode"] == "strict_fail_closed"
    assert payload["workflow"]["chain_enqueue"]["db_writes_performed"] is True
    assert payload["input"]["text"] == "health check ontology workflow"
    assert payload["input"]["ingress_cache_path"].startswith("04_RUNTIME/luci/")


def test_luci_learning_prompt_routes_to_learning_slice():
    assert is_learning_prompt("study one current source or internal artifact, extract one reusable improvement, test it, and receipt the result")
    assert is_learning_prompt("board state, algorithm, and reuse")
    assert not is_learning_prompt("health check ontology workflow")


def test_luci_source_prompt_routes_to_source_slice():
    assert is_source_prompt("read the live world from Hacker News and arXiv")
    assert is_source_prompt("current world source adapter for Reddit")
    assert not is_source_prompt("study one reusable improvement")


def test_luci_plain_text_defaults_to_operate():
    proc = _run([str(LUCI_BIN), "health check"])
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "LUCI=PASS" in proc.stdout


def test_luci_launches_and_routes_stdin_when_no_args_provided():
    proc = subprocess.run(
        [str(LUCI_BIN)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        input="health check ontology workflow\n",
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "LUCI=PASS" in proc.stdout
    assert "REPORT_PATH=05_OUTPUTS/luci/" in proc.stdout


def test_legacy_claw_wrapper_is_not_operator_command():
    proc = _run([str(LEGACY_CLAW_BIN), "--help"])
    assert proc.returncode != 0
    assert "LUCI" in proc.stderr
    assert "./luci" in proc.stderr


def test_luci_wrapper_does_not_depend_on_legacy_claw_wrapper():
    text = LUCI_BIN.read_text(encoding="utf-8")
    assert '"$ROOT/claw"' not in text
    assert "target/release/luci" in text
