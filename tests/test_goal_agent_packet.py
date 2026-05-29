import json
import subprocess
import sys
from pathlib import Path

from scripts import goal_agent_packet


def test_goal_agent_packet_builds_coding_only_cheapest_capable_prompt(tmp_path):
    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text("- Goal: G\n- Next action: edit scripts/x.py and run pytest\n")
    (tmp_path / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md").write_text("Cheapest Capable Model\n")
    (tmp_path / "GOALS" / "plugin_build_mode_bootstrap.json").write_text(json.dumps({"local_lanes":[{"name":"needle_swarm_6x"}],"cloud_lanes":[{"name":"groq"},{"name":"cohere"}]}))
    pkt = goal_agent_packet.build_packet(tmp_path, target="codex", task="edit scripts/x.py and run pytest", files=["scripts/x.py"], complexity="simple")
    assert pkt["schema"] == "lucidota.goals.agent_packet.v1"
    assert pkt["target"] == "codex"
    assert pkt["model_policy"]["main_window_model_change"] == "forbidden_without_explicit_operator_tool"
    assert pkt["model_policy"]["selected_tier"] == "tiny_or_mini"
    assert pkt["coding_prompt"]["file_ownership"] == ["scripts/x.py"]
    assert "Do not revert unrelated edits" in pkt["coding_prompt"]["constraints"]
    assert "changed files" in pkt["coding_prompt"]["return_contract"]
    assert pkt["coding_prompt"]["output_contract"]["required_output"] == ["status", "result", "next_action", "receipt_path", "evidence_refs", "decision_pairs"]
    assert pkt["coding_prompt"]["output_contract"]["envelope_style"] == "single_exact_top_level_json_object"
    assert pkt["model_policy"]["output_contract"]["decision_pairs_min"] == 2
    assert pkt["adapters"]["cloud_lanes"] == ["groq", "cohere"]
    assert pkt["adapter"]["selected"] == "needle_swarm_6x"
    assert pkt["model_calls_performed"] is False


def test_goal_agent_packet_remains_local_first_when_cloud_lanes_are_absent(tmp_path):
    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text("- Goal: G\n- Next action: edit scripts/x.py and run pytest\n")
    (tmp_path / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md").write_text("Kernel Rule: local-first.\n")
    (tmp_path / "GOALS" / "plugin_build_mode_bootstrap.json").write_text(json.dumps({"local_lanes":[{"name":"needle_swarm_6x"}],"cloud_lanes":[]}))
    pkt = goal_agent_packet.build_packet(tmp_path, task="edit scripts/x.py and run pytest", files=["scripts/x.py"], complexity="simple")
    assert pkt["adapter"]["selected"] == "needle_swarm_6x"
    assert pkt["adapters"]["cloud_lanes"] == []
    assert pkt["proof_policy"].startswith("No completion claim")


def test_goal_agent_packet_prefers_other_local_lanes_over_cloud_when_needle_missing(tmp_path):
    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text("- Goal: G\n- Next action: edit scripts/x.py and run pytest\n")
    (tmp_path / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md").write_text("Kernel Rule: local-first.\n")
    (tmp_path / "GOALS" / "plugin_build_mode_bootstrap.json").write_text(json.dumps({"local_lanes":[{"name":"mamba7b_ram"},{"name":"bonsai4b_ram"}],"cloud_lanes":[{"name":"groq"}]}))
    pkt = goal_agent_packet.build_packet(tmp_path, task="edit scripts/x.py and run pytest", files=["scripts/x.py"], complexity="simple")
    assert pkt["adapter"]["selected"] == "llama_cpp_heavy"
    assert pkt["adapters"]["cloud_lanes"] == ["groq"]


def test_goal_agent_packet_cli_writes_receipt_and_stays_under_100_loc():
    proc = subprocess.run(
        [sys.executable, "scripts/goal_agent_packet.py", "--target", "generic", "--task", "run focused tests", "--file", "tests/test_goal_agent_packet.py", "--complexity", "simple", "--json"],
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "REPORT_PATH=05_OUTPUTS/goals/goal_agent_packet_" in proc.stdout
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["coding_prompt"]["acceptance_checks"]
    source = Path("scripts/goal_agent_packet.py").read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    assert len(code_lines) <= 100


def test_goal_agent_packet_flags_main_window_model_switch_as_blocked():
    pkt = goal_agent_packet.build_packet(task="change the main window model then code", complexity="simple")
    assert "main_window_model_change_forbidden" in pkt["blockers"]
    assert "change main-window model" in pkt["model_policy"]["blocked_operations"]


def test_goal_agent_packet_is_declared_in_goals_manifests():
    policy = Path("GOALS/AGENT_ORCHESTRATION_POLICY.md").read_text()
    mode = Path("GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md").read_text()
    manifest = Path("GOALS/plugin_build_mode_bootstrap.json").read_text()
    recovery = Path("scripts/recovery_matrix.py").read_text()
    assert "goal_agent_packet.py" in policy
    assert "goal_agent_packet.py" in mode
    assert "goal_agent_packet.py" in manifest
    assert "goal_agent_packet" in recovery


def test_plugin_bootstrap_has_machine_checkable_adapter_registry():
    data = json.loads(Path("GOALS/plugin_build_mode_bootstrap.json").read_text())
    required = {"provider_type", "env_key_names", "endpoint", "dry_run_cmd", "execute_cmd", "expected_receipt_glob", "stop_or_rollback_cmd", "safety_limits"}
    for name in ["groq", "cohere", "needle_swarm_6x", "llama_cpp_heavy"]:
        assert name in data["adapter_registry"]
        assert required <= set(data["adapter_registry"][name]), name
    assert "secret_values_forbidden" in data["adapter_registry"]["groq"]["safety_limits"]
    assert "no_auto_heavy_daemon" in data["adapter_registry"]["llama_cpp_heavy"]["safety_limits"]
    assert "local-chat" in data["adapter_registry"]["llama_cpp_heavy"]["invoke_cmd"]
    assert "local-chat" in data["adapter_registry"]["needle_swarm_6x"]["invoke_cmd"]
    assert data["latest_runtime_receipts"]["swarm_usage_ledger_current"].startswith("05_OUTPUTS/goals/swarm_usage_ledger_")


def test_goal_agent_packet_uses_registry_as_adapter_source_for_heavy_local_tasks():
    pkt = goal_agent_packet.build_packet(task="wire mamba bonsai llama.cpp coding lane", complexity="integration")
    assert pkt["adapter"]["selected"] == "llama_cpp_heavy"
    assert pkt["adapter"]["execute_cmd"] == json.loads(Path("GOALS/plugin_build_mode_bootstrap.json").read_text())["adapter_registry"]["llama_cpp_heavy"]["execute_cmd"]
    assert pkt["adapters"]["registry_source"] == "GOALS/plugin_build_mode_bootstrap.json"
    assert "frontier/high only for architecture" in pkt["model_policy"]["reasoning_split"]


def test_goal_agent_packet_enforces_exact_top_level_worker_envelope():
    pkt = goal_agent_packet.build_packet(task="repair worker receipts", complexity="simple")
    contract = pkt["coding_prompt"]["output_contract"]
    assert contract["required_output"] == ["status", "result", "next_action", "receipt_path", "evidence_refs", "decision_pairs"]
    assert contract["forbidden_output"] == ["nested_result_object", "prose", "meta-summary", "hidden-reasoning"]
    assert contract["envelope_style"] == "single_exact_top_level_json_object"
    assert contract["recommended_max_tokens_floor"] == 256
    assert contract["recommended_max_tokens_ceiling"] == 512
