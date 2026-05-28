from pathlib import Path

from scripts import goal_dev_control


def test_goal_dev_control_scores_cadence_and_routes_without_model_calls(tmp_path):
    root = tmp_path
    (root / "GOALS").mkdir()
    (root / "scripts").mkdir()
    (root / "05_OUTPUTS" / "goals").mkdir(parents=True)
    (root / "GOALS" / "CURRENT_HANDOFF.md").write_text(
        "# CURRENT\n- Generated: `2026-05-26T01:00:00Z`\n- Current step: 2/4\n"
        "- Next action: Run tests and write receipts.\n"
    )
    (root / "GOALS" / "GOAL_LOG.md").write_text(
        "- Generated: `2026-05-26T00:00:00Z`\n- Generated: `2026-05-26T00:30:00Z`\n"
    )
    (root / "scripts" / "tiny.py").write_text("print('x')\n# comment\n\nprint('y')\n")
    (root / "GOALS" / "notes.md").write_text("short doc\n")
    (root / "05_OUTPUTS" / "goals" / "receipt.json").write_text("{}\n")

    report = goal_dev_control.build_report(
        root=root,
        away_minutes=15,
        text="health receipt verify test no slop",
        paths=["scripts/tiny.py", "GOALS/notes.md"],
    )

    assert report["schema"] == "lucidota.goals.dev_supply_control.v1"
    assert report["model_calls_performed"] is False
    assert report["cadence"]["handoff_count"] == 3
    assert report["cadence"]["avg_minutes_between_handoffs"] == 30
    assert report["effective_loc"]["code_lines"] == 2
    assert report["effective_loc"]["receipt_count"] == 1
    assert report["effective_loc"]["effective_loc_per_hour"] > 0
    assert report["route"]["selected_action"] in {"tiny_local", "operator_return_review"}
    assert "dev_mode_entrypoints" in report
    assert report["dev_mode_entrypoints"]["handoff"] == "GOALS/CURRENT_HANDOFF.md"


def test_goal_dev_control_helper_stays_tiny():
    source = Path(goal_dev_control.__file__).read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    assert len(code_lines) <= 100


def test_goal_dev_control_is_wired_into_policy_manifest_and_recovery():
    policy = Path("GOALS/AGENT_ORCHESTRATION_POLICY.md").read_text()
    manifest = Path("GOALS/plugin_build_mode_bootstrap.json").read_text()
    recovery = Path("scripts/recovery_matrix.py").read_text()
    assert "Dev Supply Control" in policy
    assert "scripts/goal_dev_control.py" in manifest
    assert "ALGOS.bandit_router" in manifest
    assert "goal_dev_control_check" in recovery


def test_dev_mode_feature_audit_and_integration_are_machine_checkable():
    import json
    audit = json.loads(Path("GOALS/DEV_MODE_FEATURE_AUDIT.json").read_text())
    integration = json.loads(Path("GOALS/DEV_MODE_INTEGRATION.json").read_text())
    required = {
        "crash_handoff",
        "cheapest_capable_agents",
        "external_plugin_build_mode",
        "foss_reuse",
        "local_model_fabric",
        "cloud_adapters",
        "dev_supply_control",
        "runtime_stop_control",
        "slop_control",
        "handoff_packet",
        "agent_packet_exporter",
        "adapter_registry",
    }
    assert audit["schema"] == "lucidota.goals.dev_mode_feature_audit.v1"
    features = {f["key"]: f for f in audit["features"]}
    assert required <= set(features)
    assert all(f["status"] == "verified" for f in features.values())
    for f in features.values():
        assert f["evidence"], f["key"]
        for path in f["evidence"]:
            assert Path(path).exists(), (f["key"], path)
    assert integration["schema"] == "lucidota.goals.dev_mode_integration.v1"
    assert "software" in integration["status_ledger_sections"]
    assert "hardware_runtime_targets" in integration["status_ledger_sections"]
    assert integration["single_owner"] == "GOALS"


def test_subsystem_manifest_exists_and_checks_have_real_files():
    import json
    manifest = json.loads(Path("GOALS/DEV_MODE_SUBSYSTEMS.json").read_text())
    assert manifest["schema"] == "lucidota.goals.subsystems.manifest.v1"
    ids = {s["id"]: s for s in manifest["subsystems"]}
    for required in ["handoff", "dev_supply", "status", "model_runtime", "fabric", "queue_recovery", "language_subsystem", "agent_packet_exporter"]:
        assert required in ids
    assert "scripts/lucidota_cli.py" in ids["language_subsystem"]["files"]
    for subsystem in manifest["subsystems"]:
        for path in subsystem["files"]:
            assert Path(path).exists(), (subsystem["id"], path)
        assert subsystem["check"]
