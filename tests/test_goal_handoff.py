from pathlib import Path

from scripts import goal_handoff


def test_build_handoff_contains_prefix_step_counter_and_dev_notes():
    handoff = goal_handoff.build_handoff(
        goal_title="Cryptid Goal",
        objective="Keep a crash-resumable trail.",
        step=3,
        total_steps=25,
        status="running",
        completed="Printed Hello World.",
        next_action="Fill the cabbage blank.",
        resume_command="cat GOALS/CURRENT_HANDOFF.md",
        dev_notes="Tiny proof lantern; mothman would approve.",
    )

    assert '"Save This Prompt, Pass on this Handoff:"' in handoff
    assert "Current step: 3/25" in handoff
    assert "Technical Summary Review and Dev Notes" in handoff
    assert "Tiny proof lantern" in handoff
    assert "cat GOALS/CURRENT_HANDOFF.md" in handoff


def test_update_step_writes_current_handoff_and_appends_goal_log(tmp_path):
    root = tmp_path / "GOALS"

    goal_handoff.update_step(
        root=root,
        goal_title="Crash Recovery",
        objective="Leave a handoff at every step.",
        step=1,
        total_steps=2,
        step_title="Start trail",
        status="running",
        completed="Created first marker.",
        next_action="Finish trail.",
        resume_command="cat GOALS/CURRENT_HANDOFF.md",
        dev_notes="Marker placed; no chupacabra drift.",
    )
    goal_handoff.update_step(
        root=root,
        goal_title="Crash Recovery",
        objective="Leave a handoff at every step.",
        step=2,
        total_steps=2,
        step_title="Finish trail",
        status="complete",
        completed="Closed loop.",
        next_action="Resume a new goal.",
        resume_command="cat GOALS/CURRENT_HANDOFF.md",
        dev_notes="Loop sealed; sasquatch can audit it.",
    )

    current = (root / "CURRENT_HANDOFF.md").read_text()
    log = (root / "GOAL_LOG.md").read_text()
    assert "Current step: 2/2" in current
    assert "Loop sealed" in current
    assert log.count('"Save This Prompt, Pass on this Handoff:"') == 2
    assert "## Step 1/2 — Start trail" in log
    assert "## Step 2/2 — Finish trail" in log


def test_demo_generates_25_steps_without_nested_folder_sprawl(tmp_path):
    root = tmp_path / "GOALS"

    report = goal_handoff.run_demo(root=root)

    demo_log = (root / "DEMO_25_STEP_LOG.md").read_text()
    current = (root / "CURRENT_HANDOFF.md").read_text()
    assert report["step_count"] == 25
    assert demo_log.count("## Step ") == 25
    assert "## Step 1/25 — Print Hello World" in demo_log
    assert "4+2=-1" in demo_log
    assert "cabbage are the second stinkiest salmonoid" in demo_log
    assert "Teleportato a Kronenberg to a Borg" in demo_log
    assert "Current step: 25/25" in current
    assert not [p for p in root.iterdir() if p.is_dir()]


def test_scaffold_workspace_creates_prompt_and_check_passes(tmp_path):
    root = tmp_path / "GOALS"

    goal_handoff.scaffold_workspace(root=root)
    result = goal_handoff.check_workspace(root=root)

    assert result["passed"] is True
    assert (root / "README.md").exists()
    assert (root / "GOAL_HANDOFF_PROMPT.md").exists()
    prompt = (root / "GOAL_HANDOFF_PROMPT.md").read_text()
    assert '"Save This Prompt, Pass on this Handoff:"' in prompt
    assert "append this as the final step" in prompt
    assert "Technical Summary Review and Dev Notes" in prompt

def test_goal_handoff_helper_stays_tiny():
    source = Path(goal_handoff.__file__).read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith('#')]
    assert len(code_lines) <= 100


def test_prompt_and_readme_include_yap_trap_budget(tmp_path):
    root = tmp_path / "GOALS"
    goal_handoff.scaffold_workspace(root=root)
    assert "Yap Trap" in (root / "README.md").read_text()
    assert "no yappity-yap" in (root / "GOAL_HANDOFF_PROMPT.md").read_text()


def test_recovery_matrix_advertises_goal_handoff_check():
    matrix = Path("scripts/recovery_matrix.py").read_text()
    assert "goal_handoff_check" in matrix
    assert "scripts/goal_handoff.py --root GOALS check" in matrix


def test_scaffold_includes_model_economy_policy(tmp_path):
    root = tmp_path / "GOALS"
    goal_handoff.scaffold_workspace(root=root)
    policy = (root / "AGENT_ORCHESTRATION_POLICY.md").read_text()
    prompt = (root / "GOAL_HANDOFF_PROMPT.md").read_text()
    assert "Cheapest Capable Model" in policy
    assert "do not change the main-window model" in policy
    assert "smallest capable" in policy
    assert "coding-only prompt" in policy
    assert "chunk" in policy.lower()
    assert "Cheapest Capable Model" in prompt


def test_agent_policy_preserves_capabilities_and_minimizes_mutation(tmp_path):
    root = tmp_path / "GOALS"
    goal_handoff.scaffold_workspace(root=root)
    policy = (root / "AGENT_ORCHESTRATION_POLICY.md").read_text()
    live_policy = Path("GOALS/AGENT_ORCHESTRATION_POLICY.md").read_text()
    for text in [policy, live_policy]:
        assert "Capability Preservation" in text
        assert "least mutation" in text
        assert "center-out" in text


def test_active_spec_encodes_real_feature_wargame_and_no_delete_law():
    spec = Path("00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md").read_text()
    assert "local-first sovereign data exocortex" in spec
    assert "OSINT" in spec
    assert "investigative journalism" in spec
    assert "Asymmetric dev wargame rule" in spec
    assert "No-delete rule" in spec
    assert "quarantine/archive/index" in spec


def test_agent_startup_mentions_model_economy_policy():
    agents = Path("AGENTS.md").read_text()
    index = Path("00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md").read_text()
    assert "Agent Model Economy Law" in agents
    assert "GOALS/AGENT_ORCHESTRATION_POLICY.md" in index


def test_scaffold_includes_external_plugin_build_mode(tmp_path):
    root = tmp_path / "GOALS"
    goal_handoff.scaffold_workspace(root=root)
    mode = (root / "EXTERNAL_PLUGIN_BUILD_MODE.md").read_text()
    assert "Codex-first, BYO-LLM" in mode
    assert "Adapter Contract" in mode
    assert "three next pitches" in mode
    assert "operator slop is allowed" in mode
    assert "orchestrator slop is not" in mode


def test_current_goal3_model_fabric_audit_exists_and_names_existing_lanes():
    audit = Path("GOALS/MODEL_FABRIC_AUDIT.md").read_text()
    for phrase in ["needle_swarm_6x", "mamba7b_ram", "bonsai4b_ram", "DeepSeek", "Groq", "Cohere"]:
        assert phrase in audit
    assert "secret values are not printed" in audit


def test_goal3_plugin_build_mode_bootstrap_manifest_maps_existing_system():
    import json
    manifest = json.loads(Path("GOALS/plugin_build_mode_bootstrap.json").read_text())
    assert manifest["schema"] == "lucidota.goals.plugin_build_mode_bootstrap.v1"
    assert manifest["mode"] == "codex_first_byo_llm"
    assert manifest["code_budget"]["goal_handoff_py_max_code_lines"] <= 100
    lane_names = {lane["name"] for lane in manifest["local_lanes"]}
    assert {"needle_swarm_6x", "mamba7b_ram", "bonsai4b_ram", "deepseek_r1_qwen_1p5b_gpu"} <= lane_names
    provider_names = {provider["name"] for provider in manifest["cloud_lanes"]}
    assert {"groq", "cohere"} <= provider_names
    assert manifest["lifecycle"][0]["step"] == "admission_check"
    assert "provider_ping" in [item["step"] for item in manifest["lifecycle"]]


def test_goal3_foss_reuse_audit_records_reuse_not_reinvent():
    audit = Path("GOALS/FOSS_REUSE_AUDIT.md").read_text()
    for phrase in ["LiteLLM", "OpenCode", "Aider", "Continue", "llama.cpp"]:
        assert phrase in audit
    assert "Do not build a new coding agent" in audit
    assert "GOALS remains the tiny handoff/adapter contract layer" in audit


def test_goal3_prompt_stash_has_external_plugin_prompt():
    prompts = Path("GOALS/GOAL_PROMPTS.md").read_text()
    assert "Prompt 003" in prompts
    assert "External Plugin Build Mode" in prompts
    assert "cheapest capable" in prompts
    assert "BYO LLM" in prompts


def test_goal_docs_state_cloud_is_optional_not_baseline():
    plugin = Path("GOALS/EXTERNAL_PLUGIN_BUILD_MODE.md").read_text()
    foss = Path("GOALS/FOSS_REUSE_AUDIT.md").read_text()
    fabric = Path("GOALS/MODEL_FABRIC_AUDIT.md").read_text()
    for text in [plugin, foss, fabric]:
        assert "cloud" in text.lower()
    assert "cloud is optional" in plugin.lower()
    assert "baseline dependency" in plugin.lower()
    assert "optional" in foss.lower()
    assert "baseline" in foss.lower()
    assert "optional peripherals" in fabric.lower()
