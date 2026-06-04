from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md"
PROMPT = ROOT / "GOALS" / "OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md"
GOAL_PROMPTS = ROOT / "GOALS" / "GOAL_PROMPTS.md"


def _section_from_marker(text: str, marker: str) -> str:
    assert marker in text
    start = text.index(marker)
    end = text.find("\n---\n", start)
    return text[start:] if end == -1 else text[start:end]


def test_root_rotor_reduction_laws_are_encoded_in_existing_policy() -> None:
    text = POLICY.read_text(encoding="utf-8")

    required_fragments = [
        "Root-Rotor Manual Draft Reduction Law",
        "needs_operator_label",
        "exactly 200 candidate nodes",
        "shallow typed output directories",
        "Do not flatten all receipts into `05_OUTPUTS/`",
        "live OpenAPI route enumeration",
        "verified_count, deprecated_count, and needs_operator_label_count",
        "authority docs verify",
        "01_REPOS/claudecode",
        "services/ternary_lab",
        "06_SCHEMA` routes to `SYSTEM_ARCH",
        "scripts/absurd*` route to `RUNTIME_GOVERNOR",
        "model/capability ledgers route to `lucidota_fabric",
        "systemd is read-only by default",
        "staged row -> validate hash/schema/route -> receipt -> transactional promotion",
    ]
    for fragment in required_fragments:
        assert fragment in text


def test_sendable_prompt_is_bounded_and_uses_goals_root_rotor_contract() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    prompt = _section_from_marker(text, "OPERATE: ROOT-ROTOR MANUAL DRAFT REDUCTION. NO NEW CONTROL PLANES.")

    assert len(prompt) <= 4000
    required_fragments = [
        "GOALS owns continuation",
        "exactly 200",
        "needs_operator_label",
        "no new control planes",
        "No flat 05_OUTPUTS swamp",
        "live OpenAPI route enumeration",
        "authority docs verify",
        "historical prose deprecates",
        "01_REPOS/claudecode",
        "services/ternary_lab",
        "lucidota_fabric",
        "systemd read-only by default",
        "staged -> validated -> receipted -> promoted",
        "Do not patch orchestration scripts",
    ]
    for fragment in required_fragments:
        assert fragment in prompt

    forbidden_fragments = [
        "05_OUTPUTS/session_handoff",
        "review_required",
        "flat directory: 05_OUTPUTS",
        "Commit directly",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in prompt


def test_total_migration_execution_prompt_is_saved_as_active_goal_prompt() -> None:
    text = PROMPT.read_text(encoding="utf-8")
    prompt = _section_from_marker(text, "ROOT-ROTOR TOTAL MIGRATION EXECUTION PROMPT")

    required_fragments = [
        "Objective: finish the manual node migration job, not merely run Batch 001",
        ".venv/bin/python scripts/root_rotor_manual_queue.py --batch-size 200 --source 01_REPOS/claudecode/src/services --output-dir 05_OUTPUTS/root_rotor_manuals",
        "manual_incomplete_draft_nodes",
        "Classify every candidate as exactly one of: `verified`, `deprecated`, `needs_operator_label`",
        "No arbitrary DML. No direct ad hoc DB mutation.",
        "No flat folder swamp.",
        "Begin now with Batch 001, then keep executing batches under these laws until the whole job is done or honestly blocked.",
    ]
    for fragment in required_fragments:
        assert fragment in prompt

    goal_prompts = GOAL_PROMPTS.read_text(encoding="utf-8")
    assert "Prompt 004 — Root-Rotor Total Migration Execution Prompt" in goal_prompts
    assert "GOALS/OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md" in goal_prompts
