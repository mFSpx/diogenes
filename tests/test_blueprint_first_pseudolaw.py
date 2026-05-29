#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAW = ROOT / "00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md"


def test_blueprint_first_pseudolaw_is_startup_readable() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    text = LAW.read_text(encoding="utf-8")
    assert "BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md" in agents
    assert "Blueprint first, model second" in text
    assert "workflow path belongs in readable source code" in text
    assert "A model must not be the hidden controller" in text
    assert "PocketFlow is the simplicity mirror" in text


def test_blueprint_first_pseudolaw_is_active_instruction_authority() -> None:
    registry = json.loads((ROOT / "00_PROJECT_BRAIN/instruction_authority_registry.json").read_text(encoding="utf-8"))
    active_index = (ROOT / "00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md").read_text(encoding="utf-8")
    assert "00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md" in registry["canonical_files"]
    assert any(law.get("law_key") == "blueprint_first_model_second_pocketflow_hygiene" for law in registry["active_laws"])
    assert "workflow hygiene pseudolaw" in active_index
    assert "BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md" in active_index


def test_pocketflow_clone_core_is_available_as_local_yardstick() -> None:
    core = ROOT / "01_REPOS/PocketFlow/pocketflow/__init__.py"
    assert core.exists()
    lines = core.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    assert sum(1 for line in lines if line.strip()) >= 80
