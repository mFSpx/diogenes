from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bare_steel_rule4_is_canonical_active_law() -> None:
    path = ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC" / "06_BARE_STEEL_DOCTRINE.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Rule 4. DB & Graph as Absolute Truth (Targeted & Async)" in text
    assert "Fetch only the hyper-specific, localized data required for the immediate task" in text
    assert "Read selectively, persist globally" in text
    assert "never block the main loop" in text
    assert "canonical graph materialization remains gated" in text

    index = (ROOT / "00_PROJECT_BRAIN" / "ACTIVE_INSTRUCTION_INDEX.md").read_text(encoding="utf-8")
    assert "00_PROJECT_BRAIN/ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md" in index

    registry = json.loads((ROOT / "00_PROJECT_BRAIN" / "instruction_authority_registry.json").read_text(encoding="utf-8"))
    assert "00_PROJECT_BRAIN/ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md" in registry["canonical_files"]
    assert any(law.get("law_key") == "bare_steel_db_graph_targeted_async" for law in registry["active_laws"])
