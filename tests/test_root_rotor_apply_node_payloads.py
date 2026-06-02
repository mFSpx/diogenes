from __future__ import annotations

import json
from pathlib import Path


def test_root_rotor_apply_node_payloads_loads_valid_outputs(tmp_path: Path) -> None:
    import scripts.root_rotor_apply_node_payloads as apply

    node = {
        "schema": "lucidota.root_rotor.bible_node_payload.v1",
        "source_path": "scripts/tool.py",
        "source_sha256": "a" * 64,
        "manual_id": "FLIGHT_MAN",
        "node_title": "Tool Script",
        "node_kind": "WORKFLOW",
        "ontology_tags": ["WORKFLOW", "OBJECT", "RECEIPT"],
        "payload_asd_ste100": "This script runs a tool.",
        "dependencies": ["1.0.0"],
        "affects_nodes": ["4.0.0"],
        "what_it_is_and_does": "This script runs a tool.",
        "exact_interactions": {},
        "operating_limits_failure_modes": [],
        "integration_points": {},
        "confidence": "high",
        "evidence_refs": ["scripts/tool.py:1"],
    }
    out = tmp_path / "node.json"
    out.write_text(json.dumps(node), encoding="utf-8")

    loaded = apply.load_node_payloads(tmp_path)

    assert len(loaded) == 1
    update = apply.to_update_record(loaded[0])
    assert update["source_path"] == "scripts/tool.py"
    assert update["title"] == "Tool Script"
    assert update["payload_format"] == "json"
    assert update["status"] == "verified"
    assert update["node_kind"] == "WORKFLOW"
    assert update["ontology_tags"] == ["WORKFLOW", "OBJECT", "RECEIPT"]
    assert update["dependencies"] == ["1.0.0"]
    assert update["affects_nodes"] == ["4.0.0"]
    assert json.loads(update["payload"])["payload_asd_ste100"] == "This script runs a tool."


def test_root_rotor_apply_node_payloads_rejects_wrong_schema(tmp_path: Path) -> None:
    import scripts.root_rotor_apply_node_payloads as apply

    (tmp_path / "bad.json").write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")

    assert apply.load_node_payloads(tmp_path) == []


def test_root_rotor_apply_node_payloads_keeps_only_law_of_root_edge_refs() -> None:
    import scripts.root_rotor_apply_node_payloads as apply

    payload = {
        "schema": "lucidota.root_rotor.bible_node_payload.v1",
        "source_path": ".claude/skills/vibes-delegate/SKILL.md",
        "source_sha256": "b" * 64,
        "node_title": "Vibes Delegate",
        "dependencies": ["MISTRAL_API_KEY", "1.0.0", "GROQ_API_KEY", "4.2.0"],
        "affects_nodes": ["not-a-node", "5.900.0"],
    }

    update = apply.to_update_record(payload)

    assert update["dependencies"] == ["1.0.0", "4.2.0"]
    assert update["affects_nodes"] == ["5.900.0"]
    raw_payload = json.loads(update["payload"])
    assert raw_payload["dependencies"] == ["MISTRAL_API_KEY", "1.0.0", "GROQ_API_KEY", "4.2.0"]
