from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "00_PROJECT_BRAIN" / "operator_graph_eligibility_policy.json"
ALLOWLIST = ROOT / "00_PROJECT_BRAIN" / "canonical_graph_write_allowlist.json"
SPEC = ROOT / "00_PROJECT_BRAIN" / "ACTIVE_SPEC" / "07_WORKING_REALITY_LAW.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_operator_graph_eligibility_policy_names_authorized_domains() -> None:
    policy = load(POLICY)

    assert policy["schema"] == "lucidota.operator_graph_eligibility_policy.v1"
    assert policy["status"] == "active"
    assert policy["authority_class"] == "operator_authored_assertion"
    assert policy["default_decision"] == "stage_or_materialize_with_existing_graph_gates"
    domains = {item["domain_key"]: item for item in policy["domains"]}
    assert set(domains) >= {"rickshaw_robbery", "nordby_squeeze", "operator_life_facts"}
    assert "Nordley Squeeze" in domains["nordby_squeeze"]["labels"]
    assert "NORDLEY_SQUEEZE" in domains["nordby_squeeze"]["labels"]
    for item in domains.values():
        assert item["graph_eligible"] is True
        assert item["correction_mode"] == "operator_can_mark_wrong_later_without_deleting_history"
        assert "evidence_refs_required" in item["required_gates"]
        assert "graph_promotion_gate_or_materialization_helper" in item["required_gates"]


def test_canonical_allowlist_points_to_operator_graph_policy_without_direct_write_shortcut() -> None:
    allowlist = load(ALLOWLIST)

    policy_ref = allowlist["operator_graph_eligibility_policy"]
    assert policy_ref["policy_path"] == "00_PROJECT_BRAIN/operator_graph_eligibility_policy.json"
    assert policy_ref["direct_graph_write_shortcut"] is False
    assert policy_ref["canonical_materialization_requires_existing_helper"] is True


def test_working_reality_law_records_operator_graph_eligibility_decision() -> None:
    text = SPEC.read_text(encoding="utf-8")

    assert "Operator graph eligibility decision — 2026-05-27" in text
    assert "Rickshaw Robbery" in text
    assert "Nordby Squeeze" in text
    assert "Nordley Squeeze" in text
    assert "operator-life facts" in text
    assert "wrong later" in text
