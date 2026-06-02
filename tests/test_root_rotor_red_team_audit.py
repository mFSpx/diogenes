from __future__ import annotations


def test_root_rotor_red_team_verdict_fails_on_draft_or_missing_postgrest() -> None:
    import scripts.root_rotor_red_team_audit as audit

    findings = audit.classify_findings({
        "total_nodes": 10,
        "draft_nodes": 9,
        "broken_parent_count": 0,
        "parent_cycle_count": 0,
        "dependency_cycle_count": 0,
        "postgrest_available": False,
        "postgrest_api_available": False,
        "model_payload_count": 1,
    })

    assert findings["verdict"] == "FAIL"
    assert "manual_incomplete_draft_nodes" in findings["blockers"]
    assert "postgrest_binary_missing" in findings["blockers"]
    assert "postgrest_api_unavailable" in findings["blockers"]


def test_root_rotor_red_team_verdict_passes_when_core_checks_clean() -> None:
    import scripts.root_rotor_red_team_audit as audit

    findings = audit.classify_findings({
        "total_nodes": 10,
        "draft_nodes": 0,
        "broken_parent_count": 0,
        "parent_cycle_count": 0,
        "dependency_cycle_count": 0,
        "postgrest_available": True,
        "postgrest_api_available": True,
        "model_payload_count": 10,
    })

    assert findings["verdict"] == "PASS"
    assert findings["blockers"] == []


def test_root_rotor_red_team_can_probe_postgrest_api(monkeypatch) -> None:
    import scripts.root_rotor_red_team_audit as audit

    class GoodResponse:
        status_code = 200

    monkeypatch.setattr(audit.requests, "get", lambda *_args, **_kwargs: GoodResponse())

    assert audit.postgrest_api_available("http://127.0.0.1:3000") is True
