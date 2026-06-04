from __future__ import annotations

import json


def test_preflight_audits_root_manual_route_when_openapi_exposes_it(monkeypatch, capsys):
    import scripts.codex_context_preflight as preflight

    def fake_fetch_json(path: str, query: dict[str, str] | None = None):
        if path == "":
            return 200, {"paths": {"/root_law_docs": {"get": {}}, "/manual_current": {"get": {}}}}, ""
        if path == "root_law_docs":
            return 200, [{"route_id": "root_law_docs", "status": "implemented"}], ""
        return 200, [{"route_id": path}], ""

    monkeypatch.setattr(preflight, "fetch_json", fake_fetch_json)

    assert preflight.main() == 0
    payload = json.loads(capsys.readouterr().out)
    routes = {row["route"] for row in payload["route_findings"]}
    assert "/root_law_docs" in routes
    assert payload["verified_working_state"]["root_law_docs_visible"] is True
    assert any(row["route"] == "/root_law_docs" and row["http_status"] == 200 for row in payload["route_findings"])
