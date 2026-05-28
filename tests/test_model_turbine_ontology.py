#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_model_turbine_task_injects_official_ontology(monkeypatch):
    import lucidota_model_turbine_overseer as turbine

    seen = {}

    def fake_http(port, path, payload=None, timeout=3):
        seen["payload"] = payload
        return {"ok": True, "raw": '{"choices":[{"message":{"content":"{\\"action\\":\\"ok\\",\\"test\\":\\"t\\",\\"risk\\":\\"low\\"}"}}]}', "json": {"choices": [{"message": {"content": '{"action":"ok","test":"t","risk":"low"}'}}]}}

    monkeypatch.setattr(turbine, "http_json", fake_http)
    result = turbine.task(8081, "db_watch", {"queue": "x"})
    user = seen["payload"]["messages"][1]["content"]
    assert "GO-25" in user
    assert "EVIDENCE supports CLAIM" in user
    assert result["json_ok"] is True


def test_model_turbine_parses_fenced_json_and_reasoning_content():
    import lucidota_model_turbine_overseer as turbine

    assert turbine.parse_model_json("```json\n{\"action\":\"ok\",\"test\":\"t\",\"risk\":\"low\"}\n```") == {"action": "ok", "test": "t", "risk": "low"}
    assert turbine.parse_model_json("thinking...\n{\"action\":\"ok\",\"test\":\"t\",\"risk\":\"low\"}") == {"action": "ok", "test": "t", "risk": "low"}
    assert turbine.parse_model_json('blah {"x":1} then {"action":"ok","test":"t","risk":"low"}') == {"action": "ok", "test": "t", "risk": "low"}
