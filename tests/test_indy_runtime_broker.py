from __future__ import annotations

import json

from scripts import indy_runtime_broker


def test_registry_snapshot_uses_postgrest_routes(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_fetch_json(path: str, query: dict[str, str] | None = None, *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, timeout: float = 5.0):
        calls.append((path, query))
        if path == "":
            return 200, {"paths": {"/workflow_registry": {"get": {}}, "/manual_current": {"get": {}}}}, ""
        return 200, [{"id": "row-1", "name": "registry"}], ""

    monkeypatch.setattr(indy_runtime_broker, "fetch_json", fake_fetch_json)

    snapshot = indy_runtime_broker.registry_snapshot(base_url="http://127.0.0.1:3000", routes=("workflow_registry",))
    assert snapshot["openapi_status"] == 200
    assert snapshot["route_rows"][0]["route"] == "/workflow_registry"
    assert calls[0][0] == ""
    assert calls[1][0] == "model_registry"
    assert calls[2][0] == "provider_registry"
    assert calls[3][0] == "workflow_registry"


def test_build_cloud_packet_delegates_to_prompt_api_client(monkeypatch) -> None:
    captured = {}

    def fake_cloud_packet(**kwargs):
        captured.update(kwargs)
        return {"contract_name": "prompt_api.cloud_packet.v1", "work_order_id": kwargs["work_order_id"]}

    monkeypatch.setattr(indy_runtime_broker.prompt_api_client, "cloud_packet", fake_cloud_packet)

    payload = indy_runtime_broker.build_cloud_packet(work_order_id="wo-1", max_chars=512, max_items=4, task_type="repair", target_model="codex")
    assert payload["work_order_id"] == "wo-1"
    assert captured["base_url"] == indy_runtime_broker.DEFAULT_BASE_URL
    assert captured["max_chars"] == 512
    assert captured["max_items"] == 4


def test_registry_snapshot_reports_live_local_model_role_coverage(monkeypatch) -> None:
    def fake_fetch_json(path: str, query: dict[str, str] | None = None, *, base_url: str = indy_runtime_broker.DEFAULT_BASE_URL, timeout: float = 5.0):
        if path == "":
            return 200, {"paths": {}}, ""
        if path == "model_registry":
            return 200, [
                {"model_id": "needle-26m", "role": "router", "slot_name": "router_swarm", "loadout_id": "gtx1650-special-forces-v0", "expected_vram_mb": 256, "benchmark_status": "accepted", "notes": "router"},
                {"model_id": "mamba2-1.3b-listener", "role": "listener", "slot_name": "listener", "loadout_id": "gtx1650-special-forces-v0", "expected_vram_mb": 909, "benchmark_status": "accepted", "notes": "listener"},
            ], ""
        if path == "provider_registry":
            return 200, [{"provider_key": "local_model", "active": True}], ""
        return 200, [{"route_id": "manual_current"}], ""

    monkeypatch.setattr(indy_runtime_broker, "fetch_json", fake_fetch_json)
    snapshot = indy_runtime_broker.registry_snapshot(base_url="http://127.0.0.1:3000", routes=("workflow_registry",))

    assert snapshot["local_model_roles"]["router"]["model_id"] == "needle-26m"
    assert snapshot["local_model_roles"]["thinker"] is None
    assert snapshot["provider_registry_status"] == 200
    assert indy_runtime_broker.choose_local_model(role="router", base_url="http://127.0.0.1:3000")["model_id"] == "needle-26m"
    assert indy_runtime_broker.choose_local_model(role="thinker", base_url="http://127.0.0.1:3000") is None
