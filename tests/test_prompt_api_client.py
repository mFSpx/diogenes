from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prompt_api_client


def test_build_cloud_packet_request_clamps_bounds_and_keeps_raw_bodies_off_by_default() -> None:
    payload = prompt_api_client.build_cloud_packet_request(
        work_order_id="1f2c3d4e-1111-2222-3333-444455556666",
        max_chars=999999,
        max_items=999,
        task_type="repair",
        target_model="groq",
    )

    assert payload["work_order_id"] == "1f2c3d4e-1111-2222-3333-444455556666"
    assert payload["max_chars"] <= prompt_api_client.MAX_CHARS_CAP
    assert payload["max_items"] <= prompt_api_client.MAX_ITEMS_CAP
    assert payload["task_type"] == "repair"
    assert payload["target_model"] == "groq"
    assert payload["include_raw_bodies"] is False


def test_cloud_packet_helper_uses_postgrest_rpc_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse(b'{"contract_name":"prompt_api.cloud_packet.v1","work_order_id":"wo-1"}')

    monkeypatch.setattr(prompt_api_client.urllib.request, "urlopen", fake_urlopen)

    packet = prompt_api_client.cloud_packet(
        base_url="http://127.0.0.1:3000",
        work_order_id="wo-1",
        max_chars=2048,
        max_items=8,
        task_type="repair",
        target_model="codex",
    )

    assert captured["url"] == "http://127.0.0.1:3000/rpc/cloud_packet"
    assert captured["method"] == "POST"
    assert packet["contract_name"] == "prompt_api.cloud_packet.v1"
    assert packet["work_order_id"] == "wo-1"
    assert captured["body"]["work_order_id"] == "wo-1"

