from __future__ import annotations

from pathlib import Path

import scripts.matrix_reticulum_gateway as gateway


def test_select_provider_uses_injected_binary_lookup_for_vibes(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "ROOT", tmp_path)
    env = {"MISTRAL_API_KEY": "present"}

    def fake_which(name: str):
        return "/usr/bin/vibe" if name == "vibe" else None

    assert gateway.select_provider("auto", env=env, which=fake_which) == "vibes"


def test_dry_run_no_listen_bootstrap_writes_receipt(tmp_path):
    config = gateway.GatewayConfig(
        listen=False,
        provider="dry-run",
        retain_receipt_dir=tmp_path,
    )
    runner = gateway.MatrixReticulumGateway(config)

    import asyncio

    result = asyncio.run(runner.run())

    assert result == 0
    files = list(tmp_path.glob('matrix_reticulum_gateway_bootstrap_*.json'))
    assert len(files) == 1
    payload = files[0].read_text(encoding='utf-8')
    assert 'bootstrap_only' in payload
    assert 'not_truth_runtime_only' in payload


def test_send_reticulum_frames_records_lane_id_when_backend_is_stubbed(monkeypatch):
    class DummyDestination:
        OUT = object()
        PLAIN = object()

        def __init__(self, identity, direction, mode, app_name, aspect):
            self.identity = identity
            self.direction = direction
            self.mode = mode
            self.app_name = app_name
            self.aspect = aspect

    class DummyPacket:
        sent = []

        def __init__(self, destination, frame):
            self.destination = destination
            self.frame = frame

        def send(self):
            DummyPacket.sent.append((self.destination.app_name, self.destination.aspect, len(self.frame)))

    class DummyRNS:
        Destination = DummyDestination

        @staticmethod
        def Reticulum():
            return object()

        @staticmethod
        def Identity():
            return object()

        Packet = DummyPacket

    monkeypatch.setattr(gateway, "RNS", DummyRNS)
    result = gateway.send_reticulum_frames(b"abcdef", lane_id="indy_reads_runtime")

    assert result["performed"] is True
    assert result["lane_id"] == "indy_reads_runtime"
    assert result["aspect"] == "lane_indy_reads_runtime"
    assert result["destination_map_size"] == 21
    assert "needle_19" in result["active_lane_ids"]
    assert DummyPacket.sent == [("lucidota", "lane_indy_reads_runtime", 6)]


def test_build_reticulum_destination_map_returns_unique_lane_keys(monkeypatch):
    class DummyDestination:
        OUT = object()
        PLAIN = object()

        def __init__(self, identity, direction, mode, app_name, aspect):
            self.identity = identity
            self.direction = direction
            self.mode = mode
            self.app_name = app_name
            self.aspect = aspect

    class DummyRNS:
        Destination = DummyDestination

        @staticmethod
        def Reticulum():
            return object()

        @staticmethod
        def Identity():
            return object()

    monkeypatch.setattr(gateway, "RNS", DummyRNS)
    lane_map = gateway.build_reticulum_destination_map(("indy_reads_runtime", "needle_0"))

    assert set(lane_map.keys()) == {"indy_reads_runtime", "needle_0"}
    assert lane_map["indy_reads_runtime"].aspect == "lane_indy_reads_runtime"
    assert lane_map["needle_0"].aspect == "lane_needle_0"
