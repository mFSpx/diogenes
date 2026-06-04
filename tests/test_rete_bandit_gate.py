#!/usr/bin/env python3
from __future__ import annotations

from ALGOS import minhash
from ALGOS import runtime_caps
from ALGOS import rete_bandit_gate as rbg


def test_execute_algorithm_minhash_uses_unique_shingles_once(monkeypatch):
    calls = []

    def fake_shingles(text, width):
        calls.append((text, width))
        return {"a b c d", "b c d e", "a b c d"}

    monkeypatch.setattr(minhash, "shingles", fake_shingles)
    packet = {"text_surface": "a b c d e f g h"}

    out = rbg.execute_algorithm("minhash", packet)

    assert calls and calls[0][1] == 5
    assert out["shingle_count"] == 2
    assert out["signature_head"]


def test_rete_prune_caps_payload_serialization(monkeypatch):
    seen = {}

    def fake_bounded_payload(payload, max_chars=runtime_caps.MAX_JSON_CHARS):
        seen["payload"] = payload
        seen["max_chars"] = max_chars
        return ("{" + "x" * (min(128, max_chars - 2)) + "}", False)

    monkeypatch.setattr(runtime_caps, "bounded_payload", fake_bounded_payload)
    payload = {"body": "x" * 12000}
    packet = {
        "text_surface": "evidence claim event",
        "payload": payload,
    }

    pool, hits, features = rbg.rete_prune(packet)

    assert pool
    assert seen["max_chars"] == runtime_caps.MAX_JSON_CHARS
    assert seen["payload"] is payload
