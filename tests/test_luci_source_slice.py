from __future__ import annotations

import json
from pathlib import Path

from scripts.luci_operator import is_source_prompt
from scripts.luci_source_slice import HackerNewsAdapter, choose_adapter, score_item


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_source_prompt_detection_and_adapter_choice():
    assert is_source_prompt("study the live world source adapters for Hacker News and arXiv")
    assert choose_adapter("study current world source adapters", None)[0] == "hn"
    assert choose_adapter("study arxiv preprints", None)[0] == "arxiv"
    assert choose_adapter("study reddit source", None)[0] == "reddit"


def test_hacker_news_adapter_fetches_and_normalizes_items(monkeypatch):
    from scripts import luci_source_slice as slice_mod

    def fake_urlopen(req, timeout=0):
        url = getattr(req, "full_url", req)
        if url.endswith("/topstories.json"):
            return _FakeResponse(json.dumps([101]).encode("utf-8"))
        if url.endswith("/item/101.json"):
            return _FakeResponse(
                json.dumps(
                    {
                        "id": 101,
                        "title": "Treelite routers are useful",
                        "url": "https://example.com/story",
                        "by": "operator",
                        "time": 1710000000,
                        "score": 99,
                        "descendants": 12,
                    }
                ).encode("utf-8")
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(slice_mod.urllib.request, "urlopen", fake_urlopen)
    items = HackerNewsAdapter().fetch("router", limit=1)
    assert len(items) == 1
    item = items[0]
    assert item.source_kind == "hn"
    assert item.title == "Treelite routers are useful"
    assert item.url == "https://example.com/story"
    assert item.claim_text == "Treelite routers are useful"
    assert item.metadata["score"] == 99
    score = score_item("treelite routers", item)
    assert score["score"] >= 0
    assert score["verdict"] in {"promote", "archive"}
