from __future__ import annotations

import json
from pathlib import Path


def test_compile_canonical_technical_bible_renders_manual_from_nodes(tmp_path: Path) -> None:
    import scripts.compile_canonical_technical_bible as compiler

    nodes = [
        {
            "node_id": "1.0.0",
            "manual_id": "SYSTEM_ARCH",
            "title": "System Architecture",
            "payload": "The system stores canon in database nodes.",
            "status": "verified",
            "version": 1,
            "hash_current": "a" * 64,
            "previous_hash": None,
            "source_refs": ["06_SCHEMA/144_canonical_technical_bible.sql"],
            "dependencies": [],
            "affects_nodes": ["4.1.0"],
        },
        {
            "node_id": "1.1.0",
            "manual_id": "SYSTEM_ARCH",
            "title": "Node Coordinates",
            "payload": "Each rule has one node id.",
            "status": "review_required",
            "version": 3,
            "hash_current": "b" * 64,
            "previous_hash": "a" * 64,
            "source_refs": [],
            "dependencies": ["1.0.0"],
            "affects_nodes": [],
        },
    ]

    rendered = compiler.render_manual("SYSTEM_ARCH", nodes)

    assert rendered.startswith("# System Arch")
    assert "Effective Manual Version: v3" in rendered
    assert "## 1.0.0 System Architecture" in rendered
    assert "## 1.1.0 Node Coordinates" in rendered
    assert "Status: review_required | Version: v3" in rendered
    assert "Hash: bbbbbbbbbb" in rendered
    assert "Dependencies: 1.0.0" in rendered
    assert "Blast Radius Impact: 4.1.0" in rendered


def test_compile_canonical_technical_bible_fetches_and_writes_receipt(monkeypatch, tmp_path: Path) -> None:
    import scripts.compile_canonical_technical_bible as compiler

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return [
                {
                    "node_id": "2.0.0",
                    "manual_id": "RUNTIME_GOVERNOR",
                    "title": "Runtime Governor",
                    "payload": "The governor uses Linux controls.",
                    "status": "verified",
                    "version": 2,
                    "hash_current": "c" * 64,
                    "previous_hash": None,
                    "source_refs": [],
                    "dependencies": [],
                    "affects_nodes": [],
                }
            ]

    calls: list[str] = []

    def fake_get(url: str, timeout: float):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(compiler.requests, "get", fake_get)
    result = compiler.compile_manuals(
        postgrest_url="http://localhost:3000",
        manual_ids=["RUNTIME_GOVERNOR"],
        output_dir=tmp_path / "compiled",
        receipt_dir=tmp_path / "receipts",
        timeout=0.5,
    )

    assert calls == ["http://localhost:3000/api_bible_nodes?manual_id=eq.RUNTIME_GOVERNOR&order=node_sort_key.asc"]
    assert result["schema"] == "lucidota.root_rotor.compile_manuals.v1"
    assert result["manuals_compiled"] == 1
    manual_path = Path(result["manuals"][0]["path"])
    assert manual_path.exists()
    assert len(result["manuals"][0]["sha256"]) == 64
    receipt_path = Path(result["receipt_path"])
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["manuals_compiled"] == 1


def test_compile_canonical_technical_bible_can_require_postgrest_readiness(monkeypatch, tmp_path: Path) -> None:
    import scripts.compile_canonical_technical_bible as compiler

    monkeypatch.setattr(compiler.control, "wait_for_readiness", lambda **_kwargs: {
        "ready": False,
        "admin_ready": False,
        "api_ready": False,
        "admin_last_error": "ConnectionError",
        "api_last_error": "ConnectionError",
    })

    result = compiler.compile_manuals(
        postgrest_url="http://127.0.0.1:3000",
        manual_ids=["SYSTEM_ARCH"],
        output_dir=tmp_path / "manuals",
        receipt_dir=tmp_path / "receipts",
        require_readiness=True,
    )

    assert result["status"] == "FAIL"
    assert result["errors"][0]["error"] == "postgrest_readiness_blocked"


def test_compile_canonical_technical_bible_can_compile_from_db_fetch(monkeypatch, tmp_path: Path) -> None:
    import scripts.compile_canonical_technical_bible as compiler

    def fake_fetch_nodes_db(dsn: str, manual_id: str):
        assert dsn == "postgresql:///lucidota_state"
        assert manual_id == "LEDGER"
        return [
            {
                "node_id": "5.0.0",
                "manual_id": "LEDGER",
                "title": "Ledger",
                "payload": "Receipts are events.",
                "status": "draft",
                "version": 1,
                "hash_current": "d" * 64,
                "previous_hash": None,
                "source_refs": [],
                "dependencies": [],
                "affects_nodes": [],
                "node_sort_key": [5, 0, 0],
            }
        ]

    monkeypatch.setattr(compiler, "fetch_nodes_db", fake_fetch_nodes_db)
    result = compiler.compile_manuals_db(
        dsn="postgresql:///lucidota_state",
        manual_ids=["LEDGER"],
        output_dir=tmp_path / "manuals",
        receipt_dir=tmp_path / "receipts",
    )

    assert result["schema"] == "lucidota.root_rotor.compile_manuals.v1"
    assert result["source"] == "postgres"
    assert result["manuals_compiled"] == 1
    assert Path(result["manuals"][0]["path"]).exists()
