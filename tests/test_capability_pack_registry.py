from __future__ import annotations

import json
from pathlib import Path

from scripts import capability_pack_registry as cpr


def test_discover_pack_registries_finds_sio8() -> None:
    packs = cpr.discover_pack_registries()
    assert any(p.name == "registry.json" and p.parent.name == "sio8" for p in packs)


def test_pack_to_capability_row_maps_sio8_metadata() -> None:
    source = Path("BOOKS/ontology_packs/sio8/registry.json")
    pack = json.loads(source.read_text(encoding="utf-8"))
    row = cpr.pack_to_capability_row(pack, source)
    assert row["capability_key"] == "ontology-pack-sio8"
    assert row["capability_group"] == "Ontology Packs"
    assert row["capability_name"].startswith("SIO-8 Sovereign Intelligence Ontology")
    assert row["lifecycle_status"] == "prototype"
    assert row["detail"]["recommended_first_test"]["name"] == "targets_hooks_smoke"
    assert row["detail"]["pillar_count"] == 8


def test_run_register_dry_run_writes_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cpr, "OUT_DIR", tmp_path)
    payload = cpr.run_register(execute=False, database_url=None)
    assert payload["ok"] is True
    assert payload["pack_count"] >= 1
    receipt = Path(payload["receipt_path"])
    assert receipt.exists()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.capability_pack_registry.v1"
    assert data["db_result"]["attempted"] is False


def test_upsert_capability_rows_issues_expected_sql(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            captured["sql"] = sql
            captured["params"] = params

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            captured["committed"] = True

    row = {
        "capability_key": "ontology-pack-sio8",
        "capability_group": "Ontology Packs",
        "capability_name": "SIO-8 Sovereign Intelligence Ontology [0.1-staged]",
        "lifecycle_status": "prototype",
        "run_state": "planned",
        "workflow_name": "ontology-pack-registry",
        "command": "scripts/capability_pack_registry.py register",
        "detail": {"pack_id": "sio8"},
    }
    count = cpr.upsert_capability_rows(FakeConn(), [row])
    assert count == 1
    assert "ON CONFLICT (capability_key) DO UPDATE SET" in captured["sql"]
    assert json.loads(captured["params"]["detail"]) == {"pack_id": "sio8"}
    assert captured["committed"] is True
