from __future__ import annotations

import json
from pathlib import Path


def test_root_rotor_seed_builder_creates_roots_and_one_file_node_per_manifest_entry(tmp_path: Path) -> None:
    import scripts.root_rotor_seed_bible_nodes as seed

    manifest = {
        "files": [
            {"path": "06_SCHEMA/144_canonical_technical_bible.sql", "sha256": "a" * 64, "size_bytes": 10, "bytes_read": 10, "truncated": False},
            {"path": "scripts/lucidota_bge_fleet.sh", "sha256": "b" * 64, "size_bytes": 20, "bytes_read": 20, "truncated": False},
            {"path": "ALGOS/bandit_router.py", "sha256": "c" * 64, "size_bytes": 30, "bytes_read": 30, "truncated": False},
            {"path": "luci", "sha256": "d" * 64, "size_bytes": 40, "bytes_read": 40, "truncated": False},
            {"path": "GOALS/CURRENT_HANDOFF.md", "sha256": "e" * 64, "size_bytes": 50, "bytes_read": 50, "truncated": False},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    nodes = seed.build_seed_nodes(path)
    file_nodes = [n for n in nodes if n["payload_format"] == "json"]

    assert len([n for n in nodes if n["parent_id"] is None]) == 5
    assert len(file_nodes) == 5
    assert {n["manual_id"] for n in file_nodes} == {"SYSTEM_ARCH", "RUNTIME_GOVERNOR", "AVIONICS", "FLIGHT_MAN", "LEDGER"}
    assert all(n["status"] == "draft" for n in file_nodes)
    assert all(n["node_kind"] for n in file_nodes)
    assert all(n["ontology_tags"] for n in file_nodes)
    assert all(n["source_refs"] for n in file_nodes)
    assert all(n["evidence_hashes"] for n in file_nodes)
    assert file_nodes[0]["node_id"] == "1.1.0"
    assert file_nodes[-1]["node_id"] == "5.1.0"
    by_path = {n["source_refs"][0]: n for n in file_nodes}
    assert by_path["06_SCHEMA/144_canonical_technical_bible.sql"]["node_kind"] == "SCHEMA"
    assert "OBJECT" in by_path["06_SCHEMA/144_canonical_technical_bible.sql"]["ontology_tags"]
    assert by_path["scripts/lucidota_bge_fleet.sh"]["node_kind"] == "WORKFLOW"
    assert "WORKFLOW" in by_path["scripts/lucidota_bge_fleet.sh"]["ontology_tags"]
    assert by_path["ALGOS/bandit_router.py"]["node_kind"] == "ALGORITHM"


def test_root_rotor_seed_payload_is_machine_readable_and_pending_model_analysis(tmp_path: Path) -> None:
    import scripts.root_rotor_seed_bible_nodes as seed

    manifest = {"files": [{"path": "scripts/example.py", "sha256": "f" * 64, "size_bytes": 123, "bytes_read": 100, "truncated": True}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    node = next(n for n in seed.build_seed_nodes(path) if n["payload_format"] == "json")
    payload = json.loads(node["payload"])

    assert payload["state"] == "pending_dedicated_model_analysis"
    assert payload["source_path"] == "scripts/example.py"
    assert payload["source_sha256"] == "f" * 64
    assert payload["bytes_read"] == 100
    assert payload["truncated"] is True
    assert payload["required_model_call"] is True
    assert payload["target_output_contract"] == "lucidota.root_rotor.bible_node_payload.v1"
    assert payload["node_kind"] == "WORKFLOW"
    assert "OBJECT" in payload["ontology_tags"]


def test_root_rotor_seed_can_deprecate_stale_manifest_sources() -> None:
    import scripts.root_rotor_seed_bible_nodes as seed

    nodes = [
        {"source_refs": ["scripts/a.py"]},
        {"source_refs": ["06_SCHEMA/x.sql"]},
        {"source_refs": []},
    ]

    assert seed.current_manifest_sources(nodes) == {"scripts/a.py", "06_SCHEMA/x.sql"}
    sql = seed.retire_stale_sql()
    assert "UPDATE lucidota_canon.bible_nodes" in sql
    assert "status = 'deprecated'" in sql
    assert "valid_to = now()" in sql
    assert "source_refs->>0 <> ALL" in sql


def test_root_rotor_seed_upsert_preserves_verified_model_sidecars() -> None:
    import scripts.root_rotor_seed_bible_nodes as seed

    sql = seed.upsert_seed_sql()

    assert "ON CONFLICT (node_id) DO UPDATE SET" in sql
    assert "WHERE lucidota_canon.bible_nodes.status IN ('draft', 'deprecated')" in sql
    assert "lucidota_canon.bible_nodes.payload_format = 'json'" in sql
    assert "payload::jsonb->>'schema' = 'lucidota.root_rotor.file_seed_payload.v1'" in sql
    assert "node_kind = EXCLUDED.node_kind" in sql
    assert "ontology_tags = EXCLUDED.ontology_tags" in sql


def test_root_rotor_seed_can_deprecate_duplicate_source_coordinates() -> None:
    import scripts.root_rotor_seed_bible_nodes as seed

    sql = seed.retire_duplicate_sources_sql()

    assert "row_number() OVER" in sql
    assert "PARTITION BY source_refs->>0" in sql
    assert "status = 'deprecated'" in sql
    assert "ranked.keep_rank > 1" in sql
