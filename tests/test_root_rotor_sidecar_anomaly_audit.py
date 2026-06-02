from __future__ import annotations

import json
from pathlib import Path


def write_manifest(path: Path) -> None:
    manifest = {
        "schema": "lucidota.root_rotor.audit_dump.v1",
        "excluded_dirs": ["KRAMPUSCHEWING", "03_VAULT"],
        "included_prefixes": ["01_REPOS/claudecode/", "KRAMPUSCHEWING/dirty/"],
        "dirty_nested_repo_prefixes": ["01_REPOS/claudecode/", "KRAMPUSCHEWING/dirty/"],
        "files_written": 2,
        "files": [
            {"path": "scripts/a.py", "sha256": "a" * 64, "size_bytes": 10, "bytes_read": 10, "truncated": False},
            {"path": "06_SCHEMA/x.sql", "sha256": "b" * 64, "size_bytes": 20, "bytes_read": 20, "truncated": False},
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_sidecar_anomaly_audit_reports_coverage_hash_and_symbolic_edge_anomalies(tmp_path: Path) -> None:
    import scripts.root_rotor_sidecar_anomaly_audit as audit

    manifest = tmp_path / "manifest.json"
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    write_manifest(manifest)
    (nodes / "a.json").write_text(json.dumps({
        "schema": "lucidota.root_rotor.bible_node_payload.v1",
        "source_path": "scripts/a.py",
        "source_sha256": "WRONG",
        "node_title": "A",
        "what_it_is_and_does": "A runs.",
        "payload_asd_ste100": "A runs.",
        "dependencies": ["MISTRAL_API_KEY", "1.0.0"],
        "affects_nodes": ["not-a-node"],
    }), encoding="utf-8")

    result = audit.run_audit(manifest_path=manifest, node_dir=nodes, write_receipt=False)

    assert result["verdict"] == "FAIL"
    assert result["metrics"]["manifest_files"] == 2
    assert result["metrics"]["valid_sidecars"] == 1
    assert result["metrics"]["missing_sidecars"] == 1
    assert "sidecar_hash_mismatch" in result["blockers"]
    assert "sidecar_edge_symbol_anomaly" in result["warnings"]
    assert result["anomalies"]["hash_mismatches"][0]["source_path"] == "scripts/a.py"
    assert result["anomalies"]["symbolic_edge_values"][0]["value"] == "MISTRAL_API_KEY"
    assert result["anomalies"]["included_prefixes_under_excluded_dirs"][0] == "KRAMPUSCHEWING/dirty/"


def test_sidecar_anomaly_audit_passes_when_every_manifest_file_has_valid_sidecar(tmp_path: Path) -> None:
    import scripts.root_rotor_sidecar_anomaly_audit as audit

    manifest = tmp_path / "manifest.json"
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    write_manifest(manifest)
    for source_path, sha in [("scripts/a.py", "a" * 64), ("06_SCHEMA/x.sql", "b" * 64)]:
        (nodes / f"{source_path.replace('/', '_')}.json").write_text(json.dumps({
            "schema": "lucidota.root_rotor.bible_node_payload.v1",
            "source_path": source_path,
            "source_sha256": sha,
            "node_title": source_path,
            "what_it_is_and_does": "This file exists in the active audit.",
            "payload_asd_ste100": "This file exists in the active audit.",
            "dependencies": ["1.0.0"],
            "affects_nodes": [],
        }), encoding="utf-8")

    result = audit.run_audit(manifest_path=manifest, node_dir=nodes, write_receipt=False)

    assert result["verdict"] == "PASS"
    assert result["blockers"] == []
    assert result["metrics"]["coverage_ratio"] == 1.0
