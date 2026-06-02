#!/usr/bin/env python3
"""Audit Root-Rotor JSON sidecars against the active-source manifest.

The sidecar JSON is the fast summary layer. This script checks it against the
hash-addressed manifest and flags anomalies before sidecars become law.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "GOALS" / "ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_DUMP.manifest.json"
DEFAULT_NODE_DIR = ROOT / "05_OUTPUTS" / "root_rotor_nodes"
DEFAULT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "root_rotor_manuals"
TARGET_SCHEMA = "lucidota.root_rotor.bible_node_payload.v1"
LAW_OF_ROOT_NODE_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sidecars(node_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if not node_dir.exists():
        return valid, invalid
    for path in sorted(node_dir.glob("*.json")):
        try:
            obj = load_json(path)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc), "kind": "unreadable_json"})
            continue
        if obj.get("schema") != TARGET_SCHEMA:
            invalid.append({"path": str(path), "schema": obj.get("schema"), "kind": "wrong_schema"})
            continue
        obj["_sidecar_path"] = str(path)
        valid.append(obj)
    return valid, invalid


def _top_prefix(path: str) -> str:
    if "/" not in path:
        return path
    return path.split("/", 1)[0] + "/"


def audit_sidecars(manifest: dict[str, Any], sidecars: list[dict[str, Any]], invalid_sidecars: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_files = {str(entry["path"]): entry for entry in manifest.get("files", [])}
    manifest_paths = set(manifest_files)
    sidecars_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sidecar in sidecars:
        sidecars_by_source[str(sidecar.get("source_path", ""))].append(sidecar)

    sidecar_sources = set(sidecars_by_source)
    missing = sorted(manifest_paths - sidecar_sources)
    unknown = sorted(sidecar_sources - manifest_paths)
    duplicates = sorted(source for source, rows in sidecars_by_source.items() if len(rows) > 1)

    hash_mismatches: list[dict[str, Any]] = []
    missing_required_fields: list[dict[str, Any]] = []
    symbolic_edge_values: list[dict[str, Any]] = []
    placeholder_payloads: list[dict[str, Any]] = []
    required = ["source_path", "source_sha256", "node_title", "what_it_is_and_does", "payload_asd_ste100"]

    for sidecar in sidecars:
        source_path = str(sidecar.get("source_path", ""))
        manifest_entry = manifest_files.get(source_path)
        if manifest_entry and sidecar.get("source_sha256") != manifest_entry.get("sha256"):
            hash_mismatches.append({
                "source_path": source_path,
                "sidecar_sha256": sidecar.get("source_sha256"),
                "manifest_sha256": manifest_entry.get("sha256"),
                "sidecar_path": sidecar.get("_sidecar_path"),
            })
        missing_fields = [field for field in required if not sidecar.get(field)]
        if missing_fields:
            missing_required_fields.append({
                "source_path": source_path,
                "missing_fields": missing_fields,
                "sidecar_path": sidecar.get("_sidecar_path"),
            })
        for field in ["dependencies", "affects_nodes"]:
            for value in sidecar.get(field) or []:
                value_s = str(value)
                if not LAW_OF_ROOT_NODE_RE.match(value_s):
                    symbolic_edge_values.append({
                        "source_path": source_path,
                        "field": field,
                        "value": value_s,
                        "sidecar_path": sidecar.get("_sidecar_path"),
                    })
        if "pending_dedicated_model_analysis" in json.dumps(sidecar, sort_keys=True):
            placeholder_payloads.append({
                "source_path": source_path,
                "sidecar_path": sidecar.get("_sidecar_path"),
            })

    excluded_dir_names = set(manifest.get("excluded_dirs") or [])
    included_prefixes_under_excluded_dirs = [
        prefix for prefix in manifest.get("included_prefixes") or []
        if Path(str(prefix)).parts and Path(str(prefix)).parts[0] in excluded_dir_names
    ]
    files_under_excluded_dirs = [
        path for path in manifest_paths
        if Path(path).parts and Path(path).parts[0] in excluded_dir_names
    ]

    manifest_by_top_prefix = Counter(_top_prefix(path) for path in manifest_paths)
    sidecars_by_top_prefix = Counter(_top_prefix(path) for path in sidecar_sources)

    blockers: list[str] = []
    warnings: list[str] = []
    if missing:
        blockers.append("sidecar_coverage_incomplete")
    if unknown:
        blockers.append("sidecar_source_not_in_manifest")
    if duplicates:
        blockers.append("duplicate_sidecars_for_source")
    if hash_mismatches:
        blockers.append("sidecar_hash_mismatch")
    if missing_required_fields:
        blockers.append("sidecar_required_fields_missing")
    if invalid_sidecars:
        blockers.append("invalid_sidecar_json")
    if files_under_excluded_dirs:
        blockers.append("manifest_contains_excluded_dir_files")
    if symbolic_edge_values:
        warnings.append("sidecar_edge_symbol_anomaly")
    if placeholder_payloads:
        warnings.append("sidecar_placeholder_payload")
    if included_prefixes_under_excluded_dirs:
        warnings.append("included_prefix_under_excluded_dir")

    metrics = {
        "manifest_files": len(manifest_paths),
        "valid_sidecars": len(sidecars),
        "invalid_sidecars": len(invalid_sidecars),
        "missing_sidecars": len(missing),
        "unknown_source_sidecars": len(unknown),
        "duplicate_source_sidecars": len(duplicates),
        "hash_mismatches": len(hash_mismatches),
        "missing_required_field_sidecars": len(missing_required_fields),
        "symbolic_edge_values": len(symbolic_edge_values),
        "placeholder_payloads": len(placeholder_payloads),
        "coverage_ratio": round((len(manifest_paths & sidecar_sources) / len(manifest_paths)) if manifest_paths else 1.0, 6),
        "manifest_by_top_prefix": dict(sorted(manifest_by_top_prefix.items())),
        "sidecars_by_top_prefix": dict(sorted(sidecars_by_top_prefix.items())),
    }
    anomalies = {
        "missing_sidecars_sample": missing[:50],
        "unknown_source_sidecars": unknown[:50],
        "duplicate_source_sidecars": duplicates[:50],
        "hash_mismatches": hash_mismatches[:50],
        "missing_required_fields": missing_required_fields[:50],
        "symbolic_edge_values": symbolic_edge_values[:50],
        "placeholder_payloads": placeholder_payloads[:50],
        "invalid_sidecars": invalid_sidecars[:50],
        "included_prefixes_under_excluded_dirs": included_prefixes_under_excluded_dirs,
        "files_under_excluded_dirs_sample": files_under_excluded_dirs[:50],
    }
    return {
        "metrics": metrics,
        "anomalies": anomalies,
        "blockers": blockers,
        "warnings": warnings,
        "verdict": "PASS" if not blockers else "FAIL",
    }


def run_audit(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    node_dir: Path = DEFAULT_NODE_DIR,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
    write_receipt: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    sidecars, invalid = load_sidecars(node_dir)
    audit = audit_sidecars(manifest, sidecars, invalid)
    result = {
        "schema": "lucidota.root_rotor.sidecar_anomaly_audit.v1",
        "generated_at": now(),
        "manifest_path": str(manifest_path),
        "node_dir": str(node_dir),
        "why_this_exists": "The manifest is the sanitized active-source sweep. Each manifest file must receive one JSON sidecar before it can become a DB-coordinate manual node.",
        **audit,
    }
    if write_receipt:
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = receipt_dir / f"sidecar_anomaly_audit_{result['generated_at'].replace(':', '').replace('-', '')}.json"
        result["receipt_path"] = str(receipt_path)
        receipt_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Root-Rotor sidecar coverage and anomalies.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--node-dir", default=str(DEFAULT_NODE_DIR))
    parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        manifest_path=Path(args.manifest),
        node_dir=Path(args.node_dir),
        receipt_dir=Path(args.receipt_dir),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"ROOT_ROTOR_SIDECAR_AUDIT={result['verdict']}")
        print(f"BLOCKERS={','.join(result['blockers'])}")
        print(f"RECEIPT={result.get('receipt_path', '')}")
    return 0 if result["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
